"""Predict rotating native MRI planes with the supplied TC, reconstruct in MNI,
then integrate CEM43 from the reconstructed temperature sequence.

All outputs are new artifacts; source images and trained weights are untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label, map_coordinates, distance_transform_edt
from cooling import estimate_cooling, combine_plane_estimates, bounded_cooling


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def load_mask(path):
    path = Path(path)
    with path.open('rb') as f:
        signature = f.read(2)
    return nib.load(str(path)) if signature == b'\x1f\x8b' else nib.Nifti1Image.from_bytes(path.read_bytes())


def save_nifti(path, array, reference, dtype):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(array, dtype=dtype), reference.affine, header)
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    nib.save(image, str(path))


def dose_interval(t0, t1, seconds):
    """Exact integral of standard CEM43 rate for a linear temperature ramp.

    R=0.25 below 43C, R=0.5 at/above 43C; split crossing intervals at 43C.
    Dose is in minutes. Temperatures and doses are not artificially capped.
    """
    t0, t1 = np.asarray(t0, np.float64), np.asarray(t1, np.float64)
    lo, hi = np.minimum(t0, t1), np.maximum(t0, t1)
    width = hi - lo
    stationary = width < 1e-8
    out = np.zeros_like(lo)
    for lower, upper, slope in [(lo, np.minimum(hi, 43.), math.log(4.)),
                                (np.maximum(lo, 43.), hi, math.log(2.))]:
        active = (~stationary) & (upper > lower)
        exponent = slope * (upper[active] - 43.)
        if np.any(exponent > 700):
            raise FloatingPointError('CEM43 exponent exceeds float64 capacity; temperature predictions are unusable.')
        out[active] += np.exp(exponent) * (-np.expm1(-slope * (upper[active] - lower[active]))) / (slope * width[active])
    slope = np.where(lo[stationary] < 43., math.log(4.), math.log(2.))
    out[stationary] = np.exp(slope * (lo[stationary] - 43.))
    result = out * float(seconds) / 60.
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite thermal dose.')
    return result


def dicom_point(ds, center_rc):
    iop = np.asarray(ds.ImageOrientationPatient, np.float64)
    spacing = np.asarray(ds.PixelSpacing, np.float64)
    return np.asarray(ds.ImagePositionPatient, np.float64) + center_rc[0] * spacing[0] * iop[3:] + center_rc[1] * spacing[1] * iop[:3]


def sample(array, xyz, order=1, cval=0.):
    return map_coordinates(array, np.moveaxis(xyz, -1, 0), order=order,
                           mode='constant', cval=cval, prefilter=False)


def measured_baseline_on_points(points_lps, arrays, references, default_temperature):
    """Use the trajectory's available measured baseline plane(s) for its start."""
    numerator = np.zeros(points_lps.shape[:-1], np.float64)
    denominator = np.zeros_like(numerator)
    for a, ds in zip(arrays, references):
        iop = np.asarray(ds.ImageOrientationPatient, np.float64)
        spacing = np.asarray(ds.PixelSpacing, np.float64)
        delta = points_lps - np.asarray(ds.ImagePositionPatient, np.float64)
        coords = np.stack([(delta @ iop[3:]) / spacing[0], (delta @ iop[:3]) / spacing[1]])
        valid = a['foreground_mask'].astype(bool) & np.isfinite(a['baseline_C']) & (a['baseline_C'] > 0)
        support = map_coordinates(valid.astype(np.float32), coords, order=1, mode='constant', cval=0., prefilter=False)
        temperature = map_coordinates(a['baseline_C'] * valid, coords, order=1, mode='constant', cval=0., prefilter=False)
        normal = np.cross(iop[:3], iop[3:]); normal /= np.linalg.norm(normal)
        distance = np.abs(delta @ normal)
        weight = 1. / (distance ** 2 + .01)
        numerator += weight * temperature
        denominator += weight * support
    return np.divide(numerator, denominator, out=np.full_like(numerator, default_temperature), where=denominator > 1e-6).astype(np.float32)


def apply_matrix(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def fan_profiles(predictions, center_rc, axis_dimension):
    # Arrange each plane as [radial, axial], retaining fractional center indices.
    planes = predictions if axis_dimension == 1 else predictions.transpose(0, 2, 1)
    radial_center = center_rc[1 - axis_dimension]
    radius = int(math.floor(min(radial_center, planes.shape[1] - 1 - radial_center)))
    rr, aa = np.meshgrid(np.arange(radius + 1), np.arange(planes.shape[2]), indexing='ij')
    profiles = [map_coordinates(p, [radial_center + sign * rr, aa], order=1, mode='nearest', prefilter=False)
                for sign in (1, -1) for p in planes]
    profiles.append(profiles[0])
    return np.stack(profiles), radius


def reconstruction_coordinates(points_lps, pivot, axis, radial0, radial1, center_rc, spacing_rc, axis_dimension, nplanes):
    disp = points_lps - pivot
    rx, ry = disp @ radial0, disp @ radial1
    radius = np.sqrt(rx * rx + ry * ry) / spacing_rc[1 - axis_dimension]
    theta = np.mod(np.arctan2(ry, rx), 2 * math.pi) / (math.pi / nplanes)
    axial = center_rc[axis_dimension] + (disp @ axis) / spacing_rc[axis_dimension]
    maximum = math.floor(min(center_rc[1 - axis_dimension], 511 - center_rc[1 - axis_dimension]))
    valid = (radius <= maximum) & (axial >= 0) & (axial <= 511)
    return np.stack([theta, radius, axial]), valid


def rate_stats(values):
    return {'min': float(np.min(values)), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(np.max(values))}


def region_stats(dose, region, threshold):
    values = dose[region]
    if not len(values): return {'voxels': 0}
    return {'voxels': int(len(values)), 'volume_mL': len(values) / 1000.,
            'CEM43_min_stats': rate_stats(values),
            'fraction_ge_threshold': float(np.mean(values >= threshold)),
            'volume_ge_threshold_mL': float(np.sum(values >= threshold) / 1000.)}


def heat_rgba(dose, threshold):
    import matplotlib.pyplot as plt
    out = np.zeros((*dose.shape, 4), np.float32)
    low = dose < threshold
    strength = np.log1p(np.maximum(dose, 0)) / np.log1p(threshold)
    out[low] = plt.get_cmap('Greens')(.2 + .75 * np.clip(strength[low], 0, 1))
    high = np.log10(np.maximum(dose[~low], threshold) / threshold + 1)
    out[~low] = plt.get_cmap('Reds')(.3 + .68 * np.clip(high / 4, 0, 1))
    out[..., 3] = .65
    out[dose < .01, 3] = 0
    return out


def qc_figure(path, mri, dose, temperature, component, points_lps, axes, mni, threshold, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    center = points_lps.mean(0)
    axis, radial0, radial1 = axes
    inv = np.linalg.inv(mni.affine)
    rr, cc = np.meshgrid(np.arange(-55., 56.), np.arange(-70., 71.), indexing='ij')
    fig, axs = plt.subplots(3, 3, figsize=(12, 11))
    fig.subplots_adjust(left=.015, right=.925, bottom=.025, top=.865, wspace=.035, hspace=.28)
    temperature_artist = None
    for row, degrees in enumerate([0, 60, 120]):
        angle = np.deg2rad(degrees)
        radial = np.cos(angle) * radial0 + np.sin(angle) * radial1
        lps = center + rr[..., None] * radial + cc[..., None] * axis
        ras = lps * np.array([-1., -1., 1.])
        coords = apply_matrix(ras, inv)
        mr, dd, tt = sample(mri, coords), sample(dose, coords), sample(temperature, coords)
        mask = sample(component.astype(np.uint8), coords, order=0) > 0
        for col, name in enumerate(['MRI + supplied TC', 'Final temperature (C)', 'CEM43 cumulative (min)']):
            ax = axs[row, col]
            ax.imshow(mr, cmap='gray', vmin=0, vmax=np.percentile(mri[mri > 0], 99))
            if col == 1:
                v = np.ma.masked_where((mr <= 0) | (tt < 37), tt)
                im = ax.imshow(v, cmap='RdYlGn_r', vmin=35, vmax=max(85,float(np.ceil(temperature[component].max()/5)*5)), alpha=.75)
                temperature_artist = im
            elif col == 2:
                ax.imshow(heat_rgba(dd, threshold))
                if dd.min() <= threshold <= dd.max(): ax.contour(dd, [threshold], colors='red', linewidths=.8)
            if mask.any(): ax.contour(mask, [.5], colors='cyan', linewidths=1.)
            ax.axhline(55, color='yellow', linestyle='--', linewidth=.65, alpha=.7)
            ax.set_title(f'{degrees} degrees | {name}', fontsize=8, pad=5)
            ax.axis('off')
    fig.suptitle(title + f'\nPlanes contain fiber axis (yellow dashed); cyan = TC; green < {threshold:g} min, red >= {threshold:g} min', fontsize=10)
    cax = fig.add_axes([.943, .29, .014, .40])
    fig.colorbar(temperature_artist, cax=cax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    if args.normalize_only:
        im = load_mask(args.mask)
        a = np.asarray(im.dataobj)
        assert np.isfinite(a).all() and set(np.unique(a)) <= {0, 1}
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f'{args.case_id}_TC_mni152.nii.gz'
        save_nifti(path, a, im, np.uint8)
        write_json(args.output / 'mask_source.json', {'original_file': str(args.mask),
                   'original_sha256': hashlib.sha256(args.mask.read_bytes()).hexdigest(),
                   'original_is_gzip': args.mask.read_bytes()[:2] == b'\x1f\x8b',
                   'normalized_file': str(path), 'shape': list(im.shape), 'affine': im.affine.tolist()})
        print(path, flush=True)
        return
    import ants
    import pydicom
    import torch
    import torch.nn.functional as F

    args.output.mkdir(parents=True, exist_ok=True)
    case = Path(os.environ.get('TEMP_NET_CASE_DIR', '.'))
    data = Path(os.environ.get('TEMP_NET_TRAINING_DATA', '.'))
    segments_root = Path(os.environ.get('TEMP_NET_SEGMENTS_ROOT', '.'))
    overview = Path(os.environ.get('TEMP_NET_OVERVIEW_ROOT', '.'))
    trajectory_lesions = [('A', 0), ('B', 1)]
    trajectory_planes = {'A': [2], 'B': [1, 2]}
    model_dir = Path(os.environ.get('TEMP_NET_MODEL_ROOT', '.'))
    sys.path.insert(0, str(model_dir))
    from model import DynamicsUNet
    geom_path = Path(os.environ.get('TEMP_NET_GEOMETRY', Path(__file__).resolve().with_name('test_rotating_planes_3d.py')))
    spec = importlib.util.spec_from_file_location('geometry', geom_path)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    norm = json.loads((data / 'normalization.json').read_text())
    mni_path, native_path = case / 'mni152/t2_1mm.nii.gz', case / 'native/t2_original.nii.gz'
    mni, native = nib.load(mni_path), nib.load(native_path)
    mri_mni, raw_native = np.asarray(mni.dataobj, np.float32), np.asarray(native.dataobj, np.float32)
    mask_image = load_mask(args.mask)
    if mask_image.shape != mni.shape or not np.allclose(mask_image.affine, mni.affine, atol=1e-4):
        raise ValueError('Supplied MNI mask does not match the registered T2 grid.')
    tc = np.asarray(mask_image.dataobj) > 0
    components, count = label(tc)
    sizes = np.bincount(components.ravel()); order = np.argsort(sizes[1:])[::-1] + 1
    if count != 2: raise ValueError(f'Expected exactly two TC components, found {count}.')
    # Components are assigned by the observed heat-source location, not size.
    # The order below must be independently checked against each trajectory.
    if sorted(args.component_order) != [1, 2]:
        raise ValueError('component-order must contain labels 1 and 2 exactly once.')
    component_masks = [components == n for n in args.component_order]
    for i, component in enumerate(component_masks, 1):
        save_nifti(args.output / f'lesion_{i:02d}_TC_mni152.nii.gz', component, mni, np.uint8)
    save_nifti(args.output / f'{args.case_id}_TC_mni152.nii.gz', tc, mni, np.uint8)
    save_nifti(args.output / f'{args.case_id}_T2_mni152.nii.gz', mri_mni, mni, np.float32)
    transform = case / 'transforms/native_to_mni152_0.mat'
    fixed, moving = ants.image_read(str(mni_path)), ants.image_read(str(native_path))
    native_tc = ants.apply_transforms(fixed=moving, moving=ants.image_read(str(args.mask)),
                transformlist=[str(transform)], whichtoinvert=[True], interpolator='genericLabel')
    ants.image_write(native_tc, str(args.output / f'{args.case_id}_TC_native.nii.gz'))
    brain_mni = np.asarray(nib.load(case / 'mni152/brain_mask.nii.gz').dataobj) > 0
    save_nifti(args.output / f'{args.case_id}_brain_mask_mni152.nii.gz', brain_mni, mni, np.uint8)
    valid_mni = brain_mni & np.isfinite(mri_mni) & (mri_mni > .02 * np.max(mri_mni))
    valid_mni |= tc
    mni_indices = np.argwhere(valid_mni).astype(np.float64)
    flat = np.flatnonzero(valid_mni)
    print(f'MNI brain voxels for 3D reconstruction: {len(flat)}', flush=True)
    # Obtain the image-resampling coordinate map through ANTs itself. This
    # avoids guessing the direction or LPS/RAS convention of the affine file.
    native_indices = []
    for dim in range(3):
        shape = [1, 1, 1]; shape[dim] = native.shape[dim]
        grid = np.broadcast_to(np.arange(native.shape[dim], dtype=np.float32).reshape(shape), native.shape).copy()
        coordinate_image = ants.from_numpy(grid, origin=moving.origin, spacing=moving.spacing, direction=moving.direction)
        warped = ants.apply_transforms(fixed=fixed, moving=coordinate_image, transformlist=[str(transform)], interpolator='linear')
        native_indices.append(warped.numpy().ravel()[flat])
        del grid, coordinate_image, warped
    native_indices = np.stack(native_indices, axis=1).astype(np.float64)
    good = np.all((native_indices > 1) & (native_indices < np.asarray(native.shape) - 2), axis=1)
    if not good.all(): raise ValueError('Some MNI brain/TC voxels are outside native-image coverage.')
    fit_indices = np.linspace(0, len(flat) - 1, 2048, dtype=int)
    coeff = np.linalg.lstsq(np.c_[mni_indices[fit_indices], np.ones(len(fit_indices))], native_indices[fit_indices], rcond=None)[0]
    voxel_map = np.eye(4); voxel_map[:3] = coeff.T
    residual = np.max(np.linalg.norm(apply_matrix(mni_indices, voxel_map) - native_indices, axis=1))
    if residual > .02: raise ValueError(f'Coordinate-map affine fit residual too large: {residual} native voxels')
    native_to_mni_index = np.linalg.inv(voxel_map)
    native_points_lps = apply_matrix(native_indices, native.affine) * np.array([-1., -1., 1.])
    component_native_centers = [native_points_lps[c.ravel()[flat]].mean(0) for c in component_masks]
    patient_temperature = geometry.read_patient_temperature(overview / f'{args.case_id}_trajA_normal/export_info')
    assignment = []
    for trajectory, lesion_index in trajectory_lesions:
        seed_points = []
        for plane in trajectory_planes[trajectory]:
            seed_file = data / 'npz' / f'{args.sample_prefix}_Traj{trajectory}_P{plane}_S001.npz'
            with np.load(seed_file, allow_pickle=False) as seed_npz:
                center_seed = seed_npz['probe_center_rc']
            ds_seed = pydicom.dcmread(str(segments_root / f'Traj{trajectory}/P{plane}/segments/segment-001/structural.dcm'), stop_before_pixels=True)
            seed_points.append(dicom_point(ds_seed, center_seed))
        seed_mean = np.mean(seed_points, axis=0)
        distances = [float(np.linalg.norm(seed_mean - center)) for center in component_native_centers]
        print(f'Traj{trajectory} seed-to-ordered-lesion distances mm: {distances}', flush=True)
        if int(np.argmin(distances)) != lesion_index:
            raise ValueError(f'{trajectory} heat-source seed does not match supplied component order: {distances}')
        assignment.append({'trajectory': trajectory, 'lesion': lesion_index + 1, 'connected_component': args.component_order[lesion_index],
                           'archived_seed_lps_mm': seed_mean.tolist(), 'distance_to_lesion_centers_mm': distances})
    print('Verified lesion assignment', json.dumps(assignment), flush=True)
    if args.preflight_only:
        write_json(args.output / 'lesion_assignment.json', {'assignment':assignment,'mni_voxel_to_native_voxel':voxel_map.tolist()})
        return
    state = np.zeros(mni.shape, np.float32); state[valid_mni] = patient_temperature
    cumulative = np.zeros(len(flat), np.float64)
    lesion_doses = [np.zeros(len(flat), np.float64), np.zeros(len(flat), np.float64)]
    coverage_union = np.zeros(len(flat), bool)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    model = DynamicsUNet(time_scale=135, width=24, hidden=64, local_hidden=16, enhanced=True).to(device)
    model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    del checkpoint
    report = {'patient': args.case_id, 'source_patient_folder': args.case_id, 'model_weights': str(args.weights),
              'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(), 'mask_input': str(args.mask),
              'patient_initial_temperature_C': patient_temperature, 'num_planes': args.num_planes,
              'frame_seconds': args.frame_seconds, 'CEM_threshold_min': args.threshold,
              'coordinate_fit_max_residual_native_voxels': float(residual),
              'mni_voxel_to_native_voxel': voxel_map.tolist(),
              'tumor_components_mni_voxels': [int(c.sum()) for c in component_masks],
              'model_patient_split': 'train', 'segments': [],
              'experiment': f'{args.case_id} corrected TC v2, measured inter-segment cooling applied before every inherited segment; A/B independent starts.',
              'cooling_policy': {'ROI':'active TC in same observed plane, valid foreground, previous real end > first real baseline + 2 C',
                'delta':'max(0, equal-plane mean of mean(previous measured end - current measured baseline))',
                'application':'T <- T - min(delta, max(T - first trajectory initial 3D temperature, 0)) within brain only',
                'waiting_dose':'excluded for comparability; no dose is subtracted',
                'uses_current_segment_heating_targets_for_cooling':False},
              'trajectory_available_reference_planes': trajectory_planes,
              'component_trajectory_assignment': assignment,
              'mask_input_mode': args.mask_input_mode,
              'assumptions': ['Prediction uses the supplied TC, not the old manual tumor mask.',
                'Each segment uses the reviewed probe center stored in the training-ready NPZ and center manifest.',
                'A and B independently start from their own measured initial baseline approximation. A uses only corrected P2; B uses P1/P2. No missing reference plane is fabricated.',
                'Within-trajectory measured cooling corrects the inherited initial temperature; waiting dose is still excluded. A/B remain independent.',
                'Twelve predicted 2D longitudinal planes are interpolated in cylindrical coordinates onto the MNI 3D brain grid; unobserved angles are model-generated, not measured.',
                'CEM43 is computed AFTER temperature reconstruction, using exact integration under a linear-between-frame temperature assumption.',
                'Trajectory doses are saved separately. Lesion1 dose equals A, lesion2 dose equals B; patient dose equals A+B. Per-lesion threshold statistics use its assigned trajectory, not the patient dose.',
                'No prediction-temperature clipping or dose saturation is applied; the explicitly recorded initial-state cooling correction is bounded by the initial thermal excess. Dose is float64.',
                'Supplied TC boundary is a segmentation estimate; reconstructed CEM43 is not a validated tissue-injury ground truth.']}
    rr, cc = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
    last_axes, last_entries = {}, {}
    for trajectory, lesion_index in trajectory_lesions:
        trajectory_dose = np.zeros(len(flat), np.float64)
        active_tc = component_masks[lesion_index]
        input_tc = active_tc if args.mask_input_mode == 'target' else tc
        trajectory_output = args.output / f'trajectory_{trajectory}'
        trajectory_output.mkdir(exist_ok=True)
        export = overview / f'{args.case_id}_traj{trajectory}_normal/export_info'
        trajectories, registrations = geometry.read_export_info(export)
        available_planes = trajectory_planes[trajectory]
        actual_dirs = sorted((segments_root / f'Traj{trajectory}' / f'P{available_planes[0]}/segments').glob('segment-*'))
        if not actual_dirs: raise FileNotFoundError(f'No source segments for Traj{trajectory}')
        fixed_selected = None
        previous_observations = {}
        initial_plane_baselines = {}
        first_initial_volume = None
        for directory in actual_dirs:
            reset_b_now = directory == actual_dirs[0]
            metadata = json.loads((directory / 'segment_metadata.json').read_text())
            source = metadata.get('source_record') or {
                'power_W': metadata['power_W'],
                'expected_temperature_T_range': metadata['source_range_T'],
                'source_clause': metadata.get('source_clause')}
            arrays, references, centers_lps = [], [], []
            for plane in available_planes:
                npz_file = data / 'npz' / f'{args.sample_prefix}_Traj{trajectory}_P{plane}_{directory.name.replace("segment-", "S")}.npz'
                if not npz_file.exists(): raise FileNotFoundError(npz_file)
                with np.load(npz_file, allow_pickle=False) as npz:
                    a = {k: npz[k] for k in ['probe_center_rc', 'target_time_indices', 'spacing_rc_mm', 'baseline_C', 'foreground_mask']}
                plane_dir = segments_root / f'Traj{trajectory}/P{plane}/segments/{directory.name}'
                ds = pydicom.dcmread(str(plane_dir / 'structural.dcm'), stop_before_pixels=True)
                # The NPZ center is the reviewed probe center. It is shared by
                # the training input, Gaussian map, and physical reconstruction.
                iop_cal = np.asarray(ds.ImageOrientationPatient, np.float64)
                sp_cal = np.asarray(ds.PixelSpacing, np.float64)
                pts_cal = np.asarray(ds.ImagePositionPatient, np.float64) + rr[..., None] * sp_cal[0] * iop_cal[3:] + cc[..., None] * sp_cal[1] * iop_cal[:3]
                native_xyz_cal = apply_matrix(pts_cal * [-1., -1., 1.], np.linalg.inv(native.affine))
                mni_xyz_cal = apply_matrix(native_xyz_cal, native_to_mni_index)
                projected_active = sample(active_tc.astype(np.uint8), mni_xyz_cal, order=0) > 0
                if not projected_active.any(): raise ValueError(f'Active lesion absent from {plane_dir}')
                center = np.asarray(a['probe_center_rc'], dtype=np.float32)
                if center.shape != (2,) or not np.isfinite(center).all():
                    raise ValueError(f'Invalid reviewed probe center in {npz_file}')
                a['mni_xyz_reference'] = mni_xyz_cal
                a['projected_active_tc'] = projected_active
                a['reference_plane'] = plane
                a['reference_directory'] = plane_dir
                if reset_b_now:
                    initial_plane_baselines[plane] = a['baseline_C'].copy()
                arrays.append(a); references.append(ds); centers_lps.append(dicom_point(ds, a['probe_center_rc']))
            before_cooling = state.copy()
            cooling_estimates = []
            if reset_b_now:
                cooling_delta, cooling_raw_mean = 0., 0.
            else:
                for a, ds in zip(arrays,references):
                    plane=a['reference_plane']; old=previous_observations[plane]
                    if not (np.allclose(old['IOP'],ds.ImageOrientationPatient,atol=1e-5)
                            and np.allclose(old['IPP'],ds.ImagePositionPatient,atol=1e-4)
                            and np.allclose(old['spacing'],ds.PixelSpacing,atol=1e-5)):
                        raise ValueError('Cooling observations must refer to the same physical plane and pixel grid')
                    estimate=estimate_cooling(old['temperature'],a['baseline_C'],initial_plane_baselines[plane],
                        a['projected_active_tc'],a['foreground_mask'],rise_threshold_C=2.)
                    current_meta=json.loads((a['reference_directory']/'segment_metadata.json').read_text())
                    def filename_seconds(name):
                        stamp=name.split('_')[1]
                        return int(stamp[:2])*3600+int(stamp[2:4])*60+float(stamp[4:6]+'.'+stamp[6:])
                    baseline_original=current_meta['original_filenames']['baseline']
                    estimate.update({'plane':plane,'previous_segment':old['segment'],
                        'previous_end_frame_original':old['original_frame'],'current_baseline_original':baseline_original,
                        'gap_seconds_from_filename':filename_seconds(baseline_original)-filename_seconds(old['original_frame'])})
                    cooling_estimates.append(estimate)
                cooling_delta,cooling_raw_mean=combine_plane_estimates(cooling_estimates)
                state=bounded_cooling(state,first_initial_volume,brain_mni,cooling_delta)
                if np.any(state>before_cooling+1e-5):raise AssertionError('Cooling increased a temperature')
                if not np.array_equal(state[~brain_mni],before_cooling[~brain_mni]):raise AssertionError('Cooling changed nonbrain voxels')
                print(f'Traj{trajectory}/{directory.name} initial cooling: raw={cooling_raw_mean:.4f}, applied={cooling_delta:.4f} C',flush=True)
            cooling_record={'trajectory':trajectory,'segment':directory.name,
                'first_segment_uses_measured_initial_baseline':reset_b_now,'raw_equal_plane_mean_drop_C':cooling_raw_mean,
                'applied_delta_C':cooling_delta,'plane_estimates':cooling_estimates,
                'brain_voxels_changed':int(np.count_nonzero(before_cooling!=state)),
                'waiting_dose_integrated':False,'cumulative_dose_subtracted':False}
            pivot = np.mean(centers_lps, axis=0)
            if fixed_selected is None:
                fixed_selected, _ = geometry.select_geometry(trajectories, registrations, pivot, f'Traj {trajectory}')
            axis = fixed_selected['target_lps_mm'] - fixed_selected['entry_lps_mm']; axis /= np.linalg.norm(axis)
            ds = references[0]; iop = np.asarray(ds.ImageOrientationPatient, np.float64)
            dirs = [iop[3:], iop[:3]]
            axis_dimension = int(np.argmax([abs(axis @ d) for d in dirs]))
            grid_alignment = float(abs(axis @ dirs[axis_dimension]))
            plane_normal = np.cross(iop[:3],iop[3:]); plane_normal /= np.linalg.norm(plane_normal)
            normal_component = float(abs(axis @ plane_normal))
            # An oblique IN-PLANE fiber is valid. The fan below is explicitly
            # resampled from native 3D anatomy in a fiber-aligned basis, rather
            # than assuming that the DICOM rows/columns already follow it.
            if normal_component > .1: raise ValueError(f'Fiber leaves observed reference plane: normal component={normal_component}')
            if axis @ dirs[axis_dimension] < 0: axis *= -1
            radial0 = dirs[1 - axis_dimension] - (dirs[1 - axis_dimension] @ axis) * axis; radial0 /= np.linalg.norm(radial0)
            radial1 = np.cross(axis, radial0); radial1 /= np.linalg.norm(radial1)
            center = arrays[0]['probe_center_rc'].astype(np.float64)
            spacing = np.asarray(ds.PixelSpacing, np.float64)
            angles = np.arange(args.num_planes) * math.pi / args.num_planes
            fan_raw, fan_base, fan_tc = [], [], []
            inactive_mask_pixels = 0
            for angle in angles:
                radial = math.cos(angle) * radial0 + math.sin(angle) * radial1
                row_direction, col_direction = (radial, axis) if axis_dimension == 1 else (axis, radial)
                pts = pivot + (rr - center[0])[..., None] * spacing[0] * row_direction + (cc - center[1])[..., None] * spacing[1] * col_direction
                native_xyz = apply_matrix(pts * np.array([-1., -1., 1.]), np.linalg.inv(native.affine))
                mni_xyz = apply_matrix(native_xyz, native_to_mni_index)
                fan_raw.append(sample(raw_native, native_xyz))
                if reset_b_now:
                    sampled_state = measured_baseline_on_points(pts, arrays, references, patient_temperature)
                else:
                    state_support = sample(valid_mni.astype(np.float32), mni_xyz)
                    sampled_state = sample(state, mni_xyz)
                    sampled_state = np.divide(sampled_state, state_support,
                        out=np.full_like(sampled_state, patient_temperature), where=state_support > 1e-6)
                fan_base.append(sampled_state)
                input_mask = sample(input_tc.astype(np.uint8), mni_xyz, order=0)
                other_mask = sample((tc & ~active_tc).astype(np.uint8), mni_xyz, order=0)
                inactive_mask_pixels += int(np.sum((input_mask > 0) & (other_mask > 0)))
                fan_tc.append(input_mask)
            raw = np.stack(fan_raw).astype(np.float32)
            baseline = np.stack(fan_base).astype(np.float32)
            tumor = np.stack(fan_tc).astype(np.float32)
            if args.mask_input_mode == 'target' and inactive_mask_pixels:
                raise AssertionError('Inactive lesion leaked into target-only network mask channel.')
            structural, stats = geometry.foreground_zscore_planes(raw, .02)
            foreground = raw > (.02 * raw.max(axis=(1, 2))[:, None, None])
            baseline[~foreground] = 0.
            dist2 = ((rr - center[0]) * spacing[0]) ** 2 + ((cc - center[1]) * spacing[1]) ** 2
            gaussian = np.exp(-dist2 / (2 * norm['probe_sigma_mm'] ** 2)).astype(np.float32)
            xx = np.stack([structural, (baseline - 35.) / 20., tumor,
                           np.broadcast_to(gaussian, structural.shape), np.zeros_like(structural)], axis=1)
            condition = np.tile([source['power_W'] / norm['power_scale_train_max_W'], 2 * center[0] / 511 - 1, 2 * center[1] / 511 - 1, 0], (args.num_planes, 1)).astype(np.float32)
            coords, covered = reconstruction_coordinates(native_points_lps, pivot, axis, radial0, radial1, center, spacing, axis_dimension, args.num_planes)
            support_profiles, _ = fan_profiles(foreground.astype(np.float32), center, axis_dimension)
            reconstruction_support = map_coordinates(support_profiles, coords, order=1, mode='nearest', prefilter=False)
            covered &= reconstruction_support > 1e-6
            previous = state.ravel()[flat].astype(np.float64)
            if reset_b_now:
                previous = np.full(len(flat), patient_temperature, np.float64)
                baseline_profiles, _ = fan_profiles(baseline * foreground, center, axis_dimension)
                previous[covered] = map_coordinates(baseline_profiles, coords[:, covered], order=1, mode='nearest', prefilter=False) / reconstruction_support[covered]
            start_TC_temperature = rate_stats(previous[component_masks[lesion_index].ravel()[flat]])
            initial_ready=np.zeros(mni.shape,np.float32);initial_ready.ravel()[flat]=previous.astype(np.float32)
            if directory == actual_dirs[0]:
                initial_volume = np.zeros(mni.shape, np.float32)
                initial_volume.ravel()[flat] = previous.astype(np.float32)
                save_nifti(trajectory_output / 'initial_temperature_mni152.nii.gz', initial_volume, mni, np.float32)
                first_initial_volume=initial_volume.copy()
                before_cooling=initial_volume.copy()
            segment_output=trajectory_output/'segments'/directory.name
            segment_output.mkdir(parents=True,exist_ok=True)
            save_nifti(segment_output/'initial_temperature_mni152.nii.gz',initial_ready,mni,np.float32)
            dose_start = cumulative.copy()
            x = torch.from_numpy(xx).to(device); cond = torch.from_numpy(condition).to(device); fg = torch.from_numpy(foreground.astype(np.float32)[:, None]).to(device)
            times = arrays[0]['target_time_indices'].astype(int)
            if any(not np.array_equal(times, a['target_time_indices']) for a in arrays[1:]): raise ValueError('Available reference-plane frame offsets differ.')
            expected = np.arange(1, times.max() + 1)
            # Missing observed frames do not collapse elapsed time: all fixed
            # 3.6-s model steps are reconstructed and integrated.
            nframes = int(times.max())
            with torch.inference_mode():
                emb0 = model.condition(cond); feature = x; skips = []
                for block, film in zip(model.enc, model.ef): feature = film(block(feature), emb0); skips.append(feature)
                latent = model.temporal_in(F.avg_pool2d(feature, 2)); hidden = torch.tanh(latent)
                local_latent = model.local_in(skips[-2]); local_hidden = torch.tanh(local_latent)
                maximum_temperature = float(previous.max())
                for step in range(1, nframes + 1):
                    c = cond.clone(); c[:, 3] = step / model.time_scale; emb = model.condition(c)
                    hidden = model.gru(latent + model.temporal_condition(emb)[:, :, None, None], hidden)
                    local_hidden = model.local_gru(local_latent + model.local_condition(emb)[:, :, None, None], local_hidden)
                    predictions = []
                    for start in range(0, args.num_planes, args.batch_size):
                        sl = slice(start, start + args.batch_size)
                        pred, _ = model.decode(feature[sl], [s[sl] for s in skips], hidden[sl], local_hidden[sl], emb[sl], x[sl], fg[sl], torch.full((len(x[sl]),), step, device=device))
                        predictions.append(pred[:, 0].cpu().numpy())
                    predictions = np.concatenate(predictions)
                    if not np.isfinite(predictions).all(): raise FloatingPointError('Nonfinite model prediction.')
                    profiles, _ = fan_profiles(predictions * foreground, center, axis_dimension)
                    current = previous.copy()
                    current[covered] = map_coordinates(profiles, coords[:, covered], order=1, mode='nearest', prefilter=False) / reconstruction_support[covered]
                    increment = dose_interval(previous, current, args.frame_seconds)
                    cumulative += increment
                    lesion_doses[lesion_index] += increment
                    trajectory_dose += increment
                    previous = current
                    maximum_temperature = max(maximum_temperature, float(current.max()))
                    if step % 20 == 0: print(f'Traj{trajectory}/{directory.name}: {step}/{nframes} frames reconstructed; max={current.max():.2f} C', flush=True)
            state.ravel()[flat] = previous.astype(np.float32)
            coverage_union |= covered
            closest_component = int(np.argmin([np.linalg.norm(pivot - c) for c in component_native_centers]))
            if closest_component != lesion_index: raise ValueError('Heat source no longer matches assigned TC component.')
            node = {'trajectory': trajectory, 'segment': directory.name, 'lesion': lesion_index + 1, 'power_W': source['power_W'],
                    'original_T_range': source['expected_temperature_T_range'], 'model_steps': nframes,
                    'observed_frame_offsets': times.tolist(), 'pivot_lps_mm': pivot.tolist(),
                    'axis_dimension': axis_dimension, 'center_rc': center.tolist(),
                    'original_grid_axis_alignment': grid_alignment,
                    'axis_out_of_reference_plane_component': normal_component,
                    'fiber_aligned_resampling_rotation_degrees': float(np.degrees(np.arccos(np.clip(grid_alignment,-1,1)))),
                    'planned_axis_entry_lps_mm': fixed_selected['entry_lps_mm'].tolist(),
                    'axis_direction_lps': axis.tolist(),
                    'available_reference_planes': available_planes,
                    'center_calibration_plane_distance_mm': float(np.linalg.norm(centers_lps[0] - centers_lps[1])) if len(centers_lps)>1 else None,
                    'reconstruction_coverage_fraction': float(covered.mean()), 'max_temperature_C': maximum_temperature,
                    'dose_increment_max_min': float((cumulative - dose_start).max()),
                    'warning_temperature_above_100C': maximum_temperature > 100}
            node['baseline_policy'] = f'{trajectory} initial measured-plane baseline' if reset_b_now else 'original within-trajectory inheritance'
            node['baseline_policy'] = 'initial measured-plane baseline' if reset_b_now else 'inherited prediction with measured inter-segment bounded cooling correction'
            node['cooling_correction']=cooling_record
            node['reference_plane_errors']=[]
            for a,ds in zip(arrays,references):
                plane=a['reference_plane'];pd=a['reference_directory'];roi=a['projected_active_tc']
                frame=sorted((pd/'temperature').glob('*.dcm'))[-1]
                target=pydicom.dcmread(frame).pixel_array.astype(np.float64)/10.
                pred=sample(state,a['mni_xyz_reference'])
                pre=sample(before_cooling,a['mni_xyz_reference'])
                post=sample(initial_ready,a['mni_xyz_reference'])
                error=pred-target
                plane_metrics={'trajectory':trajectory,'segment':directory.name,'plane':plane,'frame':frame.name,
                    'TC_pixels':int(roi.sum()),'end_TC_MAE_C':float(np.mean(abs(error[roi]))),
                    'end_TC_bias_C':float(np.mean(error[roi])),
                    'initial_before_TC_MAE_C':float(np.mean(abs(pre[roi]-a['baseline_C'][roi]))),
                    'initial_after_TC_MAE_C':float(np.mean(abs(post[roi]-a['baseline_C'][roi])))}
                node['reference_plane_errors'].append(plane_metrics)
                mm=json.loads((pd/'segment_metadata.json').read_text())
                previous_observations[plane]={'temperature':target,'segment':directory.name,
                    'original_frame':mm['original_filenames']['temperature'][frame.name],
                    'IOP':np.asarray(ds.ImageOrientationPatient,float),'IPP':np.asarray(ds.ImagePositionPatient,float),
                    'spacing':np.asarray(ds.PixelSpacing,float)}
            save_nifti(segment_output/'end_temperature_mni152.nii.gz',state,mni,np.float32)
            segment_dose=np.zeros(mni.shape,np.float64);segment_dose.ravel()[flat]=trajectory_dose
            save_nifti(segment_output/'end_cem43_cumulative_mni152.nii.gz',segment_dose,mni,np.float64)
            write_json(segment_output/'cooling_correction.json',cooling_record)
            write_json(segment_output/'reference_plane_errors.json',node['reference_plane_errors'])
            node['start_TC_temperature_C'] = start_TC_temperature
            node['mask_input_mode'] = args.mask_input_mode
            node['inactive_lesion_mask_pixels'] = inactive_mask_pixels
            node['structural_fan_sha256'] = hashlib.sha256(structural.tobytes()).hexdigest()
            node['power_W'] = source['power_W']
            report['segments'].append(node)
            write_json(args.output / 'progress.json', report)
            print(f'Completed Traj{trajectory}/{directory.name}: {nframes} steps, {source["power_W"]} W; max CEM43={cumulative.max():.6g} min', flush=True)
            last_axes[lesion_index] = (axis.copy(), radial0.copy(), radial1.copy())
            last_entries[lesion_index] = fixed_selected['entry_lps_mm'].copy()
            del x, fg, feature, skips, hidden, local_hidden, latent, local_latent
            if device.type == 'cuda': torch.cuda.empty_cache()
        trajectory_output = args.output / f'trajectory_{trajectory}'
        trajectory_output.mkdir(exist_ok=True)
        trajectory_volume = np.zeros(mni.shape, np.float64)
        trajectory_volume.ravel()[flat] = trajectory_dose
        save_nifti(trajectory_output / 'end_cem43_increment_mni152.nii.gz', trajectory_volume, mni, np.float64)
        save_nifti(trajectory_output / 'end_temperature_mni152.nii.gz', state, mni, np.float32)
        if trajectory in ['A', 'B']:
            index = lesion_index; directory = args.output / f'lesion_{index + 1:02d}'; directory.mkdir(exist_ok=True)
            dose_volume = np.zeros(mni.shape, np.float64); dose_volume.ravel()[flat] = lesion_doses[index]
            save_nifti(directory / 'end_cem43_cumulative_mni152.nii.gz', dose_volume, mni, np.float64)
            save_nifti(directory / 'end_cem43_log10_1p_mni152.nii.gz', np.log10(1 + dose_volume), mni, np.float32)
            save_nifti(directory / 'end_temperature_mni152.nii.gz', state, mni, np.float32)
            component = component_masks[index]
            inner_distance = distance_transform_edt(component)
            outer_distance = distance_transform_edt(~component)
            inside = component & (inner_distance <= 1.00001)
            outside = (~component) & (outer_distance <= 1.00001) & valid_mni
            summary = {'lesion': index + 1, 'included_trajectories': [trajectory],
                       'included_segments': len(actual_dirs), 'tumor': region_stats(dose_volume, component, args.threshold),
                       'inner_1mm': region_stats(dose_volume, inside, args.threshold),
                       'outer_1mm': region_stats(dose_volume, outside, args.threshold),
                       'combined_2mm': region_stats(dose_volume, inside | outside, args.threshold),
                       'dose_is_measured': False}
            write_json(directory / 'end_summary.json', summary)
            # Map native axes into MNI LPS via the verified coordinate map.
            lps_to_native_index = np.linalg.inv(native.affine) @ np.diag([-1., -1., 1., 1.])
            native_lps_to_mni_lps = np.diag([-1., -1., 1., 1.]) @ mni.affine @ native_to_mni_index @ lps_to_native_index
            native_center = component_native_centers[index]
            line_axis = last_axes[index][0]
            line_entry = last_entries[index]
            native_center = line_entry + np.dot(native_center - line_entry, line_axis) * line_axis
            center_mni_lps = apply_matrix(native_center[None], native_lps_to_mni_lps)
            axes_mni = []
            for direction in last_axes[index]:
                d = native_lps_to_mni_lps[:3, :3] @ direction; axes_mni.append(d / np.linalg.norm(d))
            axes_mni[1] -= (axes_mni[1] @ axes_mni[0]) * axes_mni[0]; axes_mni[1] /= np.linalg.norm(axes_mni[1])
            axes_mni[2] = np.cross(axes_mni[0], axes_mni[1])
            qc_figure(directory / 'end_cem43_fiber_planes.png', mri_mni, dose_volume, state, component, center_mni_lps, axes_mni, mni, args.threshold,
                      f'{args.case_id} lesion {index + 1}: {summary["included_segments"]} segments, axis Traj{trajectory}, reference planes {available_planes}, MODEL PREDICTION')
    whole_dose = np.zeros(mni.shape, np.float64); whole_dose.ravel()[flat] = cumulative
    save_nifti(args.output / 'patient_end_cem43_cumulative_mni152.nii.gz', whole_dose, mni, np.float64)
    save_nifti(args.output / 'patient_end_cem43_log10_1p_mni152.nii.gz', np.log10(1 + whole_dose), mni, np.float32)
    save_nifti(args.output / 'patient_end_temperature_mni152.nii.gz', state, mni, np.float32)
    coverage = np.zeros(mni.shape, np.uint8); coverage.ravel()[flat] = coverage_union
    save_nifti(args.output / 'reconstruction_coverage_mni152.nii.gz', coverage, mni, np.uint8)
    report['completed'] = True; report['logical_segments'] = len(report['segments'])
    report['total_model_steps'] = sum(s['model_steps'] for s in report['segments'])
    report['warnings_temperature_above_100C'] = [s for s in report['segments'] if s['warning_temperature_above_100C']]
    report['valid_tissue_interpolation'] = 'Normalize temperature samples by foreground support; zero-valued outside-brain background does not enter tissue-temperature interpolation.'
    report['final_brain_temperature_stats_C'] = rate_stats(state[valid_mni])
    write_json(args.output / 'reconstruction_summary.json', report)
    print('COMPLETED', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-id', required=True, help='Patient folder name used in the overview exports and output filenames.')
    parser.add_argument('--sample-prefix', required=True, help='NPZ sample prefix, e.g. <batch>_<case-id>.')
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--num-planes', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--frame-seconds', type=float, default=3.6)
    parser.add_argument('--threshold', type=float, default=10.)
    parser.add_argument('--normalize-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--mask-input-mode', choices=['all', 'target'], default='target')
    parser.add_argument('--component-order', nargs=2, type=int, required=True)
    main(parser.parse_args())
