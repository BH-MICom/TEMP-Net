"""Full-cohort inference and transparent final coverage reporting.

Source records, weights, previous experiments and masks are read-only.
This is retrospective, early-measurement-calibrated prediction, not a
prospective treatment plan or a measured 3-D thermal-dose ground truth.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates

from prepare_cohort import TRAIN, GEOMETRY, canonical, load_image, sha, write_json

OLD_CODE = Path(os.environ.get("TEMP_NET_3D_SUPPORT", Path(__file__).resolve().parent / "support"))
sys.path.insert(0, str(OLD_CODE))
spec = importlib.util.spec_from_file_location("old_cooling_runner", OLD_CODE / "single_case_runner.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
spec = importlib.util.spec_from_file_location("geometry", GEOMETRY)
geom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geom)


def stamp_seconds(name):
    match = re.search(r"T\d+_(\d{6,})_", name)
    if not match:
        raise ValueError(f"No acquisition timestamp in {name}")
    s = match.group(1)
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + float(s[4:6] + "." + (s[6:] or "0"))


def time_difference(later, earlier):
    value = stamp_seconds(later) - stamp_seconds(earlier)
    if value < -43200:
        value += 86400
    if value < -1e-5:
        raise ValueError(f"Negative time interval: {earlier} -> {later}: {value}")
    return max(0., value)


def vol_save(path, array, ref, dtype=np.float32):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old.save_nifti(path, array, ref, dtype)


def read_volume(path, ref):
    im = load_image(path)
    if im.shape != ref.shape or not np.allclose(im.affine, ref.affine, atol=1e-4):
        raise ValueError(f"Grid mismatch: {path}")
    array = np.asarray(im.dataobj)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite volume: {path}")
    return array


def sample_supported(volume, valid, coords, default=0.):
    support = old.sample(valid.astype(np.float32), coords)
    value = old.sample(volume, coords)
    return np.divide(value, support, out=np.full_like(value, default), where=support > 1e-6)


def regions(mask, brain, spacing):
    inner = mask & (distance_transform_edt(mask, sampling=spacing) <= 1.00001)
    outer = ~mask & brain & (distance_transform_edt(~mask, sampling=spacing) <= 1.00001)
    return {"tumor": mask, "inner_1mm": inner, "outer_1mm": outer, "combined_2mm": inner | outer}


def coverage(dose, roi, voxel_mL):
    values = np.asarray(dose[roi], np.float64)
    n = len(values)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite CEM43 cannot enter a coverage table")
    return {"voxels": n, "volume_mL": n * voxel_mL, "comparison": ">=",
            **{f"gte_{t}_min": {"covered_voxels": int((values >= t).sum()),
                              "covered_volume_mL": float((values >= t).sum() * voxel_mL),
                              "coverage_fraction": float((values >= t).mean()) if n else None}
               for t in (2, 10, 60)}}


def cover_all(dose, mask, brain, ref):
    spacing = nib.affines.voxel_sizes(ref.affine)
    voxel_mL = abs(np.linalg.det(ref.affine[:3, :3])) / 1000.
    return {k: coverage(dose, r, voxel_mL) for k, r in regions(mask, brain, spacing).items()}


def summarize_case(record, dose, mask, brain, ref, destination, **extra):
    result = {"patient": record["patient"], "session": record["session"],
              "model_split": record["model_split"], "status": "complete",
              "action": record["action"], "calculation_policy": record["calculation_policy"],
              "mask_sources": record["mask_sources"], "mask_sha256": record["mask_sha256"],
              "logical_segments": record["logical_segments"],
              "trajectory_chains": len(record["trajectories"]),
              "coverage": cover_all(dose, mask, brain, ref), **extra}
    components, count = label(mask)
    result["connected_regions_not_clinical_lesions"] = []
    for n in range(1, count + 1):
        roi = components == n
        result["connected_regions_not_clinical_lesions"].append(
            {"component": n, **coverage(dose, roi, abs(np.linalg.det(ref.affine[:3, :3])) / 1000.)})
    write_json(destination / "case_complete.json", result)
    return result


def plot_case(ref, mri, mask, dose, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coordinates = np.argwhere(mask)
    center = np.round(coordinates.mean(0)).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), facecolor="white")
    for row, dimension in enumerate((0, 1, 2)):
        mr = np.take(mri, center[dimension], axis=dimension).T
        mm = np.take(mask, center[dimension], axis=dimension).T
        dd = np.take(dose, center[dimension], axis=dimension).T
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(mr, origin="lower", cmap="gray", vmin=0, vmax=np.percentile(mri[mri > 0], 99))
            if col:
                threshold = (2, 10, 60)[col - 1]
                ax.imshow(old.heat_rgba(dd, threshold), origin="lower")
                ax.set_title(f"CEM43 > {threshold} min")
            else:
                ax.set_title("MRI + analysis mask")
            if mm.any():
                ax.contour(mm, [.5], origin="lower", colors="cyan", linewidths=.6)
            ax.axis("off")
    fig.suptitle(title + "\nMODEL 3D DOSE; not measured injury. Cyan: supplied mask.", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def build_coordinate_map(ref, native, transform, valid):
    import ants
    fixed, moving = ants.image_read(ref.get_filename()), ants.image_read(native.get_filename())
    flat = np.flatnonzero(valid)
    mni_indices = np.argwhere(valid).astype(np.float64)
    native_indices = []
    for dimension in range(3):
        shape = [1, 1, 1]
        shape[dimension] = native.shape[dimension]
        grid = np.broadcast_to(np.arange(native.shape[dimension], dtype=np.float32).reshape(shape), native.shape).copy()
        coordinate_image = ants.from_numpy(grid, origin=moving.origin, spacing=moving.spacing, direction=moving.direction)
        warped = ants.apply_transforms(fixed=fixed, moving=coordinate_image,
                                       transformlist=[str(transform)], interpolator="linear")
        native_indices.append(warped.numpy().ravel()[flat])
    native_indices = np.stack(native_indices, 1).astype(np.float64)
    interior = np.all((native_indices > 1) & (native_indices < np.asarray(native.shape) - 2), axis=1)
    if interior.sum() < 1000:
        raise ValueError("Insufficient native/MNI overlap for coordinate transform")
    select = np.flatnonzero(interior)[::max(1, int(interior.sum()) // 4096)]
    coefficients = np.linalg.lstsq(np.c_[mni_indices[select], np.ones(len(select))],
                                   native_indices[select], rcond=None)[0]
    matrix = np.eye(4)
    matrix[:3] = coefficients.T
    fitted = old.apply_matrix(mni_indices, matrix)
    residual = float(np.max(np.linalg.norm(fitted[interior] - native_indices[interior], axis=1)))
    if residual > .03:
        raise ValueError(f"Non-affine or incorrect coordinate transform: residual {residual}")
    supported = np.all((fitted >= 0) & (fitted <= np.asarray(native.shape) - 1), axis=1)
    if not supported.all():
        raise ValueError(f"{int((~supported).sum())} analysis voxels outside native T2 field of view")
    points_lps = old.apply_matrix(fitted, native.affine) * [-1., -1., 1.]
    return matrix, flat, points_lps, residual


def load_planes(segment, ref, native, native_to_mni, mask):
    import pydicom
    planes = []
    for row in segment["planes"]:
        source = Path(row["source_segment"])
        with np.load(TRAIN / row["npz_file"], allow_pickle=False) as z:
            a = {k: np.asarray(z[k]) for k in ("probe_center_rc", "target_time_indices", "baseline_C", "foreground_mask")}
            a["power_W"] = float(z["power_W"])
        ds = pydicom.dcmread(source / "structural.dcm", stop_before_pixels=True)
        if (int(ds.Rows), int(ds.Columns)) != (512, 512):
            raise ValueError(f"Unsupported model grid: {source}")
        if a["baseline_C"].shape != (512, 512):
            raise ValueError("Unexpected baseline array dimensions")
        rr, cc = np.meshgrid(np.arange(ds.Rows), np.arange(ds.Columns), indexing="ij")
        iop, spacing = np.array(ds.ImageOrientationPatient, float), np.array(ds.PixelSpacing, float)
        points = np.array(ds.ImagePositionPatient, float) + rr[..., None] * spacing[0] * iop[3:] + cc[..., None] * spacing[1] * iop[:3]
        xyz = old.apply_matrix(old.apply_matrix(points * [-1., -1., 1.], np.linalg.inv(native.affine)), native_to_mni)
        projected = old.sample(mask.astype(np.uint8), xyz, order=0) > 0
        foreground = a["foreground_mask"].astype(bool)
        calibration_roi = projected & foreground
        calibration_policy = "updated_3d_mask_projected"
        if not calibration_roi.any():
            # A plane can miss a valid 3-D lesion. Keep the already reviewed
            # measured-center estimate rather than inventing a mask intersection.
            calibration_policy = "archived_measured_center_no_updated_mask_intersection"
        frames = sorted((source / "temperature").glob("T*.dcm"))
        meta = json.loads((source / "segment_metadata.json").read_text())
        original = meta["original_filenames"]
        if not frames:
            raise ValueError(f"No observed heating frames in {source}")
        baseline = pydicom.dcmread(source / "baseline/baseline.dcm").pixel_array.astype(np.float32) / 10.
        if not np.allclose(baseline, a["baseline_C"], atol=1e-4):
            raise ValueError("Training-ready baseline differs from standardized DICOM")
        # The reviewed center stored in the training-ready NPZ is authoritative.
        # Do not re-estimate it from early heating frames.
        center = np.asarray(a["probe_center_rc"], dtype=np.float32)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Invalid reviewed probe center in training-ready NPZ")
        a["probe_center_rc"] = center
        bnum = int(re.search(r"T(\d+)", original["baseline"]).group(1))
        actual_offsets = np.array([int(f.stem[1:]) - bnum for f in frames], int)
        if not np.array_equal(a["target_time_indices"], actual_offsets):
            raise ValueError("Frame offsets disagree with training manifest")
        times = {int(k): time_difference(original["temperature"][f.name], original["baseline"])
                 for k, f in zip(actual_offsets, frames)}
        frame_by_offset = {int(k): f for k, f in zip(actual_offsets, frames)}
        planes.append({**a, "source": source, "ds": ds, "xyz": xyz, "projected": projected,
                       "row": row, "plane": row["plane"], "original": original, "frames": frames,
                       "frame_by_offset": frame_by_offset, "times": times, "calibration_policy": calibration_policy,
                       "center_lps": old.dicom_point(ds, a["probe_center_rc"]),
                       "grid": [int(ds.Rows), int(ds.Columns)], "IOP": iop,
                       "IPP": np.array(ds.ImagePositionPatient, float), "spacing": spacing})
    if not planes or len({p["power_W"] for p in planes}) != 1:
        raise ValueError("Missing planes or conflicting power between reference planes")
    return planes


def select_axis(chain, planes, pivot):
    candidates = []
    for export in chain["export_candidates"]:
        trajectories, registrations = geom.read_export_info(export["path"])
        names = [t["name"] for t in trajectories
                 if re.sub(r"\s+", "", t["name"]).lower() == ("traj" + chain["letter"]).lower()]
        for name in names:
            try:
                selected, choices = geom.select_geometry(trajectories, registrations, pivot, name)
            except RuntimeError:
                continue
            for choice in choices:
                axis = choice["target_lps_mm"] - choice["entry_lps_mm"]
                axis /= np.linalg.norm(axis)
                off = max(abs(axis @ np.cross(p["IOP"][:3], p["IOP"][3:])) for p in planes)
                if choice["center_to_axis_mm"] <= 10 and off <= .1:
                    candidates.append((choice["center_to_axis_mm"] + 30 * off, choice, export, off))
    if not candidates:
        raise ValueError(f"No axis agrees with heat-source position and observed plane(s): {chain['id']}")
    candidates.sort(key=lambda x: x[0])
    score, choice, export, off = candidates[0]
    return choice, export, float(off)


def integrate_step_times(planes):
    grouped = {}
    for p in planes:
        for index, seconds in p["times"].items():
            grouped.setdefault(index, []).append(seconds)
    offsets = sorted(grouped)
    if offsets[0] <= 0:
        raise ValueError("Heating frame is not later than baseline")
    if any(max(values) - min(values) > 1.0 for values in grouped.values()):
        raise ValueError("Reference planes have inconsistent relative acquisition times")
    observed = np.array([0.] + [float(np.median(grouped[k])) for k in offsets])
    if np.any(np.diff(observed) <= 0):
        raise ValueError("Nonincreasing heating timestamps")
    step_times = np.interp(np.arange(offsets[-1] + 1), [0] + offsets, observed)
    return np.diff(step_times), offsets


def compute_chain(record, chain, plan, destination, context, model, device):
    import pydicom
    import torch
    import torch.nn.functional as F
    ref, native, mri, raw, mask, brain, valid, flat, points, native_to_mni, norm = context
    directory = destination / "trajectories" / chain["id"]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "chain_complete.json").exists():
        return json.loads((directory / "chain_complete.json").read_text())
    state = np.zeros(ref.shape, np.float32)
    initial = None
    heating = np.zeros(len(flat), np.float64)
    waiting = np.zeros(len(flat), np.float64)
    nodes = []
    fixed_axis = None
    patient_temperature = None
    rr, cc = np.meshgrid(np.arange(512), np.arange(512), indexing="ij")
    all_covered = np.zeros(len(flat), bool)
    for segment in chain["segments"]:
        node_dir = directory / "segments" / segment["segment"]
        node_dir.mkdir(parents=True, exist_ok=True)
        all_planes = load_planes(segment, ref, native, native_to_mni, mask)
        holdout_plane = plan.get("holdout_plane")
        planes = [p for p in all_planes if holdout_plane is None or p["plane"] != holdout_plane]
        if not planes:
            raise ValueError(f"The holdout plane removed all observed planes in {chain['id']}/{segment['segment']}")
        pivot = np.mean([p["center_lps"] for p in planes], axis=0)
        selected, export, off = select_axis(chain, planes, pivot)
        if fixed_axis is None:
            fixed_axis = selected
            patient_temperature = float(export["temperature"])
        else:
            old_axis = fixed_axis["target_lps_mm"] - fixed_axis["entry_lps_mm"]
            new_axis = selected["target_lps_mm"] - selected["entry_lps_mm"]
            cosine = abs(np.dot(old_axis, new_axis) / np.linalg.norm(old_axis) / np.linalg.norm(new_axis))
            if cosine < .99:
                raise ValueError("Planned fiber-axis direction changed within a trajectory chain")
        axis = selected["target_lps_mm"] - selected["entry_lps_mm"]
        axis /= np.linalg.norm(axis)
        first_plane = planes[0]
        directions = [first_plane["IOP"][3:], first_plane["IOP"][:3]]
        axis_dimension = int(np.argmax([abs(axis @ d) for d in directions]))
        if axis @ directions[axis_dimension] < 0:
            axis *= -1
        radial0 = directions[1 - axis_dimension] - np.dot(directions[1 - axis_dimension], axis) * axis
        radial0 /= np.linalg.norm(radial0)
        radial1 = np.cross(axis, radial0)
        center, spacing = first_plane["probe_center_rc"].astype(float), first_plane["spacing"]
        # Paper protocol: each segment starts from its own measured baseline
        # field. Inter-segment cooling and waiting dose are not included.
        estimates, gaps = [], []
        delta, raw_delta, duration = 0.0, 0.0, 0.0
        wait_increment = np.zeros_like(waiting)
        fan_raw, fan_base, fan_mask = [], [], []
        for angle in np.arange(plan["num_planes"]) * math.pi / plan["num_planes"]:
            radial = math.cos(angle) * radial0 + math.sin(angle) * radial1
            rd, cd = (radial, axis) if axis_dimension == 1 else (axis, radial)
            pts = pivot + (rr - center[0])[..., None] * spacing[0] * rd + (cc - center[1])[..., None] * spacing[1] * cd
            native_xyz = old.apply_matrix(pts * [-1., -1., 1.], np.linalg.inv(native.affine))
            mni_xyz = old.apply_matrix(native_xyz, native_to_mni)
            fan_raw.append(old.sample(raw, native_xyz))
            b = old.measured_baseline_on_points(pts, planes, [p["ds"] for p in planes], patient_temperature)
            fan_base.append(b)
            fan_mask.append(old.sample(mask.astype(np.uint8), mni_xyz, order=0))
        raw_planes = np.stack(fan_raw).astype(np.float32)
        baseline = np.stack(fan_base).astype(np.float32)
        tumor = np.stack(fan_mask).astype(np.float32)
        structural, zstats = geom.foreground_zscore_planes(raw_planes, .02)
        foreground = raw_planes > (.02 * raw_planes.max(axis=(1, 2))[:, None, None])
        baseline[~foreground] = 0.
        gaussian = np.exp(-(((rr - center[0]) * spacing[0]) ** 2 + ((cc - center[1]) * spacing[1]) ** 2)
                          / (2 * norm["probe_sigma_mm"] ** 2)).astype(np.float32)
        xx = np.stack([structural, (baseline - 35.) / 20., tumor,
                       np.broadcast_to(gaussian, structural.shape), np.zeros_like(structural)], axis=1)
        conditions = np.tile([first_plane["power_W"] / norm["power_scale_train_max_W"],
                              2 * center[0] / 511 - 1, 2 * center[1] / 511 - 1, 0.],
                             (plan["num_planes"], 1)).astype(np.float32)
        coords, covered = old.reconstruction_coordinates(points, pivot, axis, radial0, radial1, center,
                                                         spacing, axis_dimension, plan["num_planes"])
        support_profiles, _ = old.fan_profiles(foreground.astype(np.float32), center, axis_dimension)
        support = map_coordinates(support_profiles, coords, order=1, mode="nearest", prefilter=False)
        covered &= support > 1e-6
        previous = state.ravel()[flat].astype(np.float64)
        profiles, _ = old.fan_profiles(baseline * foreground, center, axis_dimension)
        previous[covered] = map_coordinates(profiles, coords[:, covered], order=1, mode="nearest",
                                            prefilter=False) / support[covered]
        state.ravel()[flat] = previous.astype(np.float32)
        initial = state.copy()
        vol_save(node_dir / "initial_temperature_mni152.nii.gz", initial, ref)
        vol_save(node_dir / "initial_temperature_mni152.nii.gz", state, ref)
        durations, observed_offsets = integrate_step_times(planes)
        node = {"segment": segment["segment"], "power_W": first_plane["power_W"],
                "model_steps": len(durations), "heating_seconds": float(durations.sum()),
                "actual_step_seconds_min_max": [float(durations.min()), float(durations.max())],
                "observed_offsets_union": observed_offsets,
                "missing_model_offsets": sorted(set(range(1, len(durations) + 1)) - set(observed_offsets)),
                "pivot_lps_mm": pivot.tolist(), "center_rc": center.tolist(),
                "axis_lps": axis.tolist(), "entry_lps_mm": selected["entry_lps_mm"].tolist(),
                "export_info": export["path"], "axis_out_of_plane": off,
                "mask_input_policy": plan["mask_input_policy"],
                "calibration": {p["plane"]: p["calibration_policy"] for p in all_planes},
                "holdout_plane": holdout_plane,
                "cooling": {"delta_C": 0.0, "raw_mean_C": 0.0, "estimates": [],
                            "waiting_seconds": 0.0, "waiting_dose_included": False,
                            "policy": "segment-specific measured baseline; no inter-segment waiting dose"},
                "reference_plane_errors": []}
        print(f'{record["id"]} {chain["id"]}/{segment["segment"]}: {len(durations)} steps, '
              f'{first_plane["power_W"]}W, measured segment baseline, no waiting dose', flush=True)
        x = torch.from_numpy(xx).to(device)
        cond = torch.from_numpy(conditions).to(device)
        fg = torch.from_numpy(foreground.astype(np.float32)[:, None]).to(device)
        maximum = float(previous.max())
        captured = {}
        online_anchor_frames = int(plan.get("online_anchor_frames", 0))
        online_anchor_applied = False
        online_delta = 0.0
        first_anchor_step = min((min(p["times"]) for p in planes if p["times"]), default=10**9)
        with torch.inference_mode():
            emb0 = model.condition(cond)
            feature, skips = x, []
            for block, film in zip(model.enc, model.ef):
                feature = film(block(feature), emb0)
                skips.append(feature)
            latent = model.temporal_in(F.avg_pool2d(feature, 2))
            hidden = torch.tanh(latent)
            local_latent = model.local_in(skips[-2])
            local_hidden = torch.tanh(local_latent)
            for step, seconds in enumerate(durations, 1):
                c = cond.clone()
                c[:, 3] = step / model.time_scale
                emb = model.condition(c)
                hidden = model.gru(latent + model.temporal_condition(emb)[:, :, None, None], hidden)
                local_hidden = model.local_gru(local_latent + model.local_condition(emb)[:, :, None, None], local_hidden)
                prediction = []
                for start in range(0, len(x), 4):
                    sl = slice(start, start + 4)
                    pred, _ = model.decode(feature[sl], [s[sl] for s in skips], hidden[sl], local_hidden[sl],
                                           emb[sl], x[sl], fg[sl], torch.full((len(x[sl]),), step, device=device))
                    prediction.append(pred[:, 0].cpu().numpy())
                prediction = np.concatenate(prediction)
                if not np.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite predicted temperature")
                profiles, _ = old.fan_profiles(prediction * foreground, center, axis_dimension)
                current = previous.copy()
                current[covered] = map_coordinates(profiles, coords[:, covered], order=1, mode="nearest",
                                                   prefilter=False) / support[covered]
                if online_anchor_frames > 0 and not online_anchor_applied and step >= first_anchor_step:
                    deltas = []
                    for p in planes:
                        offset = min(p["times"])
                        frame = p["frame_by_offset"].get(int(offset))
                        if frame is None:
                            continue
                        volume = np.zeros(ref.shape, np.float32)
                        volume.ravel()[flat] = current.astype(np.float32)
                        predicted_plane = sample_supported(volume, valid, p["xyz"])
                        target_plane = pydicom.dcmread(frame).pixel_array.astype(np.float64) / 10.0
                        roi = p["projected"] & p["foreground_mask"].astype(bool) \
                              & np.isfinite(target_plane) & (target_plane > 0) \
                              & np.isfinite(predicted_plane)
                        if roi.any():
                            deltas.append(float(np.median(target_plane[roi] - predicted_plane[roi])))
                    if deltas:
                        online_delta = float(np.clip(np.median(deltas), -5.0, 5.0))
                        online_anchor_applied = True
                if online_anchor_applied:
                    current[covered] = current[covered] + online_delta
                heating += old.dose_interval(previous, current, float(seconds))
                previous = current
                maximum = max(maximum, float(current.max()))
                for p in planes:
                    if step == int(max(p["times"])):
                        v = np.zeros(ref.shape, np.float32)
                        v.ravel()[flat] = current.astype(np.float32)
                        captured[p["plane"]] = sample_supported(v, valid, p["xyz"])
                if step % 40 == 0:
                    print(f'  {step}/{len(durations)}; max T={current.max():.2f} C', flush=True)
        node["online_anchor"] = {"enabled": bool(online_anchor_frames),
                                  "anchor_frames": online_anchor_frames,
                                  "delta_C": online_delta,
                                  "applied": online_anchor_applied}
        state.ravel()[flat] = previous.astype(np.float32)
        all_covered |= covered
        for p in all_planes:
            if p["plane"] not in captured:
                v = np.zeros(ref.shape, np.float32)
                v.ravel()[flat] = current.astype(np.float32)
                captured[p["plane"]] = sample_supported(v, valid, p["xyz"])
            frame = p["frames"][-1]
            target = pydicom.dcmread(frame).pixel_array.astype(np.float64) / 10.
            prediction = captured[p["plane"]]
            roi = p["projected"] & p["foreground_mask"].astype(bool) & np.isfinite(target) & (target > 0)
            error = prediction - target
            node["reference_plane_errors"].append(
                {"plane": p["plane"], "frame": frame.name, "pixels": int(roi.sum()),
                 "tumor_MAE_C": float(abs(error[roi]).mean()) if roi.any() else None,
                 "tumor_bias_C": float(error[roi].mean()) if roi.any() else None})
        node["max_predicted_temperature_C"] = maximum
        node["temperature_above_100C_warning"] = maximum > 100
        vol_save(node_dir / "end_temperature_mni152.nii.gz", state, ref)
        for name, values in (("heating", heating), ("waiting", waiting), ("total", heating + waiting)):
            volume = np.zeros(ref.shape, np.float64)
            volume.ravel()[flat] = values
            vol_save(node_dir / f"end_cem43_{name}_cumulative_mni152.nii.gz", volume, ref, np.float64)
        node["coverage"] = cover_all(volume, mask, brain, ref)
        write_json(node_dir / "segment_complete.json", node)
        nodes.append(node)
        write_json(directory / "progress.json", {"segments_done": len(nodes), "segments_total": len(chain["segments"]),
                                                "last_segment": segment["segment"]})
        del x, cond, fg, feature, skips, latent, hidden, local_latent, local_hidden
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for name, values in (("heating", heating), ("waiting", waiting), ("total", heating + waiting)):
        volume = np.zeros(ref.shape, np.float64)
        volume.ravel()[flat] = values
        vol_save(directory / f"end_cem43_{name}_mni152.nii.gz", volume, ref, np.float64)
    vol_save(directory / "end_temperature_mni152.nii.gz", state, ref)
    recon = np.zeros(ref.shape, np.uint8)
    recon.ravel()[flat] = all_covered
    vol_save(directory / "reconstruction_support_mni152.nii.gz", recon, ref, np.uint8)
    report = {"chain": chain["id"], "completed": True, "segments": nodes,
              "dose_file": str(directory / "end_cem43_total_mni152.nii.gz"),
              "heating_file": str(directory / "end_cem43_heating_mni152.nii.gz"),
              "waiting_file": str(directory / "end_cem43_waiting_mni152.nii.gz"),
              "waiting_seconds": sum(n["cooling"]["waiting_seconds"] for n in nodes),
              "heating_seconds": sum(n["heating_seconds"] for n in nodes),
              "cross_trajectory_gap_included": False}
    write_json(directory / "chain_complete.json", report)
    return report


def predict_case(record, plan, destination):
    import torch
    model_src = Path(os.environ.get("TEMP_NET_MODEL_SRC", Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(model_src))
    from temp_net.models.dynamics_unet import DynamicsUNet
    case = Path(record["case_dir"])
    ref = nib.load(case / "mni152/t2_1mm.nii.gz")
    native = nib.load(case / "native/t2_original.nii.gz")
    mri, raw = np.asarray(ref.dataobj, np.float32), np.asarray(native.dataobj, np.float32)
    mask = read_volume(record["mask"], ref) > 0
    brain = read_volume(case / "mni152/brain_mask.nii.gz", ref) > 0
    valid = (brain & np.isfinite(mri) & (mri > .02 * mri.max())) | mask
    matrix, flat, points, residual = build_coordinate_map(ref, native, case / "transforms/native_to_mni152_0.mat", valid)
    write_json(destination / "geometry.json", {"mni_to_native_voxel": matrix.tolist(),
                                              "fit_max_residual_native_voxels": residual})
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently run full cohort on CPU")
    device = torch.device("cuda")
    model = DynamicsUNet(time_scale=135, width=24, hidden=64, local_hidden=16, enhanced=True).to(device)
    checkpoint = torch.load(plan["weights"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    del checkpoint
    norm = json.loads((TRAIN / "normalization.json").read_text())
    context = (ref, native, mri, raw, mask, brain, valid, flat, points, np.linalg.inv(matrix), norm)
    reports = []
    for chain in record["trajectories"]:
        reports.append(compute_chain(record, chain, plan, destination, context, model, device))
    total = np.zeros(ref.shape, np.float64)
    for r in reports:
        total += read_volume(r["dose_file"], ref)
    dose_file = destination / "patient_end_cem43_cumulative_mni152.nii.gz"
    vol_save(dose_file, total, ref, np.float64)
    vol_save(destination / "mask_mni152.nii.gz", mask, ref, np.uint8)
    result = summarize_case(record, total, mask, brain, ref, destination,
                           dose_file=str(dose_file), chains=[r["chain"] for r in reports],
                           waiting_seconds=sum(r["waiting_seconds"] for r in reports),
                           heating_seconds=sum(r["heating_seconds"] for r in reports),
                           cross_trajectory_gap_included=False,
                           temperature_above_100C_segments=sum(
                               n["temperature_above_100C_warning"] for r in reports for n in r["segments"]))
    plot_case(ref, mri, mask, total, destination / "end_coverage_2_10_60.png", record["id"])
    return result


def aggregate(output, plan):
    records, totals = [], {"completed": 0, "failed": 0, "pending": 0}
    for record in plan["records"]:
        directory = output / "cases" / record["patient"] / record["session"]
        complete, failure = directory / "case_complete.json", directory / "failure.json"
        if complete.exists():
            data = json.loads(complete.read_text())
            totals["completed"] += 1
        elif failure.exists():
            data = {**record, "status": "failed", "failure": json.loads(failure.read_text())}
            totals["failed"] += 1
        else:
            data = {**record, "status": "pending"}
            totals["pending"] += 1
        records.append(data)
    write_json(output / "coverage_summary.json", {"counts": totals, "records": records,
                                                  "thresholds_min": [2, 10, 60], "comparison": ">="})
    fields = ["Patient", "Session", "Split", "Status", "Calculation policy", "Mask type",
              "Segments", "Tumor mL", "Tumor >=2 min %", "Tumor >=10 min %", "Tumor >=60 min %",
              "Inner 1mm >=2 %", "Inner 1mm >=10 %", "Inner 1mm >=60 %",
              "Outer 1mm >=2 %", "Outer 1mm >=10 %", "Outer 1mm >=60 %",
              "Combined 2mm >=2 %", "Combined 2mm >=10 %", "Combined 2mm >=60 %", "Reason or warning"]
    def fmt(value, fraction=False):
        return "NA" if value is None else f"{value * (100 if fraction else 1):.2f}"
    table = []
    for data in records:
        cov = data.get("coverage", {})
        row = [data["patient"], data["session"], data["model_split"], data["status"],
               data["calculation_policy"], "+".join(sorted({s["label_type"] for s in data["mask_sources"]})),
               str(data["logical_segments"]), fmt(cov.get("tumor", {}).get("volume_mL"))]
        for region in ("tumor", "inner_1mm", "outer_1mm", "combined_2mm"):
            row.extend(fmt(cov.get(region, {}).get(f"gte_{t}_min", {}).get("coverage_fraction"), True) for t in (2, 10, 60))
        row.append(data.get("policy_warning") or data.get("failure", {}).get("error") or "")
        table.append(row)
    # TSV is an interchange table, not an authored Excel workbook.
    with (output / "final_coverage.tsv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(fields)
        writer.writerows(table)
    lines = ["# 60-patient / 62-series CEM43 coverage", "",
             f'Completed {totals["completed"]}/62; failed {totals["failed"]}; pending {totals["pending"]}.', "",
             "Inclusive >= thresholds. NA is unavailable, not 0%. Predictions are not measured tissue damage.", "",
             "Legacy reuse rows exclude cooling/waiting and must not be pooled with corrected results.", "",
             "| Patient | Series | Split | Status | Policy | Tumor mL | >=2 min | >=10 min | >=60 min |",
             "|---|---|---|---|---|---:|---:|---:|---:|"]
    for r in table:
        lines.append("| " + " | ".join(r[:5] + r[7:11]) + " |")
    (output / "FINAL_COVERAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "run_status.json", {**totals, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           "complete": totals["completed"] == 62})
    return totals


def queue(args, plan):
    import fcntl
    lock = (args.output / "cohort.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(args.output / "job.json", {"pid": os.getpid(), "python": sys.executable,
                                         "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    records = sorted(plan["records"], key=lambda r: r["id"])
    for record in records:
        directory = args.output / "cases" / record["patient"] / record["session"]
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "case_complete.json").exists():
            continue
        while True:
            usage = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                              "--format=csv,noheader,nounits"], text=True).strip().splitlines()
            if int(usage[0]) < 2500:
                break
            write_json(args.output / "queue_wait.json", {"patient": record["id"],
                                                        "gpu_used_MiB": int(usage[0]),
                                                        "state": "waiting_for_gpu"})
            time.sleep(30)
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output), "--case", record["id"]]
        print("START", record["id"], record["action"], flush=True)
        with (directory / "run.log").open("a") as log:
            rc = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        if rc and not (directory / "failure.json").exists():
            write_json(directory / "failure.json", {"error": f"Worker exited {rc}; see run.log"})
        print("FINISH", record["id"], "exit", rc, aggregate(args.output, plan), flush=True)
    totals = aggregate(args.output, plan)
    write_json(args.output / "queue_finished.json", totals)
    print("QUEUE_FINISHED", totals, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    plan = json.loads((args.output / "cohort_plan.json").read_text())
    if args.aggregate:
        print(aggregate(args.output, plan))
        return
    if args.case:
        record = next(r for r in plan["records"] if r["id"] == args.case)
        destination = args.output / "cases" / record["patient"] / record["session"]
        destination.mkdir(parents=True, exist_ok=True)
        try:
            if record["issues"]:
                raise ValueError("Input issues: " + "; ".join(record["issues"]))
            predict_case(record, plan, destination)
        except Exception as e:
            write_json(destination / "failure.json", {"error": str(e), "traceback": traceback.format_exc()})
            raise
    else:
        queue(args, plan)


if __name__ == "__main__":
    main()
