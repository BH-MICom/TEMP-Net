"""Rotate 2-D inference planes around a planned trajectory and rebuild a 3-D map.

The trained 2-D model is unchanged. Real 3-D T2 data supplies the structural
channel. With the currently provided files, baseline temperature and tumor mask
exist only in one reference plane and are reused at every angle. Therefore the
result is an exploratory pseudo-3-D estimate, not a clinical temperature map.
"""

import argparse
import base64
import json
import math
import os
import re
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cfg", required=True)
    p.add_argument("--weights")
    p.add_argument("--reference-npz", required=True)
    p.add_argument("--reference-dicom", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--structural-zip")
    source.add_argument("--structural-dicom-dir")
    p.add_argument("--export-info", required=True)
    p.add_argument(
        "--tumor-vtp",
        help="3-D tumor surface in the export_info Basis coordinate system",
    )
    p.add_argument("--trajectory", default="auto")
    p.add_argument(
        "--center-mode",
        choices=["reference", "planned-axis-projection"],
        default="reference",
        help=(
            "reference keeps the previously detected DICOM center exactly at "
            "(253,247); planned-axis-projection uses the closest point on the "
            "untranslated planned trajectory"
        ),
    )
    p.add_argument("--num-planes", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--power-W", type=float)
    p.add_argument("--time-index", type=float)
    p.add_argument("--output-dir", default="rotating_planes_3d_output")
    p.add_argument("--geometry-only", action="store_true")
    return p.parse_args()


def require_file(path, label):
    if not path:
        raise ValueError(f"{label} is required")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def safe_extract(zip_path, destination):
    destination = os.path.abspath(destination)
    os.makedirs(destination, exist_ok=True)
    marker = os.path.join(destination, ".extraction_complete")
    if os.path.isfile(marker):
        return destination
    with zipfile.ZipFile(zip_path) as archive:
        for item in archive.infolist():
            target = os.path.abspath(os.path.join(destination, item.filename))
            if os.path.commonpath([destination, target]) != destination:
                raise RuntimeError(f"Unsafe ZIP member: {item.filename}")
        archive.extractall(destination)
    Path(marker).touch()
    return destination


def read_export_info(path):
    text = Path(path).read_bytes().decode("utf-16-be", errors="ignore")
    trajectories = []
    for block in re.findall(r"<trajectory>(.*?)</trajectory>", text, re.S | re.I):
        def field(name):
            found = re.search(fr"<{name}>(.*?)</{name}>", block, re.S | re.I)
            return found.group(1).strip() if found else None
        name, entry, target = field("name"), field("entry"), field("target")
        if name and entry and target:
            trajectories.append({
                "name": name,
                "entry_basis_mm": np.fromstring(entry, sep=",", dtype=np.float64),
                "target_basis_mm": np.fromstring(target, sep=",", dtype=np.float64),
            })
    registrations = []
    for block in re.findall(
        r"<registrationInfo>(.*?)</registrationInfo>", text, re.S | re.I
    ):
        uid = re.search(r"<uuid>(.*?)</uuid>", block, re.S | re.I)
        matrix = re.search(r"<regMatrix>(.*?)</regMatrix>", block, re.S | re.I)
        if uid and matrix:
            values = np.fromstring(matrix.group(1), sep=",", dtype=np.float64)
            if values.size == 16:
                registrations.append({
                    "uuid": uid.group(1).strip(),
                    "matrix": values.reshape(4, 4),
                })
    if not trajectories:
        raise RuntimeError("No valid trajectories found in export_info")
    if not registrations:
        raise RuntimeError("No valid regMatrix found in export_info")
    return trajectories, registrations


def read_patient_temperature(export_info_path):
    text = Path(export_info_path).read_bytes().decode(
        "utf-16-be", errors="ignore"
    )
    found = re.search(
        r"<patientTemprature>(.*?)</patientTemprature>", text, re.S | re.I
    )
    if not found:
        raise RuntimeError("patientTemprature not found in export_info")
    value = float(found.group(1).strip())
    if not np.isfinite(value) or not 20.0 <= value <= 45.0:
        raise RuntimeError(f"Invalid patientTemprature: {value}")
    return value


def read_volume_model_metadata(export_info_path, model_uuid):
    text = Path(export_info_path).read_bytes().decode(
        "utf-16-be", errors="ignore"
    )
    for block in re.findall(
        r"<volumeModelModel>(.*?)</volumeModelModel>", text, re.S | re.I
    ):
        if model_uuid.lower() not in block.lower():
            continue
        def field(name):
            found = re.search(fr"<{name}>(.*?)</{name}>", block, re.S | re.I)
            return found.group(1).strip() if found else None
        volume = field("contourVolume")
        return {
            "uuid": model_uuid,
            "name": field("volumeModelName"),
            "coordinate_space": field("volumeInfoName"),
            "contour_volume_mm3": float(volume) if volume else None,
        }
    raise RuntimeError(f"VTP model UUID {model_uuid} not found in export_info")


def _decode_vtk_compressed_array(encoded, offset, dtype):
    """Decode one VTK XML appended/base64/zlib data array."""
    encoded = encoded[offset:]
    first = base64.b64decode(encoded[:16])
    number_of_blocks, block_size, last_block_size = struct.unpack(
        "<III", first[:12]
    )
    header_bytes = 4 * (3 + number_of_blocks)
    header_chars = 4 * ((header_bytes + 2) // 3)
    header = base64.b64decode(encoded[:header_chars])[:header_bytes]
    values = struct.unpack(
        "<" + "I" * (3 + number_of_blocks), header
    )
    compressed_sizes = values[3:]
    compressed_bytes = sum(compressed_sizes)
    compressed_chars = 4 * ((compressed_bytes + 2) // 3)
    compressed = base64.b64decode(
        encoded[header_chars:header_chars + compressed_chars]
    )[:compressed_bytes]
    output = []
    cursor = 0
    for compressed_size in compressed_sizes:
        block = compressed[cursor:cursor + compressed_size]
        output.append(zlib.decompress(block))
        cursor += compressed_size
    raw = b"".join(output)
    expected = (
        (number_of_blocks - 1) * block_size + last_block_size
        if number_of_blocks else 0
    )
    if len(raw) != expected:
        raise RuntimeError(
            f"Invalid VTP array length: decoded={len(raw)}, expected={expected}"
        )
    return np.frombuffer(raw, dtype=dtype)


def read_vtp_mesh(path):
    """Read the compressed VTP format used by the supplied Sinovation export."""
    raw = Path(path).read_bytes()
    appended_tag = raw.find(b"<AppendedData")
    underscore = raw.find(b"_", appended_tag)
    appended_end = raw.rfind(b"</AppendedData>")
    if appended_tag < 0 or underscore < 0 or appended_end < 0:
        raise RuntimeError("VTP has no valid AppendedData section")
    header = raw[:appended_tag].decode("utf-8", errors="strict")
    appended = re.sub(rb"\s+", b"", raw[underscore + 1:appended_end])
    point_match = re.search(
        r"<Points>.*?<DataArray[^>]*type=\"(\w+)\"[^>]*"
        r"Name=\"Points\"[^>]*offset=\"(\d+)\"",
        header, re.S,
    )
    polys_match = re.search(
        r"<Polys>.*?<DataArray[^>]*type=\"(\w+)\"[^>]*"
        r"Name=\"connectivity\"[^>]*offset=\"(\d+)\".*?"
        r"<DataArray[^>]*type=\"(\w+)\"[^>]*"
        r"Name=\"offsets\"[^>]*offset=\"(\d+)\"",
        header, re.S,
    )
    if not point_match or not polys_match:
        raise RuntimeError("Could not locate Points/Polys arrays in VTP header")
    vtk_dtypes = {
        "Float32": "<f4", "Float64": "<f8",
        "Int32": "<i4", "Int64": "<i8",
        "UInt32": "<u4", "UInt64": "<u8",
    }
    try:
        points = _decode_vtk_compressed_array(
            appended, int(point_match.group(2)), vtk_dtypes[point_match.group(1)]
        ).reshape(-1, 3)
        connectivity = _decode_vtk_compressed_array(
            appended, int(polys_match.group(2)), vtk_dtypes[polys_match.group(1)]
        )
        offsets = _decode_vtk_compressed_array(
            appended, int(polys_match.group(4)), vtk_dtypes[polys_match.group(3)]
        )
    except KeyError as error:
        raise RuntimeError(f"Unsupported VTP scalar type: {error}") from error
    sizes = np.diff(np.r_[0, offsets])
    if not np.all(sizes == 3):
        raise RuntimeError("Tumor VTP must contain triangular polygons only")
    triangles = connectivity.reshape(-1, 3).astype(np.int64)
    if triangles.size == 0 or triangles.min() < 0 or triangles.max() >= len(points):
        raise RuntimeError("Invalid VTP polygon connectivity")
    return points.astype(np.float64), triangles


def load_largest_dicom_series(root):
    import SimpleITK as sitk
    candidates = []
    for directory, _, files_in_directory in os.walk(root):
        if not files_in_directory:
            continue
        for series_id in sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory) or []:
            files = list(
                sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, series_id)
            )
            candidates.append((len(files), directory, series_id, files))
    if not candidates:
        raise RuntimeError(f"No DICOM series found below {root}")
    count, directory, series_id, files = max(candidates, key=lambda x: x[0])
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    if image.GetDimension() != 3 or image.GetSize()[2] < 2:
        raise RuntimeError(f"Selected series is not 3-D: {image.GetSize()}")
    return image, {
        "directory": directory, "series_id": series_id, "number_of_files": count
    }


def normalize_structural(raw, threshold, mean, std):
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid structural std: {std}")
    return np.where(raw > threshold, (raw - mean) / std, 0.0).astype(np.float32)


def foreground_zscore_planes(raw_planes, foreground_ratio, eps=1e-8):
    """Apply the original NPZ preprocessing independently to every plane."""
    if not np.isfinite(foreground_ratio) or not 0 < foreground_ratio < 1:
        raise ValueError(f"Invalid structural_foreground_ratio: {foreground_ratio}")
    normalized = np.zeros_like(raw_planes, dtype=np.float32)
    statistics = []
    for index, image in enumerate(raw_planes):
        finite = np.isfinite(image)
        if not np.any(finite):
            raise RuntimeError(f"Structural plane {index} has no finite pixels")
        maximum = float(image[finite].max())
        if maximum <= 0:
            raise RuntimeError(
                f"Structural plane {index} maximum is not positive: {maximum}"
            )
        threshold = maximum * foreground_ratio
        foreground = finite & (image > threshold)
        if not np.any(foreground):
            raise RuntimeError(
                f"Structural plane {index} has no pixels above {threshold}"
            )
        values = image[foreground]
        mean = float(values.mean())
        std = float(values.std())
        if std < eps:
            raise RuntimeError(
                f"Structural plane {index} foreground std is too small: {std}"
            )
        normalized[index, foreground] = (values - mean) / (std + eps)
        statistics.append({
            "plane_index": index,
            "maximum": maximum,
            "threshold": threshold,
            "foreground_mean": mean,
            "foreground_std": std,
            "foreground_pixels": int(foreground.sum()),
        })
    return normalized, statistics


def reference_alignment(reference_image, reference):
    """Verify the exact array convention used to build the training NPZ."""
    import SimpleITK as sitk
    raw = sitk.GetArrayFromImage(reference_image)
    if raw.ndim == 3:
        if raw.shape[0] != 1:
            raise RuntimeError(f"Reference DICOM must have one plane: {raw.shape}")
        raw = raw[0]
    expected = reference["data"][0].astype(np.float32)
    normal = normalize_structural(
        raw.astype(np.float32),
        float(reference["structural_foreground_threshold"]),
        float(reference["structural_foreground_mean"]),
        float(reference["structural_foreground_std"]),
    )
    candidates = {
        "identity": normal,
        "flip_rows": np.flipud(normal),
        "flip_cols": np.fliplr(normal),
        "flip_both": np.flip(normal, (0, 1)),
        "transpose": normal.T,
        "transpose_flip_rows": np.flipud(normal.T),
        "transpose_flip_cols": np.fliplr(normal.T),
        "transpose_flip_both": np.flip(normal.T, (0, 1)),
    }
    scores = {
        name: float(np.corrcoef(expected.ravel(), value.ravel())[0, 1])
        if value.shape == expected.shape else -1.0
        for name, value in candidates.items()
    }
    best = max(scores, key=scores.get)
    mae = float(np.mean(np.abs(expected - candidates[best])))
    if best != "transpose" or scores[best] < 0.999:
        raise RuntimeError(
            "Expected verified NPZ convention 'transpose', but detected "
            f"{best} (correlation={scores[best]:.6f})"
        )
    return {"array_transform": best, "correlation": scores[best], "mae": mae}


def apply_affine(matrix, point):
    return (matrix @ np.r_[point, 1.0])[:3]


def closest_point_on_line(point, entry, target):
    vector = target - entry
    fraction = float(np.dot(point - entry, vector) / np.dot(vector, vector))
    closest = entry + fraction * vector
    return closest, fraction, float(np.linalg.norm(point - closest))


def select_geometry(trajectories, registrations, center_lps, requested_name):
    choices = []
    for registration in registrations:
        matrix = registration["matrix"]
        for matrix_name, basis_to_lps in (
            ("regMatrix", matrix), ("inverse_regMatrix", np.linalg.inv(matrix))
        ):
            for trajectory in trajectories:
                if (
                    requested_name.lower() != "auto"
                    and trajectory["name"].lower() != requested_name.lower()
                ):
                    continue
                entry = apply_affine(basis_to_lps, trajectory["entry_basis_mm"])
                target = apply_affine(basis_to_lps, trajectory["target_basis_mm"])
                pivot, fraction, distance = closest_point_on_line(
                    center_lps, entry, target
                )
                choices.append({
                    "registration_uuid": registration["uuid"],
                    "matrix_direction": matrix_name,
                    "basis_to_lps": basis_to_lps,
                    "trajectory": trajectory,
                    "entry_lps_mm": entry,
                    "target_lps_mm": target,
                    "pivot_lps_mm": pivot,
                    "line_fraction": fraction,
                    "center_to_axis_mm": distance,
                })
    if not choices:
        raise RuntimeError(f"Requested trajectory not found: {requested_name}")
    choices.sort(key=lambda x: x["center_to_axis_mm"])
    selected = choices[0]
    if selected["center_to_axis_mm"] > 10.0:
        raise RuntimeError(
            "No trajectory/matrix interpretation is within 10 mm of the known "
            f"center; best={selected['center_to_axis_mm']:.3f} mm"
        )
    return selected, choices


def build_plane_basis(reference_image, entry, target):
    direction = np.asarray(reference_image.GetDirection()).reshape(3, 3)
    # NPZ array = DICOM array.T, so NPZ row is DICOM x and NPZ col is DICOM y.
    npz_row_direction = direction[:, 0]
    npz_col_direction = direction[:, 1]
    axis = target - entry
    axis /= np.linalg.norm(axis)
    # Depending on the thermometry plane export, the fibre axis is represented
    # by either NPZ rows or NPZ columns.  Choose the strongly aligned in-plane
    # direction and use the other direction as the initial radial vector.
    candidates = [(npz_row_direction, npz_col_direction, "row"),
                  (npz_col_direction, npz_row_direction, "column")]
    aligned_direction, radial_seed, aligned_name = max(
        candidates, key=lambda item: abs(float(np.dot(axis, item[0])))
    )
    if np.dot(axis, aligned_direction) < 0:
        axis *= -1.0
    alignment = float(np.dot(axis, aligned_direction))
    if alignment < 0.95:
        raise RuntimeError(
            "Trajectory is not aligned with either reference NPZ in-plane direction: "
            f"dot={alignment:.6f}"
        )
    radial_zero = radial_seed - np.dot(radial_seed, axis) * axis
    radial_zero /= np.linalg.norm(radial_zero)
    radial_quarter = np.cross(axis, radial_zero)
    radial_quarter /= np.linalg.norm(radial_quarter)
    return axis, radial_zero, radial_quarter, alignment


def physical_to_indices(image, points):
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    return ((points - origin) @ direction) / spacing


def rasterize_closed_mesh(points_xyz, triangles, output_size_xyz):
    """Voxelize a closed triangular surface by x-axis scanline parity."""
    points = np.asarray(points_xyz, dtype=np.float64)
    faces = points[np.asarray(triangles, dtype=np.int64)]
    size_x, size_y, size_z = map(int, output_size_xyz)
    mask = np.zeros((size_z, size_y, size_x), dtype=np.uint8)
    lower = np.maximum(np.floor(points.min(axis=0)).astype(int) - 1, 0)
    upper = np.minimum(
        np.ceil(points.max(axis=0)).astype(int) + 1,
        np.asarray([size_x - 1, size_y - 1, size_z - 1]),
    )
    y0, z0 = faces[:, 0, 1], faces[:, 0, 2]
    y1, z1 = faces[:, 1, 1], faces[:, 1, 2]
    y2, z2 = faces[:, 2, 1], faces[:, 2, 2]
    denominator = (y1 - y2) * (z0 - z2) + (z2 - z1) * (y0 - y2)
    valid_projection = np.abs(denominator) > 1e-10
    y_min, y_max = faces[:, :, 1].min(axis=1), faces[:, :, 1].max(axis=1)
    z_min, z_max = faces[:, :, 2].min(axis=1), faces[:, :, 2].max(axis=1)
    odd_scanlines = 0
    for z_index in range(lower[2], upper[2] + 1):
        z_value = float(z_index)
        z_candidates = valid_projection & (z_value >= z_min) & (z_value <= z_max)
        for y_index in range(lower[1], upper[1] + 1):
            y_value = float(y_index)
            candidates = (
                z_candidates & (y_value >= y_min) & (y_value <= y_max)
            )
            if not np.any(candidates):
                continue
            ids = np.flatnonzero(candidates)
            den = denominator[ids]
            weight0 = (
                (y1[ids] - y2[ids]) * (z_value - z2[ids])
                + (z2[ids] - z1[ids]) * (y_value - y2[ids])
            ) / den
            weight1 = (
                (y2[ids] - y0[ids]) * (z_value - z2[ids])
                + (z0[ids] - z2[ids]) * (y_value - y2[ids])
            ) / den
            weight2 = 1.0 - weight0 - weight1
            inside = (
                (weight0 >= -1e-8)
                & (weight1 >= -1e-8)
                & (weight2 >= -1e-8)
            )
            if not np.any(inside):
                continue
            selected_faces = faces[ids[inside]]
            intersections = (
                weight0[inside] * selected_faces[:, 0, 0]
                + weight1[inside] * selected_faces[:, 1, 0]
                + weight2[inside] * selected_faces[:, 2, 0]
            )
            intersections.sort()
            if intersections.size > 1:
                keep = np.r_[True, np.diff(intersections) > 1e-5]
                intersections = intersections[keep]
            if intersections.size % 2:
                odd_scanlines += 1
                intersections = intersections[:-1]
            for left, right in intersections.reshape(-1, 2):
                start = max(int(math.ceil(left)), 0)
                stop = min(int(math.floor(right)), size_x - 1)
                if stop >= start:
                    mask[z_index, y_index, start:stop + 1] = 1
    if not np.any(mask):
        raise RuntimeError("Tumor VTP rasterization produced an empty mask")
    return mask, {
        "mesh_bounds_index_xyz": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "mask_bounds_index_xyz": [lower.tolist(), upper.tolist()],
        "odd_scanlines": odd_scanlines,
        "tumor_voxels": int(mask.sum()),
    }


def sample_rotating_planes(
    image, volume_zyx, pivot, axis, radial_zero, radial_quarter,
    center_rc, shape, spacing_rc, number_of_planes, cval=0.0,
):
    from scipy.ndimage import map_coordinates
    height, width = shape
    row0, col0 = center_rc
    row_mm = (np.arange(height, dtype=np.float64) - row0) * spacing_rc[0]
    col_mm = (np.arange(width, dtype=np.float64) - col0) * spacing_rc[1]
    rr, cc = np.meshgrid(row_mm, col_mm, indexing="ij")
    angles = np.arange(number_of_planes) * math.pi / number_of_planes
    planes = []
    for angle in angles:
        radial = math.cos(angle) * radial_zero + math.sin(angle) * radial_quarter
        points = pivot + rr[..., None] * radial + cc[..., None] * axis
        xyz = physical_to_indices(image, points)
        coords = np.stack([xyz[..., 2], xyz[..., 1], xyz[..., 0]], axis=0)
        plane = map_coordinates(
            volume_zyx, coords, order=1, mode="constant", cval=float(cval),
            prefilter=False,
        )
        planes.append(plane.astype(np.float32))
    return angles.astype(np.float64), np.stack(planes)


def load_model_and_predict(
    cfg_path, weights_path, structural, baseline, mask, center_rc,
    power_w, time_index, batch_size,
):
    import torch
    from core.config import config
    from dataset.common import validate_condition_scales
    from dataset.dataloader import make_probe_heatmap
    from models import unet
    from utils.utils import determine_device, update_config

    update_config(config, cfg_path)
    device = determine_device()
    model = unet.UNet(config).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    state = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    if next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    if not (isinstance(checkpoint, dict) and "condition_scales" in checkpoint):
        raise RuntimeError(
            "Checkpoint has no condition_scales. Use a v2 checkpoint so power/time "
            "normalization remains training-split-only."
        )
    scales = validate_condition_scales(checkpoint["condition_scales"])
    height, width = structural.shape[-2:]
    heatmap = make_probe_heatmap(
        center_rc, height, width, config.MODEL.EXTRA.PROBE_SIGMA_PIXELS
    )[0]
    center_norm = np.asarray([
        2.0 * center_rc[0] / max(height - 1, 1) - 1.0,
        2.0 * center_rc[1] / max(width - 1, 1) - 1.0,
    ], dtype=np.float32)
    if baseline.ndim == 2:
        baseline_planes = np.repeat(baseline[None], len(structural), axis=0)
    elif baseline.ndim == 3 and baseline.shape == structural.shape:
        baseline_planes = baseline
    else:
        raise RuntimeError(
            f"baseline must be (H,W) or match structural {structural.shape}, "
            f"got {baseline.shape}"
        )
    if mask.ndim == 2:
        mask_planes = np.repeat(mask[None], len(structural), axis=0)
    elif mask.ndim == 3 and mask.shape == structural.shape:
        mask_planes = mask
    else:
        raise RuntimeError(
            f"mask must be (H,W) or match structural {structural.shape}, "
            f"got {mask.shape}"
        )
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(structural), batch_size):
            structural_batch = structural[start:start + batch_size]
            count = len(structural_batch)
            images = np.stack([
                structural_batch,
                baseline_planes[start:start + count],
                mask_planes[start:start + count],
                np.repeat(heatmap[None], count, axis=0),
            ], axis=1).astype(np.float32)
            image_tensor = torch.from_numpy(images).to(device)
            power = torch.full(
                (count, 1), power_w / scales["power"],
                dtype=torch.float32, device=device,
            )
            time = torch.full(
                (count, 1), time_index / scales["time"],
                dtype=torch.float32, device=device,
            )
            center = torch.from_numpy(
                np.repeat(center_norm[None], count, axis=0)
            ).to(device)
            prediction = model(image_tensor, power, center, time)[0]
            predictions.append(prediction[:, 0].cpu().numpy().astype(np.float32))
    return np.concatenate(predictions), scales, str(device)


def reconstruct_dense_volume(
    predictions, image, raw_volume, pivot, axis, radial_zero, radial_quarter,
    center_rc, spacing_rc, structural_threshold,
):
    """Interpolate the fan predictions in cylindrical coordinates."""
    from scipy.ndimage import map_coordinates
    number_of_planes, height, width = predictions.shape
    row0, col0 = center_rc
    max_radius = int(math.floor(min(height - 1 - row0, row0)))
    radii = np.arange(max_radius + 1)
    profiles = []
    for plane in predictions:
        profiles.append(plane[(row0 + radii).astype(int), :])
    for plane in predictions:
        profiles.append(plane[(row0 - radii).astype(int), :])
    profiles = np.stack(profiles).astype(np.float32)
    profiles = np.concatenate([profiles, profiles[:1]], axis=0)
    angular_step = 2.0 * math.pi / (2 * number_of_planes)

    size_x, size_y, size_z = image.GetSize()
    output = np.full((size_z, size_y, size_x), np.nan, dtype=np.float32)
    coverage = np.zeros_like(output, dtype=np.uint8)
    spacing_xyz = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    x_grid, y_grid = np.meshgrid(
        np.arange(size_x, dtype=np.float64),
        np.arange(size_y, dtype=np.float64), indexing="xy",
    )
    for z_index in range(size_z):
        indices = np.stack(
            [x_grid, y_grid, np.full_like(x_grid, z_index)], axis=-1
        )
        points = origin + (indices * spacing_xyz) @ direction.T
        displacement = points - pivot
        axial_mm = displacement @ axis
        radial_x = displacement @ radial_zero
        radial_y = displacement @ radial_quarter
        radius_pixels = np.sqrt(radial_x ** 2 + radial_y ** 2) / spacing_rc[0]
        angles = np.mod(np.arctan2(radial_y, radial_x), 2.0 * math.pi)
        angle_indices = angles / angular_step
        axial_indices = col0 + axial_mm / spacing_rc[1]
        valid = (
            (radius_pixels <= max_radius)
            & (axial_indices >= 0)
            & (axial_indices <= width - 1)
            & (raw_volume[z_index] > structural_threshold)
        )
        coordinates = np.stack(
            [angle_indices, radius_pixels, axial_indices], axis=0
        )
        sampled = map_coordinates(
            profiles, coordinates, order=1, mode="nearest", prefilter=False
        ).astype(np.float32)
        output[z_index][valid] = sampled[valid]
        coverage[z_index][valid] = 1
    return output, coverage


def save_qc_figure(path, angles, structural, predictions, masks, center_rc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    temperature_cmap = LinearSegmentedColormap.from_list(
        "temperature_green_red",
        ["#087f23", "#fff59d", "#ff7f00", "#b30000"],
    )
    selected = np.linspace(
        0, len(angles) - 1, min(6, len(angles)), dtype=int
    )
    fig, axes = plt.subplots(
        len(selected), 2, figsize=(10, 4 * len(selected)), squeeze=False
    )
    for row, index in enumerate(selected):
        mask = masks[index] if masks.ndim == 3 else masks
        axes[row, 0].imshow(structural[index], cmap="gray")
        axes[row, 0].contour(mask, levels=[0.5], colors=["cyan"], linewidths=0.8)
        axes[row, 0].scatter(
            center_rc[1], center_rc[0], marker="x", color="yellow"
        )
        axes[row, 0].set_title(
            f"Structural plane {math.degrees(angles[index]):.1f} deg"
        )
        shown = axes[row, 1].imshow(
            predictions[index], cmap=temperature_cmap, vmin=30, vmax=50
        )
        axes[row, 1].contour(mask, levels=[0.5], colors=["cyan"], linewidths=0.8)
        axes[row, 1].scatter(
            center_rc[1], center_rc[0], marker="x", color="yellow"
        )
        axes[row, 1].set_title("Predicted temperature (deg C)")
        for axis_object in axes[row]:
            axis_object.axis("off")
    fig.colorbar(
        shown, ax=axes[:, 1], shrink=0.65, label="Temperature (deg C)"
    )
    fig.suptitle("Rotating planes around the selected planned trajectory")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_structural_qc(path, angles, structural, center_rc, masks=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = 4
    rows = int(math.ceil(len(angles) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.5 * rows), squeeze=False)
    for index, axis_object in enumerate(axes.ravel()):
        axis_object.axis("off")
        if index >= len(angles):
            continue
        axis_object.imshow(structural[index], cmap="gray", vmin=-1.3, vmax=3.0)
        if masks is not None and np.any(masks[index]):
            axis_object.contour(
                masks[index], levels=[0.5], colors=["cyan"], linewidths=0.8
            )
        axis_object.scatter(
            center_rc[1], center_rc[0], marker="x", color="yellow", s=28
        )
        axis_object.set_title(f"{math.degrees(angles[index]):.1f} deg")
    fig.suptitle("3-D T2 resampled into planes around the planned trajectory")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def geometry_report(
    selected, alternatives, axis_alignment, center_lps,
    rotation_center_lps, center_mode,
):
    return {
        "selected_trajectory": selected["trajectory"]["name"],
        "registration_uuid": selected["registration_uuid"],
        "matrix_direction": selected["matrix_direction"],
        "entry_basis_mm": selected["trajectory"]["entry_basis_mm"].tolist(),
        "target_basis_mm": selected["trajectory"]["target_basis_mm"].tolist(),
        "entry_lps_mm": selected["entry_lps_mm"].tolist(),
        "target_lps_mm": selected["target_lps_mm"].tolist(),
        "reference_center_lps_mm": center_lps.tolist(),
        "axis_pivot_lps_mm": selected["pivot_lps_mm"].tolist(),
        "rotation_center_lps_mm": rotation_center_lps.tolist(),
        "center_mode": center_mode,
        "center_to_axis_mm": selected["center_to_axis_mm"],
        "reference_column_axis_dot": axis_alignment,
        "candidates": [
            {
                "trajectory": item["trajectory"]["name"],
                "matrix_direction": item["matrix_direction"],
                "center_to_axis_mm": item["center_to_axis_mm"],
            }
            for item in alternatives
        ],
    }


def main(args):
    import SimpleITK as sitk
    if args.num_planes < 3:
        raise ValueError("--num-planes must be at least 3")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    reference_npz_path = require_file(args.reference_npz, "Reference NPZ")
    reference_dicom_path = require_file(args.reference_dicom, "Reference DICOM")
    export_info_path = require_file(args.export_info, "export_info")
    if args.structural_zip:
        dicom_root = safe_extract(
            require_file(args.structural_zip, "Structural ZIP"),
            os.path.join(output_dir, "extracted_dicom"),
        )
    else:
        dicom_root = os.path.abspath(args.structural_dicom_dir)
        if not os.path.isdir(dicom_root):
            raise NotADirectoryError(dicom_root)

    volume_image, series = load_largest_dicom_series(dicom_root)
    raw_volume = sitk.GetArrayFromImage(volume_image).astype(np.float32)
    reference_image = sitk.ReadImage(reference_dicom_path)
    with np.load(reference_npz_path, allow_pickle=False) as archive:
        reference = {key: np.asarray(archive[key]) for key in archive.files}
    data = reference["data"].astype(np.float32)
    if data.ndim != 3 or data.shape[0] != 4:
        raise RuntimeError(f"Expected reference data (4,H,W), got {data.shape}")
    power_w = (
        float(reference["power_W"]) if args.power_W is None else args.power_W
    )
    time_index = (
        float(reference["time_index"])
        if args.time_index is None else args.time_index
    )
    if power_w <= 0 or time_index <= 0:
        raise ValueError("power_W and time_index must be positive")
    reference_baseline = data[1]

    array_alignment = reference_alignment(reference_image, reference)
    center_rc = reference["probe_center_rc"].astype(np.float64)
    # Verified transpose: NPZ (row,col) maps to DICOM continuous index (x=row,y=col).
    center_lps = np.asarray(
        reference_image.TransformContinuousIndexToPhysicalPoint(
            (float(center_rc[0]), float(center_rc[1]), 0.0)
        )
    )
    trajectories, registrations = read_export_info(export_info_path)
    patient_temperature = read_patient_temperature(export_info_path)
    selected, alternatives = select_geometry(
        trajectories, registrations, center_lps, args.trajectory
    )
    axis, radial_zero, radial_quarter, axis_alignment = build_plane_basis(
        reference_image, selected["entry_lps_mm"], selected["target_lps_mm"]
    )
    reference_spacing = reference_image.GetSpacing()
    spacing_rc = (float(reference_spacing[0]), float(reference_spacing[1]))
    rotation_center = (
        center_lps
        if args.center_mode == "reference"
        else selected["pivot_lps_mm"]
    )
    angles, raw_planes = sample_rotating_planes(
        volume_image, raw_volume, rotation_center, axis,
        radial_zero, radial_quarter, center_rc, data.shape[-2:], spacing_rc,
        args.num_planes,
    )
    foreground_ratio = float(reference["structural_foreground_ratio"])
    structural, plane_statistics = foreground_zscore_planes(
        raw_planes, foreground_ratio
    )
    reference_background = data[0] == 0
    if not np.any(reference_background):
        raise RuntimeError("Reference structural image has no background pixels")
    background_temperature = float(
        np.median(reference_baseline[reference_background])
    )
    finite_volume = np.isfinite(raw_volume)
    volume_maximum = float(raw_volume[finite_volume].max())
    volume_foreground_threshold = foreground_ratio * volume_maximum
    volume_foreground = finite_volume & (
        raw_volume > volume_foreground_threshold
    )
    baseline_volume = np.full(
        raw_volume.shape, background_temperature, dtype=np.float32
    )
    baseline_volume[volume_foreground] = patient_temperature
    _, baseline_planes = sample_rotating_planes(
        volume_image, baseline_volume, rotation_center, axis,
        radial_zero, radial_quarter, center_rc, data.shape[-2:],
        spacing_rc, args.num_planes, cval=background_temperature,
    )
    tumor_report = None
    if args.tumor_vtp:
        tumor_vtp_path = require_file(args.tumor_vtp, "Tumor VTP")
        uuid_match = re.search(
            r"\{[0-9a-fA-F-]{36}\}", os.path.basename(tumor_vtp_path)
        )
        if not uuid_match:
            raise RuntimeError("Tumor VTP filename has no model UUID")
        model_metadata = read_volume_model_metadata(
            export_info_path, uuid_match.group(0)
        )
        if (model_metadata["coordinate_space"] or "").lower() != "basis":
            raise RuntimeError(
                "Tumor VTP was expected in Basis coordinates, got "
                f"{model_metadata['coordinate_space']!r}"
            )
        points_basis, triangles = read_vtp_mesh(tumor_vtp_path)
        points_lps = (
            np.c_[points_basis, np.ones(len(points_basis))]
            @ selected["basis_to_lps"].T
        )[:, :3]
        points_index = physical_to_indices(volume_image, points_lps)
        tumor_mask_3d, voxelization = rasterize_closed_mesh(
            points_index, triangles, volume_image.GetSize()
        )
        voxel_volume = float(np.prod(volume_image.GetSpacing()))
        rasterized_volume = float(tumor_mask_3d.sum() * voxel_volume)
        expected_volume = model_metadata["contour_volume_mm3"]
        relative_error = (
            abs(rasterized_volume - expected_volume) / expected_volume
            if expected_volume and expected_volume > 0 else None
        )
        if relative_error is not None and relative_error > 0.10:
            raise RuntimeError(
                "Rasterized tumor volume differs from export_info by more than "
                f"10%: rasterized={rasterized_volume:.3f}, "
                f"expected={expected_volume:.3f} mm3"
            )
        _, sampled_masks = sample_rotating_planes(
            volume_image, tumor_mask_3d.astype(np.float32), rotation_center,
            axis, radial_zero, radial_quarter, center_rc, data.shape[-2:],
            spacing_rc, args.num_planes,
        )
        mask_planes = (sampled_masks > 0.5).astype(np.float32)
        if not np.any(mask_planes):
            raise RuntimeError("3-D tumor does not intersect any rotating plane")
        tumor_image = sitk.GetImageFromArray(tumor_mask_3d)
        tumor_image.CopyInformation(volume_image)
        sitk.WriteImage(
            tumor_image, os.path.join(output_dir, "tumor_mask_3d.nii.gz")
        )
        tumor_report = {
            **model_metadata,
            **voxelization,
            "points": int(len(points_basis)),
            "triangles": int(len(triangles)),
            "mesh_bounds_basis_mm": [
                points_basis.min(axis=0).tolist(),
                points_basis.max(axis=0).tolist(),
            ],
            "mesh_bounds_lps_mm": [
                points_lps.min(axis=0).tolist(),
                points_lps.max(axis=0).tolist(),
            ],
            "rasterized_volume_mm3": rasterized_volume,
            "volume_relative_error": relative_error,
            "plane_tumor_pixels": [
                int(mask.sum()) for mask in mask_planes
            ],
        }
        auxiliary_input_mode = (
            "synthetic_3d_recorded_temperature_plus_true_3d_vtp_tumor_masks"
        )
    else:
        mask_planes = np.repeat(
            (data[3:4] > 0.5).astype(np.float32),
            args.num_planes, axis=0,
        )
        auxiliary_input_mode = (
            "synthetic_3d_recorded_temperature_plus_reused_reference_2d_mask"
        )
    report = geometry_report(
        selected, alternatives, axis_alignment, center_lps,
        rotation_center, args.center_mode,
    )
    report.update({
        "reference_alignment": array_alignment,
        "series": series,
        "volume_size_xyz": list(volume_image.GetSize()),
        "volume_spacing_xyz_mm": list(volume_image.GetSpacing()),
        "npz_center_rc": center_rc.tolist(),
        "plane_spacing_rc_mm": list(spacing_rc),
        "number_of_planes": args.num_planes,
        "angles_degrees": np.degrees(angles).tolist(),
        "auxiliary_input_mode": auxiliary_input_mode,
        "tumor_vtp": tumor_report,
        "initial_state": {
            "source":
                "3-D synthetic field using export_info patientTemprature",
            "baseline_file": str(reference.get("baseline_file", "")),
            "patient_temperature_C": patient_temperature,
            "background_temperature_C": background_temperature,
            "background_temperature_source":
                "median of reference baseline where structural_zscore is zero",
            "volume_foreground_rule":
                "raw_3d_structural > foreground_ratio * volume_maximum",
            "volume_foreground_threshold":
                volume_foreground_threshold,
            "baseline_plane_min_C": float(np.min(baseline_planes)),
            "baseline_plane_max_C": float(np.max(baseline_planes)),
            "power_W": power_w,
            "time_index": time_index,
            "frame_interval_seconds": 3.6,
            "elapsed_time_seconds": time_index * 3.6,
            "gaussian_center_rc": center_rc.tolist(),
            "gaussian_center_policy":
                "same physical reference center at the same pixel in every plane",
        },
        "structural_normalization": {
            "method": "per-plane foreground Z-score",
            "foreground_rule": "pixel > foreground_ratio * plane_maximum",
            "background_value": 0.0,
            "foreground_ratio": foreground_ratio,
            "plane_statistics": plane_statistics,
        },
    })
    report_path = os.path.join(output_dir, "geometry_report.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    np.savez_compressed(
        os.path.join(output_dir, "rotating_structural_planes.npz"),
        structural_zscore=structural,
        baseline_temperature_C=baseline_planes,
        tumor_mask=mask_planes,
        angles_degrees=np.degrees(angles).astype(np.float32),
        probe_center_rc=center_rc.astype(np.float32),
    )
    save_structural_qc(
        os.path.join(output_dir, "rotating_structural_planes_qc.png"),
        angles, structural, center_rc, mask_planes,
    )
    print(
        f"Selected {selected['trajectory']['name']} using "
        f"{selected['matrix_direction']}; center-to-axis="
        f"{selected['center_to_axis_mm']:.3f} mm"
    )
    print(
        f"NPZ/DICOM alignment={array_alignment['array_transform']}, "
        f"correlation={array_alignment['correlation']:.6f}, "
        f"MAE={array_alignment['mae']:.6g}"
    )
    if args.geometry_only:
        print(f"Geometry-only outputs saved to {output_dir}")
        return

    weights = require_file(args.weights, "Model weights")
    predictions, condition_scales, device = load_model_and_predict(
        args.cfg, weights, structural, baseline_planes, mask_planes, center_rc,
        power_w, time_index, args.batch_size,
    )
    np.savez_compressed(
        os.path.join(output_dir, "rotating_plane_predictions.npz"),
        structural_zscore=structural,
        baseline_temperature_C=baseline_planes,
        tumor_mask=mask_planes,
        predicted_temperature_C=predictions,
        angles_degrees=np.degrees(angles).astype(np.float32),
        probe_center_rc=center_rc.astype(np.float32),
        power_W=np.float32(power_w),
        time_index=np.float32(time_index),
    )
    reconstructed, coverage = reconstruct_dense_volume(
        predictions, volume_image, raw_volume, rotation_center, axis,
        radial_zero, radial_quarter, center_rc, spacing_rc,
        float(raw_volume[np.isfinite(raw_volume)].max()) * foreground_ratio,
    )
    temperature_image = sitk.GetImageFromArray(reconstructed)
    temperature_image.CopyInformation(volume_image)
    sitk.WriteImage(
        temperature_image,
        os.path.join(output_dir, "predicted_temperature_3d.nii.gz"),
    )
    coverage_image = sitk.GetImageFromArray(coverage)
    coverage_image.CopyInformation(volume_image)
    sitk.WriteImage(
        coverage_image, os.path.join(output_dir, "coverage_mask_3d.nii.gz")
    )
    save_qc_figure(
        os.path.join(output_dir, "rotating_planes_qc.png"),
        angles, structural, predictions, mask_planes, center_rc,
    )
    report.update({
        "power_W": power_w,
        "time_index": time_index,
        "condition_scales": condition_scales,
        "device": device,
        "covered_voxels": int(coverage.sum()),
        "covered_fraction": float(coverage.mean()),
        "warning": (
            "Pseudo-3D exploratory result: baseline planes were sampled from a "
            "synthetic 3-D field using recorded patient temperature and the "
            "reference baseline background value. Tumor masks came from the "
            "true 3-D VTP when --tumor-vtp was supplied."
        ),
    })
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"3-D prediction and QC outputs saved to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
