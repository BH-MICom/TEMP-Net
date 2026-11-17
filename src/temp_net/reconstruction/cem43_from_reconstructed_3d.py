"""Recalculate CEM43 from a reconstructed 3-D temperature at every time point."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import batch_predict_3d_cem43 as engine


FLOAT32_MAX = np.finfo(np.float32).max


def _finite_float32(values):
    """Preserve a readable NIfTI without turning extreme CEM43 into infinity."""
    return np.nan_to_num(
        values, nan=0.0, posinf=FLOAT32_MAX, neginf=0.0
    ).clip(0.0, FLOAT32_MAX).astype(np.float32)


def _segment_report_map(report):
    return {item["source_name"]: item for item in report.get("segments", [])}


def recompute_chain_cem43_3d(args, segments, runner, chain_output, report):
    """Overwrite segment CEM43 files using frame-by-frame reconstructed 3-D T."""
    chain_output = Path(chain_output)
    reports = _segment_report_map(report)
    if len(reports) < len(segments):
        for summary_path in sorted(chain_output.glob("[0-9][0-9][0-9]_*/summary.json")):
            item = json.loads(summary_path.read_text(encoding="utf-8"))
            reports[item["source_name"]] = item
    expected_sources = [segment.source_name for segment in segments]
    if all(source in reports for source in expected_sources) and report.get("cem43_domain") == (
        "frame-by-frame reconstructed 3-D temperature"
    ):
        print(f"[{segments[0].patient}/Traj{segments[0].trajectory}] 3-D CEM43 already complete")
        return report

    patient, trajectory = segments[0].patient, segments[0].trajectory
    overview = engine.find_overview_directory(Path(args.patient_root), patient, trajectory)
    export_info = overview / "export_info"
    volume_image, _ = engine.load_largest_dicom_series(str(overview / "T2_3D"))
    raw_volume = sitk.GetArrayFromImage(volume_image).astype(np.float32)
    trajectories, registrations = engine.read_export_info(export_info)
    selection_center = np.median(
        np.asarray([plane.center_lps for segment in segments for plane in segment.planes.values()]),
        axis=0,
    )
    selected, _ = engine.select_geometry(
        trajectories, registrations, selection_center, f"Traj {trajectory}"
    )
    _, _, first_reference, _ = engine.select_axis_aligned_plane(
        segments[0], selected["entry_lps_mm"], selected["target_lps_mm"]
    )
    axis, radial_zero, radial_quarter, _ = engine.build_plane_basis(
        first_reference, selected["entry_lps_mm"], selected["target_lps_mm"]
    )
    tumor_image = sitk.ReadImage(str(chain_output / "tumor_mask_3d.nii.gz"))
    tumor_volume = sitk.GetArrayFromImage(tumor_image).astype(np.uint8)
    finite = np.isfinite(raw_volume)
    structural_threshold = args.foreground_ratio * float(raw_volume[finite].max())
    patient_temperature = float(report["patient_temperature_C"])
    background_temperature = float(report["background_temperature_C"])
    volume_foreground = finite & (raw_volume > structural_threshold)
    initial_temperature = np.full(raw_volume.shape, background_temperature, dtype=np.float32)
    initial_temperature[volume_foreground] = patient_temperature
    cumulative = np.zeros(raw_volume.shape, dtype=np.float64)
    dt_minutes = args.frame_seconds / 60.0
    progress_path = chain_output / "cem43_3d_progress.json"
    start_index = 0
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start_index = int(progress.get("processed_segments", 0))
        if start_index:
            cumulative = sitk.GetArrayFromImage(
                sitk.ReadImage(progress["cumulative_cem43_path"])
            ).astype(np.float64)
            print(f"[{patient}/Traj{trajectory}] resume 3-D CEM43 at segment {start_index + 1}")

    previous_segment_final = initial_temperature
    if start_index:
        previous_report = reports[segments[start_index - 1].source_name]
        previous_segment_final = sitk.GetArrayFromImage(
            sitk.ReadImage(str(Path(previous_report["output_directory"]) / "final_temperature_3d.nii.gz"))
        ).astype(np.float32)

    for segment_index, segment in enumerate(segments):
        if segment_index < start_index:
            continue
        segment_report = reports[segment.source_name]
        output_folder = Path(segment_report["output_directory"])
        print(
            f"[{patient}/Traj{trajectory}] 3-D CEM43 {segment_index + 1}/{len(segments)} "
            f"{segment.source_name}: {len(segment.frame_numbers)} reconstructions"
        )
        _, fan_reference_plane, reference, _ = engine.select_axis_aligned_plane(
            segment, selected["entry_lps_mm"], selected["target_lps_mm"]
        )
        shape = engine.read_structural(fan_reference_plane.structural_path).shape
        spacing_rc = tuple(float(value) for value in reference.GetSpacing()[:2])
        center_rc = fan_reference_plane.center_rc.astype(np.float64)
        rotation_center = segment.center_lps.astype(np.float64)
        _, raw_fan = engine.sample_rotating_planes(
            volume_image, raw_volume, rotation_center, axis, radial_zero,
            radial_quarter, center_rc, shape, spacing_rc, args.num_planes,
        )
        structural_fan = engine.foreground_zscore_planes(
            raw_fan, args.foreground_ratio
        )[0]
        _, baseline_fan = engine.sample_rotating_planes(
            volume_image, previous_segment_final, rotation_center, axis,
            radial_zero, radial_quarter, center_rc, shape, spacing_rc,
            args.num_planes, cval=background_temperature,
        )
        _, tumor_fan_raw = engine.sample_rotating_planes(
            volume_image, tumor_volume.astype(np.float32), rotation_center, axis,
            radial_zero, radial_quarter, center_rc, shape, spacing_rc,
            args.num_planes, cval=0,
        )
        tumor_fan = (tumor_fan_raw > 0.5).astype(np.float32)
        previous_frame_3d = previous_segment_final.copy()
        segment_increment = np.zeros(raw_volume.shape, dtype=np.float64)
        final_frame_3d = None
        for chunk_start in range(0, len(segment.frame_numbers), args.time_batch_size):
            chunk = segment.frame_numbers[chunk_start:chunk_start + args.time_batch_size]
            structural_batch = np.tile(structural_fan, (len(chunk), 1, 1))
            baseline_batch = np.tile(baseline_fan, (len(chunk), 1, 1))
            mask_batch = np.tile(tumor_fan, (len(chunk), 1, 1))
            centers = np.repeat(
                center_rc[None], len(chunk) * args.num_planes, axis=0
            )
            powers = np.full(
                len(chunk) * args.num_planes, segment.power_w, dtype=np.float32
            )
            times = np.repeat(
                np.arange(chunk_start + 1, chunk_start + 1 + len(chunk), dtype=np.float32),
                args.num_planes,
            )
            predictions = runner.predict(
                structural_batch, baseline_batch, mask_batch, centers, powers, times
            ).reshape(len(chunk), args.num_planes, *shape)
            for prediction_fan in predictions:
                reconstructed, coverage = engine.reconstruct_dense_volume(
                    prediction_fan, volume_image, raw_volume, rotation_center, axis,
                    radial_zero, radial_quarter, center_rc, spacing_rc,
                    structural_threshold,
                )
                valid = (coverage > 0) & np.isfinite(reconstructed)
                current_frame_3d = previous_segment_final.copy()
                current_frame_3d[valid] = reconstructed[valid]
                interval = valid & np.isfinite(previous_frame_3d)
                midpoint = 0.5 * (
                    previous_frame_3d[interval].astype(np.float64)
                    + current_frame_3d[interval].astype(np.float64)
                )
                segment_increment[interval] += dt_minutes * engine.thermal_dose_rate(midpoint)
                previous_frame_3d = current_frame_3d
                final_frame_3d = current_frame_3d
        if final_frame_3d is None:
            raise RuntimeError(f"No 3-D frames reconstructed: {segment.source_name}")
        cumulative += segment_increment
        existing_final = sitk.GetArrayFromImage(
            sitk.ReadImage(str(output_folder / "final_temperature_3d.nii.gz"))
        ).astype(np.float32)
        final_difference = float(np.nanmax(np.abs(existing_final - final_frame_3d)))
        engine.write_nifti(
            output_folder / "cem43_segment_increment_3d.nii.gz",
            _finite_float32(segment_increment), volume_image,
        )
        engine.write_nifti(
            output_folder / "cem43_cumulative_3d.nii.gz",
            _finite_float32(cumulative), volume_image,
        )
        engine.write_nifti(
            output_folder / "cem43_cumulative_log10_3d.nii.gz",
            np.log10(1.0 + np.maximum(cumulative, 0.0)).astype(np.float32), volume_image,
        )
        center_xyz = engine.physical_to_indices(
            volume_image, segment.center_lps[None]
        )[0]
        engine.save_3d_qc(
            output_folder / "final_temperature_and_cem43.png", raw_volume,
            existing_final, cumulative, tumor_volume, center_xyz,
            args.cem_threshold, args.qc_crop_size,
        )
        segment_report.update(
            {
                "cem43_domain": "frame-by-frame reconstructed 3-D temperature",
                "cem43_intervals": len(segment.frame_numbers),
                "cem43_segment_max_minutes": float(segment_increment.max()),
                "cem43_cumulative_max_minutes": float(cumulative.max()),
                "cem43_voxels_ge_threshold": int((cumulative >= args.cem_threshold).sum()),
                "cem43_raw_nifti_clipped_to_float32": bool(np.any(cumulative > FLOAT32_MAX)),
                "second_pass_final_temperature_max_difference_C": final_difference,
            }
        )
        engine.write_json(output_folder / "summary.json", segment_report)
        engine.write_json(
            progress_path,
            {
                "processed_segments": segment_index + 1,
                "cumulative_cem43_path": str(
                    (output_folder / "cem43_cumulative_3d.nii.gz").resolve()
                ),
            },
        )
        previous_segment_final = existing_final
    report["cem43_domain"] = "frame-by-frame reconstructed 3-D temperature"
    report["cem43_integration"] = (
        "For each 3.6-second interval: reconstruct current 3-D temperature, "
        "average previous/current 3-D temperatures, then apply the CEM43 rate."
    )
    engine.write_json(chain_output / "chain_summary.json", report)
    return report
