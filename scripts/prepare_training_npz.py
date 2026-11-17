#!/usr/bin/env python3
"""Prepare one compressed NPZ per segmented LITT plane segment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from temp_net.cohort_config import load_cohort_config  # noqa: E402

T_RE = re.compile(r"T(\d+)", re.I)
POWER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:W|w|Ｗ|瓦)")
PATIENT_ALIASES: dict[str, str] = {}  # filled from the private cohort config
SIGMA_MM = 5.39064
TEMP_OFFSET = 35.0
TEMP_SCALE = 20.0


def patient_group(name: str) -> str:
    compact = re.sub(r"[^a-z0-9_]", "", name.lower())
    compact = PATIENT_ALIASES.get(compact, compact)
    return compact.replace("_", "")


def t_number(name: str) -> int:
    match = T_RE.search(name)
    if not match:
        raise ValueError(f"No T number in {name}")
    return int(match.group(1))


def read_temperature(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = np.squeeze(ds.pixel_array).astype(np.float32) / 10.0
    if arr.shape != (512, 512) or not np.isfinite(arr).all():
        raise ValueError(f"Invalid temperature array: {path} {arr.shape}")
    return arr


def read_structural(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = pydicom.dcmread(path)
    raw = np.squeeze(ds.pixel_array).astype(np.float32)
    raw = raw * float(ds.get("RescaleSlope", 1.0)) + float(ds.get("RescaleIntercept", 0.0))
    valid = np.isfinite(raw)
    threshold = 0.02 * float(np.max(raw[valid]))
    foreground = valid & (raw > threshold)
    if foreground.sum() < 100:
        raise ValueError(f"Invalid structural foreground: {path}")
    mean = float(raw[foreground].mean())
    std = float(raw[foreground].std())
    if std <= 1e-8:
        raise ValueError(f"Invalid structural standard deviation: {path}")
    zscore = np.zeros_like(raw, dtype=np.float32)
    zscore[foreground] = (raw[foreground] - mean) / std
    return zscore, foreground.astype(np.uint8), np.asarray(ds.PixelSpacing, dtype=np.float32)


def parse_power(readme: Path) -> float:
    text = readme.read_text(encoding="utf-8-sig", errors="replace")
    values = [float(x) for x in POWER_RE.findall(text)]
    if not values or not all(0 < value < 100 for value in values):
        raise ValueError(f"Cannot parse valid power from {readme}")
    return values[0]


def npz_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path: Path, arrays: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def choose_splits(groups: dict[str, dict], seed: int, locked_test: set[str]) -> dict[str, set[str]]:
    rng = np.random.default_rng(seed)
    all_groups = sorted(groups)
    locked = {name for name in locked_test if name in groups}
    remaining = [name for name in all_groups if name not in locked]

    all_batches = sorted({batch for value in groups.values() for batch in value["batch_frames"]})
    total_frames = sum(value["frames"] for value in groups.values())
    total_batch_frames = {batch: sum(value["batch_frames"].get(batch, 0) for value in groups.values()) for batch in all_batches}

    def assign_balanced(candidates, count, existing, fraction):
        chosen = set(existing)
        shuffled = list(candidates); rng.shuffle(shuffled)
        while len(chosen) < count:
            current_frames = sum(groups[name]["frames"] for name in chosen)
            current_batch_frames = {batch: sum(groups[name]["batch_frames"].get(batch, 0) for name in chosen) for batch in all_batches}
            represented = {batch for name in chosen for batch in groups[name]["batch_frames"]}
            best = None; best_score = None
            for name in shuffled:
                if name in chosen:
                    continue
                candidate = chosen | {name}
                new_frames = current_frames + groups[name]["frames"]
                new_batch_frames = {batch: current_batch_frames[batch] + groups[name]["batch_frames"].get(batch, 0) for batch in all_batches}
                target_frames = total_frames * fraction
                score = ((new_frames - target_frames) / max(target_frames, 1.0)) ** 2
                for batch in all_batches:
                    target = total_batch_frames[batch] * fraction
                    score += 2.0 * ((new_batch_frames[batch] - target) / max(target, 1.0)) ** 2
                # Make each split cover each available batch where possible.
                missing_batches = [batch for batch in all_batches if batch not in represented and batch not in groups[name]["batch_frames"]]
                score += 3.0 * len(missing_batches)
                if best_score is None or score < best_score:
                    best, best_score = name, score
            chosen.add(best)
        return chosen

    test = assign_balanced(remaining, 12, locked, 0.20)
    remaining2 = [name for name in remaining if name not in test]
    val = assign_balanced(remaining2, 6, set(), 0.10)
    train = set(all_groups) - test - val
    if train & test or train & val or test & val:
        raise RuntimeError("Patient split overlap")
    return {"train": train, "val": val, "test": test}


def load_center_manifest(path: Path) -> dict[str, np.ndarray]:
    """Load reviewed row/column centers keyed by sample_id."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = {}
        for sample_id, value in payload.items():
            if isinstance(value, dict):
                value = value.get("center_rc")
            result[str(sample_id)] = np.asarray(value, dtype=np.float32)
        return result
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            result[row["sample_id"]] = np.asarray(
                [float(row["center_row"]), float(row["center_col"])], dtype=np.float32
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--center-manifest",
        required=True,
        type=Path,
        help="JSON or CSV mapping each sample_id to the reviewed physical center in row/column pixels.",
    )
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--cohort-config", type=Path, default=None,
                        help="Private cohort config JSON (or set TEMP_NET_COHORT_CONFIG).")
    args = parser.parse_args()
    cohort = load_cohort_config(args.cohort_config, required=True)
    PATIENT_ALIASES.update(cohort["patient_aliases"])
    center_manifest = load_center_manifest(args.center_manifest)
    root = args.dataset_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    npz_root = output / "npz"
    npz_root.mkdir()

    records = []
    for batch in ("batch-01", "batch-02", "batch-03", "batch-04"):
        source_root = root / "raw/segmented_records" / batch
        for segment in sorted(source_root.glob("*/*/P*/segments/segment-*")):
            metadata_path = segment / "segment_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            patient = metadata["patient"]
            trajectory = metadata["trajectory"]
            plane = metadata["plane"]
            segment_name = metadata["segment"].replace("segment-", "S")
            group = patient_group(patient)
            sample_id = f"B{batch[-2:]}_{patient}_{trajectory}_{plane}_{segment_name}"
            sample_id = re.sub(r"[^A-Za-z0-9_.-]", "_", sample_id)
            npz_path = npz_root / f"{sample_id}.npz"

            structural_path = segment / "structural.dcm"
            baseline_path = segment / "baseline/baseline.dcm"
            mask_path = segment / "tumor_mask.nii.gz"
            readme_path = segment / "readme.txt"
            structural, foreground, spacing = read_structural(structural_path)
            baseline = read_temperature(baseline_path)
            mask = np.squeeze(nib.load(mask_path).get_fdata()).T
            tumor = (mask > 0).astype(np.float32)
            if tumor.shape != structural.shape or not tumor.any():
                raise ValueError(f"Invalid tumor mask: {segment}")
            temps = sorted((segment / "temperature").glob("T*.dcm"), key=lambda p: t_number(p.name))
            targets = np.stack([read_temperature(path) for path in temps]).astype(np.float32)
            target_numbers = np.asarray([t_number(path.name) for path in temps], dtype=np.int32)
            baseline_number = t_number(json.loads(metadata_path.read_text(encoding="utf-8"))["original_filenames"]["baseline"])
            time_indices = target_numbers - baseline_number
            if np.any(time_indices <= 0) or len(set(target_numbers.tolist())) != len(target_numbers):
                raise ValueError(f"Invalid target time indices: {segment}")

            center = center_manifest.get(sample_id)
            if center is None:
                metadata_center = metadata.get("probe_center_rc", metadata.get("center_rc"))
                if metadata_center is None:
                    raise KeyError(
                        f"No reviewed probe center for {sample_id}. Add it to --center-manifest "
                        "using center_rc=[row, col]."
                    )
                center = np.asarray(metadata_center, dtype=np.float32)
            center = np.asarray(center, dtype=np.float32)
            if center.shape != (2,) or not np.isfinite(center).all():
                raise ValueError(f"Invalid reviewed probe center for {sample_id}: {center}")
            if not (0 <= center[0] < structural.shape[0] and 0 <= center[1] < structural.shape[1]):
                raise ValueError(f"Reviewed probe center outside image for {sample_id}: {center}")
            rows = np.arange(structural.shape[0], dtype=np.float32)[:, None]
            cols = np.arange(structural.shape[1], dtype=np.float32)[None, :]
            distance_sq = ((rows - center[0]) * spacing[0]) ** 2 + ((cols - center[1]) * spacing[1]) ** 2
            gaussian = np.exp(-distance_sq / (2.0 * 5.39064 ** 2)).astype(np.float32)
            # Scale is filled after the split is known; use physical power for now.
            power = parse_power(readme_path)
            static_raw = {
                "structural_zscore": structural,
                "baseline_C": baseline,
                "tumor_mask": tumor.astype(np.uint8),
                "foreground_mask": foreground,
                "probe_gaussian": gaussian,
                "power_W": np.float32(power),
                "targets_C": targets,
                "target_time_indices": time_indices,
                "probe_center_rc": center.astype(np.float32),
                "spacing_rc_mm": spacing.astype(np.float32),
                "target_frame_names": np.asarray([p.name for p in temps]),
                "baseline_frame_name": np.asarray(metadata["original_filenames"]["baseline"]),
                "patient": np.asarray(patient),
                "patient_group": np.asarray(group),
                "trajectory": np.asarray(trajectory),
                "plane": np.asarray(plane),
                "batch": np.int32(int(batch[-2:])),
                "format_version": np.asarray("autoreg_segment_v1"),
                "source_segment": np.asarray(str(segment)),
            }
            save_npz(npz_path, static_raw)
            missing = sorted(set(range(1, int(time_indices.max()) + 1)) - set(time_indices.tolist()))
            records.append({
                "sample_id": sample_id,
                "npz_file": str(npz_path.relative_to(output)),
                "batch": int(batch[-2:]),
                "patient": patient,
                "patient_group": group,
                "trajectory": trajectory,
                "plane": plane,
                "segment": segment_name,
                "frames": int(len(targets)),
                "time_min": int(time_indices.min()),
                "time_max": int(time_indices.max()),
                "missing_time_indices": ",".join(map(str, missing)),
                "power_W": power,
                "height": int(structural.shape[0]),
                "width": int(structural.shape[1]),
                "source_segment": str(segment),
                "npz_sha256": npz_sha256(npz_path),
            })
            if len(records) % 25 == 0:
                print(f"prepared {len(records)} segments", flush=True)

    groups = {}
    for group in sorted({row["patient_group"] for row in records}):
        group_rows = [row for row in records if row["patient_group"] == group]
        batch_frames = defaultdict(int)
        for row in group_rows:
            batch_frames[row["batch"]] += row["frames"]
        groups[group] = {"frames": sum(batch_frames.values()), "batch_frames": dict(batch_frames)}
    locked_test = set(cohort["locked_test_patients"])
    splits = choose_splits(groups, args.seed, locked_test)
    split_for_group = {group: split for split, names in splits.items() for group in names}
    for row in records:
        row["split"] = split_for_group[row["patient_group"]]

    # Power source maps are generated with the training split maximum and saved in each NPZ.
    power_scale = max(row["power_W"] for row in records if row["split"] == "train")
    time_scale = max(row["time_max"] for row in records if row["split"] == "train")
    for row in records:
        path = output / row["npz_file"]
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["baseline_norm"] = ((arrays["baseline_C"] - TEMP_OFFSET) / TEMP_SCALE).astype(np.float32)
        arrays["power_source"] = arrays["probe_gaussian"] * (float(arrays["power_W"]) / power_scale)
        arrays["missing_time_indices"] = np.asarray(
            [int(x) for x in row["missing_time_indices"].split(",") if x], dtype=np.int32
        )
        save_npz(path, arrays)
        row["npz_sha256"] = npz_sha256(path)

    fields = list(records[0])
    (output / "manifests").mkdir(exist_ok=True)
    with (output / "manifests/segment_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    rollout_rows = []
    for row in records:
        times = [0] + [int(x) for x in np.load(output / row["npz_file"], allow_pickle=False)["target_time_indices"]]
        for from_time, to_time in zip(times, times[1:]):
            rollout_rows.append({
                "sample_id": row["sample_id"],
                "split": row["split"],
                "from_time_index": from_time,
                "to_time_index": to_time,
                "rollout_steps": to_time - from_time,
                "missing_steps_between": max(0, to_time - from_time - 1),
                "loss_enabled": True,
            })
    with (output / "manifests/rollout_index.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rollout_rows[0]))
        writer.writeheader(); writer.writerows(rollout_rows)
    split_summary = []
    for split, names in splits.items():
        subset = [row for row in records if row["split"] == split]
        split_summary.append({"split": split, "patients": len(names), "segments": len(subset), "frames": sum(row["frames"] for row in subset), "patient_groups": ",".join(sorted(names))})
    (output / "splits").mkdir(exist_ok=True)
    with (output / "splits/patient_split_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_summary[0])); writer.writeheader(); writer.writerows(split_summary)
    split_protocol = {"seed": args.seed, "locked_test_patients": sorted(locked_test), "splits": {key: sorted(value) for key, value in splits.items()}, "patient_group_rule": cohort["patient_group_rule"]}
    (output / "splits/patient_split.json").write_text(json.dumps(split_protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset_summary = {
        "format_version": "autoreg_segment_v1",
        "segments": len(records),
        "patients": len(groups),
        "temperature_frames": sum(row["frames"] for row in records),
        "power_scale_train_max_W": power_scale,
        "time_scale_train_max_index": time_scale,
        "probe_sigma_mm": SIGMA_MM,
        "split_summary": split_summary,
        "known_policy": "missing target frames are not synthesized; rollout advances fixed 3.6 s steps and loss is applied only to observed targets",
    }
    (output / "dataset_summary.json").write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "normalization.json").write_text(json.dumps({"temperature_offset_C": TEMP_OFFSET, "temperature_scale_C": TEMP_SCALE, "power_scale_train_max_W": power_scale, "time_scale_train_max_index": time_scale, "probe_sigma_mm": SIGMA_MM}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
