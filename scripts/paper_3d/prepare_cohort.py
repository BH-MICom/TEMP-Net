"""Create a provenance-preserving cohort plan. Never alter source data."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import re
import zipfile
import os

import nibabel as nib
import numpy as np
import pydicom

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from temp_net.cohort_config import load_cohort_config  # noqa: E402

ROOT = Path(os.environ.get("TEMP_NET_DATA_ROOT", "."))
TRAIN = ROOT / "LITT_2D_4PLANE_DATASET/training_ready_autoreg_v1_20260925"
GEOMETRY = Path(os.environ.get("TEMP_NET_GEOMETRY", Path(__file__).resolve().parent / "support" / "test_rotating_planes_3d.py"))


COHORT = load_cohort_config()


def canonical(name):
    key = re.sub("[^a-z0-9]", "", name.lower())
    for old, new in COHORT["canonical_aliases"].items():
        key = key.replace(old, new)
    return key


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_image(path):
    path = Path(path)
    with path.open("rb") as f:
        signature = f.read(2)
    return nib.load(str(path)) if signature == b"\x1f\x8b" else nib.Nifti1Image.from_bytes(path.read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def series_date(uid):
    match = re.search(r"(20\d{6})\d{6}", uid)
    return match.group(1) if match else None


def main(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    extracted = out / "inputs/revision_20260924"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            target = (extracted / item.filename).resolve()
            if not target.is_relative_to(extracted.resolve()):
                raise ValueError("Unsafe ZIP member")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(archive.read(item))
    updates = list(extracted.rglob("*.nii")) + list(extracted.rglob("*.nii.gz"))
    revision = {canonical(p.name.split("_")[0].split("-")[0]): [] for p in updates}
    for p in updates:
        revision[canonical(p.name.split("_")[0].split("-")[0])].append(p)
    supplement = out / "inputs/revision_202610"
    october = dict(COHORT["supplementary_masks"])
    series = list(csv.DictReader((ROOT / "LITT_3D_DATASET/manifests/series_manifest.csv").open(encoding="utf-8-sig")))
    rows = list(csv.DictReader((TRAIN / "manifests/segment_manifest.csv").open(encoding="utf-8-sig")))
    mapping = list(csv.DictReader((ROOT / "LITT_3D_DATASET/manifests/trajectory_mapping.csv").open(encoding="utf-8-sig")))
    split = json.loads((TRAIN / "splits/patient_split.json").read_text())["splits"]
    patient_lookup = {canonical(r["patient"]): r["patient"] for r in series}
    overview_roots = [Path(value) for value in os.environ.get("TEMP_NET_OVERVIEW_ROOTS", "").split(os.pathsep) if value]
    if not overview_roots:
        raise ValueError("Set TEMP_NET_OVERVIEW_ROOTS to one or more overview roots")
    exports = []
    spec = importlib.util.spec_from_file_location("geometry", GEOMETRY)
    geom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geom)
    for overview_root in overview_roots:
        for export in overview_root.glob("*/export_info"):
            try:
                trajectories, registrations = geom.read_export_info(export)
                exports.append({"path": str(export), "name": export.parent.name,
                                "patient_key": canonical(re.split("_traj", export.parent.name, flags=re.I)[0]),
                                "temperature": geom.read_patient_temperature(export),
                                "trajectory_names": [t["name"] for t in trajectories],
                                "registration_uids": [r["uuid"] for r in registrations]})
            except Exception as e:
                exports.append({"path": str(export), "error": str(e), "name": export.parent.name})
    groups = collections.defaultdict(list)
    unmapped = []
    for row in rows:
        key = canonical(row["patient"])
        key = COHORT["row_aliases"].get(key, key)
        patient = patient_lookup.get(key)
        if patient is None:
            unmapped.append(row)
            continue
        candidates = [s for s in series if s["patient"] == patient]
        if len(candidates) == 1:
            session = candidates[0]["session"]
        elif patient in COHORT["session_by_suffix"]:
            session = "ses-01" if row["patient"].endswith("_1") else "ses-02"
        else:
            ds = pydicom.dcmread(Path(row["source_segment"]) / "structural.dcm", stop_before_pixels=True)
            dates = {str(getattr(ds, k, "")) for k in ("StudyDate", "SeriesDate", "AcquisitionDate")}
            matched = [s for s in candidates if series_date(s["series_uid"]) in dates]
            if len(matched) != 1:
                unmapped.append({"reason": "ambiguous session date", "dicom_dates": sorted(dates), **row})
                continue
            session = matched[0]["session"]
        groups[(patient, session)].append(row)
    records = []
    for s in series:
        patient, session = s["patient"], s["session"]
        case = Path(s["case_dir"])
        ref = nib.load(case / "mni152/t2_1mm.nii.gz")
        record = {**s, "id": f"{patient}/{session}", "issues": [],
                  "model_split": next((k for k, v in split.items() if canonical(patient) in [canonical(x) for x in v]), "unknown"),
                  "mask_sources": [], "trajectories": []}
        existing = ROOT / "LITT_3D_NNUNET_SEGMENTATION/mni152_space" / patient / session / "prediction_TC.nii.gz"
        sources = []
        if patient in october and (supplement / october[patient]).exists():
            sources = [supplement / october[patient]]
            record["action"] = "predict"
        else:
            files = revision.get(canonical(patient), [])
            only_session = COHORT["revision_mask_session_only"].get(patient)
            if only_session is not None:
                files = files if session == only_session else []
            sources = files or [existing]
            if patient in COHORT["merge_existing_and_revision_masks"]:
                sources = [existing] + files
            record["action"] = "predict"
        if record["action"] == "predict":
            record["calculation_policy"] = "segment_baseline_no_waiting_dose"
        union = np.zeros(ref.shape, bool)
        for source in sources:
            image = load_image(source)
            if image.shape != ref.shape or not np.allclose(image.affine, ref.affine, atol=1e-4):
                record["issues"].append(f"mask_grid_mismatch:{source}")
                continue
            data = np.asarray(image.dataobj)
            if not np.isfinite(data).all() or not set(np.unique(data)) <= {0, 1}:
                record["issues"].append(f"nonbinary_or_nonfinite_mask:{source}")
                continue
            union |= data > 0
            record["mask_sources"].append({"path": str(source), "sha256": sha(source),
                                           "label_type": "WT" if "_WT" in source.name else "TC",
                                           "voxels": int((data > 0).sum())})
        if not union.any():
            record["issues"].append("empty_mask")
        dest = out / "inputs/masks" / patient / session / "tumor_mask_mni152.nii.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(union.astype(np.uint8), ref.affine, ref.header), dest)
        record.update(mask=str(dest), mask_sha256=sha(dest),
                      mask_volume_mL=float(union.sum() * abs(np.linalg.det(ref.affine[:3, :3])) / 1000))
        case_rows = groups[(patient, session)]
        chain_groups = collections.defaultdict(list)
        for r in case_rows:
            chain_groups[(r["batch"], r["patient"], r["trajectory"])].append(r)
        for (batch, source_patient, trajectory), chain_rows in sorted(chain_groups.items()):
            letter = re.match(r"Traj([A-Z])", trajectory).group(1)
            overview_names = [r["trajectory_overview"] for r in mapping if r["patient"] == patient and r["session"] == session]
            pool = [e for e in exports if "error" not in e and (
                e["name"] in overview_names or e.get("patient_key") in {canonical(source_patient), canonical(patient)})]
            requested = [e for e in pool if any(re.sub(r"\s+", "", t).lower() == ("traj" + letter).lower() for t in e["trajectory_names"])]
            preferred = [e for e in requested if re.search(rf"_traj{letter}_", e["name"], re.I)]
            if "_G" in trajectory:
                g = trajectory.rsplit("_G", 1)[1]
                preferred = [e for e in (preferred or requested) if e["name"].endswith("_" + g)]
            pool = preferred or requested
            # All geometrically plausible exports are retained; worker selects using
            # physical source position and reference-plane normals, never first-match.
            segments = collections.defaultdict(list)
            for r in chain_rows:
                segments[r["segment"]].append(r)
            chain = {"id": f"B{int(batch):02d}_{source_patient}_{trajectory}", "batch": int(batch),
                     "source_patient": source_patient, "trajectory": trajectory, "letter": letter,
                     "export_candidates": pool, "segments": [
                         {"segment": k, "planes": sorted(v, key=lambda r: r["plane"])}
                         for k, v in sorted(segments.items())]}
            if not pool:
                record["issues"].append(f"no_export_info_for:{chain['id']}")
            record["trajectories"].append(chain)
        record["logical_segments"] = sum(len(t["segments"]) for t in record["trajectories"])
        record["reference_plane_segments"] = len(case_rows)
        if not case_rows:
            record["issues"].append("no_training_ready_temperature_records")
        records.append(record)
    payload = {"expected_patients": 60, "expected_series": 62,
               "patients": len({r["patient"] for r in records}), "series": len(records),
               "thresholds_min": [2, 10, 60], "comparison": ">=",
               "weights": os.environ.get("TEMP_NET_WEIGHTS", ""),
               "num_planes": int(args.num_planes), "frame_seconds": 3.6, "holdout_plane": args.holdout_plane,
               "mask_input_policy": "full_updated_mask_unless_explicit_trajectory_mask",
               "records": records, "unmapped_training_rows": unmapped,
               "archive": str(args.archive), "archive_sha256": sha(args.archive)}
    write_json(out / "cohort_plan.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k not in ("records", "unmapped_training_rows")}, ensure_ascii=False))
    print("unmapped", len(unmapped), "issues", [(r["id"], r["issues"]) for r in records if r["issues"]], flush=True)
    print("chains", sum(len(r["trajectories"]) for r in records), "segments", sum(r["logical_segments"] for r in records))
    print("mask_revisions", [(r["id"], [Path(s["path"]).name for s in r["mask_sources"]]) for r in records if any("revision_" in x["path"] for x in r["mask_sources"])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-planes", type=int, default=180)
    parser.add_argument("--holdout-plane", type=int, choices=[1, 2], default=None)
    main(parser.parse_args())
