#!/usr/bin/env python3
"""Finalize an already prepared one-NPZ-per-segment autoregressive dataset."""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from temp_net.cohort_config import load_cohort_config  # noqa: E402

from prepare_training_npz import choose_splits


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--cohort-config", type=Path, default=None,
                    help="Private cohort config JSON (or set TEMP_NET_COHORT_CONFIG).")
    args = ap.parse_args()
    cohort = load_cohort_config(args.cohort_config, required=True)
    root = args.output.resolve()
    files = sorted(root.glob("npz/*.npz"))
    if len(files) != 1340:
        raise RuntimeError(f"Expected 1340 NPZ files, found {len(files)}")
    rows = []
    groups = {}
    for path in files:
        with np.load(path, allow_pickle=False) as a:
            required = {"structural_zscore", "baseline_C", "baseline_norm", "tumor_mask", "foreground_mask", "probe_gaussian", "power_source", "targets_C", "target_time_indices", "power_W", "patient", "patient_group", "trajectory", "plane", "batch", "source_segment"}
            missing = sorted(required - set(a.files))
            if missing:
                raise RuntimeError(f"{path.name} missing keys: {missing}")
            patient = str(a["patient"]); group = str(a["patient_group"]); trajectory = str(a["trajectory"]); plane = str(a["plane"]); batch = int(a["batch"]); source_segment = str(a["source_segment"])
            targets = a["targets_C"]; times = a["target_time_indices"].astype(np.int64)
            power = float(a["power_W"])
            if targets.ndim != 3 or times.ndim != 1 or targets.shape[0] != times.shape[0]:
                raise RuntimeError(f"Invalid target arrays: {path.name}")
            if not np.isfinite(targets).all() or not np.isfinite(a["power_source"]).all():
                raise RuntimeError(f"Non-finite arrays: {path.name}")
        missing_times = sorted(set(range(1, int(times.max()) + 1)) - set(times.tolist()))
        sample_id = path.stem
        segment_name = sample_id.split("_")[-1]
        rows.append({"sample_id": sample_id, "npz_file": str(path.relative_to(root)), "batch": batch, "patient": patient, "patient_group": group, "trajectory": trajectory, "plane": plane, "segment": segment_name, "frames": int(len(times)), "time_min": int(times.min()), "time_max": int(times.max()), "missing_time_indices": ",".join(map(str, missing_times)), "power_W": power, "height": int(targets.shape[1]), "width": int(targets.shape[2]), "source_segment": source_segment, "npz_sha256": sha256(path)})
        groups.setdefault(group, {"frames": 0, "batch_frames": defaultdict(int)})
        groups[group]["frames"] += len(times); groups[group]["batch_frames"][batch] += len(times)
    groups = {k: {"frames": v["frames"], "batch_frames": dict(v["batch_frames"])} for k, v in groups.items()}
    locked=set(cohort["locked_test_patients"])
    splits=choose_splits(groups,args.seed,locked); split_for={g:s for s,names in splits.items() for g in names}
    for row in rows: row["split"] = split_for[row["patient_group"]]
    power_scale=max(row["power_W"] for row in rows if row["split"]=="train")
    time_scale=max(row["time_max"] for row in rows if row["split"]=="train")

    (root/"splits").mkdir(exist_ok=True); (root/"manifests").mkdir(exist_ok=True)
    fields=list(rows[0])
    with (root/"manifests/segment_manifest.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(rows,key=lambda x:x["sample_id"]))
    rollout=[]
    for row in rows:
        with np.load(root/row["npz_file"],allow_pickle=False) as a: times=[int(x) for x in a["target_time_indices"]]
        prev=0
        for t in times:
            rollout.append({"sample_id":row["sample_id"],"split":row["split"],"from_time_index":prev,"to_time_index":t,"rollout_steps":t-prev,"missing_steps_between":max(0,t-prev-1),"loss_enabled":True});prev=t
    with (root/"manifests/rollout_index.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rollout[0]));w.writeheader();w.writerows(rollout)
    split_summary=[]
    for split,names in splits.items():
        sub=[r for r in rows if r["split"]==split];split_summary.append({"split":split,"patients":len(names),"segments":len(sub),"frames":sum(r["frames"] for r in sub),"patient_groups":",".join(sorted(names))})
    with (root/"splits/patient_split_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(split_summary[0]));w.writeheader();w.writerows(split_summary)
    protocol={"seed":args.seed,"locked_test_patients":sorted(locked),"splits":{k:sorted(v) for k,v in splits.items()},"patient_group_rule":cohort["patient_group_rule"]}
    (root/"splits/patient_split.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary={"format_version":"autoreg_segment_v1","segments":len(rows),"patients":len(groups),"temperature_frames":sum(r["frames"] for r in rows),"rollout_pairs":len(rollout),"power_scale_train_max_W":power_scale,"time_scale_train_max_index":time_scale,"probe_sigma_mm":5.39064,"split_summary":split_summary,"known_policy":"missing target frames are not synthesized; rollout advances fixed 3.6 s steps and loss is applied only to observed targets"}
    (root/"dataset_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (root/"normalization.json").write_text(json.dumps({"temperature_offset_C":35.0,"temperature_scale_C":20.0,"power_scale_train_max_W":power_scale,"time_scale_train_max_index":time_scale,"probe_sigma_mm":5.39064},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
