#!/usr/bin/env python3
"""Evaluate a DynamicsUNet checkpoint on a fixed patient split."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from temp_net.data.direct_dataset import DirectSequenceDataset
from temp_net.models.dynamics_unet import DynamicsUNet

@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    values = defaultdict(list)
    for index in range(len(dataset)):
        batch = dataset[index]
        x = batch["x"][None].to(device)
        condition = batch["condition"][None].to(device)
        foreground = batch["foreground"][None].to(device)
        times = batch["times"][None].to(device)
        target = batch["target"][None].to(device)
        prediction, _ = model.forward_sequence(x, condition, foreground, times)
        valid = times[0] >= 0
        prediction = prediction[0][valid]
        truth = target[0][valid]
        tumor = x[0, 2:3].expand_as(prediction)
        error = (prediction - truth).abs()
        tumor_mae = (error * tumor).flatten(1).sum(1) / tumor.flatten(1).sum(1).clamp_min(1)
        for value in tumor_mae.detach().cpu().numpy():
            values[batch["patient_group"]].append(float(value))
    patient_values = [np.mean(v) for v in values.values()]
    return {
        "patients": len(patient_values),
        "samples": int(sum(len(v) for v in values.values())),
        "tumor_mae": float(np.mean([x for v in values.values() for x in v])),
        "patient_macro_tumor_mae": float(np.mean(patient_values)),
        "by_patient": {k: float(np.mean(v)) for k, v in values.items()},
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DirectSequenceDataset(args.data, args.split, training=False, window=4)
    model = DynamicsUNet(time_scale=135, width=24, hidden=64, local_hidden=16, enhanced=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {"split": args.split, "checkpoint": str(args.checkpoint), "metrics": evaluate(model, dataset, device)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
