#!/usr/bin/env python3
"""ConvLSTM-U-Net temporal baseline."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_autoreg_rollout import SequenceWindowDataset, collate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        groups = 8 if cout % 8 == 0 else 1
        self.main = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, padding=1, bias=False),
            nn.GroupNorm(groups, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
        )
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class ConvLSTM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 4 * channels, 3, padding=1)

    def forward(self, x, h, c):
        i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
        i, f, o = i.sigmoid(), f.sigmoid(), o.sigmoid()
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class TemporalMapUNet(nn.Module):
    def __init__(self, width=24, hidden=64):
        super().__init__()
        channels = [width * (2 ** i) for i in range(5)]
        self.enc = nn.ModuleList([
            Block(7 if i == 0 else channels[i - 1], c, 1 if i == 0 else 2)
            for i, c in enumerate(channels)
        ])
        self.state_in = nn.Conv2d(channels[-1], hidden, 1)
        self.lstm = ConvLSTM(hidden)
        self.state_out = nn.Conv2d(hidden, channels[-1], 1)
        self.up = nn.ModuleList([nn.Conv2d(channels[i + 1], channels[i], 1) for i in range(3, -1, -1)])
        self.dec = nn.ModuleList([Block(2 * channels[i], channels[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_step(self, x, support, state=None):
        features = x
        skips = []
        for block in self.enc:
            features = block(features)
            skips.append(features)
        latent = self.state_in(features)
        if state is None:
            h, c = torch.tanh(latent), latent
        else:
            h, c = state
        h, c = self.lstm(latent, h, c)
        state = (h, c)
        features = features + self.state_out(h)
        for j, (up, block) in enumerate(zip(self.up, self.dec)):
            skip = skips[-j - 2]
            features = up(F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            features = block(torch.cat([features, skip], 1))
        previous_norm = x[:, 1:2]
        support = torch.maximum(support.float(), x[:, 3:4].float())
        prediction_norm = previous_norm + self.head(features) * support
        return prediction_norm * 20.0 + 35.0, state


def rollout_batch(model, batch, device, teacher_probability):
    static = batch["static"].to(device)
    support = batch["support"].to(device)
    previous = batch["start"].to(device)
    current = batch["start_time"].to(device)
    targets = batch["targets"].to(device)
    target_times = batch["target_times"].to(device)
    state = None
    total = previous.new_tensor(0.0)
    count = 0
    records = []
    for j in range(target_times.shape[1]):
        target_time = target_times[:, j]
        steps = (target_time - current).clamp_min(0)
        for step in range(int(steps.max().item())):
            now = current + step + 1
            time_map = (now.float() / 135.0)[:, None, None, None].expand(-1, 1, static.shape[-2], static.shape[-1])
            x = torch.cat([static[:, 0:1], (previous - 35.0) / 20.0, static[:, 1:], time_map], 1)
            candidate, new_state = model.forward_step(x, support, state)
            active = (steps > step)[:, None, None, None]
            previous = torch.where(active, candidate, previous)
            state = new_state
        valid = target_time >= 0
        if valid.any():
            prediction = previous[valid]
            truth = targets[valid, j]
            tumor = static[valid, 2:3]
            error = (prediction - truth).abs()
            tumor_error = (error * tumor).flatten(1).sum(1) / tumor.flatten(1).sum(1).clamp_min(1.0)
            total = total + error.mean() + tumor_error.mean()
            count += 1
            valid_indices = torch.where(valid)[0]
            for k, batch_index in enumerate(valid_indices.tolist()):
                records.append((batch["patient_group"][batch_index], float(tumor_error[k].item())))
            if teacher_probability > 0:
                use_truth = (torch.rand(prediction.shape[0], device=device) < teacher_probability)[:, None, None, None]
                previous[valid] = torch.where(use_truth, truth, prediction)
        current = torch.where(valid, target_time, current)
    return total / max(count, 1), records


@torch.no_grad()
def evaluate_window(model, loader, device):
    model.eval()
    by_patient = defaultdict(list)
    all_values = []
    for batch in loader:
        _, records = rollout_batch(model, batch, device, 0.0)
        for patient, value in records:
            by_patient[patient].append(value)
            all_values.append(value)
    return {
        "tumor_mae": float(np.mean(all_values)),
        "patient_macro_tumor_mae": float(np.mean([np.mean(values) for values in by_patient.values()])),
        "samples": len(all_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--init", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalMapUNet().to(device)
    checkpoint = torch.load(args.init, map_location=device, weights_only=False)
    previous_state = checkpoint["model"]
    own_state = model.state_dict()
    loaded = 0
    for key, value in previous_state.items():
        if key == "enc.0.main.0.weight":
            own_state[key][:, :value.shape[1]] = value
            loaded += 1
        elif key in own_state and own_state[key].shape == value.shape:
            own_state[key] = value
            loaded += 1
    model.load_state_dict(own_state)
    print("warm_started_parameters", loaded, flush=True)
    train_dataset = SequenceWindowDataset(args.data, "train", args.window, True)
    val_dataset = SequenceWindowDataset(args.data, "val", args.window, False)
    sampler = WeightedRandomSampler(train_dataset.weights, num_samples=args.steps_per_epoch * args.batch_size, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = float("inf"); bad = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); iterator = iter(train_loader); losses = []
        teacher_probability = max(0.15, 1.0 - 0.85 * (epoch - 1) / max(1, args.epochs - 1))
        for _ in range(args.steps_per_epoch):
            batch = next(iterator); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss, _ = rollout_batch(model, batch, device, teacher_probability)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate_window(model, val_loader, device)
        row = {"epoch": epoch, "teacher_probability": teacher_probability, "train_loss": float(np.mean(losses)), "val_window": validation}
        history.append(row); print(json.dumps(row), flush=True)
        if validation["patient_macro_tumor_mae"] < best:
            best = validation["patient_macro_tumor_mae"]; bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": validation, "args": vars(args)}, args.output / "model_best.pt")
        else:
            bad += 1
        if bad >= args.patience:
            break
    (args.output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print("training_finished", flush=True)


if __name__ == "__main__":
    main()
