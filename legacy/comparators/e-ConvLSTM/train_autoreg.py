#!/usr/bin/env python3
"""Train a map-only autoregressive temperature model on prepared segment NPZ files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8 if cout % 8 == 0 else 1, cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8 if cout % 8 == 0 else 1, cout),
        )
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class AutoregressiveUNet(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()
        channels = [width * (2 ** i) for i in range(5)]
        self.enc = nn.ModuleList([
            ResidualBlock(6 if i == 0 else channels[i - 1], c, 1 if i == 0 else 2)
            for i, c in enumerate(channels)
        ])
        self.up = nn.ModuleList([nn.Conv2d(channels[i + 1], channels[i], 1) for i in range(3, -1, -1)])
        self.dec = nn.ModuleList([ResidualBlock(2 * channels[i], channels[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward_step(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        previous_norm = x[:, 1:2]
        features = x
        skips = []
        for block in self.enc:
            features = block(features); skips.append(features)
        for j, (up, block) in enumerate(zip(self.up, self.dec)):
            skip = skips[-j - 2]
            features = up(F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            features = block(torch.cat([features, skip], dim=1))
        delta_norm = self.head(features)
        support = torch.maximum(support.float(), x[:, 3:4].float())
        prediction_norm = previous_norm + delta_norm * support
        return prediction_norm * 20.0 + 35.0


def tumor_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = (pred - target).abs()
    flat_mask = mask.float().flatten(1)
    flat_error = error.flatten(1)
    denom = flat_mask.sum(1).clamp_min(1.0)
    return ((flat_error * flat_mask).sum(1) / denom).mean()


class TransitionDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False):
        self.root = root
        self.augment = augment
        with (root / "manifests/segment_manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
            segments = list(csv.DictReader(handle))
        self.records = []
        self.cache = OrderedDict()
        for row in segments:
            if row["split"] != split:
                continue
            path = root / row["npz_file"]
            with np.load(path, allow_pickle=False) as data:
                times = data["target_time_indices"].astype(np.int64)
            previous = 0
            for target_pos, target_time in enumerate(times.tolist()):
                self.records.append({
                    "path": path,
                    "sample_id": row["sample_id"],
                    "patient_group": row["patient_group"],
                    "target_pos": target_pos,
                    "from_time": previous,
                    "to_time": int(target_time),
                    "rollout_steps": int(target_time - previous),
                })
                previous = int(target_time)
        self.patient_counts = defaultdict(int)
        for row in self.records:
            self.patient_counts[row["patient_group"]] += 1
        n = max(1, len(self.records)); p = max(1, len(self.patient_counts))
        self.weights = [0.5 / n + 0.5 / (p * self.patient_counts[row["patient_group"]]) for row in self.records]

    def __len__(self): return len(self.records)

    def _load(self, path: Path):
        key = str(path)
        if key not in self.cache:
            with np.load(path, allow_pickle=False) as data:
                self.cache[key] = {k: data[k] for k in data.files}
            if len(self.cache) > 16: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def __getitem__(self, index):
        row = self.records[index]
        a = self._load(row["path"])
        previous = a["baseline_C"] if row["target_pos"] == 0 else a["targets_C"][row["target_pos"] - 1]
        target = a["targets_C"][row["target_pos"]]
        static = np.stack([
            a["structural_zscore"],
            (previous - 35.0) / 20.0,
            a["baseline_norm"],
            a["tumor_mask"],
            a["probe_gaussian"],
            a["power_source"],
        ]).astype(np.float32)
        support = a["foreground_mask"].astype(np.float32)[None]
        target = target.astype(np.float32)[None]
        if self.augment:
            if np.random.rand() < 0.5:
                static = static[:, :, ::-1].copy(); support = support[:, :, ::-1].copy(); target = target[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                static = static[:, ::-1, :].copy(); support = support[:, ::-1, :].copy(); target = target[:, ::-1, :].copy()
            if np.random.rand() < 0.25:
                static[0] *= np.random.uniform(0.9, 1.1)
        return {
            "x": torch.from_numpy(static), "support": torch.from_numpy(support), "target": torch.from_numpy(target),
            "rollout_steps": row["rollout_steps"], "patient_group": row["patient_group"], "sample_id": row["sample_id"],
        }


def collate(batch):
    return {
        "x": torch.stack([r["x"] for r in batch]),
        "support": torch.stack([r["support"] for r in batch]),
        "target": torch.stack([r["target"] for r in batch]),
        "rollout_steps": [r["rollout_steps"] for r in batch],
        "patient_group": [r["patient_group"] for r in batch],
        "sample_id": [r["sample_id"] for r in batch],
    }


def loss_fn(pred, target, support):
    global_l1 = F.l1_loss(pred, target)
    tumor_l1_value = tumor_l1(pred, target, support)
    return 0.25 * global_l1 + 1.0 * tumor_l1_value, global_l1.detach(), tumor_l1_value.detach()


@torch.no_grad()
def evaluate_one_step(model, loader, device):
    model.eval(); total = 0; mae = []; tumor = []; by_patient = defaultdict(list)
    for batch in loader:
        x=batch["x"].to(device, non_blocking=True); support=batch["support"].to(device, non_blocking=True); target=batch["target"].to(device, non_blocking=True)
        pred=model.forward_step(x,support)
        err=(pred-target).abs().mean(dim=(1,2,3)).cpu().numpy()
        mask=(x[:,3:4]>0.5)
        terr=((pred-target).abs()*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)
        terr=terr.cpu().numpy()
        for i,p in enumerate(batch["patient_group"]):by_patient[p].append(float(terr[i]))
        mae.extend(err.tolist()); tumor.extend(terr.tolist()); total += len(err)
    patient_macro=float(np.mean([np.mean(v) for v in by_patient.values()])) if by_patient else float('nan')
    return {"mae":float(np.mean(mae)),"tumor_mae":float(np.mean(tumor)),"patient_macro_tumor_mae":patient_macro,"samples":total}


@torch.no_grad()
def rollout_evaluate(model, root: Path, split: str, device, limit_segments: int | None = None):
    model.eval();
    with (root / "manifests/segment_manifest.csv").open(newline="", encoding="utf-8-sig") as handle: rows=[r for r in csv.DictReader(handle) if r["split"]==split]
    if limit_segments: rows=rows[:limit_segments]
    records=[]; by_patient=defaultdict(list)
    for row in rows:
        with np.load(root/row["npz_file"],allow_pickle=False) as a:
            target_times=a["target_time_indices"].astype(np.int64); targets=a["targets_C"].astype(np.float32); baseline=a["baseline_C"].astype(np.float32)
            static=np.stack([a["structural_zscore"],np.zeros_like(baseline),a["baseline_norm"],a["tumor_mask"],a["probe_gaussian"],a["power_source"]]).astype(np.float32)
            support=torch.from_numpy(a["foreground_mask"].astype(np.float32)[None,None]).to(device)
            previous=torch.from_numpy(baseline[None,None]).to(device)
            target_by_time={int(t):targets[i] for i,t in enumerate(target_times)}
            max_time=int(target_times.max())
            for step in range(1,max_time+1):
                x=static.copy();x[1]=previous[0,0].detach().cpu().numpy()/20.0-35.0/20.0
                pred=model.forward_step(torch.from_numpy(x[None]).to(device),support)
                previous=pred
                if step in target_by_time:
                    target=torch.from_numpy(target_by_time[step][None,None]).to(device)
                    err=(pred-target).abs();mask=torch.from_numpy(a["tumor_mask"].astype(np.float32)[None,None]).to(device)
                    tm=float((err*mask).sum().item()/max(1.0,mask.sum().item()));gm=float(err.mean().item())
                    records.append({"patient_group":row["patient_group"],"sample_id":row["sample_id"],"time_index":step,"mae":gm,"tumor_mae":tm})
                    by_patient[row["patient_group"]].append(tm)
    return {"split":split,"samples":len(records),"mae":float(np.mean([r["mae"] for r in records])) if records else None,"tumor_mae":float(np.mean([r["tumor_mae"] for r in records])) if records else None,"patient_macro_tumor_mae":float(np.mean([np.mean(v) for v in by_patient.values()])) if by_patient else None,"records":records}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--epochs',type=int,default=50);parser.add_argument('--steps-per-epoch',type=int,default=1000);parser.add_argument('--batch-size',type=int,default=2);parser.add_argument('--lr',type=float,default=1e-4);parser.add_argument('--seed',type=int,default=20260925);parser.add_argument('--patience',type=int,default=10);args=parser.parse_args()
    seed_everything(args.seed);args.output.mkdir(parents=True,exist_ok=True);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds=TransitionDataset(args.data,'train',augment=True);val_ds=TransitionDataset(args.data,'val',augment=False)
    sampler=WeightedRandomSampler(train_ds.weights,num_samples=args.steps_per_epoch*args.batch_size,replacement=True)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True,collate_fn=collate)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=2,pin_memory=True,collate_fn=collate)
    model=AutoregressiveUNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=float('inf');bad=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();losses=[];iterator=iter(train_loader)
        for _ in range(args.steps_per_epoch):
            batch=next(iterator);x=batch['x'].to(device,non_blocking=True);support=batch['support'].to(device,non_blocking=True);target=batch['target'].to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True);prev=x[:,1:2]*20+35
            # One-step transitions are the dominant case; gap transitions are unrolled.
            pred=None
            rollout_steps = torch.as_tensor(batch['rollout_steps'], device=device)
            for step in range(int(rollout_steps.max().item())):
                x_step=x.clone();x_step[:,1:2]=(prev-35)/20
                with torch.autocast('cuda',enabled=device.type=='cuda'):
                    candidate=model.forward_step(x_step,support)
                active=(rollout_steps > step)[:,None,None,None]
                prev=torch.where(active,candidate,prev)
            pred=prev
            with torch.autocast('cuda',enabled=device.type=='cuda'):
                loss,_,_=loss_fn(pred,target,x[:,3:4])
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update();losses.append(float(loss.detach().cpu()))
        val=evaluate_one_step(model,val_loader,device);row={'epoch':epoch,'train_loss':float(np.mean(losses)),'val':val};history.append(row);print(json.dumps(row),flush=True)
        if val['patient_macro_tumor_mae']<best:
            best=val['patient_macro_tumor_mae'];bad=0;torch.save({'model':model.state_dict(),'epoch':epoch,'val':val,'args':vars(args)},args.output/'model_best.pt')
        else:bad+=1
        if bad>=args.patience:break
    (args.output/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
    checkpoint=torch.load(args.output/'model_best.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['model']);
    final={'best_checkpoint':checkpoint['val'],'val_rollout':rollout_evaluate(model,args.data,'val',device),'test_rollout':rollout_evaluate(model,args.data,'test',device)}
    (args.output/'final_metrics.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
