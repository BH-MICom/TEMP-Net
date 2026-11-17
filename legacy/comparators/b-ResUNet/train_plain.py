#!/usr/bin/env python3
"""Plain one-stage ResUNet baseline: adjacent-frame teacher forcing only."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import torch
from resunet import ResUNetAR
from train import seed_all, train_stage

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--steps",type=int,default=500); ap.add_argument("--batch-size",type=int,default=2); ap.add_argument("--max-targets",type=int,default=64); ap.add_argument("--patience",type=int,default=6); ap.add_argument("--seed",type=int,default=20261011); a=ap.parse_args(); seed_all(a.seed)
    d=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=ResUNetAR().to(d); a.output.mkdir(parents=True,exist_ok=True)
    class Args: pass
    cfg=Args(); cfg.epochs=[a.epochs,0]; cfg.lr=[1e-4,0.]; cfg.steps=a.steps; cfg.batch_size=a.batch_size; cfg.max_targets=a.max_targets; cfg.patience=a.patience
    best=train_stage(model,a.data,1,a.output/"stage1",d,cfg,None)
    ck=torch.load(best,map_location=d,weights_only=False); torch.save({"model":ck["model"],"protocol":"plain ResUNet adjacent-frame teacher forcing"},a.output/"model_final.pt")

if __name__=="__main__": main()
