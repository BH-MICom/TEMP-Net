#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch
from resunet import ResUNetAR
from data import ResDataset
from train import evaluate
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",type=Path,required=True);ap.add_argument("--run",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--max-targets",type=int,default=64);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);d=torch.device("cuda" if torch.cuda.is_available() else "cpu");out={}
 for name in ["stage1","stage2","model_final"]:
  p=a.run/f"{name}/model_best.pt" if name.startswith("stage") else a.run/"model_final.pt"
  if not p.exists():continue
  m=ResUNetAR().to(d);m.load_state_dict(torch.load(p,map_location=d,weights_only=False)["model"]);out[name]=evaluate(m,ResDataset(a.data,"test","sequence",a.max_targets),d,a.max_targets)
 (a.output/"test_metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
