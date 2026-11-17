#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from train_temporal_map import TemporalMapUNet

@torch.no_grad()
def evaluate(model,root,split,device):
 with (root/'manifests/segment_manifest.csv').open(newline='',encoding='utf-8-sig') as f:rows=[r for r in csv.DictReader(f) if r['split']==split]
 by=defaultdict(list);mae=[];tumor=[]
 for row in rows:
  with np.load(root/row['npz_file'],allow_pickle=False) as a:
   times=a['target_time_indices'].astype(np.int64);targets=a['targets_C'];prev=torch.from_numpy(a['baseline_C'][None,None]).to(device);support=torch.from_numpy(a['foreground_mask'][None,None].astype(np.float32)).to(device);static=np.stack([a['structural_zscore'],a['baseline_norm'],a['tumor_mask'],a['probe_gaussian'],a['power_source']]).astype(np.float32);target_map={int(t):targets[i] for i,t in enumerate(times)};state=None
   for step in range(1,int(times.max())+1):
    timemap=torch.full((1,1,static.shape[-2],static.shape[-1]),step/135.0,device=device);x=torch.cat([torch.from_numpy(static[0:1][None]).to(device),(prev-35)/20,torch.from_numpy(static[1:][None]).to(device),timemap],1);pred,state=model.forward_step(x,support,state);prev=pred
    if step in target_map:
     truth=torch.from_numpy(target_map[step][None,None]).to(device);mask=torch.from_numpy(a['tumor_mask'].astype(np.float32)[None,None]).to(device);err=(pred-truth).abs();gm=float(err.mean());tm=float((err*mask).sum().item()/max(1.0,mask.sum().item()));mae.append(gm);tumor.append(tm);by[row['patient_group']].append(tm)
 return {'split':split,'samples':len(mae),'mae':float(np.mean(mae)),'tumor_mae':float(np.mean(tumor)),'patient_macro_tumor_mae':float(np.mean([np.mean(v) for v in by.values()])),'by_patient':{k:float(np.mean(v)) for k,v in by.items()}}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path);ap.add_argument('--checkpoint',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=TemporalMapUNet().to(device);d=torch.load(args.checkpoint,map_location=device,weights_only=False);model.load_state_dict(d['model']);model.eval();result={'checkpoint_val_window':d.get('val'),'val_rollout':evaluate(model,args.data,'val',device),'test_rollout':evaluate(model,args.data,'test',device)};args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:{z:v.get(z) for z in ['split','samples','mae','tumor_mae','patient_macro_tumor_mae'] if z in v} for k,v in result.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
