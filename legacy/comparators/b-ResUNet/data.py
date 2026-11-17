#!/usr/bin/env python3
from __future__ import annotations
import csv, json, collections
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset


class ResDataset(Dataset):
    def __init__(self, root, split, mode="adjacent", max_targets=8):
        self.root=Path(root); self.mode=mode; self.max_targets=max_targets; self.cache=collections.OrderedDict()
        with (self.root/"manifests/segment_manifest.csv").open(encoding="utf-8-sig",newline="") as f:
            self.rows=[r for r in csv.DictReader(f) if r["split"]==split]
        norm=json.loads((self.root/"normalization.json").read_text()); self.power_scale=float(norm["power_scale_train_max_W"])
        self.items=[]
        if mode=="adjacent":
            for i,row in enumerate(self.rows):
                a=self._load(self.root/row["npz_file"]); times=a["target_time_indices"].astype(int)
                for j,t in enumerate(times):
                    if t==1 or (j>0 and times[j-1]==t-1): self.items.append((i,j))
        else: self.items=list(range(len(self.rows)))

    def _load(self,path):
        key=str(path)
        if key not in self.cache:
            with np.load(path,allow_pickle=False) as z:self.cache[key]={k:z[k] for k in z.files}
            if len(self.cache)>16:self.cache.popitem(last=False)
        self.cache.move_to_end(key); return self.cache[key]

    def __len__(self): return len(self.items)

    def make_input(self,a,current_norm,time_index):
        structural=a["structural_zscore"].astype(np.float32); tumor=a["tumor_mask"].astype(np.float32); fg=a["foreground_mask"].astype(np.float32); g=a["probe_gaussian"].astype(np.float32)
        baseline=(a["baseline_C"].astype(np.float32)-35.)/20.; p=float(a["power_W"])/self.power_scale
        power=g*p; time=np.full_like(g,float(time_index)/135.,dtype=np.float32)
        return np.stack([structural,current_norm.astype(np.float32),baseline,tumor,g,power,time,fg]).astype(np.float32)

    def __getitem__(self,index):
        if self.mode=="adjacent":
            i,j=self.items[index]; a=self._load(self.root/self.rows[i]["npz_file"]); times=a["target_time_indices"].astype(int); t=int(times[j]); prev=((a["baseline_C"] if t==1 else a["targets_C"][j-1])-35.)/20.
            return {"x":torch.from_numpy(self.make_input(a,prev,t)),"target":torch.from_numpy(a["targets_C"][j][None].astype(np.float32)),"patient_group":self.rows[i]["patient_group"]}
        i=self.items[index]; a=self._load(self.root/self.rows[i]["npz_file"]); times=a["target_time_indices"].astype(int); k=min(len(times),self.max_targets); return {"a_path":str(self.root/self.rows[i]["npz_file"]),"times":torch.from_numpy(times[:k]),"targets":torch.from_numpy(a["targets_C"][:k,None].astype(np.float32)),"baseline":torch.from_numpy(((a["baseline_C"]-35.)/20.)[None].astype(np.float32)),"patient_group":self.rows[i]["patient_group"],"sample_id":self.rows[i]["sample_id"]}


def collate_sequence(batch):
    k=max(int(x["times"].numel()) for x in batch); h,w=batch[0]["targets"].shape[-2:]; times=torch.full((len(batch),k),-1,dtype=torch.long); targets=torch.zeros((len(batch),k,1,h,w));
    for i,x in enumerate(batch): n=int(x["times"].numel()); times[i,:n]=x["times"]; targets[i,:n]=x["targets"]
    return {"a_path":[x["a_path"] for x in batch],"times":times,"targets":targets,"baseline":torch.stack([x["baseline"] for x in batch]),"patient_group":[x["patient_group"] for x in batch],"sample_id":[x["sample_id"] for x in batch]}
