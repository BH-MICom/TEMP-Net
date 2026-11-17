from __future__ import annotations
import collections, csv, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class PaperSequenceDataset(Dataset):
    """Patient-filtered dataset for paper-style five-fold and two-stage training."""
    def __init__(self, root, patients, training, window=4, horizon=None):
        self.root=Path(root); self.training=training; self.window=window; self.horizon=horizon
        with (self.root/'manifests/segment_manifest.csv').open(newline='',encoding='utf-8-sig') as f:
            all_rows=list(csv.DictReader(f))
        wanted={str(x) for x in patients}; self.rows=[r for r in all_rows if r['patient_group'] in wanted]
        norm=json.loads((self.root/'normalization.json').read_text())
        self.power_scale=float(norm['power_scale_train_max_W'])
        summary=json.loads((self.root/'dataset_summary.json').read_text())
        self.time_scale=int(summary['time_scale_train_max_index'])
        self.cache=collections.OrderedDict(); self.items=[]
        for i,row in enumerate(self.rows):
            with np.load(self.root/row['npz_file'],allow_pickle=False) as a:
                times=a['target_time_indices'].astype(np.int64)
            if self.horizon is not None: times=times[times<=int(self.horizon)]
            if self.training:
                self.items.extend((i,j) for j in range(len(times)))
        counts=collections.Counter(r['patient_group'] for r in self.rows)
        n=max(1,len(self.items)); p=max(1,len(counts))
        self.weights=[0.5/n+0.5/(p*counts[self.rows[i]['patient_group']]) for i,j in self.items]
    def __len__(self): return len(self.items) if self.training else len(self.rows)
    def _load(self,path):
        key=str(path)
        if key not in self.cache:
            with np.load(path,allow_pickle=False) as a: self.cache[key]={k:a[k] for k in a.files}
            if len(self.cache)>12: self.cache.popitem(last=False)
        self.cache.move_to_end(key); return self.cache[key]
    def __getitem__(self,index):
        if self.training: i,j=self.items[index]
        else: i,j=index,0
        row=self.rows[i]; a=self._load(self.root/row['npz_file'])
        all_times=a['target_time_indices'].astype(np.int64); target_map={int(t):a['targets_C'][k] for k,t in enumerate(all_times)}
        times=np.array(sorted(t for t in target_map if self.horizon is None or t<=int(self.horizon)),dtype=np.int64)
        if self.training:
            length=min(self.window,len(times)); start=max(0,min(j-1,len(times)-length)); chosen=times[start:start+length]; slots=self.window
        else:
            chosen=times; slots=len(chosen)
        h,w=a['structural_zscore'].shape; center=a['probe_center_rc']
        x=np.stack([a['structural_zscore'],a['baseline_norm'],a['tumor_mask'],a['probe_gaussian'],np.zeros_like(a['probe_gaussian'])]).astype(np.float32)
        condition=np.array([float(a['power_W'])/self.power_scale,2*center[0]/max(h-1,1)-1,2*center[1]/max(w-1,1)-1,0.],np.float32)
        t=np.full(slots,-1,np.int64); target=np.zeros((slots,1,h,w),np.float32); t[:len(chosen)]=chosen
        for k,value in enumerate(chosen): target[k,0]=target_map[int(value)]
        result={'x':torch.from_numpy(x),'condition':torch.from_numpy(condition),'target':torch.from_numpy(target),'foreground':torch.from_numpy(a['foreground_mask'].astype(np.float32)[None]),'times':torch.from_numpy(t),'patient_group':row['patient_group'],'sample_id':row['sample_id']}
        if self.training:
            if np.random.rand()<.5:
                result['x']=torch.flip(result['x'],[-1]); result['target']=torch.flip(result['target'],[-1]); result['foreground']=torch.flip(result['foreground'],[-1]); result['condition'][2]*=-1
            if np.random.rand()<.5:
                result['x']=torch.flip(result['x'],[-2]); result['target']=torch.flip(result['target'],[-2]); result['foreground']=torch.flip(result['foreground'],[-2]); result['condition'][1]*=-1
        return result
