import collections,csv,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def manifest(root,split):
    with (Path(root)/'manifests/segment_manifest.csv').open(encoding='utf-8-sig') as f:
        return [r for r in csv.DictReader(f) if r['split']==split]

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def make_input(a,temperatures,time_indices,power_scale):
    s=np.asarray(a['structural_zscore'],np.float32)
    baseline=(np.asarray(a['baseline_C'],np.float32)-35)/20
    tumor=np.asarray(a['tumor_mask'],np.float32);g=np.asarray(a['probe_gaussian'],np.float32)
    fg=np.asarray(a['foreground_mask'],np.float32)
    frames=[]
    for temperature,t in zip(temperatures,time_indices):
        frames.append(np.stack([s,(temperature-35)/20,baseline,tumor,g,
                               g*float(a['power_W'])/power_scale,np.full_like(g,t/135),fg]))
    return np.stack(frames).astype(np.float32)

class TrainingWindows(Dataset):
    """Four measured history frames -> next four temperature frames; no fake labels."""
    def __init__(self,root,history=4):
        self.root=Path(root);self.history=history;self.rows=manifest(root,'train')
        self.power_scale=json.loads((self.root/'normalization.json').read_text())['power_scale_train_max_W']
        self.items=[];self.cache=collections.OrderedDict();self.skipped_missing_history=0
        for ri,row in enumerate(self.rows):
            with np.load(self.root/row['npz_file']) as z: times=z['target_time_indices'].astype(int)
            available=set(times.tolist())|{0};last=int(times[-1])
            for origin in range(last):
                hist=[max(0,origin-self.history+1+j) for j in range(self.history)]
                if not all(t in available for t in hist):self.skipped_missing_history+=1;continue
                if any(t in available for t in range(origin+1,min(last,origin+self.history)+1)):
                    self.items.append((ri,origin))
            # Include a cold-baseline startup window for every short/long segment.
        counts=collections.Counter(self.rows[ri]['patient_group'] for ri,_ in self.items)
        self.weights=[1/counts[self.rows[ri]['patient_group']] for ri,_ in self.items]
    def __len__(self):return len(self.items)
    def _load(self,ri):
        if ri not in self.cache:
            self.cache[ri]=load_npz(self.root/self.rows[ri]['npz_file'])
            if len(self.cache)>6:self.cache.popitem(last=False)
        self.cache.move_to_end(ri);return self.cache[ri]
    def __getitem__(self,index):
        ri,origin=self.items[index];a=self._load(ri);h=self.history
        available={int(t):a['targets_C'][i] for i,t in enumerate(a['target_time_indices'])}
        available[0]=a['baseline_C']
        hist=[max(0,origin-h+1+j) for j in range(h)]
        x=make_input(a,[available[t] for t in hist],hist,self.power_scale)
        hh,ww=a['tumor_mask'].shape;targets=np.zeros((h,1,hh,ww),np.float32)
        valid=np.zeros(h,bool)
        for j in range(h):
            if origin+j+1 in available:targets[j,0]=available[origin+j+1];valid[j]=True
        return {'x':torch.from_numpy(x),'targets':torch.from_numpy(targets),
                'valid':torch.from_numpy(valid),'origin':origin,'sample_id':self.rows[ri]['sample_id']}
