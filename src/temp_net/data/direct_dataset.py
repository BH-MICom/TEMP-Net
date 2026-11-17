import collections
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class DirectSequenceDataset(Dataset):
    def __init__(self, root, split, training=False, window=4):
        self.root=Path(root);self.training=training;self.window=window;self.cache=collections.OrderedDict()
        with (self.root/'manifests/segment_manifest.csv').open(newline='',encoding='utf-8-sig') as f:
            rows=[r for r in csv.DictReader(f) if r['split']==split]
        self.rows=rows
        self.scales={
            'power_W':float((self.root/'normalization.json').read_text().split('power_scale_train_max_W')[1].split(':')[1].split(',')[0]),
            'time_index':int((self.root/'dataset_summary.json').read_text().split('time_scale_train_max_index')[1].split(':')[1].split(',')[0]),
        }
        self.items=[]
        for i,row in enumerate(rows):
            with np.load(self.root/row['npz_file'],allow_pickle=False) as a:count=len(a['target_time_indices'])
            self.items.extend([(i,j) for j in range(count)])
        counts=collections.Counter(r['patient_group'] for r in rows);n=max(1,len(self.items));p=max(1,len(counts))
        self.weights=[.5/n+.5/(p*counts[rows[i]['patient_group']]) for i,j in self.items]

    def __len__(self):return len(self.items) if self.training else len(self.rows)
    def _load(self,path):
        key=str(path)
        if key not in self.cache:
            with np.load(path,allow_pickle=False) as a:self.cache[key]={k:a[k] for k in a.files}
            if len(self.cache)>12:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]

    def __getitem__(self,index):
        if self.training:i,j=self.items[index]
        else:i,j=index,0
        row=self.rows[i];a=self._load(self.root/row['npz_file']);times=a['target_time_indices'].astype(np.int64);targets=a['targets_C'].astype(np.float32)
        if self.training:
            length=min(self.window,len(times));start=max(0,min(j-1,len(times)-length));indices=list(range(start,start+length));slots=self.window
        else:indices=list(range(len(times)));slots=len(indices)
        h,w=a['structural_zscore'].shape;center=a['probe_center_rc'];
        # Old best architecture input contract: MRI, normalized baseline, tumor, Gaussian, zero compatibility channel.
        x=np.stack([a['structural_zscore'],a['baseline_norm'],a['tumor_mask'],a['probe_gaussian'],np.zeros_like(a['probe_gaussian'])]).astype(np.float32)
        condition=np.array([float(a['power_W'])/self.scales['power_W'],2*center[0]/max(h-1,1)-1,2*center[1]/max(w-1,1)-1,0.],np.float32)
        t=np.full(slots,-1,np.int64);target=np.zeros((slots,1,h,w),np.float32);t[:len(indices)]=times[indices];target[:len(indices),0]=targets[indices]
        foreground=torch.from_numpy(a['foreground_mask'].astype(np.float32)[None]);
        result={'x':torch.from_numpy(x),'condition':torch.from_numpy(condition),'target':torch.from_numpy(target),'foreground':foreground,'times':torch.from_numpy(t),'patient_group':row['patient_group'],'sample_id':row['sample_id']}
        if self.training:
            if np.random.rand()<.5:
                result['x']=torch.flip(result['x'],[-1]);result['target']=torch.flip(result['target'],[-1]);result['foreground']=torch.flip(result['foreground'],[-1]);result['condition'][2]*=-1
            if np.random.rand()<.5:
                result['x']=torch.flip(result['x'],[-2]);result['target']=torch.flip(result['target'],[-2]);result['foreground']=torch.flip(result['foreground'],[-2]);result['condition'][1]*=-1
            if np.random.rand()<.25:result['x'][0]*=np.random.uniform(.9,1.1)
        return result
