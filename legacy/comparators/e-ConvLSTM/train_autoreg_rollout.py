#!/usr/bin/env python3
"""Scheduled-sampling multi-step fine-tuning for the autoregressive model."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_autoreg import AutoregressiveUNet, rollout_evaluate


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class SequenceWindowDataset(Dataset):
    def __init__(self, root: Path, split: str, window: int, training: bool):
        self.root=root;self.window=window;self.training=training;self.cache=OrderedDict();self.rows=[]
        with (root/'manifests/segment_manifest.csv').open(newline='',encoding='utf-8-sig') as f: segments=list(csv.DictReader(f))
        self.segments=[r for r in segments if r['split']==split]
        for i,r in enumerate(self.segments): self.rows.extend([(i,j) for j in range(max(1,int(r['frames'])-1))])
        counts=defaultdict(int)
        for r in self.segments:counts[r['patient_group']]+=1
        self.weights=[]; n=max(1,len(self.rows)); p=max(1,len(counts))
        for i,j in self.rows:
            group=self.segments[i]['patient_group']; self.weights.append(.5/n+.5/(p*counts[group]))

    def __len__(self): return len(self.rows)
    def _load(self,path):
        key=str(path)
        if key not in self.cache:
            with np.load(path,allow_pickle=False) as a:self.cache[key]={k:a[k] for k in a.files}
            if len(self.cache)>12:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]

    def __getitem__(self,index):
        seg_index,row_index=self.rows[index];row=self.segments[seg_index];a=self._load(self.root/row['npz_file'])
        times=a['target_time_indices'].astype(np.int64);targets=a['targets_C'].astype(np.float32)
        if self.training:
            # Half of windows begin at baseline; the rest begin at an observed target.
            start_pos=-1 if np.random.rand()<0.5 else int(np.random.randint(0,max(1,len(times)-1)))
        else:start_pos=-1
        start_time=0 if start_pos<0 else int(times[start_pos]);start_frame=a['baseline_C'] if start_pos<0 else targets[start_pos]
        begin=start_pos+1;chosen=list(range(begin,min(len(times),begin+self.window)))
        static=np.stack([a['structural_zscore'],a['baseline_norm'],a['tumor_mask'],a['probe_gaussian'],a['power_source']]).astype(np.float32)
        target=np.zeros((self.window,1,*start_frame.shape),np.float32);target_times=np.full(self.window,-1,np.int64)
        for j,pos in enumerate(chosen):target[j,0]=targets[pos];target_times[j]=times[pos]
        support=a['foreground_mask'].astype(np.float32)[None]
        if self.training:
            if np.random.rand()<0.5: static=static[:,:,::-1].copy();support=support[:,:,::-1].copy();target=target[:,:,:,::-1].copy()
            if np.random.rand()<0.5: static=static[:,::-1,:].copy();support=support[:,::-1,:].copy();target=target[:,:,::-1,:].copy()
            if np.random.rand()<0.25: static[0]*=np.random.uniform(.9,1.1)
        return {'static':torch.from_numpy(static),'support':torch.from_numpy(support),'start':torch.from_numpy(start_frame[None]),'start_time':start_time,'targets':torch.from_numpy(target),'target_times':torch.from_numpy(target_times),'patient_group':row['patient_group'],'sample_id':row['sample_id']}


def collate(batch):
    return {'static':torch.stack([x['static'] for x in batch]),'support':torch.stack([x['support'] for x in batch]),'start':torch.stack([x['start'] for x in batch]),'start_time':torch.tensor([x['start_time'] for x in batch]),'targets':torch.stack([x['targets'] for x in batch]),'target_times':torch.stack([x['target_times'] for x in batch]),'patient_group':[x['patient_group'] for x in batch],'sample_id':[x['sample_id'] for x in batch]}


def rollout_batch(model,batch,device,teacher_probability):
    static=batch['static'].to(device);support=batch['support'].to(device);prev=batch['start'].to(device);current=batch['start_time'].to(device);target=batch['targets'].to(device);times=batch['target_times'].to(device)
    total=prev.new_tensor(0.);count=0;records=[]
    for j in range(times.shape[1]):
        target_time=times[:,j];steps=(target_time-current).clamp_min(0);max_steps=int(steps.max().item())
        for step in range(max_steps):
            x=torch.cat([static[:,0:1],(prev-35.)/20.,static[:,1:]],dim=1)
            candidate=model.forward_step(x,support)
            active=(steps>step)[:,None,None,None];prev=torch.where(active,candidate,prev)
        valid=target_time>=0
        if valid.any():
            pred=prev[valid];truth=target[valid,j];tumor=static[valid,2:3]
            err=(pred-truth).abs();global_l1=err.mean();den=tumor.flatten(1).sum(1).clamp_min(1);tumor_l1=(err*tumor).flatten(1).sum(1)/den
            total=total+0.25*global_l1+tumor_l1.mean();count+=1
            for k in range(pred.shape[0]):records.append((batch['patient_group'][int(torch.where(valid)[0][k])],float((err[k]*tumor[k]).sum().item()/max(1.0,tumor[k].sum().item()))))
            if teacher_probability>0:
                use_truth=(torch.rand(pred.shape[0],device=device)<teacher_probability)[:,None,None,None]
                replacement=torch.where(use_truth,truth,pred);prev[valid]=replacement
        current=torch.where(valid,target_time,current)
    return total/max(count,1),records


@torch.no_grad()
def evaluate_windows(model,loader,device):
    model.eval();by=defaultdict(list);allv=[]
    for batch in loader:
        _,records=rollout_batch(model,batch,device,0.0)
        for p,v in records:by[p].append(v);allv.append(v)
    return {'tumor_mae':float(np.mean(allv)),'patient_macro_tumor_mae':float(np.mean([np.mean(v) for v in by.values()])),'samples':len(allv)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--init',required=True,type=Path);ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--steps-per-epoch',type=int,default=500);ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--window',type=int,default=4);ap.add_argument('--lr',type=float,default=2e-5);ap.add_argument('--patience',type=int,default=6);ap.add_argument('--seed',type=int,default=20260926);args=ap.parse_args()
    seed_everything(args.seed);args.output.mkdir(parents=True,exist_ok=True);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=AutoregressiveUNet().to(device);checkpoint=torch.load(args.init,map_location=device,weights_only=False);model.load_state_dict(checkpoint['model']);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    train_ds=SequenceWindowDataset(args.data,'train',args.window,True);val_ds=SequenceWindowDataset(args.data,'val',args.window,False);sampler=WeightedRandomSampler(train_ds.weights,num_samples=args.steps_per_epoch*args.batch_size,replacement=True)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True,collate_fn=collate);val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=2,pin_memory=True,collate_fn=collate)
    best=float('inf');bad=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();losses=[];teacher=max(.15,1.0-.85*(epoch-1)/max(1,args.epochs-1));it=iter(train_loader)
        for _ in range(args.steps_per_epoch):
            batch=next(it);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=device.type=='cuda'):loss,_=rollout_batch(model,batch,device,teacher)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update();losses.append(float(loss.detach().cpu()))
        val=evaluate_windows(model,val_loader,device);row={'epoch':epoch,'teacher_probability':teacher,'train_loss':float(np.mean(losses)),'val_window':val};history.append(row);print(json.dumps(row),flush=True)
        if val['patient_macro_tumor_mae']<best:best=val['patient_macro_tumor_mae'];bad=0;torch.save({'model':model.state_dict(),'epoch':epoch,'val':val,'args':vars(args)},args.output/'model_best.pt')
        else:bad+=1
        if bad>=args.patience:break
    (args.output/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8');checkpoint=torch.load(args.output/'model_best.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['model'])
    final={'best_window_checkpoint':checkpoint['val'],'val_rollout':rollout_evaluate(model,args.data,'val',device),'test_rollout':rollout_evaluate(model,args.data,'test',device)}
    (args.output/'final_metrics.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:{z:v.get(z) for z in ['split','samples','mae','tumor_mae','patient_macro_tumor_mae'] if z in v} for k,v in final.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
