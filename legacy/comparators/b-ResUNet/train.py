#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random,time,itertools
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from resunet import ResUNetAR
from data import ResDataset,collate_sequence

def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def move(b,d): return {k:(v.to(d,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in b.items()}
def loss_fn(p,y,x):
 e=(p-y).abs(); tumor=x[:,3:4].expand_as(e); fg=x[:,7:8].expand_as(e); support=torch.maximum(tumor,fg); hot=1+1.5*torch.sigmoid((y-41.5)/1.5)
 def reg(v,m):
  n=m.flatten(1).sum(1).clamp_min(1); z=(v*m).flatten(1).sum(1)/n; good=m.flatten(1).sum(1)>0; return z[good].mean() if good.any() else v.sum()*0
 return .05*e.mean()+.45*reg(e*hot,support)+1.5*reg(e*hot,tumor)
def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
@torch.no_grad()
def evaluate(model,ds,device,max_targets):
 model.eval(); loader=DataLoader(ds,batch_size=2,shuffle=False,num_workers=2,pin_memory=True,collate_fn=collate_sequence); vals=defaultdict(list)
 for raw in loader:
  b=move(raw,device); preds=[]; xlist=[]
  for i,path in enumerate(b["a_path"]):
   a=load_npz(path); current=b["baseline"][i:i+1]; out=[]
   for j,t in enumerate(b["times"][i].tolist()):
    if t<0:break
    # The baseline-map branch is the only persistent state for this plain AR baseline.
    x=torch.from_numpy(ds.make_input(a,current[0,0].cpu().numpy(),int(t))).to(device)[None]
    predicted_c=model(x); out.append(predicted_c[0,0]); current=(predicted_c-35.0)/20.0
   preds.append(torch.stack(out) if out else torch.zeros((0,512,512),device=device))
  for i,p in enumerate(preds):
   n=p.shape[0]; y=b["targets"][i,:n,0]; tumor=torch.from_numpy(load_npz(b["a_path"][i])["tumor_mask"].astype(np.float32)).to(device); err=(p-y).abs(); den=tumor.sum().clamp_min(1); score=(err*tumor).flatten(1).sum(1)/den; vals[b["patient_group"][i]].extend(score.cpu().tolist())
 allv=[x for v in vals.values() for x in v]; return {"samples":len(allv),"patients":len(vals),"tumor_mae":float(np.mean(allv)),"patient_macro_tumor_mae":float(np.mean([np.mean(v) for v in vals.values()]))}
def train_stage(model,root,stage,out,device,args,init=None):
 out.mkdir(parents=True,exist_ok=True)
 mode="adjacent" if stage==1 else "sequence"; train=ResDataset(root,"train",mode,args.max_targets); val=ResDataset(root,"val","sequence",args.max_targets); loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True,collate_fn=None if stage==1 else collate_sequence); 
 if init: model.load_state_dict(torch.load(init,map_location=device,weights_only=False)["model"])
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr[stage-1],weight_decay=1e-4); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda"); best=1e9;bad=0;hist=[]
 for ep in range(1,args.epochs[stage-1]+1):
  model.train();it=iter(loader);losses=[];teacher=max(.2,1-.8*(ep-1)/max(1,args.epochs[stage-1]-1))
  for step in range(args.steps):
   b=move(next(it),device);opt.zero_grad(set_to_none=True)
   with torch.autocast("cuda",enabled=device.type=="cuda",dtype=torch.bfloat16):
    if stage==1: p=model(b["x"]); loss=loss_fn(p,b["target"],b["x"])
    else:
     # Keep this baseline strictly autoregressive: predicted frame becomes the next input.
     total=0.;count=0
     for i,path in enumerate(b["a_path"]):
      a=load_npz(path); current=b["baseline"][i:i+1]; targets=b["targets"][i]; times=b["times"][i];
      for j,t in enumerate(times.tolist()):
       if t<0:break
       x=torch.from_numpy(train.make_input(a,current[0,0].detach().cpu().numpy(),int(t))).to(device)[None]; pred=model(x); total=total+loss_fn(pred,targets[j:j+1].to(device),x)
       if random.random()<teacher: current=targets[j:j+1].to(device)[:,0:1].sub(35).div(20)
       else: current=(pred-35).div(20)
       count+=1
     loss=total/max(1,count)
   scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update();losses.append(float(loss.detach().cpu()))
  vm=evaluate(model,val,device,args.max_targets); row={"stage":stage,"epoch":ep,"teacher_probability":teacher if stage==2 else 0,"train_loss":float(np.mean(losses)),"val":vm};hist.append(row);print(json.dumps(row),flush=True)
  if vm["patient_macro_tumor_mae"]<best:best=vm["patient_macro_tumor_mae"];bad=0;torch.save({"model":model.state_dict(),"epoch":ep,"val":vm},out/"model_best.pt")
  else:bad+=1
  if bad>=args.patience:break
 (out/"history.json").write_text(json.dumps(hist,ensure_ascii=False,indent=2));return out/"model_best.pt"
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--epochs",type=int,nargs=2,default=[15,12]);ap.add_argument("--lr",type=float,nargs=2,default=[1e-4,3e-5]);ap.add_argument("--steps",type=int,default=400);ap.add_argument("--batch-size",type=int,default=2);ap.add_argument("--max-targets",type=int,default=8);ap.add_argument("--patience",type=int,default=5);ap.add_argument("--seed",type=int,default=20261011);a=ap.parse_args();seed_all(a.seed);d=torch.device("cuda" if torch.cuda.is_available() else "cpu");m=ResUNetAR().to(d);a.output.mkdir(parents=True,exist_ok=True);ck=None
 for s in (1,2):ck=train_stage(m,a.data,s,a.output/f"stage{s}",d,a,ck)
 m.load_state_dict(torch.load(ck,map_location=d,weights_only=False)["model"]);torch.save({"model":m.state_dict(),"protocol":"ResUNet autoregressive adjacent + scheduled sampling"},a.output/"model_final.pt")
if __name__=="__main__":main()
