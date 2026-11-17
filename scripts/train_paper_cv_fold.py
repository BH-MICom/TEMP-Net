#!/usr/bin/env python3
"""Paper-protocol training for one patient-level CV fold.

Stage I uses a short temporal horizon. Stage II initializes from Stage I and
uses the complete observed horizon. Both stages supervise up to four sampled
MRT target times while recurrent states advance through intervening steps.
"""
from __future__ import annotations
import argparse,json,random,time,sys
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from temp_net.data.paper_dataset import PaperSequenceDataset
from temp_net.models.dynamics_unet import DynamicsUNet

def loss_region(error,mask):
 n=mask.flatten(1).sum(1); v=(error*mask).flatten(1).sum(1)/n.clamp_min(1); return v[n>0].mean() if (n>0).any() else error.sum()*0

def loss_fn(pred,target,x,fg):
 e=(pred-target).abs(); tumor=x[:,2:3].expand_as(e); roi=torch.maximum(tumor,fg); return .05*e.mean()+.5*loss_region(e,roi)+2.*loss_region(e,tumor)

def sequence_loss(pred,aux,b):
 valid=b['times']>=0; x=b['x'][:,None].expand(-1,valid.shape[1],-1,-1,-1)[valid]; fg=b['foreground'][:,None].expand(-1,valid.shape[1],-1,-1,-1)[valid]; target=b['target'][valid]; main=loss_fn(pred[valid],target,x,fg); deep=main*0
 for weight,a in zip((.125,.25),aux):
  shape=a.shape[-2:]; deep += weight*loss_fn(a[valid],F.interpolate(target,size=shape,mode='bilinear',align_corners=False),F.interpolate(x,size=shape,mode='nearest'),F.interpolate(fg,size=shape,mode='nearest'))
 return main+deep

def collate(batch):
 return {k:torch.stack([x[k] for x in batch]) if isinstance(batch[0].get(k),torch.Tensor) else [x[k] for x in batch] for k in batch[0]}

@torch.no_grad()
def evaluate(model,data,device):
 model.eval(); by={}
 for i in range(len(data)):
  b=data[i]; x=b['x'][None].to(device); pred,_=model.forward_sequence(x,b['condition'][None].to(device),b['foreground'][None].to(device),b['times'][None].to(device)); valid=b['times']>=0; truth=b['target'][valid].to(device); pred=pred[0][valid]; tumor=x[0,2:3].expand_as(pred); err=(pred-truth).abs(); vals=((err*tumor).flatten(1).sum(1)/tumor.flatten(1).sum(1).clamp_min(1)).cpu().numpy(); by.setdefault(b['patient_group'],[]).extend(vals.tolist())
 pv=[np.mean(v) for v in by.values()]; return {'patients':len(pv),'samples':sum(map(len,by.values())),'tumor_mae':float(np.mean([x for v in by.values() for x in v])),'patient_macro_tumor_mae':float(np.mean(pv)),'by_patient':{k:float(np.mean(v)) for k,v in by.items()}}

def run_stage(model,train,val,device,epochs,steps,lr,output,stage):
 sampler=WeightedRandomSampler(train.weights,num_samples=steps*2,replacement=True); loader=DataLoader(train,batch_size=2,sampler=sampler,num_workers=0,collate_fn=collate); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda'); best=1e9; best_epoch=0; history=[]
 for epoch in range(1,epochs+1):
  model.train(); it=iter(loader); losses=[]; tic=time.time()
  for _ in range(steps):
   b=next(it); b={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()}; opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',enabled=device.type=='cuda',dtype=torch.bfloat16): pred,aux=model.forward_sequence(b['x'],b['condition'],b['foreground'],b['times'],12); loss=sequence_loss(pred,aux,b)
   scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
  valm=evaluate(model,val,device); row={'stage':stage,'epoch':epoch,'loss':float(np.mean(losses)),'val':valm,'seconds':time.time()-tic}; history.append(row); print(json.dumps(row),flush=True)
  if valm['patient_macro_tumor_mae']<best: best=valm['patient_macro_tumor_mae']; best_epoch=epoch; torch.save({'model':model.state_dict(),'stage':stage,'epoch':epoch,'val':valm},output/f'model_{stage}_best.pt')
 return history,best_epoch

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--folds',type=Path,required=True); p.add_argument('--fold',type=int,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--short-horizon',type=int,default=16); p.add_argument('--stage1-epochs',type=int,default=30); p.add_argument('--stage2-epochs',type=int,default=40); p.add_argument('--steps-per-epoch',type=int,default=500); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=20261112); args=p.parse_args()
 random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 folds=json.loads(args.folds.read_text()); held=set(folds['folds'][str(args.fold)]); allp=set(folds['patient_to_fold']); remaining=sorted(allp-held)
 rng=np.random.default_rng(args.seed+args.fold); rng.shuffle(remaining); internal_val_count=max(1,round(len(remaining)*0.2)); internal_val=set(remaining[:internal_val_count]); trainp=sorted(set(remaining)-internal_val)
 args.output.mkdir(parents=True,exist_ok=True); (args.output/'fold_patients.json').write_text(json.dumps({'fold':args.fold,'train':trainp,'internal_validation':sorted(internal_val),'heldout_test':sorted(held)},ensure_ascii=False,indent=2),encoding='utf-8')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); internal_val_data=PaperSequenceDataset(args.data,internal_val,False,4,None); heldout_test=PaperSequenceDataset(args.data,held,False,4,None); stage1=PaperSequenceDataset(args.data,trainp,True,4,args.short_horizon); full=PaperSequenceDataset(args.data,trainp,True,4,None); model=DynamicsUNet(time_scale=135,width=24,hidden=64,local_hidden=16,enhanced=True).to(device)
 hist1,e1=run_stage(model,stage1,internal_val_data,device,args.stage1_epochs,args.steps_per_epoch,args.lr,args.output,'stage1_short'); ck=torch.load(args.output/'model_stage1_short_best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model']); hist2,e2=run_stage(model,full,internal_val_data,device,args.stage2_epochs,args.steps_per_epoch,args.lr,args.output,'stage2_full'); ck2=torch.load(args.output/'model_stage2_full_best.pt',map_location=device,weights_only=False); model.load_state_dict(ck2['model']); result={'fold':args.fold,'stage1_best_epoch':e1,'stage2_best_epoch':e2,'internal_validation':evaluate(model,internal_val_data,device),'heldout_test':evaluate(model,heldout_test,device),'protocol':'patient-level five-fold; held-out fold used only once for final evaluation; internal validation selects checkpoints; Stage I short horizon followed by Stage II full horizon; up to four targets; no predicted-temperature feedback'}; (args.output/'final_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); (args.output/'history.json').write_text(json.dumps(hist1+hist2,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
