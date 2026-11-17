#!/usr/bin/env python3
import argparse,csv,json,random,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from temp_net.data.direct_dataset import DirectSequenceDataset
from temp_net.models.dynamics_unet import DynamicsUNet

def loss_region(error,mask):
 n=mask.flatten(1).sum(1);v=(error*mask).flatten(1).sum(1)/n.clamp_min(1);return v[n>0].mean() if (n>0).any() else error.sum()*0
def loss_fn(pred,target,x,fg):
 e=(pred-target).abs();tumor=x[:,2:3].expand_as(e);roi=torch.maximum(tumor,fg);return .05*e.mean()+.5*loss_region(e,roi)+2.*loss_region(e,tumor)
def sequence_loss(pred,aux,b):
 valid=b['times']>=0;k=valid.shape[1];x=b['x'][:,None].expand(-1,k,-1,-1,-1)[valid];fg=b['foreground'][:,None].expand(-1,k,-1,-1,-1)[valid];target=b['target'][valid];main=loss_fn(pred[valid],target,x,fg);deep=main*0
 for weight,a in zip([.125,.25],aux):
  shape=a.shape[-2:];xx=F.interpolate(x,size=shape,mode='nearest');ff=F.interpolate(fg,size=shape,mode='nearest');yy=F.interpolate(target,size=shape,mode='bilinear',align_corners=False);deep=deep+weight*loss_fn(a[valid],yy,xx,ff)
 return main+deep
@torch.no_grad()
def evaluate(model,dataset,device,out_dir=None):
 model.eval();records=[]
 for i in range(len(dataset)):
  b=dataset[i];x=b['x'][None].to(device);condition=b['condition'][None].to(device);fg=b['foreground'][None].to(device);times=b['times'][None].to(device);target=b['target'][None].to(device)
  pred,_=model.forward_sequence(x,condition,fg,times);pred=pred[0];valid=times[0]>=0;pred=pred[valid];truth=target[0][valid];tumor=x[0,2:3].expand_as(pred);e=(pred-truth).abs();tm=(e*tumor).flatten(1).sum(1)/tumor.flatten(1).sum(1).clamp_min(1)
  for j in range(int(valid.sum())):records.append({'patient_group':b['patient_group'],'sample_id':b['sample_id'],'time_index':int(times[0,j]),'mae':float(e[j].mean()),'tumor_mae':float(tm[j])})
 by=defaultdict(list)
 for r in records:by[r['patient_group']].append(r['tumor_mae'])
 result={'samples':len(records),'mae':float(np.mean([r['mae'] for r in records])),'tumor_mae':float(np.mean([r['tumor_mae'] for r in records])),'patient_macro_tumor_mae':float(np.mean([np.mean(v) for v in by.values()])),'records':records}
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--steps-per-epoch',type=int,default=500);ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--window',type=int,default=4);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--patience',type=int,default=12);ap.add_argument('--seed',type=int,default=20260930);args=ap.parse_args()
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 args.output.mkdir(parents=True,exist_ok=True);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');train=DirectSequenceDataset(args.data,'train',True,args.window);val=DirectSequenceDataset(args.data,'val',False,args.window);test=DirectSequenceDataset(args.data,'test',False,args.window);sampler=WeightedRandomSampler(train.weights,num_samples=args.steps_per_epoch*args.batch_size,replacement=True);loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True,collate_fn=lambda b:{k:torch.stack([x[k] for x in b]) if k in ['x','condition','target','foreground','times'] else [x[k] for x in b] for k in b[0]});model=DynamicsUNet(time_scale=135,width=24,hidden=64,local_hidden=16,enhanced=True).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda');best=1e9;bad=0;history=[]
 for epoch in range(1,args.epochs+1):
  model.train();it=iter(loader);losses=[];tic=time.time()
  for _ in range(args.steps_per_epoch):
   b=next(it);b={k:v.to(dev,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in b.items()};opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',enabled=dev.type=='cuda',dtype=torch.bfloat16):pred,aux=model.forward_sequence(b['x'],b['condition'],b['foreground'],b['times'],12);loss=sequence_loss(pred,aux,b)
   scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();losses.append(float(loss.detach().cpu()))
  valm=evaluate(model,val,dev);row={'epoch':epoch,'loss':float(np.mean(losses)),'val':{k:v for k,v in valm.items() if k!='records'},'seconds':time.time()-tic};history.append(row);print(json.dumps(row),flush=True)
  if valm['patient_macro_tumor_mae']<best:best=valm['patient_macro_tumor_mae'];bad=0;torch.save({'model':model.state_dict(),'epoch':epoch,'val':valm,'args':vars(args)},args.output/'model_best.pt')
  else:bad+=1
  if bad>=args.patience:break
 (args.output/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8');ck=torch.load(args.output/'model_best.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['model']);final={'best':ck['val'],'val':evaluate(model,val,dev),'test':evaluate(model,test,dev)};(args.output/'final_metrics.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:{z:v.get(z) for z in ['samples','mae','tumor_mae','patient_macro_tumor_mae'] if z in v} for k,v in final.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
