"""Train official multi-frame SimVP adaptation and cold-start free-rollout evaluate."""
import argparse,csv,hashlib,json,random,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from adapter import TemperatureSimVP
from data import TrainingWindows,make_input,load_npz,manifest

def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def region(error,mask):
    n=mask.flatten(1).sum(1);values=(error*mask).flatten(1).sum(1)/n.clamp_min(1)
    return values[n>0].mean() if (n>0).any() else error.sum()*0
def loss_fn(pred,y,x,valid):
    b,t=valid.shape
    if not valid.any():raise ValueError('Batch has no observed targets')
    e=(pred[valid]-y[valid]).abs()
    tm=x[:,-1,3:4][:,None].expand(-1,t,-1,-1,-1)[valid]
    fg=x[:,-1,7:8][:,None].expand(-1,t,-1,-1,-1)[valid]
    hot=1+1.5*torch.sigmoid((y[valid]-41.5)/1.5)
    return .05*e.mean()+.45*region(e*hot,torch.maximum(tm,fg))+1.5*region(e*hot,tm)

@torch.inference_mode()
def predict_segment(model,a,power_scale,device):
    h=model.history;times=a['target_time_indices'].astype(int)
    # Cold start: no measured heating frame is used as a prediction input.
    buffer=[a['baseline_C'].astype(np.float32).copy() for _ in range(h)]
    buffer_times=[0]*h;out=[];origin=0
    while origin<int(times[-1]):
        x=torch.from_numpy(make_input(a,buffer,buffer_times,power_scale))[None].to(device)
        pred=model(x)[0,:,0].cpu().numpy()
        if not np.isfinite(pred).all():raise FloatingPointError('Non-finite free rollout')
        for j in range(h):
            out.append(pred[j]);buffer.append(pred[j]);buffer_times.append(origin+j+1)
        buffer=buffer[-h:];buffer_times=buffer_times[-h:];origin+=h
    return np.stack(out)[times-1]

@torch.inference_mode()
def evaluate(model,root,split,device,power_scale,max_targets=None,examples_dir=None):
    model.eval();rows=manifest(root,split);by=defaultdict(list);counts=0
    sum_mae=sum_fg=sum_tm=sum_bias=0.;valid_tm=0;recs=[]
    for i,row in enumerate(rows):
        a=load_npz(Path(root)/row['npz_file'])
        if max_targets:
            a['target_time_indices']=a['target_time_indices'][:max_targets]
            a['targets_C']=a['targets_C'][:max_targets]
        pred=predict_segment(model,a,power_scale,device);truth=a['targets_C'].astype(np.float32)
        error=pred-truth;tm=a['tumor_mask']>.5;fg=a['foreground_mask']>.5
        mae=np.abs(error).mean(axis=(1,2));sum_mae+=float(mae.sum());counts+=len(mae)
        fgvals=np.abs(error)[:,fg].mean(axis=1) if fg.any() else np.full(len(mae),np.nan)
        sum_fg+=float(fgvals.sum())
        if tm.any():
            values=np.abs(error)[:,tm].mean(axis=1);bias=error[:,tm].mean(axis=1)
            sum_tm+=float(values.sum());sum_bias+=float(bias.sum());valid_tm+=len(values)
            by[row['patient_group']].extend(values.tolist())
            for j,t in enumerate(a['target_time_indices']):
                recs.append({'sample_id':row['sample_id'],'patient_group':row['patient_group'],'time_index':int(t),
                             'tumor_mae':float(values[j]),'tumor_bias':float(bias[j])})
        if examples_dir and i<6:
            np.savez_compressed(examples_dir/(row['sample_id']+'.npz'),prediction=pred,truth=truth,times=a['target_time_indices'],structural=a['structural_zscore'],tumor_mask=a['tumor_mask'])
    return {'samples':counts,'patients':len(by),'mae':sum_mae/counts,'foreground_mae':sum_fg/counts,
            'tumor_mae':sum_tm/valid_tm,'patient_macro_tumor_mae':float(np.mean([np.mean(v) for v in by.values()])),
            'tumor_bias':sum_bias/valid_tm,'by_patient':{p:float(np.mean(v)) for p,v in by.items()},'records':recs}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--steps',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--patience',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=20261012)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    dev=torch.device('cuda');ds=TrainingWindows(a.data);model=TemperatureSimVP().to(dev)
    sampler=WeightedRandomSampler(ds.weights,num_samples=a.steps*a.batch_size,replacement=True)
    dl=DataLoader(ds,batch_size=a.batch_size,sampler=sampler,num_workers=2,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda');norm=json.loads((a.data/'normalization.json').read_text())
    if a.smoke:
        b=next(iter(dl));x=b['x'].to(dev);y=b['targets'].to(dev);valid=b['valid'].to(dev)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x);loss=loss_fn(pred,y,x,valid)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite smoke loss')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise RuntimeError('Nonfinite gradient')
            scaler.step(opt);scaler.update()
        print(json.dumps({'smoke':'passed','shape':list(pred.shape),'loss':float(loss),
                          'train_windows':len(ds),'parameters':sum(p.numel() for p in model.parameters()),
                          'max_gpu_MiB':torch.cuda.max_memory_allocated()/1024**2}),flush=True);return
    a.output.mkdir(parents=True,exist_ok=False);status=a.output/'status.json'
    config={'architecture':'official CVPR2022 SimVP encoder/Mid_Xnet Inception/decoder, temperature readout adaptation',
            'official_commit':'4dcd4451282e014573ca4346ced8223e342f7893','history_frames':4,'output_frames':4,
            'hid_S':16,'hid_T':128,'N_S':4,'N_T':4,'input_channels':8,'single_stage':True,
            'input_temperature_policy':'four measured history frames during training; baseline cold-start and predicted history only during evaluation',
            'gap_policy':'advance every 3.6-second index; loss/metrics at observed targets only',
            'patient_split_sha256':sha(a.data/'splits/patient_split.json'),'data':str(a.data),
            'seed':a.seed,'training_windows':len(ds),'skipped_windows_missing_history':ds.skipped_missing_history,
            'epochs_max':a.epochs,'steps':a.steps,'batch_size':a.batch_size,'lr':a.lr,
            'validation_selection':'all observed frames, free rollout patient macro tumor MAE',
            'note':'Different history and block-output architecture from single-frame baselines; not a pure backbone-only ablation.'}
    write(a.output/'protocol.json',config);history=[];best=float('inf');bad=0
    try:
        for epoch in range(1,a.epochs+1):
            model.train();losses=[];tic=time.time();write(status,{'state':'training','epoch':epoch,'epochs_max':a.epochs})
            for b in dl:
                x=b['x'].to(dev,non_blocking=True);y=b['targets'].to(dev,non_blocking=True);valid=b['valid'].to(dev,non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x);loss=loss_fn(pred,y,x,valid)
                if not torch.isfinite(loss):raise FloatingPointError('Training loss nonfinite')
                scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
            write(status,{'state':'validating_full_rollout','epoch':epoch})
            v=evaluate(model,a.data,'val',dev,norm['power_scale_train_max_W']);score=v['patient_macro_tumor_mae']
            compact={k:z for k,z in v.items() if k!='records'}
            r={'epoch':epoch,'train_loss':float(np.mean(losses)),'val':compact,'seconds':time.time()-tic}
            history.append(r);write(a.output/'history.json',history);print(json.dumps(r),flush=True)
            if score<best:
                best=score;bad=0;torch.save({'model':model.state_dict(),'epoch':epoch,'val':compact,'protocol':config},a.output/'model_best.pt')
            else:bad+=1
            if bad>=a.patience:break
        ck=torch.load(a.output/'model_best.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['model'])
        result={'best_epoch':ck['epoch'],'validation_all':ck['val'],'test':{}}
        for name,limit in [('first_8',8),('first_64',64),('all_observed_frames',None)]:
            write(status,{'state':'testing','range':name,'best_epoch':ck['epoch']})
            test=evaluate(model,a.data,'test',dev,norm['power_scale_train_max_W'],limit)
            result['test'][name]={k:z for k,z in test.items() if k!='records'}
            write(a.output/('per_frame_'+name+'.json'),test['records']);write(a.output/'test_metrics.json',result)
            print(json.dumps({'test_range':name,**result['test'][name]}),flush=True)
        write(status,{'state':'complete','best_epoch':ck['epoch'],'finished_unix':time.time()})
    except Exception as e:write(status,{'state':'failed','error':str(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
