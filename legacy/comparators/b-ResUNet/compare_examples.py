"""Paired held-out examples: frozen DynamicsUNet vs frozen plain ResUNet."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import os
# Directory containing the legacy DynamicsUNet `model.py`.
LEGACY=Path(os.environ.get('TEMP_NET_LEGACY_MODEL_DIR','.'))
sys.path.insert(0,str(LEGACY))
from model import DynamicsUNet
sys.path.insert(0,str(Path(__file__).resolve().parent))
from resunet import ResUNetAR

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def dose_linear(previous,current,seconds):
    # Analytic integration of CEM43 rate under linearly varying temperature.
    lo=np.minimum(previous,current).astype(np.float64)
    hi=np.maximum(previous,current).astype(np.float64)
    diff=hi-lo
    def integral(a,b,k):
        return np.exp(k*(a-43))*np.expm1(k*(b-a))/k
    low_hi=np.minimum(hi,43.); high_lo=np.maximum(lo,43.)
    result=np.zeros_like(lo)
    low=lo<low_hi; high=hi>high_lo
    result[low]+=integral(lo[low],low_hi[low],np.log(4.))
    result[high]+=integral(high_lo[high],hi[high],np.log(2.))
    flat=diff<1e-8
    result[~flat]/=diff[~flat]
    result[flat]=np.exp(np.where(lo[flat]<43,np.log(4.),np.log(2.))*(lo[flat]-43))
    return result*seconds/60.

def dose_sequence(values,times,baseline):
    total=np.zeros_like(baseline,dtype=np.float64); prev=baseline; tprev=0
    for t,v in zip(times,values):
        total+=dose_linear(prev,v,3.6*(int(t)-tprev)); prev=v; tprev=int(t)
    assert np.isfinite(total).all()
    return total

def roi_crop(a):
    mask=a['tumor_mask']>0
    rr,cc=np.where(mask)
    pr,pc=a['probe_center_rc']
    h,w=mask.shape
    if len(rr):
        r0=max(0,int(min(rr.min(),pr))-48); r1=min(h,int(max(rr.max(),pr))+49)
        c0=max(0,int(min(cc.min(),pc))-48); c1=min(w,int(max(cc.max(),pc))+49)
    else:r0,r1,c0,c1=0,h,0,w
    return (slice(r0,r1),slice(c0,c1))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--best',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--patients',nargs='+',required=True,help='Predefined test patient groups to illustrate.'); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); device=torch.device('cuda')
    best=DynamicsUNet(time_scale=135,width=24,hidden=64,local_hidden=16,enhanced=True).to(device).eval()
    best.load_state_dict(torch.load(args.best,map_location='cpu',weights_only=False)['model'])
    base=ResUNetAR().to(device).eval()
    base.load_state_dict(torch.load(args.baseline,map_location='cpu',weights_only=False)['model'])
    norm=json.loads((args.data/'normalization.json').read_text())
    with (args.data/'manifests/segment_manifest.csv').open(encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r['split']=='test']
    # Predefined patients; longest sequence up to the evaluated 64-frame prefix.
    patients=args.patients
    selected=[]
    for patient in patients:
        choices=[r for r in rows if r['patient_group']==patient]
        if choices: selected.append(sorted(choices,key=lambda r:(-min(int(r['frames']),64),r['sample_id']))[0])
    results=[]; examples=[]
    for r in selected:
        with np.load(args.data/r['npz_file'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
        times=a['target_time_indices'][:64].astype(np.int64); truth=a['targets_C'][:64].astype(np.float32)
        s=a['structural_zscore'].astype(np.float32); tumor=a['tumor_mask'].astype(np.float32)
        fg=a['foreground_mask'].astype(np.float32); g=a['probe_gaussian'].astype(np.float32)
        baseline=a['baseline_C'].astype(np.float32); h,w=s.shape; cr,cc=a['probe_center_rc']
        x=np.stack([s,(baseline-35)/20,tumor,g,np.zeros_like(g)]).astype(np.float32)
        c=np.array([float(a['power_W'])/norm['power_scale_train_max_W'],2*cr/(h-1)-1,2*cc/(w-1)-1,0],np.float32)
        bestpred,_=best.forward_sequence(torch.from_numpy(x)[None].to(device),torch.from_numpy(c)[None].to(device),torch.from_numpy(fg)[None,None].to(device),torch.from_numpy(times)[None].to(device))
        bestpred=bestpred[0,:,0].cpu().numpy()
        current=(baseline-35)/20; basepred=[]
        # Match the existing ResUNet evaluator: one update per observed frame.
        for t in times:
            inputs=np.stack([s,current,(baseline-35)/20,tumor,g,g*c[0],np.full_like(g,t/135),fg]).astype(np.float32)
            predicted=base(torch.from_numpy(inputs)[None].to(device))[0,0].cpu().numpy()
            basepred.append(predicted); current=(predicted-35)/20
        basepred=np.stack(basepred)
        assert bestpred.shape==basepred.shape==truth.shape
        assert all(np.isfinite(v).all() for v in (truth,bestpred,basepred))
        mask=tumor>0.5; crop=roi_crop(a)
        info={'sample_id':r['sample_id'],'split':r['split'],'frames':len(times),'time_index':int(times[-1]),'elapsed_s':float(times[-1]*3.6),'power_W':float(a['power_W']),'missing_steps_within_prefix':int(times[-1]-len(times))}
        for name,values in [('best',bestpred),('baseline',basepred)]:
            info[name+'_end_tumor_MAE_C']=float(np.abs(values[-1]-truth[-1])[mask].mean())
            info[name+'_sequence_tumor_MAE_C']=float(np.abs(values-truth)[:,mask].mean())
        dose={name:dose_sequence(v,times,baseline) for name,v in [('truth',truth),('best',bestpred),('baseline',basepred)]}
        for name,v in dose.items():info[name+'_CEM43_coverage_pct']={str(t):float((v[mask]>t).mean()*100) for t in (2,10,60)}
        np.savez_compressed(args.output/(r['sample_id']+'.npz'),times=times,truth_C=truth,best_C=bestpred,baseline_prediction_C=basepred,structural=s,tumor_mask=tumor,**{k+'_cem43':v for k,v in dose.items()})
        results.append(info);examples.append((s,tumor,truth[-1],bestpred[-1],basepred[-1],dose,crop,info))
        print(json.dumps(info),flush=True)
    fig,axes=plt.subplots(len(examples),6,figsize=(20,3.2*len(examples)),squeeze=False,layout='constrained')
    tempcmap=plt.get_cmap('RdYlGn_r').copy();tempcmap.set_over('#600000')
    for i,(s,tm,gt,pb,pr,dose,crop,info) in enumerate(examples):
        for ax in axes[i]:ax.set_axis_off()
        axes[i,0].imshow(s[crop],cmap='gray',interpolation='nearest')
        axes[i,0].contour(tm[crop],levels=[.5],colors='cyan',linewidths=.7)
        axes[i,0].set_title(info['sample_id']+'\n'+f"t={info['elapsed_s']:.1f}s; {info['power_W']:g}W",fontsize=8)
        for j,v,title in [(1,gt,'Measured temperature'),(2,pb,'Best: DynamicsUNet'),(3,pr,'Baseline: ResUNet')]:
            im=axes[i,j].imshow(v[crop],cmap=tempcmap,vmin=35,vmax=80,interpolation='nearest')
            axes[i,j].contour(tm[crop],levels=[.5],colors='cyan',linewidths=.7)
            axes[i,j].set_title(title,fontsize=9)
        for j,v,key,title in [(4,pb,'best','Best absolute error'),(5,pr,'baseline','Baseline absolute error')]:
            er=axes[i,j].imshow(np.abs(v-gt)[crop],cmap='magma',vmin=0,vmax=15,interpolation='nearest')
            axes[i,j].contour(tm[crop],levels=[.5],colors='cyan',linewidths=.7)
            axes[i,j].set_title(title+f"\nTumor MAE={info[key+'_end_tumor_MAE_C']:.2f} C",fontsize=9)
    fig.colorbar(im,ax=axes[:,1:4].ravel().tolist(),label='Temperature (C)',shrink=.4,extend='both')
    fig.colorbar(er,ax=axes[:,4:].ravel().tolist(),label='Absolute error (C)',shrink=.4,extend='max')
    fig.suptitle('Paired test examples: segment-level prediction, no temperature calibration',fontsize=13)
    fig.savefig(args.output/'temperature_comparison.png',dpi=180);plt.close(fig)
    # Threshold coverage contours retain normal tissue around the tumor.
    fig,axes=plt.subplots(len(examples),4,figsize=(15,3.2*len(examples)),squeeze=False,layout='constrained')
    colors=['#ffee00','#ff8000','#ff3030']
    for i,(s,tm,gt,pb,pr,dose,crop,info) in enumerate(examples):
        for j,key in enumerate([None,'truth','best','baseline']):
            ax=axes[i,j];ax.imshow(s[crop],cmap='gray',interpolation='nearest');ax.set_axis_off()
            ax.contour(tm[crop],levels=[.5],colors='cyan',linewidths=.8)
            if key:
                v=dose[key][crop]
                for t,color in zip((2,10,60),colors):
                    if v.min()<t<v.max():ax.contour(v,levels=[t],colors=color,linewidths=.8)
                cv=info[key+'_CEM43_coverage_pct']
                ax.set_title(key+f"\n2/10/60 min: {cv['2']:.1f}/{cv['10']:.1f}/{cv['60']:.1f}%",fontsize=8)
            else:ax.set_title(info['sample_id']+'\nMRI + tumor',fontsize=8)
    fig.suptitle('Current-segment CEM43: cyan=tumor; yellow=2, orange=10, red=60 min contours',fontsize=12)
    fig.savefig(args.output/'cem43_comparison.png',dpi=180);plt.close(fig)
    summary={'best_weights':str(args.best),'best_sha256':digest(args.best),'baseline_weights':str(args.baseline),'baseline_sha256':digest(args.baseline),'selection':'predefined 6 test patients, longest prefix up to 64 observed frames; independent of prediction errors','calibration':False,'cem_policy':'analytic linear-temperature integration; current segment only, no preceding segments or waiting intervals','gap_policy':'matches existing evaluators; DynamicsUNet advances every index, ResUNet once per observed frame','examples':results}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (args.output/'README.md').write_text('# Paired prediction examples\n\nSame test segments, baseline, power, center and timestamps. Frozen weights; no temperature calibration. Cyan outlines mark the tumor. MRI is foreground z-score.\n\nTemperature panels: shared 35-80 C scale; error panels: shared 0-15 C scale. ROI crop contains the tumor and probe center with a 48-pixel margin.\n\nCEM43 contours use analytic integration of linearly changing temperature, inclusive CEM43 >=2/>=10/>=60 min coverage. Dose includes only the selected segment, not the preceding trajectory.\n\nThis compares existing fitted systems, not an architecture-only ablation. Where frames are missing, the existing evaluators have different rollout policies; missing-step counts are recorded in summary.json.\n',encoding='utf-8')

if __name__=='__main__':main()

