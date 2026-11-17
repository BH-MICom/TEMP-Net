"""Single-stage U-Net training, followed by frozen test evaluations."""
import argparse, hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch
from model import UNetAR

# Reuse the frozen ResUNet data/loss/evaluator to keep the baseline protocol identical.
import os
BASELINE=Path(os.environ.get('TEMP_NET_RESUNET_DIR',Path(__file__).resolve().parents[1]/'b-ResUNet'))
sys.path.insert(0,str(BASELINE))
from data import ResDataset
from train import train_stage,evaluate,seed_all

def checksum(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--steps',type=int,default=500)
    p.add_argument('--seed',type=int,default=20261011);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'stage1/model_best.pt').exists():raise RuntimeError('Checkpoint exists; choose a fresh output directory')
    seed_all(args.seed);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    device=torch.device('cuda');m=UNetAR().to(device)
    cfg=SimpleNamespace(epochs=[args.epochs],lr=[1e-4],steps=args.steps,batch_size=2,max_targets=64,patience=6)
    protocol={'model':'Plain U-Net, four encoder scales, no residual blocks, FiLM or recurrent memory',
              'channels':['MRI zscore','current temperature norm','segment baseline norm','tumor mask','probe Gaussian','power map','time map','foreground mask'],
              'training':'single stage, adjacent measured-frame teacher forcing, from scratch',
              'seed':args.seed,'epochs_max':args.epochs,'steps_per_epoch':args.steps,
              'batch_size':2,'lr':1e-4,'early_stopping_patience':6,'checkpoint_selection':'validation patient macro tumor MAE, 64-target rollout',
              'data_root':str(args.data),'patient_split_sha256':checksum(args.data/'splits/patient_split.json'),
              'parameters':sum(p.numel() for p in m.parameters()),
              'eval_gap_policy':'same as existing ResUNet baseline: one update per observed target; missing targets not synthesized',
              'shared_training_code':str(BASELINE/'train.py'),'shared_training_sha256':checksum(BASELINE/'train.py')}
    write(args.output/'protocol.json',protocol)
    status=args.output/'status.json'
    try:
        write(status,{'state':'training','started_unix':time.time()})
        best=train_stage(m,args.data,1,args.output/'stage1',device,cfg)
        ck=torch.load(best,map_location=device,weights_only=False);m.load_state_dict(ck['model'])
        torch.save({'model':ck['model'],'epoch':ck['epoch'],'val':ck['val'],'protocol':protocol},args.output/'model_best.pt')
        result={'best_epoch':ck['epoch'],'validation_64':ck['val'],'test':{}}
        write(status,{'state':'evaluating','best_epoch':ck['epoch']})
        for name,limit in [('first_8',8),('first_64',64),('all_observed_frames',1000000)]:
            metrics=evaluate(m,ResDataset(args.data,'test','sequence',limit),device,limit)
            result['test'][name]=metrics
            write(args.output/'test_metrics.json',result)
            print(json.dumps({'test_range':name,**metrics}),flush=True)
        write(status,{'state':'complete','finished_unix':time.time(),'best_epoch':ck['epoch']})
    except Exception as e:
        write(status,{'state':'failed','error':str(e),'failed_unix':time.time()});raise

if __name__=='__main__':main()
