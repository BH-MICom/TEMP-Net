"""Meaningful CPU checks: temporal mixing, causality, padding and loss masking."""
import numpy as np
import torch
from adapter import TemperatureSimVP
from data import make_input
from run import predict_segment,loss_fn

torch.set_num_threads(2);torch.manual_seed(5)
model=TemperatureSimVP().eval()
# Nonzero readout is needed to observe the translator; trained head starts at zero.
torch.nn.init.normal_(model.core.dec.readout.weight,std=.01)
sample={'structural_zscore':np.random.default_rng(1).normal(size=(64,64)).astype(np.float32),
        'baseline_C':np.full((64,64),35.5,np.float32),'tumor_mask':np.ones((64,64),np.float32),
        'foreground_mask':np.ones((64,64),np.float32),'probe_gaussian':np.ones((64,64),np.float32),
        'power_W':np.float32(6),'target_time_indices':np.array([1,2,4,7]),
        'targets_C':np.full((4,64,64),99.,np.float32)}
x=torch.from_numpy(make_input(sample,[sample['baseline_C']]*4,[0,0,0,0],12))[None]
with torch.no_grad():
    p=model(x);changed=x.clone();changed[:,0,1]+=.2;q=model(changed)
assert p.shape==(1,4,1,64,64)
assert not torch.allclose(p,q,atol=1e-7),'Earliest history frame has no temporal influence'
pred=predict_segment(model,sample,12,torch.device('cpu'))
altered={**sample,'targets_C':sample['targets_C']+1000}
assert np.array_equal(pred,predict_segment(model,altered,12,torch.device('cpu'))),'Future measured temperatures affected evaluation'
target=torch.ones_like(p)*36;valid=torch.tensor([[True,True,False,False]])
loss=loss_fn(p,target,x,valid);target[:,2:]=1000
assert torch.allclose(loss,loss_fn(p,target,x,valid)),'Missing target labels affected loss'
assert np.isfinite(pred).all()
print('PASS: official temporal mixing, output [B,4,1,H,W], no future-label input leakage, missing-label mask, missing-index free rollout')
