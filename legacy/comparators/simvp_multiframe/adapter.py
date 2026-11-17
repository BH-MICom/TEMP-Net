"""Conditional temperature adapter around pinned official CVPR-2022 SimVP."""
import importlib.util
import sys
from pathlib import Path
import torch
from torch import nn

VENDOR=Path(__file__).parent/'vendor'/'SimVP'
spec=importlib.util.spec_from_file_location('simvp_official_modules',VENDOR/'modules.py')
official_modules=importlib.util.module_from_spec(spec);spec.loader.exec_module(official_modules)
sys.modules['modules']=official_modules
spec=importlib.util.spec_from_file_location('simvp_official_model',VENDOR/'model.py')
official_model=importlib.util.module_from_spec(spec);spec.loader.exec_module(official_model)

class TemperatureSimVP(nn.Module):
    def __init__(self,history=4,hid_s=16,hid_t=128,n_s=4,n_t=4):
        super().__init__();self.history=history
        self.core=official_model.SimVP((history,8,512,512),hid_S=hid_s,hid_T=hid_t,N_S=n_s,N_T=n_t)
        # Preserve official spatial encoder, temporal Translator and spatial decoder.
        # Only adapt the output readout from eight reconstructed channels to temperature.
        self.core.dec.readout=nn.Conv2d(hid_s,1,1)
        nn.init.zeros_(self.core.dec.readout.weight);nn.init.zeros_(self.core.dec.readout.bias)
    def forward(self,x):
        if x.ndim!=5 or x.shape[1:3]!=(self.history,8):raise ValueError('Expected [B,4,8,H,W]')
        b,t,c,h,w=x.shape
        embed,skip=self.core.enc(x.reshape(b*t,c,h,w))
        _,cs,hs,ws=embed.shape
        features=self.core.hid(embed.reshape(b,t,cs,hs,ws))
        delta=self.core.dec(features.reshape(b*t,cs,hs,ws),skip).reshape(b,t,1,h,w)
        current=x[:,-1:,1:2].float()*20+35
        support=torch.maximum(x[:,-1:,3:4],x[:,-1:,7:8]).float()
        return current+20*delta.float()*support
