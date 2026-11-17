"""2D conditional residual U-Net; no measured future temperature is a forward input."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(cin,cout,3,stride,padding=1,bias=False),
            nn.GroupNorm(8,cout),nn.SiLU(),nn.Conv2d(cout,cout,3,padding=1,bias=False),
            nn.GroupNorm(8,cout),nn.SiLU())
    def forward(self,x):
        return self.net(x)


class FiLM(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.affine=nn.Linear(128,2*channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
    def forward(self,x,c):
        g,b=self.affine(c).chunk(2,dim=1)
        return x*(1+g[:,:,None,None])+b[:,:,None,None]


class PhysicalConditionEncoder(nn.Module):
    """Matched condition encoder with optional explicit heating-time bases.

    Both ablation arms have exactly the same parameters.  The control zeros the
    six derived columns, while the proposed arm exposes simple deterministic
    functions of normalized power and elapsed time.  No target temperatures are
    used to construct these features.
    """
    def __init__(self,use_physics_basis=False):
        super().__init__()
        self.use_physics_basis=bool(use_physics_basis)
        self.net=nn.Sequential(nn.Linear(10,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())

    def forward(self,condition):
        p,row,col,t=condition.float().unbind(1)
        derived=torch.stack((p*t,t.square(),torch.sqrt(t.clamp_min(0)+1e-8)-1e-4,
            torch.sin(torch.pi*t),torch.cos(torch.pi*t)-1,p.square()),1)
        if not self.use_physics_basis:
            derived=torch.zeros_like(derived)
        return self.net(torch.cat((condition.float(),derived),1))


class TemperatureUNet(nn.Module):
    def __init__(self,time_scale,width=24,use_physics_basis=False):
        super().__init__()
        self.time_scale=float(time_scale)
        channels=[width*2**i for i in range(5)]
        self.condition=PhysicalConditionEncoder(use_physics_basis)
        self.enc=nn.ModuleList([Block(5 if i==0 else channels[i-1],v,1 if i==0 else 2) for i,v in enumerate(channels)])
        # Same extra input weights in BOTH arms, initially zero. Only the input
        # source values differ; channel count and parameter count match exactly.
        with torch.no_grad():self.enc[0].net[0].weight[:,4].zero_()
        self.ef=nn.ModuleList([FiLM(v) for v in channels])
        self.up=nn.ModuleList([nn.Conv2d(channels[i+1],channels[i],1) for i in range(3,-1,-1)])
        self.dec=nn.ModuleList([Block(channels[i]*2,channels[i]) for i in range(3,-1,-1)])
        self.df=nn.ModuleList([FiLM(channels[i]) for i in range(3,-1,-1)])
        self.head=nn.Conv2d(width,1,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self,x,condition,foreground):
        # x: structural z-score, (baseline-35)/20, tumor mask, probe Gaussian.
        # condition: power/train_max, row[-1,1], col[-1,1], time_index/train_max.
        embedding=self.condition(condition)
        features=x
        skips=[]
        for block,film in zip(self.enc,self.ef):
            features=film(block(features),embedding)
            skips.append(features)
        for j,(up,block,film) in enumerate(zip(self.up,self.dec,self.df)):
            skip=skips[-j-2]
            features=up(F.interpolate(features,size=skip.shape[-2:],mode='bilinear',align_corners=False))
            features=film(block(torch.cat([features,skip],1)),embedding)
        # Exact initial condition at t=0; no assumption of monotonic heating.
        gate=(condition[:,3:4]*self.time_scale).clamp(0,1)[:,:,None,None]
        baseline=x[:,1:2].float()*20.+35.
        support=torch.maximum(foreground,x[:,2:3])
        return baseline+20.*self.head(features).float()*gate*support
