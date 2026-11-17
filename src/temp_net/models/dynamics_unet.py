"""Causal dual-resolution temperature dynamics without future measurements."""
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import TemperatureUNet


class ConvGRU(nn.Module):
    def __init__(self,channels=64):
        super().__init__()
        self.gates=nn.Conv2d(2*channels,2*channels,3,padding=1)
        self.candidate=nn.Conv2d(2*channels,channels,3,padding=1)
    def forward(self,x,h):
        reset,update=self.gates(torch.cat([x,h],1)).sigmoid().chunk(2,1)
        proposal=torch.tanh(self.candidate(torch.cat([x,reset*h],1)))
        return (1-update)*h+update*proposal


class SeparableConvGRU(nn.Module):
    """Cheap local state: spatial depthwise filtering followed by channel mixing."""
    def __init__(self,channels=16):
        super().__init__();joined=2*channels
        self.gates=nn.Sequential(nn.Conv2d(joined,joined,3,padding=1,groups=joined,bias=False),nn.Conv2d(joined,2*channels,1))
        self.candidate=nn.Sequential(nn.Conv2d(joined,joined,3,padding=1,groups=joined,bias=False),nn.Conv2d(joined,channels,1))
    def forward(self,x,h):
        reset,update=self.gates(torch.cat([x,h],1)).sigmoid().chunk(2,1)
        proposal=torch.tanh(self.candidate(torch.cat([x,reset*h],1)))
        return (1-update)*h+update*proposal


class DynamicsUNet(TemperatureUNet):
    """1/32 long-range state plus an optional efficient 1/8 local state."""
    def __init__(self,time_scale,width=24,hidden=64,local_hidden=16,enhanced=False):
        super().__init__(time_scale,width,use_physics_basis=enhanced)
        self.enhanced=bool(enhanced);self.hidden_channels=hidden;self.local_hidden_channels=local_hidden
        self.temporal_in=nn.Conv2d(width*16,hidden,1)
        self.temporal_condition=nn.Linear(128,hidden)
        self.gru=ConvGRU(hidden)
        self.temporal_out=nn.Conv2d(hidden,width*16,1)
        nn.init.zeros_(self.temporal_out.weight);nn.init.zeros_(self.temporal_out.bias)
        self.local_in=nn.Conv2d(width*8,local_hidden,1)
        self.local_condition=nn.Linear(128,local_hidden)
        self.local_gru=SeparableConvGRU(local_hidden)
        self.local_out=nn.Conv2d(local_hidden,width*8,1)
        nn.init.zeros_(self.local_out.weight);nn.init.zeros_(self.local_out.bias)
        self.aux_heads=nn.ModuleList([nn.Conv2d(width*4,1,1),nn.Conv2d(width*2,1,1)])
        for head in self.aux_heads:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def decode(self,encoded,skips,coarse_state,local_state,embedding,x,foreground,time_indices):
        features=encoded+F.interpolate(self.temporal_out(coarse_state),size=encoded.shape[-2:],mode='bilinear',align_corners=False)
        base=x[:,1:2].float()*20+35;support=torch.maximum(foreground,x[:,2:3])
        gate=time_indices.float().clamp(0,1)[:,None,None,None];auxiliaries=[]
        for j,(up,block,film) in enumerate(zip(self.up,self.dec,self.df)):
            skip=skips[-j-2]
            features=up(F.interpolate(features,size=skip.shape[-2:],mode='bilinear',align_corners=False))
            features=film(block(torch.cat([features,skip],1)),embedding)
            if j==0 and self.enhanced:features=features+self.local_out(local_state)
            if j in (1,2):
                delta=self.aux_heads[j-1](features).float();b=F.interpolate(base,size=delta.shape[-2:],mode='bilinear',align_corners=False)
                mask=F.interpolate(support,size=delta.shape[-2:],mode='nearest');auxiliaries.append(b+20*delta*gate*mask)
        return base+20*self.head(features).float()*gate*support,auxiliaries

    def forward_sequence(self,x,condition,foreground,times,detach_interval=0):
        if times.ndim!=2 or times.shape[0]!=x.shape[0]:raise ValueError('times must be [B,K]')
        if torch.any(times < -1) or torch.any(times != times.round()):raise ValueError('Only integer frame offsets >=0 and -1 padding are supported')
        batch,k=times.shape;h,w=x.shape[-2:]
        if h%16 or w%16:raise ValueError('Spatial size must be divisible by 16')
        static=condition.clone();static[:,3]=0;embedding0=self.condition(static);features=x;skips=[]
        for block,film in zip(self.enc,self.ef):features=film(block(features),embedding0);skips.append(features)
        coarse_latent=self.temporal_in(F.avg_pool2d(features,2));coarse_state=torch.tanh(coarse_latent)
        local_latent=self.local_in(skips[-2]);local_state=torch.tanh(local_latent)
        predictions=x.new_zeros((batch,k,1,h,w),dtype=torch.float32)
        auxiliary=[x.new_zeros((batch,k,1,h//s,w//s),dtype=torch.float32) for s in (4,2)]
        zero_b,zero_k=torch.where(times==0)
        if len(zero_b):
            baseline=x[zero_b,1:2].float()*20+35;predictions=predictions.index_put((zero_b,zero_k),baseline)
            for q,s in enumerate((4,2)):auxiliary[q]=auxiliary[q].index_put((zero_b,zero_k),F.interpolate(baseline,size=(h//s,w//s),mode='bilinear',align_corners=False))
        last_time=times.max(1).values
        for step in range(1,max(0,int(last_time.max().item()))+1):
            c=static.clone();c[:,3]=step/self.time_scale;embedding=self.condition(c)
            coarse_update=self.gru(coarse_latent+self.temporal_condition(embedding)[:,:,None,None],coarse_state)
            active=(last_time>=step)[:,None,None,None];coarse_state=torch.where(active,coarse_update,coarse_state)
            if self.enhanced:
                local_update=self.local_gru(local_latent+self.local_condition(embedding)[:,:,None,None],local_state)
                local_state=torch.where(active,local_update,local_state)
            bi,ki=torch.where(times==step)
            if len(bi):
                pred,aux=self.decode(features[bi],[s[bi] for s in skips],coarse_state[bi],local_state[bi],embedding[bi],x[bi],foreground[bi],times[bi,ki])
                predictions=predictions.index_put((bi,ki),pred)
                for q in range(2):auxiliary[q]=auxiliary[q].index_put((bi,ki),aux[q])
            if self.training and detach_interval and step%detach_interval==0:
                coarse_state=coarse_state.detach();local_state=local_state.detach()
        return predictions,auxiliary

    def forward(self,x,condition,foreground):
        times=(condition[:,3]*self.time_scale).round().long()[:,None]
        return self.forward_sequence(x,condition,foreground,times)[0][:,0]
