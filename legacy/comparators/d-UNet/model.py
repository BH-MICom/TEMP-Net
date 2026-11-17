"""Plain U-Net with the same eight-channel contract as the ResUNet baseline."""
import torch
from torch import nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(cin,cout,3,stride,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),
            nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU())
    def forward(self,x):return self.net(x)

class UNetAR(nn.Module):
    def __init__(self,width=24):
        super().__init__(); ch=[width,width*2,width*4,width*8]
        self.enc=nn.ModuleList([DoubleConv(8,ch[0])]+[
            DoubleConv(ch[i-1],ch[i],2) for i in range(1,4)])
        self.up=nn.ModuleList([nn.Conv2d(ch[i+1],ch[i],1) for i in (2,1,0)])
        self.dec=nn.ModuleList([DoubleConv(2*ch[i],ch[i]) for i in (2,1,0)])
        self.head=nn.Conv2d(width,1,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        z=x;skips=[]
        for block in self.enc:z=block(z);skips.append(z)
        for up,block,skip in zip(self.up,self.dec,skips[-2::-1]):
            z=up(F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False))
            z=block(torch.cat([z,skip],1))
        support=torch.maximum(x[:,3:4].float(),x[:,7:8].float())
        gate=x[:,6:7].float().clamp(0,1)
        return x[:,1:2].float()*20+35+20*self.head(z).float()*gate*support
