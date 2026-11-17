#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        groups = 8 if cout % 8 == 0 else 1
        self.main = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(groups, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(groups, cout),
        )
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride)
        self.act = nn.SiLU()

    def forward(self, x): return self.act(self.main(x) + self.skip(x))


class ResUNetAR(nn.Module):
    """Plain residual U-Net autoregressive baseline; no FiLM or recurrent state."""
    def __init__(self, width=24):
        super().__init__()
        ch = [width, width * 2, width * 4, width * 8]
        self.enc = nn.ModuleList([
            ResBlock(8, ch[0]), ResBlock(ch[0], ch[1], 2),
            ResBlock(ch[1], ch[2], 2), ResBlock(ch[2], ch[3], 2)
        ])
        self.up = nn.ModuleList([
            nn.Conv2d(ch[3], ch[2], 1), nn.Conv2d(ch[2], ch[1], 1),
            nn.Conv2d(ch[1], ch[0], 1)
        ])
        self.dec = nn.ModuleList([
            ResBlock(ch[2] * 2, ch[2]), ResBlock(ch[1] * 2, ch[1]),
            ResBlock(ch[0] * 2, ch[0])
        ])
        self.head = nn.Conv2d(ch[0], 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: structural, current, baseline, tumor, Gaussian, power, time, foreground
        skips=[]; z=x
        for block in self.enc:
            z=block(z); skips.append(z)
        for up, dec, skip in zip(self.up, self.dec, (skips[2], skips[1], skips[0])):
            z=up(F.interpolate(z,size=skip.shape[-2:],mode="bilinear",align_corners=False))
            z=dec(torch.cat([z,skip],1))
        current=x[:,1:2].float(); support=torch.maximum(x[:,3:4].float(),x[:,7:8].float())
        time_gate=x[:,6:7].float().clamp(0,1)
        return current*20.0+35.0+20.0*self.head(z).float()*time_gate*support
