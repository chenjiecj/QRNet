import torch
import torch.nn as nn

from nets.detr_training import weights_init
from .SFM import SFM

def autopad(k, p=None, d=1):
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True):
        super(Bottleneck, self).__init__()
        self.cv2 = SFM(c1, c2, 3, 1, 1, groups=c1)
        self.cv3 = SFM(c2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(x)) if self.add else self.cv3(self.cv2(x))

class SFMB(nn.Module):
    def __init__(self, c1, c2, n=1,  e=0.5):
        super(SFMB, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = SFM(c1, c_, 1, 1)
        self.cv2 = SFM(c1, c_, 1, 1)
        self.cv3 = SFM(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, ) for _ in range(n)])
    def forward(self, x):
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)),
                self.cv2(x)
            )
            , dim=1))



class SFMBackbone(nn.Module):
    def __init__(self,):
        super().__init__()
        base_channels =16
        base_depth = 1
        self.block1 = SFM(3,base_channels,3,2,1)

        self.block2 = nn.Sequential(
            SFM(base_channels, base_channels * 2, 3, 2, 1,groups=base_channels),
            SFMB(base_channels * 2, base_channels * 2, base_depth),
        )
        self.block3 = nn.Sequential(
            SFM(base_channels * 2, base_channels * 4, 3, 2,1,groups=base_channels * 2),
            SFMB(base_channels * 4, base_channels * 4, base_depth * 3),
        )
        self.block4 = nn.Sequential(
            SFM(base_channels * 4, base_channels * 8, 3, 2,1,groups=base_channels * 4),
            SFMB(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        self.block5 = nn.Sequential(
            SFM(base_channels * 8, base_channels * 16, 3, 2, 1,groups=base_channels * 8),
            SFMB(base_channels * 16, base_channels * 16, base_depth,),
        )
        weights_init(self)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat1 = x
        x = self.block4(x)
        feat2 = x
        x = self.block5(x)
        feat3 = x
        return feat1, feat2, feat3
