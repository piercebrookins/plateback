"""Pinned Apache-2.0 DINOv2 backbone with a locally trained dense segmentation head."""
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
REVISION='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
def backbone(pretrained=False):
    repo=ROOT/'data/source/dinov2'
    if not (repo/'hubconf.py').exists():raise ValueError('Pinned local DINOv2 source is missing; run ml/train_dino.py setup first')
    return torch.hub.load(str(repo),'dinov2_vits14',source='local',pretrained=pretrained,verbose=False)
def head():return nn.Sequential(nn.Conv2d(384,128,1),nn.GELU(),nn.Dropout2d(.1),nn.Conv2d(128,5,1))
class DinoSegmenter(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=backbone(False);self.head=head()
    def forward(self,x):
        f=self.backbone.forward_features(x)['x_norm_patchtokens'];h,w=x.shape[-2:];f=f.transpose(1,2).reshape(len(x),384,h//14,w//14)
        return {'out':nn.functional.interpolate(self.head(f),size=(h,w),mode='bilinear',align_corners=False)}

class SpatialHead(nn.Module):
    """V4 logits plus a learned spatial residual at twice the patch resolution."""
    def __init__(self):
        super().__init__();self.base=head()
        self.project=nn.Sequential(nn.Conv2d(384,64,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GELU())
        self.refine=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Conv2d(32,5,1))
        nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)
    def forward(self,x):
        size=(x.shape[-2]*2,x.shape[-1]*2)
        base=nn.functional.interpolate(self.base(x),size=size,mode='bilinear',align_corners=False)
        detail=nn.functional.interpolate(self.project(x),size=size,mode='bilinear',align_corners=False)
        return base+self.refine(detail)

class SpatialDinoSegmenter(DinoSegmenter):
    def __init__(self):
        super().__init__();self.head=SpatialHead()
