"""Local five-class semantic segmentation; torchvision BSD-3-Clause architecture."""
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large,DeepLabV3_MobileNet_V3_Large_Weights
from torch import nn
LABELS=['rice','pasta','broccoli','chicken']
def make_model(pretrained=False):
    model=deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None,weights_backbone=None,aux_loss=True)
    model.classifier[4]=nn.Conv2d(256,5,1);model.aux_classifier[4]=nn.Conv2d(model.aux_classifier[4].in_channels,5,1)
    return model
def normalize(x):
    return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
