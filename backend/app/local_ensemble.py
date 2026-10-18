"""Validation-selected local mask sources with an optional trained presence gate."""
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .mask_filter import LABELS,keep_mask
_CACHE={}
def predict(path,folder,settings):
    import torch
    from torch import nn
    from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
    from ultralytics import YOLO
    key=str(Path(folder).resolve())
    if key not in _CACHE:
        models={'candidate':YOLO(str(Path(folder)/'best.pt')),'baseline':YOLO(str(Path(folder)/'baseline.pt'))}
        for m in models.values():
            if [m.names[i] for i in range(len(m.names))]!=LABELS:raise ValueError('Unexpected local ensemble class map')
        state=torch.load(Path(folder)/'presence.pt',map_location='cpu',weights_only=True);backbone=mobilenet_v3_small(weights=None).eval();backbone.load_state_dict(state['backbone']);head=nn.Sequential(nn.Dropout(.2),nn.Linear(576,4)).eval();head.load_state_dict(state['head'])
        _CACHE[key]=(models,backbone,head,MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms())
    models,backbone,head,transform=_CACHE[key]
    with torch.no_grad():
        x=transform(Image.open(path).convert('RGB')).unsqueeze(0);probs=head(backbone.avgpool(backbone.features(x)).flatten(1)).sigmoid().numpy()[0]
    masks=[]
    for source in sorted({r['source'] for r in settings['per_class'].values()}):
        result=models[source].predict(str(path),imgsz=416 if source=='candidate' else 320,conf=.05,device='cpu',retina_masks=True,verbose=False)[0]
        if result.masks is None:continue
        for mask,box in zip(result.masks.data.cpu().numpy(),result.boxes):
            cls=int(box.cls.item());score=float(box.conf.item());rule=settings['per_class'][LABELS[cls]];binary=mask>.5
            if rule['source']==source and probs[cls]>=rule['gate'] and keep_mask(cls,score,binary,settings):masks.append((cls,score,binary))
    return masks,probs.tolist()
