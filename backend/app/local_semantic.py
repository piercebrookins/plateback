"""Semantic probabilities and connected-component filtering shared by evaluation and serving."""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from .mask_filter import LABELS
_CACHE={}
def load(folder):
    import torch
    from .semantic_model import make_model
    key=str(Path(folder).resolve())
    if key not in _CACHE:
        state=torch.load(Path(folder)/'best.pt',map_location='cpu',weights_only=True)
        if state['labels']!=['background']+LABELS:raise ValueError('Unexpected semantic class map')
        if state.get('architecture')=='dinov2-spatial-head':
            from .dino_model import SpatialDinoSegmenter
            model=SpatialDinoSegmenter().eval()
        elif state.get('architecture')=='dinov2-head':
            from .dino_model import DinoSegmenter
            model=DinoSegmenter().eval()
        else:model=make_model(False).eval()
        model.load_state_dict(state['state_dict']);_CACHE[key]=(model,state['size'])
    return _CACHE[key]
def probabilities(path,folder):
    import torch
    from .semantic_model import normalize
    model,size=load(folder);im=Image.open(path).convert('RGB').resize((size,size),Image.Resampling.BILINEAR)
    x=torch.from_numpy(np.array(im).transpose(2,0,1).copy()).float().unsqueeze(0)/255
    with torch.inference_mode():return model(normalize(x))['out'].softmax(1)[0].numpy()
def class_mask(prob,shape,rule):
    p=cv2.resize(prob.astype(np.float32),(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
    binary=(p>=rule['confidence']).astype(np.uint8);n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    accepted=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=max(20,rule['min_area_fraction']*binary.size));accepted=accepted[accepted!=0]
    mask=np.isin(labels,accepted)
    if 'grow_confidence' in rule:
        low=rule['grow_confidence']
        if not 0<=low<=rule['confidence']:raise ValueError('Growth confidence must not exceed seed confidence')
        _,components=cv2.connectedComponents((p>=low).astype(np.uint8),8)
        selected=np.unique(components[mask]);selected=selected[selected!=0]
        mask=np.isin(components,selected)
    return mask,p
_BASE={}
def predict(path,folder,settings):
    folder=Path(folder);im=Image.open(path);shape=(im.height,im.width);probs=probabilities(path,folder);out=[]
    for c,l in enumerate(LABELS):
        rule=settings['per_class'][l]
        if rule['source']=='semantic':
            mask,p=class_mask(probs[c+1],shape,rule)
            if mask.any():out.append((c,float(p[mask].mean()),mask))
    if any(r['source']=='baseline' for r in settings['per_class'].values()):
        from ultralytics import YOLO
        key=str(folder.resolve())
        if key not in _BASE:_BASE[key]=YOLO(str(folder/'baseline.pt'))
        result=_BASE[key].predict(str(path),imgsz=320,conf=.2,device='cpu',retina_masks=True,verbose=False)[0]
        if result.masks is not None:
            for mask,box in zip(result.masks.data.cpu().numpy(),result.boxes):
                c=int(box.cls.item())
                if settings['per_class'][LABELS[c]]['source']=='baseline':out.append((c,float(box.conf.item()),mask>.5))
    return out
