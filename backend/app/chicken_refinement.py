"""Experimental v7 geometry refinement, retaining v6 detection as the starting point."""
import numpy as np
from .local_semantic import class_mask

def refine(base,probability,rule,mode,fallback):
    if not base.any():return base.copy()
    if mode=='redistribute':
        import cv2
        p=cv2.resize(probability.astype(np.float32),(base.shape[1],base.shape[0]),interpolation=cv2.INTER_LINEAR)
        radius=max(1,round(min(base.shape)*rule['radius_fraction']))
        support=cv2.dilate(base.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))).astype(bool)
        scores=p+rule['base_prior']*base
        indices=np.flatnonzero(support);count=int(base.sum())
        # Exact area budget: improve overlap by relocating pixels, not deleting them.
        order=np.argsort(-scores.flat[indices],kind='stable')[:count]
        mask=np.zeros_like(base);mask.flat[indices[order]]=True
        return mask
    mask,_=class_mask(probability,base.shape,rule)
    if mode=='intersect':mask &= base
    elif mode=='union':mask |= base
    elif mode!='replace':raise ValueError('Unknown refinement mode')
    if fallback and not mask.any():return base.copy()
    return mask
