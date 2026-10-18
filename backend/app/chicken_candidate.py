"""Experimental additive chicken rescue. Never removes a v5 pixel or changes other foods."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from .local_semantic import load,probabilities,class_mask
from .semantic_model import normalize
ROOT=Path(__file__).resolve().parents[2]
_CLASSIFIERS={}
def classifier(folder,method):
 key=(str(folder),method)
 if key not in _CLASSIFIERS:
  state=torch.load(Path(folder)/(method+'.pt'),map_location='cpu',weights_only=True)
  clf=nn.Linear(768,1) if method=='linear' else nn.Sequential(nn.Linear(768,128),nn.GELU(),nn.Dropout(.3),nn.Linear(128,1));clf.load_state_dict(state['state_dict']);clf.eval();_CLASSIFIERS[key]=(clf,state['mean'],state['std'])
 return _CLASSIFIERS[key]
def feature(path):
 model,_=load(ROOT/'artifacts/models/semantic-v5');im=Image.open(path).convert('RGB').resize((336,336));x=torch.from_numpy(np.array(im).transpose(2,0,1).copy()).float()[None]/255
 with torch.inference_mode():
  f=model.backbone.forward_features(normalize(x))['x_norm_patchtokens'];return torch.cat([f.mean(1),f.amax(1)],1)
def score(features,folder,method):
 clf,mean,std=classifier(folder,method)
 with torch.inference_mode():return clf((features-mean)/std).flatten().sigmoid().numpy()
def combine(base_probability,rescue_probability,shape,base_rule,rescue_rule,presence_score,threshold):
 base,p=class_mask(base_probability,shape,base_rule)
 if presence_score>=threshold:
  extra,_=class_mask(rescue_probability,shape,rescue_rule);base|=extra
 return base

def predict(path,folder,settings):
 """Candidate callable for experiments; active app selection is deliberately unchanged."""
 basefolder=ROOT/'artifacts/models/semantic-v5';basepolicy=json.loads((basefolder/'settings.json').read_text());probs=probabilities(path,basefolder);im=Image.open(path);shape=(im.height,im.width);rescue=settings['rescue'];source=ROOT/'artifacts/models'/rescue['source'];extra=probs if source==basefolder else probabilities(path,source);confidence=float(score(feature(path),folder,rescue['method'])[0]);out=[]
 for c,label in enumerate(['rice','pasta','broccoli','chicken']):
  mask,p=class_mask(probs[c+1],shape,basepolicy['per_class'][label])
  if label=='chicken':mask=combine(probs[4],extra[4],shape,basepolicy['per_class'][label],rescue['rule'],confidence,rescue['threshold'])
  if mask.any():out.append((c,float(p[mask].mean()),mask))
 return out
