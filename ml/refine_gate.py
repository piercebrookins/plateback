"""Validation-only mask-source/threshold/gate selection. No test results imported."""
import sys,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ml.evaluate_v2 import collect,measure
from backend.app.mask_filter import LABELS
OUT=ROOT/'artifacts/models/ensemble-v3';OUT.mkdir(parents=True,exist_ok=True)
def main():
 torch.set_num_threads(4);data=ROOT/'data/datasets/plateback-v2';sources={name:collect(ROOT/f'artifacts/models/{folder}/best.pt',data,'val',size,name+'-val') for name,folder,size in [('baseline','baseline-v1',320),('candidate','candidate-v2',416)]}
 state=torch.load(ROOT/'artifacts/models/presence-v1/best.pt',map_location='cpu',weights_only=True);model=mobilenet_v3_small(weights=None).eval();model.load_state_dict(state['backbone']);head=nn.Sequential(nn.Dropout(.2),nn.Linear(576,4)).eval();head.load_state_dict(state['head']);transform=MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms();probs=[]
 paths=sorted((data/'images/val').glob('*.jpg'))
 with torch.no_grad():
  for off in range(0,len(paths),16):
   x=torch.stack([transform(Image.open(p).convert('RGB')) for p in paths[off:off+16]]);probs.extend(head(model.avgpool(model.features(x)).flatten(1)).sigmoid().numpy().tolist())
 probs=np.array(probs);settings={'version':'v3','selection':'Validation only; maximize .55 pooled IoU + .45 presence F0.5. Require recall >= .50 when feasible. Choose class-specific source and optional presence gate.','per_class':{}};grids={}
 for c,l in enumerate(LABELS):
  choices=[]
  for source,items in sources.items():
   for gate in [0,.2,.35,.5]:
    gated=[(id,gt,[p for p in pred if p[0]==c and probs[i,c]>=gate],elapsed) for i,(id,gt,pred,elapsed) in enumerate(items)]
    for conf in [.05,.1,.15,.2,.3,.4,.5,.6]:
     for area in [0,.005,.01]:
      a=measure(gated,{'per_class':{l:{'confidence':conf,'min_area_fraction':area}}},only=l)['per_class'][l];choices.append({'source':source,'gate':gate,'confidence':conf,'min_area_fraction':area,'metrics':a,'objective':.55*a['iou']+.45*a['f0_5']})
  feasible=[x for x in choices if x['metrics']['recall']>=.5];best=max(feasible or choices,key=lambda x:x['objective']);settings['per_class'][l]={k:best[k] for k in ['source','gate','confidence','min_area_fraction']};grids[l]={'selected':best,'recall_floor_feasible':bool(feasible),'candidates':choices};print(l,best,flush=True)
 merged=[]
 for i,(id,gt,_,elapsed) in enumerate(sources['baseline']):
  masks=[]
  for c,l in enumerate(LABELS):
   rule=settings['per_class'][l]
   for cls,score,m in sources[rule['source']][i][2]:
    if cls==c and probs[i,c]>=rule['gate'] and score>=rule['confidence'] and float(m.mean())>=rule['min_area_fraction']:masks.append((cls,score,m))
  merged.append((id,gt,masks,elapsed))
 result=measure(merged,{'confidence':0});(OUT/'settings.json').write_text(json.dumps(settings,indent=2));(OUT/'calibration.json').write_text(json.dumps(grids,indent=2));(OUT/'validation.json').write_text(json.dumps(result,indent=2));print('validation macro IoU',result['macro_iou'],flush=True)
if __name__=='__main__':main()
