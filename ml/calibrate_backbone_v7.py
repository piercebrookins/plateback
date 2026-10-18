"""Validation-only cleaner-mask policy; frozen before regression evaluation."""
import json,sys,hashlib,os
from pathlib import Path
import torch,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.local_semantic import probabilities,class_mask
from backend.app.chicken_candidate import score,combine
from backend.app.chicken_refinement import refine
from evaluate_semantic import metrics,one
NAME=os.getenv('V7_MODEL','backbone-v7');OUT=ROOT/'artifacts'/('evaluation-v7' if NAME=='backbone-v7' else 'evaluation-'+NAME);OUT.mkdir(parents=True,exist_ok=True);FOLDER=ROOT/'artifacts/models'/NAME;torch.set_num_threads(2)
identity={'checkpoint_sha256':hashlib.sha256((FOLDER/'best.pt').read_bytes()).hexdigest(),'v6_policy_sha256':hashlib.sha256((ROOT/'artifacts/models/chicken-v6/settings.json').read_bytes()).hexdigest()}
stamp=OUT/'probability-cache-identity.json'
if stamp.exists():assert json.loads(stamp.read_text())==identity,'Weights changed; use a fresh calibration cache directory'
else:stamp.write_text(json.dumps(identity,indent=2))
food=[r for r in json.loads((ROOT/'ml/manifest-v4.json').read_text()) if r['split']=='val'];nutrition=[r for r in json.loads((ROOT/'ml/nutrition-v6-manifest.json').read_text()) if r['split']=='val'];v6=ROOT/'artifacts/models/chicken-v6';rescue=json.loads((v6/'settings.json').read_text())['rescue'];base_rule=json.loads((ROOT/'artifacts/models/semantic-v5/settings.json').read_text())['per_class']['chicken'];features=torch.from_numpy(np.load(ROOT/'artifacts/evaluation-v6/food-validation-presence-features.npy'));nf=torch.from_numpy(np.load(v6/'nutrition-val.npy').astype('float32'));nf=torch.cat([nf.mean((2,3)),nf.amax((2,3))],1);scores=np.concatenate([score(features,v6,rescue['method']),score(nf,v6,rescue['method'])]);samples=[]
for i,r in enumerate(food+nutrition):
 isfood='classes' in r;path=ROOT/'data/datasets/plateback-v4/images/val'/(r['id']+'.jpg') if isfood else ROOT/r['path'];im=Image.open(path);shape=(im.height,im.width);old=ROOT/'artifacts/evaluation-v6/calibration-probabilities';p=np.load(old/'semantic-v5'/(r['id']+'.npy'));q=np.load(old/rescue['source']/(r['id']+'.npy'));base=combine(p,q,shape,base_rule,rescue['rule'],scores[i],rescue['threshold']);saved=OUT/'validation-probabilities'/(r['id']+'.npy');saved.parent.mkdir(exist_ok=True)
 if not saved.exists():np.save(saved,probabilities(path,FOLDER)[4])
 gt=np.array(Image.open(ROOT/'data/datasets/plateback-v4/masks/val'/(r['id']+'.png')))==4 if isfood else None;samples.append((r,gt,base,np.load(saved)))
 if i%50==0:print('validation',i,flush=True)
reference=metrics([{'classes':{'chicken':one(gt,base)}} for r,gt,base,p in samples if gt is not None])['per_class']['chicken'];nref=sum(base.any() and 'chicken' in r['labels'] for r,gt,base,p in samples if gt is None);nfpref=sum(base.any() and 'chicken' not in r['labels'] for r,gt,base,p in samples if gt is None);choices=[]
for threshold in [.01,.025,.05,.1,.2,.3,.4,.5]:
 rule={'confidence':threshold,'min_area_fraction':0}
 for mode in ['replace','intersect','union']:
  for fallback in [False,True]:
   rows=[];ntp=nfp=0
   for r,gt,base,p in samples:
    m=refine(base,p,rule,mode,fallback)
    if gt is not None:rows.append({'classes':{'chicken':one(gt,m)}})
    else:
     ntp+=int(m.any() and 'chicken' in r['labels']);nfp+=int(m.any() and 'chicken' not in r['labels'])
   f=metrics(rows)['per_class']['chicken'];ok=f['recall']>=reference['recall'] and f['pixel_recall']>=reference['pixel_recall'] and f['fp']<=reference['fp'] and ntp>=nref and nfp<=nfpref
   choices.append({'rule':rule,'mode':mode,'fallback':fallback,'food':f,'nutrition_tp':ntp,'nutrition_fp':nfp,'eligible':bool(ok)})
valid=[r for r in choices if r['eligible']];best=max(valid,key=lambda r:r['food']['iou']) if valid else None
# No feasible improvement means unchanged v6; never use an ineligible policy.
selected=best if best and best['food']['iou']>reference['iou'] else {'mode':'identity','eligible':True,'food':reference,'nutrition_tp':int(nref)}
policy={'version':NAME+'-research','refinement':selected,'selection':'Validation-only IoU improvement with v6 presence/pixel recall and FP constraints; otherwise identity v6'};(FOLDER/'settings.json').write_text(json.dumps(policy,indent=2));(OUT/'calibration.json').write_text(json.dumps({'reference':reference,'nutrition_reference_tp':int(nref),'selected':selected,'choices':choices},indent=2));(OUT/'freeze.json').write_text(json.dumps({'weights_sha256':hashlib.sha256((FOLDER/'best.pt').read_bytes()).hexdigest(),'settings_sha256':hashlib.sha256((FOLDER/'settings.json').read_bytes()).hexdigest(),'regression_not_yet_evaluated':True},indent=2));print(json.dumps(selected,indent=2))
