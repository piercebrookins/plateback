"""Validation-only policy search. Freeze artifacts before separate test command."""
import sys,json,hashlib
from pathlib import Path
import torch,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.local_semantic import probabilities,class_mask
from backend.app.chicken_candidate import score,feature
from evaluate_semantic import one,metrics
OUT=ROOT/'artifacts/evaluation-v6';FOLDER=ROOT/'artifacts/models/chicken-v6';torch.set_num_threads(2)
def pooled(x):return torch.cat([x.mean((2,3)),x.amax((2,3))],1)
def main():
 food=[r for r in json.loads((ROOT/'ml/manifest-v4.json').read_text()) if r['split']=='val'];nutrition=[r for r in json.loads((ROOT/'ml/nutrition-v6-manifest.json').read_text()) if r['split']=='val'];features=None;nf=pooled(torch.from_numpy(np.load(FOLDER/'nutrition-val.npy').astype('float32')))
 fcache=OUT/'food-validation-presence-features.npy'
 if not fcache.exists():np.save(fcache,np.concatenate([feature(ROOT/'data/datasets/plateback-v4/images/val'/(r['id']+'.jpg')).numpy() for r in food]))
 features=torch.from_numpy(np.load(fcache));scores={m:np.concatenate([score(features,FOLDER,m),score(nf,FOLDER,m)]) for m in ['linear','mlp']};samples=[]
 base_rule=json.loads((ROOT/'artifacts/models/semantic-v5/settings.json').read_text())['per_class']['chicken']
 for r in food+nutrition:
  isfood='classes' in r;path=ROOT/'data/datasets/plateback-v4/images/val'/(r['id']+'.jpg') if isfood else ROOT/r['path'];im=Image.open(path);shape=(im.height,im.width);gt=np.array(Image.open(ROOT/'data/datasets/plateback-v4/masks/val'/(r['id']+'.png')))==4 if isfood else None;ps={}
  for source in ['semantic-v5','mil-v6']:
   saved=OUT/'calibration-probabilities'/source/(r['id']+'.npy');saved.parent.mkdir(parents=True,exist_ok=True)
   if not saved.exists():
    existing=ROOT/'artifacts/evaluation-dino448-v4/probabilities/dino448-v4'/(r['id']+'.npy')
    p=np.load(existing)[4] if source=='semantic-v5' and isfood and existing.exists() else probabilities(path,ROOT/'artifacts/models'/source)[4];np.save(saved,p)
   ps[source]=np.load(saved)
  base,_=class_mask(ps['semantic-v5'],shape,base_rule);samples.append((r,shape,gt,ps,base))
  if len(samples)%50==0:print('validation probabilities',len(samples),flush=True)
 choices=[]
 for source in ['semantic-v5','mil-v6']:
  for seed in [.25,.5,.75]:
   for area in [.0005,.002,.005]:
    rule={'confidence':seed,'min_area_fraction':area,'grow_confidence':.1};basecounts=[];rescuecounts=[];npres=[]
    for r,shape,gt,ps,base in samples:
     extra,_=class_mask(ps[source],shape,rule);combined=base|extra
     if gt is not None:basecounts.append(one(gt,base));rescuecounts.append(one(gt,combined))
     else:npres.append((bool(base.any()),bool(combined.any()),'chicken' in r['labels']))
    for method in ['linear','mlp']:
     for threshold in [.2,.35,.5,.65,.8,.9,.95]:
      selected=scores[method]>=threshold;rows=[{'classes':{'chicken':rescuecounts[i] if selected[i] else basecounts[i]}} for i in range(len(food))];f=metrics(rows)['per_class']['chicken'];pred=np.array([b if selected[len(food)+i] else a for i,(a,b,y) in enumerate(npres)]);truth=np.array([y for a,b,y in npres]);tp=int((pred&truth).sum());fp=int((pred&~truth).sum());recall=tp/max(1,truth.sum());ok=f['fp']<=32 and recall>=.8 and fp<=6
      choices.append({'source':source,'method':method,'threshold':threshold,'rule':rule,'food':f,'nutrition':{'tp':tp,'fp':fp,'positives':int(truth.sum()),'negatives':int((~truth).sum()),'recall':float(recall)},'eligible':bool(ok)})
 eligible=[r for r in choices if r['eligible']]
 # If none qualifies, preserve a research result; do not call it eligible or promote.
 selected=max(eligible,key=lambda r:r['food']['iou']) if eligible else max(choices,key=lambda r:(r['food']['fp']<=32 and r['nutrition']['fp']<=6,r['nutrition']['recall'],r['food']['iou']))
 policy={'version':'v6-research','rescue':{k:selected[k] for k in ['source','method','threshold','rule']},'validation_eligible':selected['eligible'],'selection':'Validation-only IoU among recall/FP-eligible policies; if none, research fallback prioritizes FP limits then Nutrition recall then IoU. Additive union preserves all v5 pixels; other foods unchanged.'};(FOLDER/'settings.json').write_text(json.dumps(policy,indent=2));(OUT/'calibration.json').write_text(json.dumps({'selected':selected,'eligible_policies':len(eligible),'choices':choices},indent=2));freeze={'settings_sha256':hashlib.sha256((FOLDER/'settings.json').read_bytes()).hexdigest(),'weights':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [FOLDER/'linear.pt',FOLDER/'mlp.pt',ROOT/'artifacts/models/mil-v6/best.pt',ROOT/'artifacts/models/semantic-v5/best.pt']},'test_predictions_not_read':True};(OUT/'freeze.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
