"""Freeze class thresholds on validation; then report fresh and regression tests.
No test predictions are loaded until the policy has been written to disk.
"""
import sys,json,time,argparse,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.mask_filter import keep_mask,LABELS
OUT=ROOT/'artifacts/evaluation-v2';OUT.mkdir(exist_ok=True,parents=True)
def collect(weights,data,split,size,tag):
    model=YOLO(str(weights));items=[];cache=OUT/'cache'/tag;cache.mkdir(parents=True,exist_ok=True)
    identity={'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'imgsz':size,'data':str(data),'split':split};meta=cache/'identity.json'
    if meta.exists() and json.loads(meta.read_text())!=identity:raise ValueError('Cache identity changed; use a new cache directory before rerunning')
    meta.write_text(json.dumps(identity))
    for path in sorted((data/'images'/split).glob('*.jpg')):
        gt=np.array(Image.open(data/'masks'/split/(path.stem+'.png')));saved=cache/(path.stem+'.npz')
        if saved.exists():
            a=np.load(saved);pred=[(int(c),float(s),m) for c,s,m in zip(a['cls'],a['score'],a['masks'])];elapsed=float(a['elapsed'])
        else:
            start=time.perf_counter();r=model.predict(str(path),imgsz=size,conf=.05,device='cpu',retina_masks=True,verbose=False)[0];elapsed=time.perf_counter()-start
            pred=[] if r.masks is None else [(int(b.cls.item()),float(b.conf.item()),m>.5) for m,b in zip(r.masks.data.cpu().numpy(),r.boxes)]
            np.savez_compressed(saved,cls=[x[0] for x in pred],score=[x[1] for x in pred],masks=np.array([x[2] for x in pred],dtype=bool),elapsed=elapsed)
        items.append((path.stem,gt,pred,elapsed))
    return items
def measure(items,settings,only=None):
    acc={l:dict(intersection=0,union=0,tp=0,fp=0,fn=0) for l in LABELS};neg=fpneg=0;rows=[]
    for name,gt,pred,elapsed in items:
        masks=[np.zeros_like(gt,dtype=bool) for _ in LABELS]
        for c,s,m in pred:
            if only is not None and LABELS[c]!=only:continue
            if keep_mask(c,s,m,settings):masks[c]|=m
        row={'id':name,'classes':{}}
        for c,l in enumerate(LABELS):
            if only is not None and l!=only:continue
            g=gt==c+1;p=masks[c];a=acc[l];gp=bool(g.any());pp=bool(p.any());inter=int((g&p).sum());union=int((g|p).sum())
            a['intersection']+=inter;a['union']+=union;a['tp']+=int(gp and pp);a['fp']+=int(not gp and pp);a['fn']+=int(gp and not pp)
            row['classes'][l]={'present':gp,'predicted':pp,'iou':inter/union if union else None,'intersection':inter,'union':union}
        if not (gt>0).any():neg+=1;fpneg+=int(any(m.any() for m in masks))
        rows.append(row)
    for a in acc.values():
        a['iou']=a['intersection']/max(1,a['union']);a['precision']=a['tp']/max(1,a['tp']+a['fp']);a['recall']=a['tp']/max(1,a['tp']+a['fn']);a['f0_5']=1.25*a['tp']/max(1,1.25*a['tp']+a['fp']+.25*a['fn'])
    return {'images':len(items),'settings':settings,'per_class':acc,'macro_iou':float(np.mean([a['iou'] for a in acc.values()])),'negative_food_images':neg,'negative_food_false_positive_images':fpneg,'latency_median_seconds':float(np.median([i[3] for i in items])),'latency_p95_seconds':float(np.percentile([i[3] for i in items],95)),'per_image':rows}
def write(name,x):
    (OUT/(name+'.json')).write_text(json.dumps(x,indent=2));print(name,json.dumps({k:v for k,v in x.items() if k!='per_image'}),flush=True)
def calibrate(val,size):
    settings={'imgsz':size,'confidence':.05,'per_class':{},'selection':'Validation only: maximize .55 category pooled IoU + .45 presence F0.5; prefer recall >= .50 when feasible in the search, otherwise maximize the objective. Class-specific thresholds; frozen before test.'};scores=[]
    # Each class is independent; metrics for other classes do not affect selection.
    for l in LABELS:
        choices=[]
        for conf in [.15,.2,.3,.4,.5,.6,.7,.8]:
            for area in [0,.002,.005,.01]:
                policy={'per_class':{l:{'confidence':conf,'min_area_fraction':area}}}
                a=measure(val,policy,only=l)['per_class'][l];choices.append({'confidence':conf,'min_area_fraction':area,'metrics':a,'objective':.55*a['iou']+.45*a['f0_5']})
        eligible=[x for x in choices if x['metrics']['recall']>=.5] or choices
        best=max(eligible,key=lambda x:x['objective']);settings['per_class'][l]={k:best[k] for k in ['confidence','min_area_fraction']};scores.append({'label':l,'choices':choices,'selected':best})
    return settings,scores
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline-only',action='store_true');args=ap.parse_args();torch.set_num_threads(4)
    base=ROOT/'artifacts/models/baseline-v1/best.pt';candidate=ROOT/'artifacts/models/candidate-v2/best.pt';data=ROOT/'data/datasets/plateback-v2'
    old={'confidence':.2,'imgsz':320};baseline_val=collect(base,data,'val',320,'baseline-val');write('baseline-validation',measure(baseline_val,old))
    if args.baseline_only:return
    val=collect(candidate,data,'val',416,'candidate-val');settings,scores=calibrate(val,416)
    base_settings,base_scores=calibrate(baseline_val,320)
    (OUT/'baseline-calibrated-settings.json').write_text(json.dumps(base_settings,indent=2));write('baseline-calibrated-validation',measure(baseline_val,base_settings))
    # Durable freeze before test access.
    (candidate.parent/'settings.json').write_text(json.dumps(settings,indent=2));write('calibration',{'candidates':scores});write('candidate-validation',measure(val,settings))
    objectives={tag:float(np.mean([.55*a['iou']+.45*a['f0_5'] for a in measure(items,policy)['per_class'].values()])) for tag,items,policy in [('candidate',val,settings),('baseline-calibrated',baseline_val,base_settings)]}
    write('selection-decision',{'criterion':'Mean class validation objective; computed before test access','objectives':objectives,'selected':max(objectives,key=objectives.get)})
    for split,data_path in [('fresh',data),('regression',ROOT/'data/datasets/plateback')]:
        for tag,weights,size,policy in [('baseline',base,320,old),('candidate',candidate,416,settings)]:
            items=collect(weights,data_path,'test',size,tag+'-'+split)
            write(tag+'-'+split,measure(items,policy))
            if tag=='baseline':write('baseline-calibrated-'+split,measure(items,base_settings))
    legacy=json.loads((ROOT/'artifacts/evaluation/test.json').read_text())
    regression=json.loads((OUT/'baseline-regression.json').read_text())
    assert abs(legacy['macro_iou']-regression['macro_iou'])<1e-8, 'Baseline regression does not reproduce original metric'
    write('protocol',{'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'test_use':'188 new publisher test images evaluated after validation-only selection; old 88 images are regression only.','limitations':['FoodSeg103 chicken/duck proxy, not chicken-specific ground truth.','Other-food negatives are not verified empty plates.','Presence precision is image/category level, not object detection precision.','Pixel IoU uses pooled category masks; no empty/empty inflation.','No weights are inferred.','Latency includes model prediction only, CPU, single local run.']})
if __name__=='__main__':main()
