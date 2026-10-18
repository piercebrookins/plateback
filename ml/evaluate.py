"""Select confidence on validation only, then evaluate untouched publisher test masks.
Reports union-of-category IoU including misses and false positives; no empty/empty inflation.
"""
import json,time,argparse,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.vision import rings
DATA=ROOT/'data/datasets/plateback';OUT=ROOT/'artifacts/evaluation';LABELS=['rice','pasta','broccoli','chicken_duck_proxy']
def collect(model,split):
    items=[]
    for path in sorted((DATA/'images'/split).glob('*.jpg')):
        gt=np.array(Image.open(DATA/'masks'/split/(path.stem+'.png')))
        start=time.perf_counter();r=model.predict(str(path),imgsz=320,conf=.05,device='cpu',retina_masks=True,verbose=False)[0];elapsed=time.perf_counter()-start
        masks=[]
        if r.masks is not None:
            for m,b in zip(r.masks.data.cpu().numpy(),r.boxes):masks.append((int(b.cls.item()),float(b.conf.item()),m>.5))
        items.append((path,gt,masks,elapsed))
    return items
def metrics(items,conf,save=False):
    acc={l:dict(intersection=0,union=0,tp=0,fp=0,fn=0,reference_plates=0,predicted_plates=0) for l in LABELS};failures=[];negative=0;negfp=0;per_image=[]
    for path,gt,masks,elapsed in items:
        ps=[np.zeros_like(gt,dtype=bool) for _ in LABELS];pred=[]
        for cls,score,mask in masks:
            if score<conf:continue
            ps[cls]|=mask
            for p,h in rings(mask):pred.append({'label':LABELS[cls],'score':score,'polygon':p,'holes':h})
        ious={}
        for cls,l in enumerate(LABELS):
            g=gt==cls+1;p=ps[cls];a=acc[l];intersection=int((g&p).sum());union=int((g|p).sum());gp=bool(g.any());pp=bool(p.any())
            a['intersection']+=intersection;a['union']+=union;a['tp']+=int(gp and pp);a['fp']+=int(not gp and pp);a['fn']+=int(gp and not pp);a['reference_plates']+=int(gp);a['predicted_plates']+=int(pp);ious[l]=intersection/union if union else None
        if not (gt>0).any():negative+=1;negfp+=int(any(p.any() for p in ps))
        per_image.append({'id':path.stem,'iou':ious,'latency_seconds':elapsed,'predictions':pred})
        if save:
            np.savez_compressed(OUT/'predictions'/f'{path.stem}.npz',**{l:m for l,m in zip(LABELS,ps)})
    for a in acc.values():
        a['iou']=a['intersection']/a['union'] if a['union'] else None
        a['precision']=a['tp']/(a['tp']+a['fp']) if a['tp']+a['fp'] else None
        a['recall']=a['tp']/(a['tp']+a['fn']) if a['tp']+a['fn'] else None
    return {'images':len(items),'confidence':conf,'per_class':acc,'macro_iou':float(np.mean([a['iou'] for a in acc.values() if a['iou'] is not None])),'latency_median_seconds':float(np.median([x[3] for x in items])),'latency_p95_seconds':float(np.percentile([x[3] for x in items],95)),'failed_requests':0,'negative_food_images':negative,'negative_food_false_positive_images':negfp,'empty_plate_evaluation':'Not measured: negative-food images contain other foods, not verified empty plates.','per_image':per_image}
def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
    model=YOLO(str(ROOT/'artifacts/models/plateback/best.pt'));val=collect(model,'val')
    choices=[metrics(val,c) for c in [.1,.2,.3,.4,.5]];best=max(choices,key=lambda x:x['macro_iou'])
    settings={'confidence':best['confidence'],'selection':'Maximum validation category-union macro IoU','validation_candidates':[{k:r[k] for k in ['confidence','macro_iou']} for r in choices]}
    (ROOT/'artifacts/models/plateback/settings.json').write_text(json.dumps(settings,indent=2))
    (OUT/'validation.json').write_text(json.dumps(best,indent=2));test=metrics(collect(model,'test'),best['confidence'],True)
    (OUT/'test.json').write_text(json.dumps(test,indent=2));print(json.dumps({k:v for k,v in test.items() if k!='per_image'},indent=2))
if __name__=='__main__':main()
