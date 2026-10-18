"""Score fixed shared inputs; never select thresholds from these test predictions."""
import json,time,sys,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.mask_filter import keep_mask,LABELS
from backend.app import budget
from benchmark_geometry import rasterize
OUT=ROOT/'artifacts/benchmark';manifest=json.loads((OUT/'manifest.json').read_text());torch.set_num_threads(4)
def presence_metrics(rows):
    result={}
    for l in LABELS:
        tp=sum(l in r['truth'] and r['prediction'].get(l,False) for r in rows);fp=sum(l not in r['truth'] and r['prediction'].get(l,False) for r in rows);fn=sum(l in r['truth'] and not r['prediction'].get(l,False) for r in rows)
        result[l]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),'positive_images':tp+fn}
    return {'per_class':result,'macro_f1':float(np.mean([a['f1'] for a in result.values()])),'negative_images':sum(not r['truth'] for r in rows),'negative_false_positives':sum(not r['truth'] and any(r['prediction'].values()) for r in rows)}
def mask_metrics(rows):
    acc={l:{'intersection':0,'union':0} for l in LABELS}
    for row in rows:
        for l,v in row['pixels'].items():
            for key in ['intersection','union']:acc[l][key]+=v[key]
    for a in acc.values():a['iou']=a['intersection']/max(1,a['union'])
    return {'per_class':acc,'macro_iou':float(np.mean([a['iou'] for a in acc.values()]))}
def pix(gt,ps):return {l:{'intersection':int(((gt==c+1)&ps[c]).sum()),'union':int(((gt==c+1)|ps[c]).sum())} for c,l in enumerate(LABELS)}
def local(tag,weights,settings):
    out=OUT/'local'/tag;out.mkdir(exist_ok=True,parents=True);model=YOLO(str(weights)) if settings.get('version') not in ['v3','v4','v5'] else None;pres=[];seg=[]
    for task in ['presence','segmentation']:
        for row in manifest[task]:
            path=out/(task+'-'+row['id']+'.json')
            if path.exists():
                report=json.loads(path.read_text());assert report['weights_sha256']==hashlib.sha256(weights.read_bytes()).hexdigest() and report['settings']==settings, 'Local benchmark cache identity changed'
            else:
                start=time.perf_counter();im=Image.open(ROOT/row['path']);ps=[np.zeros((im.height,im.width),bool) for _ in LABELS]
                if settings.get('version') in ['v4','v5']:
                    from backend.app.local_semantic import predict as semantic_predict
                    for c,score,mask in semantic_predict(ROOT/row['path'],weights.parent,settings):ps[c]|=mask
                elif settings.get('version')=='v3':
                    from backend.app.local_ensemble import predict
                    predictions,_=predict(ROOT/row['path'],weights.parent,settings)
                    for c,score,mask in predictions:ps[c]|=mask
                else:
                    r=model.predict(str(ROOT/row['path']),imgsz=settings.get('imgsz',320),conf=.05,device='cpu',retina_masks=True,verbose=False)[0]
                    if r.masks is not None:
                        for m,b in zip(r.masks.data.cpu().numpy(),r.boxes):
                            c=int(b.cls.item());score=float(b.conf.item())
                            if keep_mask(c,score,m>.5,settings):ps[c]|=m>.5
                elapsed=time.perf_counter()-start
                report={'id':row['id'],'status':'completed','latency_seconds':elapsed,'cost_usd':0,'prediction':{l:bool(ps[c].any()) for c,l in enumerate(LABELS)},'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'settings':settings}
                if task=='segmentation':report['pixels']=pix(np.array(Image.open(ROOT/row['truth_path'])),ps)
                path.write_text(json.dumps(report,indent=2))
            if task=='presence':report['truth']=row['truth'];pres.append(report)
            else:seg.append(report)
    return pres,seg
def classifier():
    from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
    from torch import nn
    state=torch.load(ROOT/'artifacts/models/presence-v1/best.pt',map_location='cpu',weights_only=True);model=mobilenet_v3_small(weights=None).eval();model.load_state_dict(state['backbone']);head=nn.Sequential(nn.Dropout(.2),nn.Linear(576,4)).eval();head.load_state_dict(state['head']);transform=MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms();rows=[]
    with torch.no_grad():
        for r in manifest['presence']:
            start=time.perf_counter();x=transform(Image.open(ROOT/r['path']).convert('RGB')).unsqueeze(0);p=head(model.avgpool(model.features(x)).flatten(1)).sigmoid().numpy()[0];rows.append({'id':r['id'],'truth':r['truth'],'prediction':{l:bool(p[c]>=state['thresholds'][c]) for c,l in enumerate(LABELS)},'probabilities':p.tolist(),'status':'completed','latency_seconds':time.perf_counter()-start,'cost_usd':0})
    return rows,[]
def cloud(model,normalized=False):
    directory=OUT/'responses'/model.replace('/','__');pres=[];seg=[]
    for task in ['presence','segmentation']:
        for row in manifest[task]:
            path=directory/(task+'-'+row['id']+'.json');r=json.loads(path.read_text()) if path.exists() else {'id':row['id'],'status':'missing','prediction':{}}
            if normalized and task=='segmentation' and r.get('response'):
                try:
                    content=r['response']['choices'][0]['message']['content'].strip()
                    if content.startswith('```'):content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
                    value=json.loads(content);assert isinstance(value['regions'],list)
                    for region in value['regions']:
                        assert region['label'] in LABELS
                        for ring in [region['polygon']]+region.get('holes',[]):
                            assert 3<=len(ring)<=4096 and all(len(pt)==2 and all(isinstance(v,(int,float)) and 0<=v<=1000 for v in pt) for pt in ring)
                    r['prediction']=value;r['status']='completed'
                except (ValueError,KeyError,AssertionError,TypeError,IndexError):pass
            report={k:r.get(k) for k in ['id','status','latency_seconds','cost_usd']}
            if task=='presence':report.update(truth=row['truth'],prediction=r.get('prediction',{}) if r['status']=='completed' else {});pres.append(report)
            else:
                gt=np.array(Image.open(ROOT/row['truth_path']));masks=[Image.new('1',(gt.shape[1],gt.shape[0])) for _ in LABELS]
                if r['status']=='completed':
                    masks=rasterize(r['prediction']['regions'],gt.shape)
                report['pixels']=pix(gt,[np.array(m,bool) for m in masks]);seg.append(report)
    return pres,seg
def main():
    results={};local_defs=[('Local YOLO v1','baseline-v1',{'imgsz':320,'confidence':.2}),('Local YOLO v2','candidate-v2',json.loads((ROOT/'artifacts/models/candidate-v2/settings.json').read_text()))]
    for name,folder,settings in local_defs:results[name]=local(folder,ROOT/'artifacts/models'/folder/'best.pt',settings)
    results['Local ensemble v3']=local('ensemble-v3',ROOT/'artifacts/models/ensemble-v3/best.pt',json.loads((ROOT/'artifacts/models/ensemble-v3/settings.json').read_text()))
    semantic=ROOT/'artifacts/models/semantic-v4'
    if (semantic/'settings.json').exists():results['Local semantic v4']=local('semantic-v4',semantic/'best.pt',json.loads((semantic/'settings.json').read_text()))
    refined=ROOT/'artifacts/models/semantic-v5'
    if (refined/'settings.json').exists():results['Local semantic v5']=local('semantic-v5',refined/'best.pt',json.loads((refined/'settings.json').read_text()))
    results['Local presence specialist']=classifier()
    for model in manifest['models']:results[model]=cloud(model)
    summaries=[];detail={}
    for name,(p,s) in results.items():
        pm=presence_metrics(p);sm=mask_metrics(s) if s else None;failp=sum(r['status']!='completed' for r in p);fails=sum(r['status']!='completed' for r in s);lat=[r['latency_seconds'] for r in p if r.get('latency_seconds') is not None];cost=sum(r.get('cost_usd') or 0 for r in p+s)
        rng=np.random.default_rng(93);boot=[presence_metrics([p[i] for i in rng.integers(0,len(p),len(p))])['macro_f1'] for _ in range(1000)];ci=np.percentile(boot,[2.5,97.5]).tolist()
        normalized_seg=cloud(name,normalized=True)[1] if name in manifest['models'] else s
        normalized_sm=mask_metrics(normalized_seg) if normalized_seg else None
        detail[name]={'normalized_segmentation':normalized_sm,'normalized_segmentation_failures':sum(r['status']!='completed' for r in normalized_seg),'presence':pm,'segmentation':sm,'presence_failures':failp,'segmentation_failures':fails,'presence_macro_f1_95_percentile_interval':ci,'presence_rows':p,'segmentation_rows':s,'cost_usd':cost,'presence_latency_median_seconds':float(np.median(lat)) if lat else None,'presence_latency_p95_seconds':float(np.percentile(lat,95)) if lat else None}
        summaries.append({'model':name,'presence_f1':pm['macro_f1'],'rice':f'{pm["per_class"]["rice"]["precision"]:.0%} / {pm["per_class"]["rice"]["recall"]:.0%}','chicken':f'{pm["per_class"]["chicken"]["precision"]:.0%} / {pm["per_class"]["chicken"]["recall"]:.0%}','mask_iou':normalized_sm['macro_iou'] if normalized_sm else None,'strict_mask_iou':sm['macro_iou'] if sm else None,'latency_seconds':float(np.median(lat)) if lat else None,'cost_usd':cost,'coverage':f'{len(p)-failp}/{len(p)} presence; {sum(r["status"]=="completed" for r in normalized_seg)}/{len(s)} parsed masks ({len(s)-fails} strict)' if s else f'{len(p)-failp}/{len(p)} presence; no mask head','presence_ci':ci})
    (OUT/'detailed-results.json').write_text(json.dumps(detail,indent=2));summary={'status':'completed','note':'Earlier pilot (regression evidence for v4/v5): 38 Nutrition5k presence images and 12 FoodSeg103 segmentation images. Mask scores include transport-only parsing (JSON fences removed; longer contours accepted), with strict scores recorded separately. Invalid geometry/missing outputs still count as failures.','models':summaries,'accounted_total_usd':budget.summary()['accounted_usd']};(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(ROOT/'docs/benchmark-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
