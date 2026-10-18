"""Freeze a validation-only policy with hard presence AND pixel recall floors before test."""
import json,sys,time,hashlib,argparse
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.local_semantic import probabilities,class_mask,predict
from backend.app.mask_filter import LABELS
from evaluate_v2 import collect
OUT=ROOT/'artifacts/evaluation-v4';OUT.mkdir(parents=True,exist_ok=True)
def metrics(rows):
    a={l:dict(intersection=0,union=0,truth_pixels=0,predicted_pixels=0,tp=0,fp=0,fn=0) for l in LABELS}
    for r in rows:
        for l,v in r['classes'].items():
            for k in a[l]:a[l][k]+=v[k]
    for v in a.values():
        v.update(iou=v['intersection']/max(1,v['union']),precision=v['tp']/max(1,v['tp']+v['fp']),recall=v['tp']/max(1,v['tp']+v['fn']),pixel_recall=v['intersection']/max(1,v['truth_pixels']),pixel_precision=v['intersection']/max(1,v['predicted_pixels']))
    return {'images':len(rows),'per_class':a,'macro_iou':float(np.mean([v['iou'] for v in a.values()])),'negative_images':sum(r.get('negative',False) for r in rows),'negative_false_positives':sum(r.get('negative',False) and any(v['fp'] for v in r['classes'].values()) for r in rows),'per_image':rows}
def one(g,p):
    gp=bool(g.any());pp=bool(p.any());return dict(intersection=int((g&p).sum()),union=int((g|p).sum()),truth_pixels=int(g.sum()),predicted_pixels=int(p.sum()),tp=int(gp and pp),fp=int(not gp and pp),fn=int(gp and not pp))
def baseline_rows(items):
    rows=[]
    for id,gt,pred,elapsed in items:
        masks=[np.zeros_like(gt,bool) for _ in LABELS]
        for c,s,m in pred:
            if s>=.2:masks[c]|=m
        rows.append({'id':id,'negative':not bool(gt.any()),'classes':{l:one(gt==c+1,masks[c]) for c,l in enumerate(LABELS)}})
    return rows
def eligible(candidate,reference):
    return candidate['recall']+1e-12>=reference['recall'] and candidate['pixel_recall']+1e-12>=reference['pixel_recall'] and candidate['fp']<=reference['fp']
def component_choices(prob,g,threshold,areas):
    p=cv2.resize(prob.astype(np.float32),(g.shape[1],g.shape[0]),interpolation=cv2.INTER_LINEAR)
    n,labels,stats,_=cv2.connectedComponentsWithStats((p>=threshold).astype(np.uint8),8)
    intersections=np.bincount(labels[g],minlength=n);truth=int(g.sum());out={}
    for area in areas:
        ids=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=max(20,area*g.size));ids=ids[ids!=0]
        predicted=int(stats[ids,cv2.CC_STAT_AREA].sum());inter=int(intersections[ids].sum());gp=truth>0;pp=predicted>0
        out[area]=dict(intersection=inter,union=truth+predicted-inter,truth_pixels=truth,predicted_pixels=predicted,tp=int(gp and pp),fp=int(not gp and pp),fn=int(gp and not pp))
    return out

def calibrate(folder):
    data=ROOT/'data/datasets/plateback-v4';base=baseline_rows(collect(ROOT/'artifacts/models/baseline-v1/best.pt',data,'val',320,'baseline-v4-val'));reference=metrics(base)
    (OUT/'baseline-validation.json').write_text(json.dumps(reference,indent=2))
    cache=OUT/'probabilities'/folder.name;cache.mkdir(parents=True,exist_ok=True);sha=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest();meta=cache/'identity.json'
    if meta.exists():assert json.loads(meta.read_text())['sha256']==sha,'Checkpoint changed; do not reuse probability cache'
    meta.write_text(json.dumps({'sha256':sha}));samples=[]
    for path in sorted((data/'images/val').glob('*.jpg')):
        saved=cache/(path.stem+'.npy')
        if not saved.exists():np.save(saved,probabilities(path,folder))
        samples.append((path.stem,np.array(Image.open(data/'masks/val'/(path.stem+'.png'))),np.load(saved)))
    policy={'version':'v4','confidence':.05,'imgsz':int(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['size']),'per_class':{},'selection':'Validation only, maximize pooled class IoU subject to no loss of either v1 or deployed v3 presence recall or pixel recall and no increase in false-positive images versus v1 (also versus v3 for chicken). Explicit baseline fallback if no candidate improves under ALL constraints.'};audit=[];chosen=[]
    for c,l in enumerate(LABELS):
        ref=dict(reference['per_class'][l]);previous=json.loads((ROOT/'artifacts/models/ensemble-v3/validation.json').read_text())['per_class'][l]
        if l=='chicken':ref['fp']=min(ref['fp'],previous['fp'])
        ref['recall']=max(ref['recall'],previous['recall']);ref['pixel_recall']=max(ref['pixel_recall'],previous['intersection']/max(1,ref['truth_pixels']))
        best={'source':'baseline','confidence':.2,'min_area_fraction':0};bestmetrics=reference['per_class'][l];bestrows=[{'id':r['id'],'classes':{l:r['classes'][l]}} for r in base];choices=[]
        areas=[0,.0005,.002,.005,.01,.015,.02,.03,.05] if l=='chicken' else [0,.0005,.002,.005,.01]
        for threshold in [.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.85,.9,.95,.98]:
            by_area={area:[] for area in areas}
            for id,gt,prob in samples:
                for area,counts in component_choices(prob[c+1],gt==c+1,threshold,areas).items():by_area[area].append({'id':id,'classes':{l:counts}})
            for area,rows in by_area.items():
                rule={'source':'semantic','confidence':threshold,'min_area_fraction':area}
                score=metrics(rows)['per_class'][l];ok=eligible(score,ref);choices.append({'rule':rule,'metrics':score,'eligible':ok})
                if ok and score['iou']>bestmetrics['iou']:best,bestmetrics,bestrows=rule,score,rows
        policy['per_class'][l]=best;chosen.append(bestrows);audit.append({'label':l,'reference':ref,'constraints_satisfied':eligible(bestmetrics,ref),'selected':best,'selected_metrics':bestmetrics,'choices':choices});print(l,best,bestmetrics,flush=True)
    joined=[{'id':r['id'],'negative':r['negative'],'classes':{l:chosen[c][i]['classes'][l] for c,l in enumerate(LABELS)}} for i,r in enumerate(base)]
    (folder/'settings.json').write_text(json.dumps(policy,indent=2));(OUT/'calibration.json').write_text(json.dumps(audit,indent=2));(OUT/'semantic-validation.json').write_text(json.dumps(metrics(joined),indent=2))
    return policy

def evaluate(folder,data,split,tag,policy):
    rows=[];times=[]
    for path in sorted((data/'images'/split).glob('*.jpg')):
        gt=np.array(Image.open(data/'masks'/split/(path.stem+'.png')));masks=[np.zeros_like(gt,bool) for _ in LABELS];t=time.perf_counter()
        for c,s,m in predict(path,folder,policy):masks[c]|=m
        times.append(time.perf_counter()-t);rows.append({'id':path.stem,'negative':not bool(gt.any()),'classes':{l:one(gt==c+1,masks[c]) for c,l in enumerate(LABELS)}})
    report=metrics(rows);report['latency_median_seconds']=float(np.median(times));report['settings']=policy
    (OUT/(tag+'.json')).write_text(json.dumps(report,indent=2));print(tag,{k:v for k,v in report.items() if k!='per_image'},flush=True)
    base=metrics(baseline_rows(collect(ROOT/'artifacts/models/baseline-v1/best.pt',data,split,320,'baseline-v4-'+tag)))
    (OUT/('baseline-'+tag+'.json')).write_text(json.dumps(base,indent=2))
    return report

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--folder',default='semantic-v4');ap.add_argument('--calibrate-only',action='store_true');ap.add_argument('--evaluate-only',action='store_true');args=ap.parse_args();torch.set_num_threads(4);folder=ROOT/'artifacts/models'/args.folder
    if args.folder!='semantic-v4':
        OUT=ROOT/('artifacts/evaluation-'+args.folder);OUT.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copy(ROOT/'artifacts/models/baseline-v1/best.pt',folder/'baseline.pt')
    policy=json.loads((folder/'settings.json').read_text()) if args.evaluate_only else calibrate(folder)
    if args.calibrate_only:return
    served=evaluate(folder,ROOT/'data/datasets/plateback-v4','val','serving-validation',policy)
    expected=json.loads((OUT/'semantic-validation.json').read_text())
    assert served['per_class']==expected['per_class'],'Serving predictions disagree with calibrated validation masks'
    for tag,name in [('fresh','plateback-v4'),('regression','plateback-v2'),('previous-final','plateback-final')]:evaluate(folder,ROOT/'data/datasets'/name,'test',tag,policy)
if __name__=='__main__':main()
