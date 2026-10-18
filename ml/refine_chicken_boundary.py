"""Validation-only hysteresis masks: confident seeds plus connected lower-confidence boundaries."""
import sys,json,shutil,hashlib
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_semantic import metrics,one
from backend.app.local_semantic import predict
OUT=ROOT/'artifacts/evaluation-v5';MODEL=ROOT/'artifacts/models/semantic-v5'
SEEDS=[.6,.65,.7,.75,.8,.85,.9,.95,.98,.99];AREAS=[.0005,.002,.005,.01,.015,.02,.03,.04,.05];GROW=[.05,.1,.2,.3,.4,.5,.6,.7,.8,.9]
def counts(inter,truth,pred):
    gp=truth>0;pp=pred>0;return dict(intersection=inter,union=truth+pred-inter,truth_pixels=truth,predicted_pixels=pred,tp=int(gp and pp),fp=int(not gp and pp),fn=int(gp and not pp))
def choices(prob,g,seeds=SEEDS,areas=AREAS,grow=GROW):
    p=cv2.resize(prob.astype(np.float32),(g.shape[1],g.shape[0]),interpolation=cv2.INTER_LINEAR);truth=int(g.sum());core={};out={}
    for seed in seeds:
        n,labels,stats,_=cv2.connectedComponentsWithStats((p>=seed).astype(np.uint8),8)
        ids,first=np.unique(labels,return_index=True);assert np.array_equal(ids,np.arange(n))
        core[seed]=(stats[:,cv2.CC_STAT_AREA],first)
    for low in grow:
        n,labels,stats,_=cv2.connectedComponentsWithStats((p>=low).astype(np.uint8),8);inter=np.bincount(labels[g],minlength=n);flat=labels.ravel()
        for seed,(sizes,first) in core.items():
            if low>seed:continue
            for area in areas:
                selected=np.flatnonzero(sizes>=max(20,area*g.size));selected=selected[selected!=0];low_ids=np.unique(flat[first[selected]]);low_ids=low_ids[low_ids!=0]
                out[(seed,area,low)]=counts(int(inter[low_ids].sum()),truth,int(stats[low_ids,cv2.CC_STAT_AREA].sum()))
    return out

def main():
    assert (ROOT/'ml/chicken-holdout-manifest.json').exists(),'Reserve the fresh chicken holdout first'
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);MODEL.mkdir(parents=True,exist_ok=True);source=ROOT/'artifacts/models/semantic-v4'
    for name in ['best.pt','baseline.pt','metadata.json','initialization.json','history.json']:shutil.copy(source/name,MODEL/name)
    totals={};samples=sorted((ROOT/'data/datasets/plateback-v4/masks/val').glob('*.png'))
    for i,path in enumerate(samples):
        g=np.array(Image.open(path))==4;prob=np.load(ROOT/'artifacts/evaluation-dino448-v4/probabilities/dino448-v4'/(path.stem+'.npy'))[4]
        for key,a in choices(prob,g).items():
            if key not in totals:totals[key]={k:0 for k in a}
            for k,v in a.items():totals[key][k]+=v
        if i%40==0:print('validation boundary search',i,'/',len(samples),flush=True)
    reference={'recall':42/68,'pixel_recall':.85,'fp':32,'iou':json.loads((ROOT/'artifacts/models/ensemble-v3/validation.json').read_text())['per_class']['chicken']['iou']};results=[]
    for (seed,area,low),a in totals.items():
        m=metrics([{'id':'aggregate','classes':{'chicken':a}}])['per_class']['chicken'];ok=m['recall']>=reference['recall'] and m['pixel_recall']>=reference['pixel_recall'] and m['fp']<=reference['fp'] and m['iou']>=reference['iou'];results.append({'rule':{'source':'semantic','confidence':seed,'min_area_fraction':area,'grow_confidence':low},'metrics':m,'eligible':ok})
    eligible=[r for r in results if r['eligible']];report={'reference':reference,'choices':results,'eligible_candidates':len(eligible)}
    (OUT/'calibration.json').write_text(json.dumps(report,indent=2));assert eligible,'No boundary policy meets all constraints; do not relax silently'
    chosen=max(eligible,key=lambda r:(r['metrics']['iou'],r['metrics']['recall']));report['selected']=chosen;(OUT/'calibration.json').write_text(json.dumps(report,indent=2))
    policy=json.loads((source/'settings.json').read_text());policy.update(version='v5',selection='Other foods frozen from v4. Chicken: validation-only confident seed + connected boundary search; pixel recall >=85%, presence recall >=original v1, false-positive images <=v3, IoU >=v3. Maximize IoU among eligible policies. Frozen before new chicken-focused test.');policy['per_class']['chicken']=chosen['rule'];(MODEL/'settings.json').write_text(json.dumps(policy,indent=2));meta=json.loads((MODEL/'metadata.json').read_text());meta.update(version='v5',boundary_refinement={'validation_reference':reference,'selected':chosen,'no_new_weight_fitting':True});(MODEL/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(chosen,indent=2))
    rows=[]
    for path in sorted((ROOT/'data/datasets/plateback-v4/images/val').glob('*.jpg')):
        g=np.array(Image.open(ROOT/'data/datasets/plateback-v4/masks/val'/(path.stem+'.png')));ps=[np.zeros_like(g,bool) for _ in range(4)]
        for c,s,m in predict(path,MODEL,policy):ps[c]|=m
        rows.append({'id':path.stem,'negative':not bool(g.any()),'classes':{l:one(g==c+1,ps[c]) for c,l in enumerate(['rice','pasta','broccoli','chicken'])}})
    actual=metrics(rows);assert actual['per_class']['chicken']==chosen['metrics'],'Calibration and serving disagree';(OUT/'validation.json').write_text(json.dumps(actual,indent=2));print('Serving verification passed')
if __name__=='__main__':main()
