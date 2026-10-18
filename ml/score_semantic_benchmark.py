"""Paired, post-freeze comparison on 32 new images; no API calls here."""
import sys,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_semantic import metrics,one,LABELS
from benchmark_geometry import rasterize
from backend.app.local_ensemble import predict as ensemble_predict
from backend.app import budget
OUT=ROOT/'artifacts/benchmark-v4';EV=ROOT/'artifacts/evaluation-v4'
def cloud_rows(manifest):
    rows=[];cost=0;valid=0
    for r in manifest['segmentation']:
        report=json.loads((OUT/'responses/google__gemini-4-argon'/('segmentation-'+r['id']+'.json')).read_text());cost+=report.get('cost_usd') or 0
        gt=np.array(Image.open(ROOT/r['truth_path']));masks=[Image.new('1',(gt.shape[1],gt.shape[0])) for _ in LABELS];status='invalid'
        try:
            content=report['response']['choices'][0]['message']['content'].strip()
            if content.startswith('```'):content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
            value=json.loads(content);assert isinstance(value['regions'],list)
            for region in value['regions']:
                assert region['label'] in LABELS
                for ring in [region['polygon']]+region.get('holes',[]):assert 3<=len(ring)<=4096 and all(len(p)==2 and all(isinstance(v,(int,float)) and 0<=v<=1000 for v in p) for p in ring)
            masks=rasterize(value['regions'],gt.shape)
            valid+=1;status='parsed'
        except (KeyError,TypeError,ValueError,IndexError,AssertionError):pass
        rows.append({'id':r['id'],'negative':not bool(gt.any()),'parse_status':status,'classes':{l:one(gt==c+1,np.array(masks[c],bool)) for c,l in enumerate(LABELS)}})
    return rows,cost,valid

def bootstrap(a,b):
    assert [r['id'] for r in a]==[r['id'] for r in b]
    rng=np.random.default_rng(141);diff=[];chicken=[]
    for _ in range(2000):
        ids=rng.integers(0,len(a),len(a));ma=metrics([a[i] for i in ids]);mb=metrics([b[i] for i in ids]);diff.append(ma['macro_iou']-mb['macro_iou']);chicken.append(ma['per_class']['chicken']['recall']-mb['per_class']['chicken']['recall'])
    return {'macro_iou_difference_95_percentile_interval':np.percentile(diff,[2.5,97.5]).tolist(),'chicken_presence_recall_difference_95_percentile_interval':np.percentile(chicken,[2.5,97.5]).tolist(),'resamples':2000,'seed':141}

def main():
    torch.set_num_threads(4);manifest=json.loads((OUT/'manifest.json').read_text());ids=[r['id'] for r in manifest['segmentation']];v3path=EV/'ensemble-fresh.json'
    if not v3path.exists():
        folder=ROOT/'artifacts/models/ensemble-v3';policy=json.loads((folder/'settings.json').read_text());rows=[]
        for p in sorted((ROOT/'data/datasets/plateback-v4/images/test').glob('*.jpg')):
            gt=np.array(Image.open(ROOT/'data/datasets/plateback-v4/masks/test'/(p.stem+'.png')));ps=[np.zeros_like(gt,bool) for _ in LABELS]
            for c,s,m in ensemble_predict(p,folder,policy)[0]:ps[c]|=m
            rows.append({'id':p.stem,'negative':not bool(gt.any()),'classes':{l:one(gt==c+1,ps[c]) for c,l in enumerate(LABELS)}})
        v3path.write_text(json.dumps(metrics(rows),indent=2))
    names={'Original YOLO v1':'baseline-fresh','Previous ensemble v3':'ensemble-fresh','Semantic v4':'fresh'};full={};paired={}
    for name,file in names.items():
        full[name]=json.loads((EV/(file+'.json')).read_text());byid={r['id']:r for r in full[name]['per_image']};paired[name]=metrics([byid[id] for id in ids])
    gem,cost,valid=cloud_rows(manifest);paired['Gemini 4 Argon']=metrics(gem)
    valid_indices=[i for i,r in enumerate(gem) if r['parse_status']=='parsed'];valid_local=[paired['Semantic v4']['per_image'][i] for i in valid_indices];valid_gem=[gem[i] for i in valid_indices]
    conditional={'images':len(valid_indices),'local':metrics(valid_local),'gemini':metrics(valid_gem),'difference':bootstrap(valid_local,valid_gem) if valid_indices else None}
    interval=bootstrap(paired['Semantic v4']['per_image'],gem);baseline_interval=bootstrap(full['Semantic v4']['per_image'],full['Original YOLO v1']['per_image'])
    def compact(v):return {k:x for k,x in v.items() if k not in ['per_image','settings']}
    detailed={'valid_geometry_diagnostic':conditional,'full_110':full,'paired_32':paired,'comparison':interval,'baseline_comparison':baseline_interval};(OUT/'detailed-results.json').write_text(json.dumps(detailed,indent=2))
    calibration=json.loads((EV/'calibration.json').read_text())
    result={'valid_geometry_diagnostic':{'images':conditional['images'],'local':compact(conditional['local']),'gemini':compact(conditional['gemini']),'difference':conditional['difference']},'validation_constraints_satisfied':all(r['constraints_satisfied'] for r in calibration),'status':'completed','title':'Local segmentation with recall constraints','training_images':1732,'full_evaluation_images':110,'paired_images':32,'gemini_parsed_outputs':valid,'gemini_cost_usd':cost,'accounted_total_usd':budget.summary()['accounted_usd'],'full_evaluation':{n:compact(v) for n,v in full.items()},'paired_evaluation':{n:compact(v) for n,v in paired.items()},'paired_difference':interval,'baseline_difference':baseline_interval,'note':'Fresh held-out images from the publisher training pool, screened against all earlier splits. Chicken includes duck. Food presence recall and pixel recall are reported separately. Validation constraints do not guarantee recall on new images. No tuning on these results. Public-source contamination of frontier pretraining is unknown.'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));(ROOT/'docs/evaluation-v4-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
