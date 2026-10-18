"""Earlier v4 tests are explicitly regression evidence after the v5 boundary change."""
import sys,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.local_semantic import predict
from evaluate_semantic import metrics,one,LABELS
import score_semantic_benchmark as shared
OUT=ROOT/'artifacts/evaluation-v5';folder=ROOT/'artifacts/models/semantic-v5';policy=json.loads((folder/'settings.json').read_text());torch.set_num_threads(4);rows=[]
for p in sorted((ROOT/'data/datasets/plateback-v4/images/test').glob('*.jpg')):
 gt=np.array(Image.open(ROOT/'data/datasets/plateback-v4/masks/test'/(p.stem+'.png')));ps=[np.zeros_like(gt,bool) for _ in LABELS]
 for c,s,m in predict(p,folder,policy):ps[c]|=m
 rows.append({'id':p.stem,'negative':not bool(gt.any()),'classes':{l:one(gt==c+1,ps[c]) for c,l in enumerate(LABELS)}})
r=metrics(rows);r['scope']='Regression only: v4 test results motivated this boundary refinement';(OUT/'regression-110.json').write_text(json.dumps(r,indent=2));manifest=json.loads((ROOT/'artifacts/benchmark-v4/manifest.json').read_text());byid={x['id']:x for x in rows};local=metrics([byid[x['id']] for x in manifest['segmentation']]);cloud,cost,valid=shared.cloud_rows(manifest);gem=metrics(cloud);(OUT/'regression-paired-32.json').write_text(json.dumps({'local':local,'gemini':gem,'scope':'Regression only; not another fresh performance claim'},indent=2));print('110 macro IoU',r['macro_iou'],'chicken',r['per_class']['chicken']);print('32 regression local/Gemini',local['macro_iou'],gem['macro_iou'])
