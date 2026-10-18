"""Fixed small direct-OpenAI check, scoped $3 run/$5 lifetime, no silent retry."""
import os,sys,json,concurrent.futures
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));load_dotenv(ROOT/'.env');os.environ['PLATEBACK_BUDGET_RUN']='hackathon-v7'
from backend.app import db,budget,vision
from evaluate_semantic import metrics,one,LABELS
OUT=ROOT/'artifacts'/os.getenv('OPENAI_BENCHMARK_DIR','benchmark-openai-v7');OUT.mkdir(parents=True,exist_ok=True);db.init();budget.start_run('hackathon-v7',3)
manifest=OUT/'manifest.json'
if not manifest.exists():
 a=json.loads((ROOT/'artifacts/benchmark-chicken-final/manifest.json').read_text())['segmentation'];b=json.loads((ROOT/'artifacts/benchmark-v4/manifest.json').read_text())['segmentation'][:8];manifest.write_text(json.dumps({'model':'gpt-6-luna','scope':'fixed regression subset, not fresh benchmark','segmentation':a+b},indent=2))
rows=json.loads(manifest.read_text())['segmentation']
def run(r):
 path=OUT/(r['id']+'.json')
 if path.exists():return
 path.write_text(json.dumps({'status':'pending','note':'Do not automatically retry an unknown outcome'}))
 try:
  regions,meta=vision.openai_segment(ROOT/r['path']);result={'status':'completed','regions':regions,'metadata':meta}
 except Exception as e:result={'status':'failed','error_type':type(e).__name__,'error':str(e)[:300]}
 path.write_text(json.dumps(result,indent=2));print(r['id'],result['status'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,rows))
scored=[];valid=0
for r in rows:
 result=json.loads((OUT/(r['id']+'.json')).read_text());gt=np.array(Image.open(ROOT/r['truth_path']));masks=[np.zeros(gt.shape,bool) for _ in LABELS]
 if result['status']=='completed':
  valid+=1
  for region in result['regions']:
   m=Image.new('1',(gt.shape[1],gt.shape[0]));d=ImageDraw.Draw(m);d.polygon([(x*gt.shape[1],y*gt.shape[0]) for x,y in region['polygon']],fill=1)
   for hole in region.get('holes',[]):d.polygon([(x*gt.shape[1],y*gt.shape[0]) for x,y in hole],fill=0)
   masks[LABELS.index(region['label'])]|=np.array(m,bool)
 scored.append({'id':r['id'],'negative':not bool(gt.any()),'classes':{l:one(gt==c+1,masks[c]) for c,l in enumerate(LABELS)}})
report={'model':'gpt-6-luna','valid':valid,'attempted':len(rows),'all':metrics(scored),'chicken_paired16':metrics(scored[:16]),'run_budget':budget.run_summary('hackathon-v7'),'accounted_total':budget.summary()['accounted_usd']};(OUT/'summary.json').write_text(json.dumps(report,indent=2));print('valid',valid,'run',report['run_budget'],flush=True)
