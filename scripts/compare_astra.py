"""Bounded scanner-prompt comparison. Reuses paid Luna responses; no benchmark promotion."""
import os,sys,json,time,sqlite3
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from backend.app import main,vision,budget
OUT=ROOT/'artifacts/astra-comparison';OUT.mkdir(exist_ok=True)
budget.start_run('astra-scanner-comparison',1.2);os.environ['PLATEBACK_BUDGET_RUN']='astra-scanner-comparison'
manifest_path=OUT/'manifest.json'
if not manifest_path.exists():
 old=json.loads((ROOT/'artifacts/benchmark-openai-v7-normalized/manifest.json').read_text())['segmentation']; chosen=[]
 for positive in [True,False]:
  rows=[r for r in old if bool((np.array(Image.open(ROOT/r['truth_path']))==4).any())==positive]
  chosen+=rows[:3]
 c=sqlite3.connect(ROOT/'data/plateback.db');screens=[]
 for ident in ['48c01e86-1203-491c-88ba-a442ddea347a','ec151fba-5ce6-4f95-8475-9e62e33fc353']:
  s=json.loads(c.execute('SELECT body FROM scans WHERE id=?',(ident,)).fetchone()[0]);screens.append({'id':ident,'path':f'data/uploads/{ident}.jpg','luna_regions':s['predictions']})
 manifest_path.write_text(json.dumps({'screenshots':screens,'segmentation':chosen,'selection':'First three chicken-positive and first three chicken-negative examples from the existing 24-example scanner regression manifest. Frozen before Astra inference.'},indent=2))
manifest=json.loads(manifest_path.read_text())
for row in manifest['screenshots']+manifest['segmentation']:
 out=OUT/(row['id']+'.json')
 if out.exists():continue
 start=time.time()
 try:
  regions,meta=vision.openai_segment(ROOT/row['path'],model='gpt-6-astra');result={'id':row['id'],'regions':regions,'metadata':meta,'status':'completed','latency':time.time()-start}
 except Exception as e:result={'id':row['id'],'regions':[],'status':'failed','error_type':type(e).__name__,'error':str(e)[:160],'latency':time.time()-start}
 out.write_text(json.dumps(result,indent=2));print(row['id'],result['status'],[r['label'] for r in result['regions']],result.get('metadata',{}).get('api_cost_usd'),flush=True)
print(budget.run_summary('astra-scanner-comparison'),flush=True)
