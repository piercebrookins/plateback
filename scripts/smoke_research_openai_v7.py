"""One budget-scoped live hybrid call, then isolated API replay of that same result."""
import os,sys,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env');os.environ['PLATEBACK_BUDGET_RUN']='hackathon-v7'
from backend.app import vision,budget
import torch
torch.set_num_threads(2);budget.start_run('hackathon-v7',3);out=ROOT/'artifacts/research-openai-v7-live.json';path=ROOT/'data/datasets/plateback-chicken-final/images/test/test_1248.jpg'
if not out.exists():
 out.write_text(json.dumps({'status':'pending','note':'Do not retry an unknown outcome automatically'}));result=vision.analyze(path,'research_openai');out.write_text(json.dumps({'status':'completed','result':result},indent=2))
value=json.loads(out.read_text());assert value['status']=='completed';result=value['result'];assert result['metadata']['experimental'] and result['cloud_review']['model']=='gpt-6-luna'
run=budget.run_summary('hackathon-v7');total=budget.summary()['accounted_usd']
tmp=tempfile.TemporaryDirectory();os.environ['PLATEBACK_DB']=tmp.name+'/app.db';os.environ['UPLOAD_DIR']=tmp.name+'/uploads'
from fastapi.testclient import TestClient
from backend.app import main
vision.analyze=lambda *a:result
with TestClient(main.app) as c:
 with path.open('rb') as f:s=c.post('/api/scans',files={'image':('plate.jpg',f,'image/jpeg')},data={'idempotency_key':'research-v7','demo':'true'}).json()
 response=c.post('/api/scans/'+s['id']+'/analyze',json={'mode':'research_openai'});assert response.status_code==200,response.text;s=response.json();assert s['net_grams'] is None
 assert c.put('/api/scans/'+s['id']+'/review',json={'regions':s['regions']}).status_code==200
 assert c.post('/api/scans/'+s['id']+'/confirm').status_code==200
 assert c.get('/api/dashboard').json()['scans']==0
summary={'live_hybrid_call':'passed','isolated_api_replay':'passed','regions':len(result['regions']),'reviewed_regions':len(result['cloud_review']['reviewed_region_ids']),'manual_review_regions':len(result['regions'])-len(result['cloud_review']['reviewed_region_ids']),'model':result['metadata']['model'],'openai_model':result['cloud_review']['model'],'unknown_weight_preserved':True,'run_budget':run,'accounted_total':total};(ROOT/'artifacts/research-openai-v7-smoke.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
