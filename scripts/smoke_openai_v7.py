"""Replay a paid, cached provider result through the real app API in an isolated database."""
import os,sys,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
tmp=tempfile.TemporaryDirectory();os.environ['PLATEBACK_DB']=tmp.name+'/smoke.db';os.environ['UPLOAD_DIR']=tmp.name+'/uploads'
from fastapi.testclient import TestClient
from backend.app import main,vision
cache=ROOT/'artifacts/benchmark-openai-v7-normalized';manifest=json.loads((cache/'manifest.json').read_text())['segmentation'];row=next(r for r in manifest if len(json.loads((cache/(r['id']+'.json')).read_text()).get('regions',[]))>0);result=json.loads((cache/(row['id']+'.json')).read_text());vision.openai_segment=lambda path:(result['regions'],result['metadata'])
with TestClient(main.app) as client:
 with open(ROOT/row['path'],'rb') as f:s=client.post('/api/scans',files={'image':('plate.jpg',f,'image/jpeg')},data={'idempotency_key':'openai-smoke','demo':'true'}).json()
 r=client.post('/api/scans/'+s['id']+'/analyze',json={'mode':'openai'});assert r.status_code==200,r.text;s=r.json();assert s['metadata']['provider']=='openai' and s['metadata']['mode']=='openai';assert s['net_grams'] is None;original=s['predictions'];edited=[{**x,'label':'unknown'} for x in s['regions']]
 assert client.put('/api/scans/'+s['id']+'/review',json={'regions':edited}).status_code==200
 final=client.post('/api/scans/'+s['id']+'/confirm');assert final.status_code==200;assert final.json()['predictions']==original and final.json()['net_grams'] is None
 assert client.get('/api/dashboard').json()['scans']==0
out={'provider':'openai','model':'gpt-6-luna','live_provider_responses_checked_separately':24,'api_smoke':'cached result replay, no additional paid call','editable_and_confirmable':True,'original_predictions_preserved':True,'unknown_weight_preserved':True,'demo_excluded':True};(ROOT/'artifacts/openai-v7-smoke.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
