"""Native-weight API round trip in an isolated temporary database; no cloud calls."""
import os,sys,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
with tempfile.TemporaryDirectory(prefix='plateback-v5-smoke-') as tmp:
    os.environ['PLATEBACK_DB']=str(Path(tmp)/'test.db');os.environ['UPLOAD_DIR']=str(Path(tmp)/'uploads');os.environ['LOCAL_MODEL_PATH']=str(ROOT/'artifacts/models/semantic-v5/best.pt')
    from fastapi.testclient import TestClient
    from backend.app.main import app
    with TestClient(app) as c:
        source=ROOT/'data/datasets/plateback/images/val/train_1031.jpg';r=c.post('/api/scans',files={'image':('plate.jpg',source.read_bytes(),'image/jpeg')},data={'idempotency_key':'v5-smoke','demo':'true'});assert r.status_code==200,r.text;s=r.json();id=s['id']
        r=c.post(f'/api/scans/{id}/analyze',json={'mode':'local'});assert r.status_code==200,r.text;s=r.json();assert s['metadata']['model']=='PlateBack v5 semantic';assert s['regions'];original=s['regions'];edited=json.loads(json.dumps(original));edited[0]['label']='unknown'
        assert c.put(f'/api/scans/{id}/review',json={'regions':edited}).status_code==200
        assert c.get(f'/api/scans/{id}').json()['predictions']==original
        assert c.put(f'/api/scans/{id}/review',json={'regions':original}).status_code==200
        assert c.post(f'/api/scans/{id}/confirm').status_code==200
        s=c.get(f'/api/scans/{id}').json();assert s['net_grams'] is None and s['weight_source']=='none'
        assert c.get('/api/dashboard').json()['scans']==0
        assert c.post('/api/scans',files={'image':('bad.jpg',b'not an image','image/jpeg')},data={'idempotency_key':'invalid-after-inference'}).status_code==422
        assert c.get('/api/health').json()['budget']['accounted_usd']==0
        report={'model':s['metadata']['model'],'regions':len(s['regions']),'labels':[r['label'] for r in s['regions']],'status':s['status'],'unknown_weight_preserved':s['net_grams'] is None,'edit_restore_confirm':True,'original_predictions_preserved':True,'demo_excluded_from_real_dashboard':True,'invalid_upload_after_native_inference_rejected':True,'api_calls':0,'database':'isolated temporary test database; active v3 unchanged'}
        (ROOT/'artifacts/semantic-v5-smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
