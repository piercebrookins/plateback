"""Live v3 + GPT-6 Luna integration on a validation photo; not an accuracy benchmark."""
import requests,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];base='http://127.0.0.1:8000/api';image=ROOT/'data/datasets/plateback/images/val/train_1031.jpg'
r=requests.post(base+'/scans',files={'image':('plate.jpg',image.read_bytes(),'image/jpeg')},data={'idempotency_key':'verification-hybrid-v3','demo':'true'},timeout=30);r.raise_for_status();s=r.json()
if not s['metadata']:
 r=requests.post(base+f'/scans/{s["id"]}/analyze',json={'mode':'local_openai'},timeout=120);r.raise_for_status();s=r.json()
assert s['metadata']['model']=='PlateBack v3 ensemble';assert s['cloud_review']['model']=='gpt-6-luna';assert s['net_grams'] is None
report={'scan_id':s['id'],'regions':len(s['regions']),'metadata':{k:v for k,v in s['metadata'].items() if k!='raw_bitmap_masks'},'cloud_review':s['cloud_review'],'weight':s['net_grams'],'budget':requests.get(base+'/health',timeout=30).json()['budget']};(ROOT/'artifacts/hybrid-v3-smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['budget','metadata']},indent=2));print('Accounted API:',report['budget']['accounted_usd'])
