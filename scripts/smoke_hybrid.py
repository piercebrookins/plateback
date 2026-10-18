"""Real HTTP local+OpenAI review; stored as demo, no image-derived weight."""
import requests,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];base='http://127.0.0.1:8000/api'
p=ROOT/'data/datasets/plateback/images/val/train_1031.jpg'
r=requests.post(base+'/scans',files={'image':('plate.jpg',p.read_bytes(),'image/jpeg')},data={'idempotency_key':'verification-hybrid-v1','demo':'true'},timeout=30);r.raise_for_status();s=r.json()
if not s['metadata']:
    r=requests.post(base+f'/scans/{s["id"]}/analyze',json={'mode':'local_openai'},timeout=90);r.raise_for_status();s=r.json()
report={'scan_id':s['id'],'regions':len(s['regions']),'metadata':{k:v for k,v in s['metadata'].items() if k!='raw_bitmap_masks'},'cloud_review':s['cloud_review'],'weight':s['net_grams'],'budget':requests.get(base+'/health').json()['budget']}
(ROOT/'artifacts/hybrid-smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
