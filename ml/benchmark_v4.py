"""New 32-image paired Gemini test; frozen before observing candidate performance."""
import json,hashlib,shutil,concurrent.futures
from pathlib import Path
import benchmark_frontier as runner
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/benchmark-v4'
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'manifest.json'
    if path.exists():return json.loads(path.read_text())
    rows=[r for r in json.loads((ROOT/'ml/manifest-v4.json').read_text()) if r['split']=='test'];rows.sort(key=lambda r:hashlib.sha256(('paired-v4-141'+r['id']).encode()).hexdigest());chosen=[]
    for c in range(4):chosen.extend([r for r in rows if c in r['classes'] and r not in chosen][:6])
    chosen.extend([r for r in rows if not r['classes']][:8])
    assert len(chosen)==32
    result={'segmentation':[{'id':r['id'],'path':f'data/datasets/plateback-v4/images/test/{r["id"]}.jpg','truth_path':f'data/datasets/plateback-v4/masks/test/{r["id"]}.png'} for r in chosen],'presence':[],'models':['google/gemini-4-argon'],'protocol':'32 new v4 held-out images: six unique selections per class plus eight negatives; overlapping categories can yield more than six positives. Selection seed141 before candidate results. Frozen identical prompt and transport parser from prior benchmark. No paid calls used for training or threshold selection.'}
    path.write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    manifest=freeze();runner.db.init();runner.OUT=OUT
    shutil.copy(ROOT/'artifacts/benchmark/openrouter-model-catalog.json',OUT/'openrouter-model-catalog.json')
    # Even if every call has an unknown outcome, 32 reservations cost no more than $3.20.
    missing=sum(not (OUT/'responses/google__gemini-4-argon'/('segmentation-'+r['id']+'.json')).exists() for r in manifest['segmentation'])
    assert runner.budget.summary()['accounted_usd']+missing*.1<=5,'Insufficient worst-case budget; no calls sent'
    parts=[dict(manifest,segmentation=manifest['segmentation'][i::4]) for i in range(4)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda m:runner.run_model('google/gemini-4-argon',['segmentation'],m),parts))
    print('Total accounting',runner.budget.summary()['accounted_usd'])
