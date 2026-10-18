"""Fresh, frozen 16-image paired chicken comparison; shared pessimistic $5 ledger."""
import json,shutil,concurrent.futures
from pathlib import Path
import benchmark_frontier as runner
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/benchmark-chicken-final'
if __name__=='__main__':
    manifest=json.loads((OUT/'manifest.json').read_text());runner.db.init();runner.OUT=OUT;shutil.copy(ROOT/'artifacts/benchmark/openrouter-model-catalog.json',OUT/'openrouter-model-catalog.json')
    missing=sum(not (OUT/'responses/google__gemini-4-argon'/('segmentation-'+r['id']+'.json')).exists() for r in manifest['segmentation']);assert runner.budget.summary()['accounted_usd']+missing*.1<=5
    parts=[dict(manifest,segmentation=manifest['segmentation'][i::4]) for i in range(4)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda m:runner.run_model('google/gemini-4-argon',['segmentation'],m),parts))
    print('Accounted total',runner.budget.summary()['accounted_usd'])
