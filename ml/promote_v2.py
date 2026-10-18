"""Install the validation-selected policy after evaluation; retain baseline rollback."""
import json,shutil,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'artifacts/evaluation-v2'
decision=json.loads((out/'selection-decision.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
for name in ['baseline-fresh','candidate-fresh','baseline-regression','candidate-regression']:
    assert json.loads((out/(name+'.json')).read_text())['images']>0
candidate=ROOT/'artifacts/models/candidate-v2/best.pt';assert hashlib.sha256(candidate.read_bytes()).hexdigest()==protocol['candidate_sha256']
chosen=decision['selected'];source=ROOT/'artifacts/models'/('candidate-v2' if chosen=='candidate' else 'baseline-v1')
settings=source/'settings.json' if chosen=='candidate' else out/'baseline-calibrated-settings.json'
target=ROOT/'artifacts/models/plateback';target.mkdir(exist_ok=True,parents=True)
meta=json.loads((source/'metadata.json').read_text());meta.update(deployment_selection=decision,license='AGPL-3.0',classes=['rice','pasta','broccoli','chicken'])
for src,name in [(source/'best.pt','best.pt'),(settings,'settings.json')]:
    tmp=target/(name+'.tmp');shutil.copyfile(src,tmp);os.replace(tmp,target/name)
(target/'metadata.json').write_text(json.dumps(meta,indent=2));(ROOT/'docs/model-metadata.json').write_text(json.dumps(meta,indent=2))
print('Installed',chosen,'with validation-frozen policy. Restart the backend to clear cached weights.')
