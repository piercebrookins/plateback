"""Promote the validation-selected v3 policy, preserving all prior artifacts."""
from pathlib import Path
import json,shutil,os,hashlib
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'artifacts/models/ensemble-v3';target=ROOT/'artifacts/models/plateback'
assert (ROOT/'artifacts/evaluation-v3/ensemble-final.json').exists()
meta=json.loads((source/'metadata.json').read_text())
for name,digest in meta['components'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
for name in ['best.pt','baseline.pt','presence.pt','settings.json','metadata.json']:
    temp=target/(name+'.tmp');shutil.copyfile(source/name,temp);os.replace(temp,target/name)
(ROOT/'docs/model-metadata.json').write_text(json.dumps(meta,indent=2));print('Promoted v3. Restart backend to load all components.')
