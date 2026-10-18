"""Download public data, verify SHA-256. No account/token or paid API required."""
import json,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for entry in json.loads((ROOT/'ml/source_manifest.json').read_text())['files']:
    p=ROOT/entry['path'];p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        tmp=p.with_suffix('.download');urllib.request.urlretrieve(entry['url'],tmp);tmp.replace(p)
    digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    if digest!=entry['sha256']:raise ValueError(f'Checksum mismatch: {p}')
    print(f'Verified {p.name}: {digest}')
