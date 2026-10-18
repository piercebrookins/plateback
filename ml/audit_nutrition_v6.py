"""Independent verification of saved manifest, actual images, split membership and duplicates."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ml'))
from prepare_v2 import phash
rows=json.loads((ROOT/'ml/nutrition-v6-manifest.json').read_text());old=json.loads((ROOT/'ml/nutrition-manifest.json').read_text());oldids={r['id'] for r in old};prior={}
for f,base in [('manifest.json','plateback'),('manifest-v2.json','plateback-v2'),('final-holdout-manifest.json','plateback-final'),('manifest-v4.json','plateback-v4'),('chicken-holdout-manifest.json','plateback-chicken-final')]:
 for r in json.loads((ROOT/'ml'/f).read_text()):prior[r['id']]=ROOT/'data/datasets'/base/'images'/r['split']/(r['id']+'.jpg')
for r in old:prior[r['id']]=ROOT/r['path']
ids=list(prior);hashes=[phash(Image.open(prior[id])) for id in ids];official={s:set((ROOT/f'data/source/nutrition5k-rgb_{s}_ids.txt').read_text().split()) for s in ['train','test']};matches=[]
assert len({r['id'] for r in rows})==len(rows)
for r in rows:
 assert r['id'] in official['test' if r['split']=='test' else 'train']
 if r['id'] in oldids:assert r['split']=='train';continue
 path=ROOT/r['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];ph=phash(Image.open(path));ix=np.flatnonzero(np.count_nonzero(np.array(hashes)!=ph,axis=1)<=6)
 for i in ix:matches.append([r['id'],ids[i]])
 ids.append(r['id']);hashes.append(ph)
report={'new_images':sum(r['id'] not in oldids for r in rows),'prior_unique_images':len(prior),'duplicate_matches':matches,'official_split_membership_verified':True,'new_image_sha256_verified':True,'phash_max_distance':6};(ROOT/'artifacts/evaluation-v6/independent-data-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));assert not matches
