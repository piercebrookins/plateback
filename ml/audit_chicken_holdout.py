"""Independent decoded-image audit against every earlier dataset, including Nutrition5k."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ml'))
from prepare_v2 import phash
prior={}
for file,base in [('manifest.json','plateback'),('manifest-v2.json','plateback-v2'),('final-holdout-manifest.json','plateback-final'),('manifest-v4.json','plateback-v4')]:
 for r in json.loads((ROOT/'ml'/file).read_text()):prior[r['id']]=ROOT/'data/datasets'/base/'images'/r['split']/(r['id']+'.jpg')
for r in json.loads((ROOT/'ml/nutrition-manifest.json').read_text()):prior[r['id']]=ROOT/r['path']
ids=list(prior);hashes=np.array([phash(Image.open(prior[id])) for id in ids]);matches=[]
for r in json.loads((ROOT/'ml/chicken-holdout-manifest.json').read_text()):
 ph=phash(Image.open(ROOT/'data/datasets/plateback-chicken-final/images/test'/(r['id']+'.jpg')))
 for i in np.flatnonzero(np.count_nonzero(hashes!=ph,axis=1)<=6):matches.append({'test_id':r['id'],'other_id':ids[i]})
 ids.append(r['id']);hashes=np.concatenate([hashes,ph[None]],axis=0)
report={'test_images':32,'prior_images':len(prior),'phash_max_distance':6,'cross_dataset_or_within_test_matches':matches};(ROOT/'ml/chicken-holdout-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert not matches
