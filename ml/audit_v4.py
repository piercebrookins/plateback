"""Independent check of decoded on-disk images across current and historical holdouts."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ml'))
from prepare_v2 import phash
rows=json.loads((ROOT/'ml/manifest-v4.json').read_text());train=[];held=[]
for r in rows:
 p=ROOT/'data/datasets/plateback-v4/images'/r['split']/(r['id']+'.jpg')
 (train if r['split']=='train' else held).append((r['id'],r['split'],phash(Image.open(p))))
for manifest,base in [('manifest.json','plateback'),('manifest-v2.json','plateback-v2'),('final-holdout-manifest.json','plateback-final')]:
 for r in json.loads((ROOT/'ml'/manifest).read_text()):
  if r['split']=='test':held.append((r['id'],'historical-test',phash(Image.open(ROOT/'data/datasets'/base/'images/test'/(r['id']+'.jpg')))))
arr=np.array([r[2] for r in held]);matches=[]
for id,split,ph in train:
 for j in np.flatnonzero(np.count_nonzero(arr!=ph,axis=1)<=6):matches.append({'training_id':id,'heldout_id':held[j][0],'heldout_split':held[j][1]})
newtest=[r for r in held if r[1]=='test'];other=[r for r in held if r[1]!='test'];testmatches=[]
for i,(id,split,ph) in enumerate(newtest):
 for oid,osp,oph in other+newtest[i+1:]:
  if np.count_nonzero(ph!=oph)<=6:testmatches.append({'new_test_id':id,'other_id':oid,'other_split':osp})
report={'training_images':len(train),'heldout_references':len(held),'phash_max_distance':6,'cross_split_matches':matches,'new_test_near_duplicates':testmatches};(ROOT/'ml/independent-audit-v4.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert not matches and not testmatches,'Cross-split near duplicates detected'
