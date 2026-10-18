"""Reserve another unused publisher subset for a final check after v3 policy freeze."""
import json,io,random,hashlib,sys
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ml.prepare_v2 import phash,IDS
OUT=ROOT/'data/datasets/plateback-final'
old=json.loads((ROOT/'ml/manifest.json').read_text());v2=json.loads((ROOT/'ml/manifest-v2.json').read_text());used={r['id'] for r in old+v2};refs=[]
for rows,data in [(old,'plateback'),(v2,'plateback-v2')]:
 for r in rows:refs.append(phash(Image.open(ROOT/f'data/datasets/{data}/images/{r["split"]}/{r["id"]}.jpg')))
counts=[0]*4;neg=0;chosen=[];excluded=[];source=pq.read_table(ROOT/'data/source/test.parquet').to_pylist();random.Random(127).shuffle(source)
for r in source:
 id='test_'+str(r['id']);classes=[IDS[c] for c in r['classes_on_image'] if c in IDS]
 if id in used:continue
 if classes and all(counts[c]>=20 for c in classes):continue
 if not classes and neg>=40:continue
 raw=r['image']['bytes'];im=Image.open(io.BytesIO(raw)).convert('RGB');h=phash(im)
 if any(np.count_nonzero(h!=p)<=6 for p in refs):excluded.append(id);continue
 refs.append(h);im.thumbnail((640,640));mask=np.array(Image.open(io.BytesIO(r['label']['bytes'])).resize(im.size,Image.Resampling.NEAREST));mapped=np.zeros_like(mask,np.uint8)
 for src,c in IDS.items():mapped[mask==src]=c+1
 for folder in ['images','masks']:(OUT/folder/'test').mkdir(exist_ok=True,parents=True)
 im.save(OUT/'images/test'/(id+'.jpg'),quality=95);Image.fromarray(mapped).save(OUT/'masks/test'/(id+'.png'));chosen.append({'id':id,'split':'test','source_file':'test.parquet','sha256':hashlib.sha256(raw).hexdigest(),'classes':classes})
 for c in classes:counts[c]+=1
 if not classes:neg+=1
(ROOT/'ml/final-holdout-manifest.json').write_text(json.dumps(chosen,indent=2));(ROOT/'ml/final-holdout-exclusions.json').write_text(json.dumps(excluded,indent=2));print({'images':len(chosen),'positive_counts':counts,'negative_images':neg})
