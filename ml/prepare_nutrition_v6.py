"""Expand CC-BY-4.0 ingredient supervision; reserve validation/test before fitting."""
import json,hashlib,random,concurrent.futures,sys
from pathlib import Path
import requests,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ml'))
from prepare_v2 import phash
OUT=ROOT/'data/datasets/nutrition-v6';OUT.mkdir(parents=True,exist_ok=True)
old=json.loads((ROOT/'ml/nutrition-manifest.json').read_text());used={r['id'] for r in old};mapping=json.loads((ROOT/'ml/nutrition-label-map.json').read_text());raw=[json.loads(x) for x in (ROOT/'data/source/food-nutrients-metadata.jsonl').read_text().splitlines()]
# All prior images, including training, are excluded from new examples.
prior={}
for f,base in [('manifest.json','plateback'),('manifest-v2.json','plateback-v2'),('final-holdout-manifest.json','plateback-final'),('manifest-v4.json','plateback-v4'),('chicken-holdout-manifest.json','plateback-chicken-final')]:
 for r in json.loads((ROOT/'ml'/f).read_text()):prior[r['id']]=ROOT/'data/datasets'/base/'images'/r['split']/(r['id']+'.jpg')
for r in old:prior[r['id']]=ROOT/r['path']
hashes=[phash(Image.open(p)) for p in prior.values()];excluded=[];selected=[]
for official in ['test','train']:
 ids=set((ROOT/f'data/source/nutrition5k-rgb_{official}_ids.txt').read_text().split());pool=[r for r in raw if r['id'] in ids and r['id'] not in used];random.Random(161).shuffle(pool)
 targets={'test':[30,30]} if official=='test' else {'val':[30,30],'train':[150,150]}
 # Download a bounded deterministic pool, with no model-based selection.
 groups=[[],[]]
 for r in pool:
  names={x['name'] for x in r['ingredients']};r['labels']=[k for k,v in mapping.items() if names & set(v)];groups[int('chicken' in r['labels'])].append(r)
 def fetch(r):
  path=OUT/(r['id']+'.jpg');url='https://huggingface.co/datasets/mmathys/food-nutrients/resolve/main/'+r['file_name']
  if not path.exists():
   from io import BytesIO
   resp=requests.get(url,timeout=90);resp.raise_for_status();im=Image.open(BytesIO(resp.content)).convert('RGB');im.thumbnail((640,640));im.save(path,quality=95)
  return r,path,url
 for label,group in enumerate(groups):
  limits=sum(v[label] for v in targets.values());accepted=0
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pooler:
   for r,path,url in pooler.map(fetch,group[:limits+70]):
    if accepted>=limits:break
    ph=phash(Image.open(path));dist=np.count_nonzero(np.array(hashes)!=ph,axis=1)
    if (dist<=6).any():excluded.append({'id':r['id'],'reason':'decoded phash <=6 from prior or selected image'});continue
    split=next(s for s,c in targets.items() if c[label]>0);targets[split][label]-=1;accepted+=1;hashes.append(ph)
    selected.append({'id':r['id'],'split':split,'official_split':official,'labels':r['labels'],'ingredients':[i['name'] for i in r['ingredients']],'path':str(path.relative_to(ROOT)),'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'license':'CC-BY-4.0','supervision':'ingredient presence only; no pixel or mass targets'})
 print(official,'selected',len(selected),'remaining',targets,flush=True)
# Existing training remains eligible; old test is regression only.
rows=[{**r,'split':'train','official_split':'train'} for r in old if r['split']=='train']+selected
(ROOT/'ml/nutrition-v6-manifest.json').write_text(json.dumps(rows,indent=2));(ROOT/'ml/nutrition-v6-audit.json').write_text(json.dumps({'prior_images':len(prior),'excluded':excluded,'phash_distance':6,'new_images':len(selected),'counts':{s:{'images':sum(r['split']==s for r in rows),'chicken':sum(r['split']==s and 'chicken' in r['labels'] for r in rows)} for s in ['train','val','test']},'label_limitation':'Ingredients do not guarantee visible chicken; no pixel masks'},indent=2))
