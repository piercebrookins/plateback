"""Nutrition5k-derived data: official split, exact ingredient mapping, no inferred mass."""
import requests,json,random,hashlib,concurrent.futures
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/datasets/nutrition5k';OUT.mkdir(parents=True,exist_ok=True)
MAP={'rice':{'white rice','brown rice','country rice','wild rice','fried rice'},'pasta':{'pasta','pasta salad'},'broccoli':{'broccoli'},'chicken':{'chicken','chicken thighs','chicken breast','grilled chicken','chicken salad'}}
meta=ROOT/'data/source/food-nutrients-metadata.jsonl'
if not meta.exists():
    response=requests.get('https://huggingface.co/datasets/mmathys/food-nutrients/resolve/main/metadata.jsonl',timeout=60);response.raise_for_status();meta.write_bytes(response.content)
rows=[json.loads(l) for l in meta.read_text().splitlines()];manifest=[]
for split in ['train','test']:
    u=f'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/rgb_{split}_ids.txt';r=requests.get(u,timeout=30);r.raise_for_status();(ROOT/f'data/source/nutrition5k-rgb_{split}_ids.txt').write_text(r.text);ids=set(r.text.split())
    candidates=[r for r in rows if r['id'] in ids];random.Random(93).shuffle(candidates);counts={l:0 for l in MAP};neg=0;selected=[];cap=60 if split=='train' else 10;negcap=60 if split=='train' else 12
    for r in candidates:
        names={i['name'] for i in r['ingredients']};labels=[l for l,v in MAP.items() if names&v]
        if labels and all(counts[l]>=cap for l in labels):continue
        if not labels and neg>=negcap:continue
        selected.append({**r,'official_split':split,'labels':labels})
        for l in labels:counts[l]+=1
        if not labels:neg+=1
    def download(r):
        path=OUT/'images'/split/(r['id']+'.jpg');path.parent.mkdir(exist_ok=True,parents=True);url='https://huggingface.co/datasets/mmathys/food-nutrients/resolve/main/'+r['file_name']
        rawpath=OUT/'originals'/(r['id']+'.png');rawpath.parent.mkdir(exist_ok=True)
        if not rawpath.exists():
            resp=requests.get(url,timeout=60);resp.raise_for_status();rawpath.write_bytes(resp.content)
        im=Image.open(rawpath).convert('RGB');im.thumbnail((640,640));im.save(path,quality=95)
        return {'id':r['id'],'split':split,'labels':r['labels'],'ingredients':[i['name'] for i in r['ingredients']],'path':str(path.relative_to(ROOT)),'url':url,'sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest(),'license':'CC-BY-4.0','mass_used':False}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:manifest.extend(pool.map(download,selected))
    print(split,len(selected),counts,'negatives',neg,flush=True)
( ROOT/'ml/nutrition-manifest.json').write_text(json.dumps(manifest,indent=2));(ROOT/'ml/nutrition-label-map.json').write_text(json.dumps({k:sorted(v) for k,v in MAP.items()},indent=2))
