"""Add unused publisher training data; freeze a new third-shard holdout before training."""
import io,json,random,hashlib,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ml'))
from prepare_v2 import phash,IDS
OUT=ROOT/'data/datasets/plateback-v4'
def main():
    prior=json.loads((ROOT/'ml/manifest-v2.json').read_text());known={r['id'] for r in prior};reserved=[];rows=[];excluded=[]
    for file,base in [('manifest.json','plateback'),('manifest-v2.json','plateback-v2'),('final-holdout-manifest.json','plateback-final')]:
        for r in json.loads((ROOT/'ml'/file).read_text()):
            p=ROOT/'data/datasets'/base/'images'/r['split']/(r['id']+'.jpg')
            if not any(x[0]==r['id'] for x in reserved):reserved.append((r['id'],r['split'],phash(Image.open(p))))
            known.add(r['id'])
    for r in prior:
        if r['split']=='test':continue
        for kind,ext in [('images','jpg'),('masks','png')]:
            dest=OUT/kind/r['split']/(r['id']+'.'+ext);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to(ROOT/'data/datasets/plateback-v2'/kind/r['split']/dest.name)
        rows.append(dict(r,origin='retained-v2'))
    def add(r,file,split):
        id='train_'+str(r['id']);raw=r['image']['bytes'];im=Image.open(io.BytesIO(raw)).convert('RGB');ph=phash(im)
        near=[oid for oid,osp,oph in reserved if np.count_nonzero(ph!=oph)<=6 and (split=='test' or osp!=split)]
        if near:excluded.append({'id':id,'near':near,'proposed_split':split});return False
        classes=[IDS[c] for c in r['classes_on_image'] if c in IDS]
        im.thumbnail((640,640));src=np.array(Image.open(io.BytesIO(r['label']['bytes'])).resize(im.size,Image.Resampling.NEAREST));mask=np.zeros_like(src,dtype=np.uint8)
        for c,d in IDS.items():mask[src==c]=d+1
        for kind in ['images','masks']:(OUT/kind/split).mkdir(parents=True,exist_ok=True)
        im.save(OUT/'images'/split/(id+'.jpg'),quality=95);Image.fromarray(mask).save(OUT/'masks'/split/(id+'.png'))
        rows.append({'id':id,'split':split,'classes':classes,'source_file':file,'sha256':hashlib.sha256(raw).hexdigest(),'license':'Apache-2.0','origin':'new-v4'})
        reserved.append((id,split,ph));known.add(id);return True
    third=pq.read_table(ROOT/'data/source/train-third.parquet').to_pylist();random.Random(141).shuffle(third)
    counts=np.zeros(4,int);neg=0
    for r in third:
        if 'train_'+str(r['id']) in known:continue
        cls=[IDS[c] for c in r['classes_on_image'] if c in IDS]
        if (cls and any(counts[c]<20 for c in cls)) or (not cls and neg<40):
            if add(r,'train-third.parquet','test'):
                for c in cls:counts[c]+=1
                if not cls:neg+=1
    negtrain=sum(not r['classes'] and r['split']=='train' for r in rows)
    for file in ['train.parquet','train-extra.parquet','train-third.parquet']:
        candidates=pq.read_table(ROOT/'data/source'/file).to_pylist();random.Random(141).shuffle(candidates)
        for r in candidates:
            if 'train_'+str(r['id']) in known:continue
            cls=[IDS[c] for c in r['classes_on_image'] if c in IDS]
            if not cls and negtrain>=400:continue
            if add(r,file,'train') and not cls:negtrain+=1
    (ROOT/'ml/manifest-v4.json').write_text(json.dumps(rows,indent=2));(ROOT/'ml/excluded-v4.json').write_text(json.dumps(excluded,indent=2))
    summary={s:{'images':sum(r['split']==s for r in rows),'classes':[sum(r['split']==s and c in r['classes'] for r in rows) for c in range(4)],'negative':sum(r['split']==s and not r['classes'] for r in rows)} for s in ['train','val','test']}
    summary['protocol']='Seed141, new test selected from previously unused third publisher TRAIN shard. All previous validation and test images protected by pHash distance <=6. Same original 197-image validation. No test access for training or selection.'
    (ROOT/'ml/split-audit-v4.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
