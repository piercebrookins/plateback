"""Reserve unused publisher-test poultry and hard negatives before boundary-refinement selection."""
import sys,json,io,random,hashlib
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ml'))
from prepare_v2 import phash,IDS
OUT=ROOT/'data/datasets/plateback-chicken-final'
def main():
    target=ROOT/'ml/chicken-holdout-manifest.json'
    if target.exists():print('Frozen holdout exists');return
    reserved=[];known=set()
    for name,base in [('manifest.json','plateback'),('manifest-v2.json','plateback-v2'),('final-holdout-manifest.json','plateback-final'),('manifest-v4.json','plateback-v4')]:
        for r in json.loads((ROOT/'ml'/name).read_text()):
            if r['id'] in known:continue
            known.add(r['id']);reserved.append((r['id'],phash(Image.open(ROOT/'data/datasets'/base/'images'/r['split']/(r['id']+'.jpg')))))
    arr=np.array([p for _,p in reserved]);rows=[];excluded=[];count={'chicken':0,'hard-negative':0};candidates=pq.read_table(ROOT/'data/source/test.parquet').to_pylist();random.Random(151).shuffle(candidates)
    for r in candidates:
        id='test_'+str(r['id']);classes=r['classes_on_image']
        group='chicken' if 48 in classes else ('hard-negative' if not any(c in IDS for c in classes) and any(c in [46,47,49,50,51,54] for c in classes) else None)
        if id in known or group is None or count[group]>=16:continue
        im=Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB');im.thumbnail((640,640));buf=io.BytesIO();im.save(buf,format='JPEG',quality=95);decoded=Image.open(io.BytesIO(buf.getvalue()));ph=phash(decoded)
        near=np.flatnonzero(np.count_nonzero(arr!=ph,axis=1)<=6)
        if len(near):excluded.append({'id':id,'near':[reserved[i][0] for i in near]});continue
        mask=np.array(Image.open(io.BytesIO(r['label']['bytes'])).resize(im.size,Image.Resampling.NEAREST));mapped=np.zeros_like(mask,np.uint8)
        for c,d in IDS.items():mapped[mask==c]=d+1
        for kind in ['images','masks']:(OUT/kind/'test').mkdir(parents=True,exist_ok=True)
        (OUT/'images/test'/(id+'.jpg')).write_bytes(buf.getvalue());Image.fromarray(mapped).save(OUT/'masks/test'/(id+'.png'));rows.append({'id':id,'split':'test','group':group,'classes':[IDS[c] for c in classes if c in IDS],'source_labels':classes,'sha256':hashlib.sha256(r['image']['bytes']).hexdigest(),'license':'Apache-2.0','publisher_split':'test'});count[group]+=1;reserved.append((id,ph));arr=np.concatenate([arr,ph[None]],axis=0)
    assert count=={'chicken':16,'hard-negative':16},count
    target.write_text(json.dumps(rows,indent=2));(ROOT/'ml/chicken-holdout-exclusions.json').write_text(json.dumps(excluded,indent=2));print(count)
    paired=[]
    for group in count:paired.extend([r for r in rows if r['group']==group][:8])
    bm=ROOT/'artifacts/benchmark-chicken-final';bm.mkdir(parents=True,exist_ok=True)
    value={'models':['google/gemini-4-argon'],'presence':[],'segmentation':[{'id':r['id'],'path':f'data/datasets/plateback-chicken-final/images/test/{r["id"]}.jpg','truth_path':f'data/datasets/plateback-chicken-final/masks/test/{r["id"]}.png'} for r in paired],'protocol':'Seed151; 16 new chicken/duck images +16 new other-meat/fish negatives, unused publisher test images screened against all previous datasets. Paired Gemini subset: first8 in each frozen group. Frozen before v5 boundary policy selection. Chicken metrics are primary; other foods lack balanced coverage.'};(bm/'manifest.json').write_text(json.dumps(value,indent=2))
if __name__=='__main__':main()
