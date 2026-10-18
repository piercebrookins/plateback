"""Expanded local training with other-food negatives, frozen old holdouts and a fresh test set."""
import io,json,hashlib,random
from pathlib import Path
import numpy as np,cv2,pyarrow.parquet as pq
from PIL import Image
from scipy.fft import dctn
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/datasets/plateback-v2';IDS={66:0,64:1,87:2,48:3};NAMES=['rice','pasta','broccoli','chicken']
def phash(im):
    v=dctn(np.asarray(im.convert('L').resize((32,32)),dtype=float),norm='ortho')[:8,:8].flatten()[1:];return v>np.median(v)
def main():
    old=json.loads((ROOT/'ml/manifest.json').read_text());oldmap={r['id']:r for r in old};reserved=[];rows=[];excluded=[];seen=set();counts={s:np.zeros(4,int) for s in ['train','val','test']};neg={s:0 for s in counts}
    for r in old:
        im=Image.open(ROOT/'data/datasets/plateback/images'/r['split']/(r['id']+'.jpg'));reserved.append((r['id'],r['split'],phash(im)))
    for file,source in [('train.parquet','train'),('train-extra.parquet','train'),('test.parquet','test')]:
        candidates=pq.read_table(ROOT/'data/source'/file).to_pylist();random.Random(77).shuffle(candidates)
        # Keep all existing training/validation examples before additions.
        candidates.sort(key=lambda r:0 if f'{source}_{r["id"]}' in oldmap else 1)
        for r in candidates:
            id=f'{source}_{r["id"]}';raw=r['image']['bytes'];sha=hashlib.sha256(raw).hexdigest();classes=[IDS[c] for c in r['classes_on_image'] if c in IDS]
            if sha in seen:continue
            prior=oldmap.get(id)
            if source=='test' and prior:continue
            split=prior['split'] if prior else ('test' if source=='test' else ('val' if int(sha[:8],16)%5==0 else 'train'))
            if not prior:
                cap={'train':210,'val':55,'test':40}[split];negcap={'train':180,'val':50,'test':70}[split]
                if classes and all(counts[split][c]>=cap for c in classes):continue
                if not classes and neg[split]>=negcap:continue
            im=Image.open(io.BytesIO(raw)).convert('RGB');ph=phash(im)
            near=[oid for oid,osp,oph in reserved if oid!=id and (osp!=split or split=='test') and np.count_nonzero(ph!=oph)<=6]
            if near:excluded.append({'id':id,'near':near});continue
            seen.add(sha);reserved.append((id,split,ph));im.thumbnail((640,640));mask=np.array(Image.open(io.BytesIO(r['label']['bytes'])).resize(im.size,Image.Resampling.NEAREST));mapped=np.zeros_like(mask,dtype=np.uint8);lines=[]
            for src,cls in IDS.items():
                binary=(mask==src).astype(np.uint8);mapped[binary>0]=cls+1
                contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c)<30:continue
                    p=cv2.approxPolyDP(c,1,True).reshape(-1,2)
                    if len(p)>=3:lines.append(str(cls)+' '+' '.join(f'{v:.6f}' for v in (p/np.array(im.size)).ravel()))
            for folder in ['images','masks','labels']:(OUT/folder/split).mkdir(parents=True,exist_ok=True)
            im.save(OUT/'images'/split/(id+'.jpg'),quality=95);Image.fromarray(mapped).save(OUT/'masks'/split/(id+'.png'));(OUT/'labels'/split/(id+'.txt')).write_text('\n'.join(lines))
            rows.append({'id':id,'split':split,'source_file':file,'sha256':sha,'classes':classes,'source_labels':r['classes_on_image'],'license':'Apache-2.0'})
            for c in classes:counts[split][c]+=1
            if not classes:neg[split]+=1
    (ROOT/'ml/manifest-v2.json').write_text(json.dumps(rows,indent=2));(ROOT/'ml/excluded-v2.json').write_text(json.dumps(excluded,indent=2))
    (OUT/'dataset.yaml').write_text(f'path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: '+json.dumps(NAMES)+'\n')
    print(json.dumps({s:{'images':sum(r['split']==s for r in rows),'positive_counts':counts[s].tolist(),'negative_images':neg[s]} for s in counts},indent=2))
if __name__=='__main__':main()
