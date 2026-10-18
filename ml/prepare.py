"""Reproducible FoodSeg103 subset; publisher references are independent mask truth."""
import io,json,hashlib,random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import cv2
from scipy.fft import dctn
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/datasets/plateback'
IDS={66:0,64:1,87:2,48:3}
NAMES=['rice','pasta','broccoli','chicken']
def main():
    random.seed(42); rows=[]; seen=set(); counts={}; visual_seen=[]; excluded=[]
    for source in ['train','test']:
        candidates=pq.read_table(ROOT/f'data/source/{source}.parquet').to_pylist()
        random.shuffle(candidates)
        class_counts=[0]*4; negative=0
        for row in candidates:
            classes=[IDS[c] for c in row['classes_on_image'] if c in IDS]
            if not classes:
                if negative>=20: continue
                negative+=1
            elif all(class_counts[c]>=90 if source=='train' else class_counts[c]>=20 for c in classes): continue
            raw=row['image']['bytes']; digest=hashlib.sha256(raw).hexdigest()
            if digest in seen: continue
            seen.add(digest)
            # Split by image hash before augmentation. Original publisher test remains test.
            split='test' if source=='test' else ('val' if int(digest[:8],16)%5==0 else 'train')
            im=Image.open(io.BytesIO(raw)).convert('RGB'); mask=Image.open(io.BytesIO(row['label']['bytes']))
            im.thumbnail((640,640)); mask=mask.resize(im.size,Image.Resampling.NEAREST); arr=np.array(mask)
            stem=f'{source}_{row["id"]}'
            d=dctn(np.asarray(im.convert('L').resize((32,32)),dtype=float),norm='ortho')[:8,:8].flatten()[1:]
            ph=d>np.median(d)
            near=[old for old,oldsplit,oldph in visual_seen if oldsplit!=split and np.count_nonzero(ph!=oldph)<=6]
            if near:
                excluded.append({'id':stem,'split':split,'near':near})
                for c in classes:class_counts[c]+=1
                continue
            visual_seen.append((stem,split,ph))
            for folder in ['images','labels','masks']: (OUT/folder/split).mkdir(parents=True,exist_ok=True)
            im.save(OUT/'images'/split/f'{stem}.jpg',quality=95)
            mapped=np.zeros(arr.shape,np.uint8); lines=[]
            for orig,cls in IDS.items():
                binary=(arr==orig).astype(np.uint8); mapped[binary>0]=cls+1
                contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour)<30: continue
                    contour=cv2.approxPolyDP(contour,1.0,True).reshape(-1,2)
                    if len(contour)<3: continue
                    xy=contour/np.array([im.width,im.height]); lines.append(str(cls)+' '+' '.join(f'{v:.6f}' for v in xy.ravel()))
            (OUT/'labels'/split/f'{stem}.txt').write_text('\n'.join(lines))
            Image.fromarray(mapped).save(OUT/'masks'/split/f'{stem}.png')
            rows.append(dict(id=stem,split=split,publisher_split=source,sha256=digest,classes=classes,source_id=row['id'],license='Apache-2.0',source='EduardoPacheco/FoodSeg103',poultry_label='chicken duck' if 3 in classes else None))
            for c in classes: class_counts[c]+=1
        counts[source]=class_counts
    (ROOT/'ml/excluded_duplicates.json').write_text(json.dumps(excluded,indent=2))
    (ROOT/'ml/manifest.json').write_text(json.dumps(rows,indent=2))
    (OUT/'dataset.yaml').write_text(f'path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: '+json.dumps(NAMES)+'\n')
    print(json.dumps({'images':len(rows),'counts':counts,'splits':{s:sum(r['split']==s for r in rows) for s in ['train','val','test']}},indent=2))
if __name__=='__main__':main()
