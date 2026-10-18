"""Render reproducible reference/prediction overlays, including low-IoU failures."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/evaluation'
r=json.loads((OUT/'test.json').read_text());rows=r['per_image'];colors=[(220,171,71),(216,121,72),(88,165,130),(172,129,192)]
def mean(row):
    v=[x for x in row['iou'].values() if x is not None];return sum(v)/len(v) if v else 0
ranked=sorted([row for row in rows if any(v is not None for v in row['iou'].values())],key=mean);chosen=[ranked[0],ranked[len(ranked)//3],ranked[2*len(ranked)//3],ranked[-1]]
canvas=Image.new('RGB',(1050,4*290+90),'#f7f8f4');d=ImageDraw.Draw(canvas)
d.text((20,15),'PlateBack | Held-out examples: lowest, 1/3, 2/3, highest image mean IoU',fill='#203e35')
for x,t in [(10,'Photograph'),(360,'Independent publisher reference'),(710,'Raw local prediction')]:d.text((x,45),t,fill='#203e35')
for i,row in enumerate(chosen):
    name=row['id'];im=Image.open(ROOT/f'data/datasets/plateback/images/test/{name}.jpg').convert('RGB');gt=np.array(Image.open(ROOT/f'data/datasets/plateback/masks/test/{name}.png'));pred=np.load(OUT/f'predictions/{name}.npz')
    imgs=[im]
    for masks in [[gt==c+1 for c in range(4)],[pred[l] for l in r['per_class']]]:
        a=np.array(im).astype(float)
        for mask,color in zip(masks,colors):a[mask]=a[mask]*.55+np.array(color)*.45
        imgs.append(Image.fromarray(a.astype('uint8')))
    for j,img in enumerate(imgs):
        img.thumbnail((330,240));canvas.paste(img,(j*350+10,i*290+75))
    d.text((10,i*290+320),f'{name} | image mean IoU {mean(row):.3f}',fill='#203e35')
canvas.save(OUT/'examples.jpg')
