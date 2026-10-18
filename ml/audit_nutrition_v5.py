"""Per-image Nutrition5k miss diagnostics, without changing labels or training on test."""
import json,sys
from pathlib import Path
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.local_semantic import probabilities,class_mask
OUT=ROOT/'artifacts/evaluation-v6';OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'nutrition-probabilities';cache.mkdir(exist_ok=True)
folder=ROOT/'artifacts/models/semantic-v5';rule=json.loads((folder/'settings.json').read_text())['per_class']['chicken'];torch.set_num_threads(2);rows=[];tiles=[]
for r in json.loads((ROOT/'ml/nutrition-manifest.json').read_text()):
 if r['split']!='test':continue
 path=ROOT/r['path'];im=Image.open(path).convert('RGB');saved=cache/(r['id']+'.npy')
 if not saved.exists():np.save(saved,probabilities(path,folder)[4])
 prob=np.load(saved);mask,p=class_mask(prob,(im.height,im.width),rule);_,_,stats,_=cv2.connectedComponentsWithStats((p>=.75).astype('uint8'),8);largest=float(stats[1:,cv2.CC_STAT_AREA].max()/p.size) if len(stats)>1 else 0
 true='chicken' in r['labels'];row={'id':r['id'],'ingredients':r['ingredients'],'truth_ingredient_chicken':true,'predicted':bool(mask.any()),'max_probability':float(p.max()),'largest_seed_area_fraction':largest,'miss_reason':('area_filter' if largest>0 else 'no_seed_above_0.75') if true and not mask.any() else None};rows.append(row)
 if true:
  im.thumbnail((300,230));tile=Image.new('RGB',(320,310),'white');tile.paste(im,((320-im.width)//2,0));d=ImageDraw.Draw(tile);d.text((5,235),r['id'],fill='black');d.text((5,253),f"{'HIT' if mask.any() else 'MISS'} | seed {largest:.3%} | max {p.max():.2f}",fill='black');d.text((5,273),', '.join(r['ingredients'])[:49],fill='black');tiles.append(tile)
(OUT/'nutrition-v5-miss-audit.json').write_text(json.dumps(rows,indent=2));sheet=Image.new('RGB',(320*5,310*2),'#ddd')
for i,t in enumerate(tiles):sheet.paste(t,((i%5)*320,(i//5)*310))
sheet.save(OUT/'nutrition-chicken-audit.jpg');print(json.dumps([r for r in rows if r['truth_ingredient_chicken']],indent=2))
