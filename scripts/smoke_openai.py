"""One real paid integration test; usage goes through the application's budget ledger."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from backend.app import db,vision,budget
from PIL import Image
import numpy as np
db.init()
rows=json.loads((ROOT/'ml/manifest.json').read_text());row=next(r for r in rows if r['split']=='val' and len(r['classes'])>=2)
p=ROOT/'data/datasets/plateback/images/val'/f'{row["id"]}.jpg';mask=np.array(Image.open(ROOT/'data/datasets/plateback/masks/val'/f'{row["id"]}.png'))
regions=[]
for cls in row['classes']:
    for poly,holes in vision.rings(mask==cls+1):regions.append({'id':f'r{len(regions)}','label':vision.LABELS[cls],'polygon':poly,'holes':holes,'score':None})
result=vision.review(p,regions[:8],'openai')
report={'image':row['id'],'input_geometry':'independent publisher reference masks; integration test only, not model evaluation','result':result,'budget':budget.summary()}
(ROOT/f'artifacts/openai-smoke-{result["model"]}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
