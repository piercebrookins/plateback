import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from backend.app import vision
rows=json.loads((ROOT/'ml/manifest.json').read_text());rows=[r for r in rows if r['split']=='test' and len(r['classes'])>=2]
report=[]
for row in rows[:5]:
    p=ROOT/'data/datasets/plateback/images/test'/f'{row["id"]}.jpg';start=time.perf_counter()
    regions,meta=vision.local_segment(p);report.append({'id':row['id'],'seconds_including_adapter':time.perf_counter()-start,'metadata':meta,'regions':regions})
(ROOT/'artifacts/local-smoke.json').write_text(json.dumps(report,indent=2));print([(r['id'],len(r['regions']),round(r['seconds_including_adapter'],3)) for r in report])
