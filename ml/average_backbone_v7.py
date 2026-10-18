import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];a=torch.load(ROOT/'artifacts/models/mil-v6/best.pt',map_location='cpu',weights_only=True);b=torch.load(ROOT/'artifacts/models/backbone-v7/best.pt',map_location='cpu',weights_only=True)
state={k:(v*.5+a['state_dict'][k]*.5 if v.is_floating_point() else v) for k,v in b['state_dict'].items()};out=ROOT/'artifacts/models/backbone-avg-v7';out.mkdir(exist_ok=True);torch.save({**b,'state_dict':state},out/'best.pt');(out/'metadata.json').write_text(json.dumps({'method':'50/50 parameter interpolation','sources':['mil-v6','backbone-v7'],'sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()},indent=2))
