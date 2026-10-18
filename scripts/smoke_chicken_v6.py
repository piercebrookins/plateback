"""Check actual candidate callable matches frozen evaluation cache; no paid calls."""
import sys,json,hashlib
from pathlib import Path
import torch,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.chicken_candidate import predict,combine,score
from backend.app.local_semantic import class_mask
folder=ROOT/'artifacts/models/chicken-v6';policy=json.loads((folder/'settings.json').read_text());base=json.loads((ROOT/'artifacts/models/semantic-v5/settings.json').read_text());rule=policy['rescue'];torch.set_num_threads(2);rows=[r for r in json.loads((ROOT/'ml/nutrition-v6-manifest.json').read_text()) if r['split']=='test'];chosen=[next(r for r in rows if 'chicken' in r['labels']),next(r for r in rows if 'chicken' not in r['labels'])];results=[]
for r in chosen:
 path=ROOT/r['path'];key=hashlib.sha256(path.read_bytes()).hexdigest();cache=ROOT/'artifacts/evaluation-v6/test-cache';p=np.load(cache/(key+'-semantic-v5.npy'));q=np.load(cache/(key+'-'+rule['source']+'.npy'));f=torch.from_numpy(np.load(cache/(key+'-features.npy')));s=float(score(f,folder,rule['method'])[0]);im=Image.open(path);shape=(im.height,im.width);expected=[class_mask(p[c+1],shape,base['per_class'][l])[0] for c,l in enumerate(['rice','pasta','broccoli','chicken'])];expected[3]=combine(p[4],q[4],shape,base['per_class']['chicken'],rule['rule'],s,rule['threshold']);actual=[np.zeros(shape,bool) for _ in range(4)]
 for c,confidence,m in predict(path,folder,policy):actual[c]|=m
 assert all(np.array_equal(a,b) for a,b in zip(expected,actual));results.append({'id':r['id'],'all_four_masks_exact':True})
assert hashlib.sha256((ROOT/'artifacts/models/plateback/best.pt').read_bytes()).hexdigest()==hashlib.sha256((ROOT/'artifacts/models/ensemble-v3/best.pt').read_bytes()).hexdigest()
(ROOT/'artifacts/evaluation-v6/serving-smoke.json').write_text(json.dumps({'cases':results,'active_v3_unchanged':True,'paid_calls':0},indent=2));print('Candidate serving matches cached evaluation; active v3 unchanged')
