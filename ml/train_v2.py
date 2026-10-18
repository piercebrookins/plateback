import time,json,hashlib,shutil
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    torch.set_num_threads(4);start=time.time();base=ROOT/'artifacts/models/baseline-v1';active=ROOT/'artifacts/models/plateback'
    if not (base/'best.pt').exists():
        if not (active/'metadata.json').exists() or json.loads((active/'metadata.json').read_text()).get('imgsz')!=320:raise ValueError('Bootstrap v1 first: ml/train.py followed by ml/evaluate.py')
        base.mkdir(parents=True,exist_ok=True)
        for name in ['best.pt','metadata.json','settings.json']:shutil.copy(active/name,base/name)
    model=YOLO(str(base/'best.pt'))
    model.train(data=str(ROOT/'data/datasets/plateback-v2/dataset.yaml'),epochs=40,imgsz=416,batch=8,device='mps' if torch.backends.mps.is_available() else 'cpu',workers=0,seed=77,project=str(ROOT/'runs'),name='food-v2',exist_ok=True,patience=10,amp=False,freeze=0,optimizer='AdamW',lr0=.0005,lrf=.1,mosaic=.3,close_mosaic=8,scale=.25,fliplr=.5,plots=True)
    out=ROOT/'artifacts/models/candidate-v2';out.mkdir(parents=True,exist_ok=True);shutil.copy(ROOT/'runs/food-v2/weights/best.pt',out/'best.pt')
    (out/'metadata.json').write_text(json.dumps({'version':'v2','base':'fine-tuned v1','imgsz':416,'device':'local MPS','seconds':time.time()-start,'epochs_requested':40,'seed':77,'freeze':0,'selection':'validation fitness; no test-based selection','sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'poultry_limitation':'Source chicken/duck proxy remains; not species-specific validation.'},indent=2))
