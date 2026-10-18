"""Local-only fine tuning. Run from repo root; no hosted training calls."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=30); p.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu'); p.add_argument('--name',default='food-v1'); a=p.parse_args()
    torch.set_num_threads(4); start=time.time()
    model=YOLO('yolo11n-seg.pt')
    model.train(data=str(ROOT/'data/datasets/plateback/dataset.yaml'),epochs=a.epochs,imgsz=320,batch=8,device=a.device,workers=0,seed=42,deterministic=True,project=str(ROOT/'runs'),name=a.name,exist_ok=True,patience=10,plots=True,cache=False,amp=False,freeze=10,close_mosaic=5)
    out=ROOT/'artifacts/models/plateback'; out.mkdir(parents=True,exist_ok=True)
    shutil.copy(ROOT/f'runs/{a.name}/weights/best.pt',out/'best.pt')
    meta={'base':'yolo11n-seg.pt','classes':['rice','pasta','broccoli','chicken'],'poultry_limitation':'Source class is chicken duck; chicken-specific performance is unverified.','device':a.device,'seconds':time.time()-start,'epochs_requested':a.epochs,'seed':42,'imgsz':320,'selection':'Ultralytics validation fitness; test untouched','sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'license':'AGPL-3.0 (Ultralytics model and derivative weights)'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)); print(meta)
if __name__=='__main__':main()
