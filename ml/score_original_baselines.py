"""Evaluate untouched COCO YOLO; report original DINO encoder as unsupported, not zero."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ultralytics import YOLO
from score_benchmark import pix,presence_metrics,mask_metrics,LABELS
OUT=ROOT/'artifacts/benchmark';torch.set_num_threads(2)
def main():
 weights=ROOT/'yolo11n-seg.pt';model=YOLO(str(weights));mapping={i:LABELS.index(name) for i,name in model.names.items() if name in LABELS};assert set(model.names.values()) & set(LABELS)=={'broccoli'}
 manifest=json.loads((OUT/'manifest.json').read_text());rows={};identity=hashlib.sha256(weights.read_bytes()).hexdigest();cache=OUT/'original-yolo';cache.mkdir(exist_ok=True)
 for task in ['presence','segmentation']:
  rows[task]=[]
  for r in manifest[task]:
   saved=cache/(task+'-'+r['id']+'.json')
   if saved.exists():
    row=json.loads(saved.read_text());assert row['weights_sha256']==identity
   else:
    path=ROOT/r['path'];im=Image.open(path);masks=[np.zeros((im.height,im.width),bool) for _ in LABELS];result=model.predict(str(path),imgsz=320,conf=.25,device='cpu',retina_masks=True,verbose=False)[0]
    if result.masks is not None:
     for mask,box in zip(result.masks.data.cpu().numpy(),result.boxes):
      cls=int(box.cls.item())
      if cls in mapping:masks[mapping[cls]]|=mask>.5
    row={'id':r['id'],'status':'completed','weights_sha256':identity,'prediction':{l:bool(masks[c].any()) for c,l in enumerate(LABELS)}}
    if task=='presence':row['truth']=r['truth']
    else:row['pixels']=pix(np.array(Image.open(ROOT/r['truth_path'])),masks)
    saved.write_text(json.dumps(row))
   rows[task].append(row)
 pm=presence_metrics(rows['presence']);sm=mask_metrics(rows['segmentation']);entry={'model':'Original YOLO11n · COCO','presence_f1':pm['macro_f1'],'mask_iou':sm['macro_iou'],'rice':'unsupported','chicken':'unsupported','cost_usd':0,'coverage':'Untouched pretrained weights; broccoli only. Other target classes are absent.','supported_class_iou':sm['per_class']['broccoli']['iou']}
 encoder={'model':'Original DINOv2 · encoder','presence_f1':None,'mask_iou':None,'rice':'no food head','chicken':'no food head','cost_usd':0,'coverage':'V4/v9 starting encoder; no original food classifier or mask decoder. Score unavailable.'}
 s=json.loads((OUT/'summary.json').read_text());s['models']=[m for m in s['models'] if m['model'] not in [entry['model'],encoder['model']]]+[entry,encoder];(OUT/'summary.json').write_text(json.dumps(s,indent=2))
 report={'weights_sha256':identity,'class_mapping':mapping,'presence':pm,'segmentation':sm,'rows':rows,'configuration':{'imgsz':320,'confidence':.25,'threshold_tuning':False},'limitation':'Only broccoli overlaps COCO. Four-food mean includes zero predictions for unsupported targets; not a like-for-like pretrained food specialist. Original DINOv2 cannot be scored without adding a trained/task-specific head.'};(OUT/'original-baselines.json').write_text(json.dumps(report,indent=2));print(entry)
if __name__=='__main__':main()
