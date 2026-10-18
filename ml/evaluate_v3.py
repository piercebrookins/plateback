"""Evaluate frozen v3 policy on untouched final subset, then previous regression sets."""
import sys,json,shutil,hashlib,time
from pathlib import Path
import torch,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ml.evaluate_v2 import collect,measure
from backend.app.local_ensemble import predict
OUT=ROOT/'artifacts/evaluation-v3';OUT.mkdir(exist_ok=True);MODEL=ROOT/'artifacts/models/ensemble-v3'
def save(name,value):
    observed=[a['iou'] for a in value['per_class'].values() if a['tp']+a['fn']>0];value['macro_iou_observed_classes']=float(np.mean(observed));value['observed_classes']=[l for l,a in value['per_class'].items() if a['tp']+a['fn']>0]
    (OUT/(name+'.json')).write_text(json.dumps(value,indent=2));print(name,{k:v for k,v in value.items() if k not in ['per_image','settings']},flush=True)
def main():
    torch.set_num_threads(4);settings=json.loads((MODEL/'settings.json').read_text())
    for src,name in [('candidate-v2/best.pt','best.pt'),('baseline-v1/best.pt','baseline.pt'),('presence-v1/best.pt','presence.pt')]:shutil.copyfile(ROOT/'artifacts/models'/src,MODEL/name)
    meta={'version':'v3','selection':'Class-specific source, confidence, area and presence gate selected only on 197 validation images. No test labels used for fitting.','components':{name:hashlib.sha256((MODEL/name).read_bytes()).hexdigest() for name in ['best.pt','baseline.pt','presence.pt']},'license':'YOLO components AGPL-3.0; Torchvision presence component BSD-3-Clause; data attribution in docs.','poultry_limitation':'Chicken/duck proxy remains.','note':'V3 was motivated by v2 test recall regression, so previous v2 and frontier sets are now exploratory regression comparisons. Final 83-image set was not previously scored; contains no pasta positives.'};(MODEL/'metadata.json').write_text(json.dumps(meta,indent=2))
    for split,data in [('final',ROOT/'data/datasets/plateback-final'),('regression',ROOT/'data/datasets/plateback-v2')]:
        baseline=collect(ROOT/'artifacts/models/baseline-v1/best.pt',data,'test',320,'v3-baseline-'+split);save('baseline-'+split,measure(baseline,{'confidence':.2,'imgsz':320}));items=[]
        for path in sorted((data/'images/test').glob('*.jpg')):
            gt=np.array(Image.open(data/'masks/test'/(path.stem+'.png')));start=time.perf_counter();masks,probs=predict(path,MODEL,settings);elapsed=time.perf_counter()-start;items.append((path.stem,gt,masks,elapsed))
        save('ensemble-'+split,measure(items,{'confidence':0,'policy':settings}))
if __name__=='__main__':main()
