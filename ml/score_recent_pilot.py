"""Add recent local candidates to the unchanged shared pilot; zero cloud requests."""
import json,sys,time,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from score_benchmark import presence_metrics,mask_metrics,pix,LABELS
from backend.app.local_semantic import probabilities,class_mask
from backend.app.chicken_candidate import score,feature,combine
from backend.app.chicken_refinement import refine
OUT=ROOT/'artifacts/benchmark';torch.set_num_threads(2)
def main():
 manifest=json.loads((OUT/'manifest.json').read_text());summary=json.loads((OUT/'summary.json').read_text());detail=json.loads((OUT/'detailed-results.json').read_text());basefolder=ROOT/'artifacts/models/semantic-v5';rules=json.loads((basefolder/'settings.json').read_text())['per_class'];v6=ROOT/'artifacts/models/chicken-v6';rescue=json.loads((v6/'settings.json').read_text())['rescue'];cache=ROOT/'artifacts/evaluation-v6/test-cache';cache.mkdir(exist_ok=True)
 names=[('Local chicken v6',None),('Local backbone v7','backbone-v7')]
 if (ROOT/'artifacts/models/hardpixels-v8/settings.json').exists():names.append(('Local hard-pixel v8 · rejected','hardpixels-v8'))
 if (ROOT/'artifacts/models/spatial-v9/settings.json').exists():
  selected=json.loads((ROOT/'artifacts/models/spatial-v9/settings.json').read_text())['refinement'];report=ROOT/'artifacts/evaluation-spatial-v9/results.json';failed=report.exists() and not json.loads(report.read_text())['all_regression_gates_pass'];names.append(('Local spatial v9'+(' · rejected' if selected['mode']=='identity' or failed else ' · candidate'),'spatial-v9'))
 results={name:([],[]) for name,_ in names}
 for task in ['presence','segmentation']:
  for r in manifest[task]:
   path=ROOT/r['path'];key=hashlib.sha256(path.read_bytes()).hexdigest();im=Image.open(path);shape=(im.height,im.width);t=time.perf_counter();arrays={}
   for folder in ['semantic-v5',rescue['source']]:
    saved=cache/(key+'-'+folder+'.npy')
    if not saved.exists():np.save(saved,probabilities(path,ROOT/'artifacts/models'/folder))
    arrays[folder]=np.load(saved)
   saved=cache/(key+'-features.npy')
   if not saved.exists():np.save(saved,feature(path).numpy())
   sc=float(score(torch.from_numpy(np.load(saved)),v6,rescue['method'])[0]);p=arrays['semantic-v5'];base=combine(p[4],arrays[rescue['source']][4],shape,rules['chicken'],rescue['rule'],sc,rescue['threshold']);others=[class_mask(p[c+1],shape,rules[l])[0] for c,l in enumerate(LABELS[:3])]
   for name,folder in names:
    m=base
    if folder:
     policy=json.loads((ROOT/'artifacts/models'/folder/'settings.json').read_text())['refinement']
     if folder=='spatial-v9' and policy['mode']=='identity':policy=max(json.loads((ROOT/'artifacts/evaluation-spatial-v9/calibration.json').read_text())['choices']+json.loads((ROOT/'artifacts/evaluation-spatial-v9/calibration-thresholds.json').read_text())['choices'],key=lambda r:r['food']['iou'])
     if folder=='hardpixels-v8':policy=max(json.loads((ROOT/'artifacts/evaluation-hardpixels-v8/calibration.json').read_text())['choices'],key=lambda r:r['food']['iou'])
     if policy['mode']!='identity':
      saved=OUT/'recent-probabilities'/folder/(key+'.npy');saved.parent.mkdir(parents=True,exist_ok=True)
      identity={'weights_sha256':hashlib.sha256((ROOT/'artifacts/models'/folder/'best.pt').read_bytes()).hexdigest()};stamp=saved.parent/'identity.json'
      if stamp.exists():assert json.loads(stamp.read_text())==identity,'Checkpoint changed; clear this probability cache first'
      else:stamp.write_text(json.dumps(identity))
      if not saved.exists():np.save(saved,probabilities(path,ROOT/'artifacts/models'/folder)[4])
      m=refine(base,np.load(saved),policy['rule'],policy['mode'],policy['fallback'])
    masks=others+[m];row={'id':r['id'],'status':'completed','cost_usd':0,'prediction':{l:bool(masks[c].any()) for c,l in enumerate(LABELS)}}
    if task=='presence':row['truth']=r['truth'];results[name][0].append(row)
    else:row['pixels']=pix(np.array(Image.open(ROOT/r['truth_path'])),masks);results[name][1].append(row)
   print(task,r['id'],flush=True)
 for name,(p,s) in results.items():
  pm=presence_metrics(p);sm=mask_metrics(s);entry={'model':name,'presence_f1':pm['macro_f1'],'mask_iou':sm['macro_iou'],'strict_mask_iou':sm['macro_iou'],'rice':f'{pm["per_class"]["rice"]["precision"]:.0%} / {pm["per_class"]["rice"]["recall"]:.0%}','chicken':f'{pm["per_class"]["chicken"]["precision"]:.0%} / {pm["per_class"]["chicken"]["recall"]:.0%}','cost_usd':0,'latency_seconds':None,'coverage':f'{len(p)} presence; {len(s)} masks','presence_ci':None};summary['models']=[m for m in summary['models'] if m['model']!=name]+[entry];detail[name]={'presence':pm,'segmentation':sm,'presence_rows':p,'segmentation_rows':s,'note':'Previously inspected regression pilot; cached inference has no measured latency. V8 shows its rejected validation-IoU diagnostic. V9 uses the validation-selected area-preserving rule, if eligible; otherwise its best validation diagnostic. No unchanged-v6 fallback is presented as new training.'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'detailed-results.json').write_text(json.dumps(detail,indent=2));print([(r['model'],r['mask_iou']) for r in summary['models']],flush=True)
if __name__=='__main__':main()
