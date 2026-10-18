"""Evaluate validation-selected v7 refinement on previously seen regression sets."""
import json,sys,hashlib,os
from pathlib import Path
import torch,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.local_semantic import probabilities
from backend.app.chicken_candidate import combine,score
from backend.app.chicken_refinement import refine
from evaluate_semantic import metrics,one
from backend.app import budget
CANDIDATE=os.getenv('REFINEMENT_MODEL');VERSION='v9' if CANDIDATE=='spatial-v9' else 'v8' if CANDIDATE else 'v7';OUT=ROOT/'artifacts'/('evaluation-'+CANDIDATE if CANDIDATE else 'evaluation-v7');torch.set_num_threads(2)
def main():
 candidates=[]
 for name,ev in ([(CANDIDATE,'evaluation-'+CANDIDATE)] if CANDIDATE else [('backbone-v7','evaluation-v7'),('backbone-avg-v7','evaluation-backbone-avg-v7')]):
  folder=ROOT/'artifacts/models'/name;f=json.loads((ROOT/'artifacts'/ev/'freeze.json').read_text());assert hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()==f['weights_sha256'];assert hashlib.sha256((folder/'settings.json').read_bytes()).hexdigest()==f['settings_sha256'];policy=json.loads((folder/'settings.json').read_text());candidates.append((policy['refinement']['food']['iou'],name,policy))
 _,name,policy=max(candidates,key=lambda r:r[0]);(OUT/'selected-model.json').write_text(json.dumps({'model':name,'policy':policy,'criterion':'validation IoU only'},indent=2));folder=ROOT/'artifacts/models'/name;refinement=policy['refinement'];v6=ROOT/'artifacts/models/chicken-v6';rescue=json.loads((v6/'settings.json').read_text())['rescue'];base_rule=json.loads((ROOT/'artifacts/models/semantic-v5/settings.json').read_text())['per_class']['chicken'];oldcache=ROOT/'artifacts/evaluation-v6/test-cache';oldresults=json.loads((ROOT/'artifacts/evaluation-v6/results.json').read_text());old_details=json.loads((ROOT/'artifacts/evaluation-v6/detailed-results.json').read_text());out={};detail={}
 def mask(path):
  key=hashlib.sha256(path.read_bytes()).hexdigest();p=np.load(oldcache/(key+'-semantic-v5.npy'))[4];q=np.load(oldcache/(key+'-'+rescue['source']+'.npy'))[4];sc=float(score(torch.from_numpy(np.load(oldcache/(key+'-features.npy'))),v6,rescue['method'])[0]);im=Image.open(path);base=combine(p,q,(im.height,im.width),base_rule,rescue['rule'],sc,rescue['threshold'])
  if refinement['mode']=='identity':return base
  saved=OUT/'regression-probabilities'/name/(key+'.npy');saved.parent.mkdir(parents=True,exist_ok=True)
  if not saved.exists():np.save(saved,probabilities(path,folder)[4])
  return refine(base,np.load(saved),refinement['rule'],refinement['mode'],refinement['fallback'])
 for tag,manifest in [('nutrition_new60','nutrition-v6-manifest.json'),('nutrition_old38','nutrition-manifest.json')]:
  rows=[];counts={'tp':0,'fp':0,'fn':0,'tn':0}
  for r in json.loads((ROOT/'ml'/manifest).read_text()):
   if r['split']!='test':continue
   m=mask(ROOT/r['path']);t='chicken' in r['labels'];p=bool(m.any());counts['tp' if t and p else 'fn' if t else 'fp' if p else 'tn']+=1;rows.append({'id':r['id'],'truth':t,'predicted':p})
  counts.update(recall=counts['tp']/max(1,counts['tp']+counts['fn']),precision=counts['tp']/max(1,counts['tp']+counts['fp']));out[tag]={'v6':oldresults[tag]['v6'],VERSION:counts};detail[tag]=rows;print(tag,out[tag],flush=True)
 for tag,dataset in [('food110','plateback-v4'),('chicken32','plateback-chicken-final'),('food188','plateback-v2'),('food83','plateback-final')]:
  rows=[];original={r['id']:r for r in old_details[tag]['v6']}
  for path in sorted((ROOT/'data/datasets'/dataset/'images/test').glob('*.jpg')):
   gt=np.array(Image.open(ROOT/'data/datasets'/dataset/'masks/test'/(path.stem+'.png')));m=mask(path);row=original[path.stem];rows.append({**row,'classes':{**row['classes'],'chicken':one(gt==4,m)}})
  result=metrics(rows);detail[tag]=rows;out[tag]={'v6':oldresults[tag]['v6'],VERSION:{k:v for k,v in result.items() if k!='per_image'}};print(tag,result['per_class']['chicken'],flush=True)
  if tag in ['food110','chicken32']:
   paired=old_details[tag+'_paired'];ids=[r['id'] for r in paired['v6']];byid={r['id']:r for r in rows};out[tag+'_gemini']={VERSION:metrics([byid[id] for id in ids]),'gemini':metrics(paired['gemini'])}
 gates={}
 for tag in ['nutrition_new60','nutrition_old38']:
  for metric in ['recall','fp']:gates[tag+'_'+metric]=out[tag][VERSION][metric]>=out[tag]['v6'][metric] if metric=='recall' else out[tag][VERSION][metric]<=out[tag]['v6'][metric]
 for tag in ['food110','chicken32','food188','food83']:
  a=out[tag][VERSION]['per_class']['chicken'];b=out[tag]['v6']['per_class']['chicken']
  for metric in ['recall','pixel_recall','iou','fp']:gates[tag+'_'+metric]=a[metric]>=b[metric] if metric!='fp' else a[metric]<=b[metric]
 out.update(selected_model=name,policy=policy,gates=gates,all_regression_gates_pass=all(gates.values()),refinement_applied=refinement['mode']!='identity',active_local='Evaluation does not alter deployment; see artifacts/models/local-selection.json',run_budget=budget.run_summary('hackathon-v7') if not CANDIDATE else {'new_api_usd':0,'scope':'local-only experiment'},accounted_total=budget.summary()['accounted_usd']);(OUT/'results.json').write_text(json.dumps(out,indent=2));(OUT/'detailed-results.json').write_text(json.dumps(detail,indent=2));print(gates,flush=True)
if __name__=='__main__':main()
