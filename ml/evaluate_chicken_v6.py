"""Frozen-policy test and regression evaluation. Uses cached Gemini only; no network."""
import sys,json,hashlib,time
from pathlib import Path
import torch,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.local_semantic import probabilities,class_mask
from backend.app.chicken_candidate import feature,score,combine
from backend.app.local_ensemble import predict as ensemble
from evaluate_semantic import one,metrics,LABELS
from evaluate_chicken_final import bootstrap
import score_semantic_benchmark as shared
from backend.app import budget
OUT=ROOT/'artifacts/evaluation-v6';FOLDER=ROOT/'artifacts/models/chicken-v6'
def compact(r):return {k:v for k,v in r.items() if k!='per_image'}
def main():
 torch.set_num_threads(2);freeze=json.loads((OUT/'freeze.json').read_text());assert hashlib.sha256((FOLDER/'settings.json').read_bytes()).hexdigest()==freeze['settings_sha256']
 for p,sha in freeze['weights'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha
 cache_stamp=OUT/'test-cache-identity.json';identity={'freeze':freeze}
 if cache_stamp.exists():assert json.loads(cache_stamp.read_text())==identity,'Candidate changed: use a new evaluation directory, never mix cached predictions'
 else:cache_stamp.write_text(json.dumps(identity,indent=2))
 policy=json.loads((FOLDER/'settings.json').read_text());rescue=policy['rescue'];basepolicy=json.loads((ROOT/'artifacts/models/semantic-v5/settings.json').read_text());before=budget.summary()['accounted_usd'];start=time.time()
 def masks(path):
  cache=OUT/'test-cache';cache.mkdir(exist_ok=True);identity=hashlib.sha256(path.read_bytes()).hexdigest();fp=cache/(identity+'-features.npy')
  if not fp.exists():np.save(fp,feature(path).numpy())
  presence=float(score(torch.from_numpy(np.load(fp)),FOLDER,rescue['method'])[0]);ps={}
  for source in set(['semantic-v5',rescue['source']]):
   saved=cache/(identity+'-'+source+'.npy')
   if not saved.exists():np.save(saved,probabilities(path,ROOT/'artifacts/models'/source))
   ps[source]=np.load(saved)
  im=Image.open(path);shape=(im.height,im.width);base=[class_mask(ps['semantic-v5'][i+1],shape,basepolicy['per_class'][l])[0] for i,l in enumerate(LABELS)];candidate=list(base);candidate[3]=combine(ps['semantic-v5'][4],ps[rescue['source']][4],shape,basepolicy['per_class']['chicken'],rescue['rule'],presence,rescue['threshold']);assert not (base[3]&~candidate[3]).any()
  return base,candidate,presence
 reports={};detail={}
 for name,rs in [('nutrition_new60',[r for r in json.loads((ROOT/'ml/nutrition-v6-manifest.json').read_text()) if r['split']=='test']),('nutrition_old38',[r for r in json.loads((ROOT/'ml/nutrition-manifest.json').read_text()) if r['split']=='test'])]:
  data=[];counts={m:{'tp':0,'fp':0,'fn':0,'tn':0} for m in ['v3','v5','v6']};folder=ROOT/'artifacts/models/ensemble-v3';v3policy=json.loads((folder/'settings.json').read_text())
  for r in rs:
   path=ROOT/r['path'];base,candidate,scorevalue=masks(path);truth='chicken' in r['labels'];v3=any(c==3 for c,s,m in ensemble(path,folder,v3policy)[0]);pred={'v3':v3,'v5':bool(base[3].any()),'v6':bool(candidate[3].any())};data.append({'id':r['id'],'truth':truth,'predictions':pred,'presence_score':scorevalue})
   for m,p in pred.items():counts[m]['tp' if truth and p else 'fn' if truth else 'fp' if p else 'tn']+=1
  for v in counts.values():v.update(recall=v['tp']/max(1,v['tp']+v['fn']),precision=v['tp']/max(1,v['tp']+v['fp']))
  reports[name]=counts;detail[name]=data;print(name,counts,flush=True)
 for name,dataset in [('food110','plateback-v4'),('chicken32','plateback-chicken-final'),('food188','plateback-v2'),('food83','plateback-final')]:
  rows={m:[] for m in ['v5','v6']}
  for path in sorted((ROOT/'data/datasets'/dataset/'images/test').glob('*.jpg')):
   gt=np.array(Image.open(ROOT/'data/datasets'/dataset/'masks/test'/(path.stem+'.png')));base,candidate,sc=masks(path)
   for model,ms in [('v5',base),('v6',candidate)]:rows[model].append({'id':path.stem,'negative':not bool(gt.any()),'classes':{l:one(gt==c+1,ms[c]) for c,l in enumerate(LABELS)}})
  reports[name]={m:compact(metrics(r)) for m,r in rows.items()};detail[name]=rows;print(name,{m:r['per_class']['chicken'] for m,r in reports[name].items()},flush=True)
  if name in ['food110','chicken32']:
   bdir='benchmark-v4' if name=='food110' else 'benchmark-chicken-final';shared.OUT=ROOT/'artifacts'/bdir;manifest=json.loads((shared.OUT/'manifest.json').read_text());cloud,cost,valid=shared.cloud_rows(manifest);ids=[r['id'] for r in manifest['segmentation']];byid={r['id']:r for r in rows['v6']};paired=[byid[id] for id in ids];reports[name+'_cached_gemini']={'v6':compact(metrics(paired)),'gemini':compact(metrics(cloud)),'chicken_difference':bootstrap(paired,cloud),'scope':'Previously seen regression benchmark, not fresh evidence','new_api_cost':0};detail[name+'_paired']={'v6':paired,'gemini':cloud}
 oldref=json.loads((ROOT/'artifacts/evaluation-v4/ensemble-fresh.json').read_text())['per_class']['chicken'];c=reports['food110']['v6']['per_class']['chicken'];gates={'validation_eligible':policy['validation_eligible'],'old_nutrition_recall_vs_v3':reports['nutrition_old38']['v6']['recall']>=reports['nutrition_old38']['v3']['recall'],'new_nutrition_recall_vs_v3':reports['nutrition_new60']['v6']['recall']>=reports['nutrition_new60']['v3']['recall'],'new_nutrition_fp_vs_v3':reports['nutrition_new60']['v6']['fp']<=reports['nutrition_new60']['v3']['fp'],'food110_pixel_recall_vs_v3':c['pixel_recall']>=oldref['pixel_recall'],'food110_presence_recall_vs_v3':c['recall']>=oldref['recall'],'food110_fp_vs_v3':c['fp']<=oldref['fp'],'fresh_segmentation_superiority_established':False}
 for dataset,reference in [('chicken32','artifacts/benchmark-chicken-final/ensemble-v3.json'),('food188','artifacts/evaluation-v3/ensemble-regression.json'),('food83','artifacts/evaluation-v3/ensemble-final.json')]:
  ref=json.loads((ROOT/reference).read_text())['per_class']['chicken'];current=reports[dataset]['v6']['per_class']['chicken']
  # Earlier metrics store intersection/truth_pixels even where pixel_recall was not named.
  pixel=ref.get('pixel_recall',ref['intersection']/max(1,current['truth_pixels']))
  for metric,floor in [('recall',ref['recall']),('pixel_recall',pixel)]:gates[dataset+'_'+metric+'_vs_v3']=current[metric]>=floor
  gates[dataset+'_fp_vs_v3']=current['fp']<=ref['fp']
 reports.update(deployment_gate=gates,active_model='v3 unchanged',promotion_eligible=all(gates.values()),policy=policy,accounted_total_usd=budget.summary()['accounted_usd'],new_api_cost_usd=0,seconds=time.time()-start);assert reports['accounted_total_usd']==before,'Ledger changed during evaluation; inspect independently, this script has no paid calls'
 (OUT/'results.json').write_text(json.dumps(reports,indent=2));(OUT/'detailed-results.json').write_text(json.dumps(detail,indent=2));print('gates',gates,flush=True)
if __name__=='__main__':main()
