"""Evaluate the frozen boundary policy on a new chicken-focused test; no model selection here."""
import sys,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_semantic import metrics,one,LABELS,baseline_rows
from evaluate_v2 import collect
from backend.app.local_semantic import predict as semantic
from backend.app.local_ensemble import predict as ensemble
from backend.app import budget
import score_semantic_benchmark as shared
OUT=ROOT/'artifacts/benchmark-chicken-final';DATA=ROOT/'data/datasets/plateback-chicken-final'
def evaluate(name,folder,version):
    path=OUT/(name+'.json')
    if path.exists():return json.loads(path.read_text())
    policy=json.loads((folder/'settings.json').read_text());rows=[];times=[]
    for p in sorted((DATA/'images/test').glob('*.jpg')):
        gt=np.array(Image.open(DATA/'masks/test'/(p.stem+'.png')));masks=[np.zeros_like(gt,bool) for _ in LABELS];start=time.perf_counter();pred=ensemble(p,folder,policy)[0] if version=='v3' else semantic(p,folder,policy)
        for c,s,m in pred:masks[c]|=m
        times.append(time.perf_counter()-start);rows.append({'id':p.stem,'negative':not bool(gt.any()),'classes':{l:one(gt==c+1,masks[c]) for c,l in enumerate(LABELS)}})
    report=metrics(rows);report['latency_median_seconds']=float(np.median(times));report['settings']=policy;path.write_text(json.dumps(report,indent=2));return report

def bootstrap(a,b):
    assert [r['id'] for r in a]==[r['id'] for r in b]
    rng=np.random.default_rng(151);out={k:[] for k in ['iou','recall','pixel_recall']}
    for _ in range(2000):
        ix=rng.integers(0,len(a),len(a));ma=metrics([a[i] for i in ix])['per_class']['chicken'];mb=metrics([b[i] for i in ix])['per_class']['chicken']
        for k in out:out[k].append(ma[k]-mb[k])
    return {k+'_difference_95_percentile_interval':np.percentile(v,[2.5,97.5]).tolist() for k,v in out.items()}

def main():
    torch.set_num_threads(4);assert (ROOT/'artifacts/evaluation-v5/validation.json').exists(),'Validate serving and freeze policy first'
    all_results={}
    all_results['Original YOLO v1']=metrics(baseline_rows(collect(ROOT/'artifacts/models/baseline-v1/best.pt',DATA,'test',320,'baseline-chicken-final')))
    for name,folder,v in [('Previous ensemble v3','ensemble-v3','v3'),('Semantic v4 candidate','semantic-v4','v4'),('Boundary-refined v5','semantic-v5','v5')]:all_results[name]=evaluate(folder,ROOT/'artifacts/models'/folder,v)
    manifest=json.loads((OUT/'manifest.json').read_text());ids=[r['id'] for r in manifest['segmentation']];shared.OUT=OUT;cloud,cost,valid=shared.cloud_rows(manifest);paired={}
    for name,v in all_results.items():
        byid={r['id']:r for r in v['per_image']};paired[name]=metrics([byid[id] for id in ids])
    paired['Gemini 4 Argon']=metrics(cloud);ci=bootstrap(paired['Boundary-refined v5']['per_image'],cloud);oldci=bootstrap(all_results['Boundary-refined v5']['per_image'],all_results['Previous ensemble v3']['per_image'])
    final=all_results['Boundary-refined v5']['per_class']['chicken'];refs=[all_results[n]['per_class']['chicken'] for n in ['Original YOLO v1','Previous ensemble v3']]
    decision={'presence_recall_preserved':final['recall']>=max(r['recall'] for r in refs),'pixel_recall_preserved':final['pixel_recall']>=max(r['pixel_recall'] for r in refs),'false_positives_not_increased':final['fp']<=all_results['Previous ensemble v3']['per_class']['chicken']['fp'],'iou_not_decreased':final['iou']>=all_results['Previous ensemble v3']['per_class']['chicken']['iou']}
    regression=json.loads((ROOT/'artifacts/evaluation-v5/regression-110.json').read_text())['per_class']['chicken'];old_regression=json.loads((ROOT/'artifacts/evaluation-v4/ensemble-fresh.json').read_text())['per_class']['chicken']
    decision.update(regression_presence_recall_preserved=regression['recall']>=old_regression['recall'],regression_pixel_recall_preserved=regression['pixel_recall']>=old_regression['pixel_recall'],regression_false_positives_not_increased=regression['fp']<=old_regression['fp'])
    presence=json.loads((ROOT/'artifacts/benchmark/detailed-results.json').read_text());np5=presence['Local semantic v5']['presence']['per_class']['chicken'];np3=presence['Local ensemble v3']['presence']['per_class']['chicken'];decision['nutrition_presence_recall_preserved']=np5['recall']>=np3['recall'];decision['promotion_eligible']=all(decision.values())
    details={'full':all_results,'paired':paired,'gemini_difference':ci,'v3_difference':oldci};(OUT/'detailed-results.json').write_text(json.dumps(details,indent=2))
    def compact(r):return {'images':r['images'],'chicken':r['per_class']['chicken'],'latency_median_seconds':r.get('latency_median_seconds')}
    summary={'active_model':'v3 retained','regression_caveat':f'On the earlier 110-image regression set, v5 chicken pixel recall is {regression["pixel_recall"]:.1%} versus v3 {old_regression["pixel_recall"]:.1%}. Nutrition5k chicken presence recall is also 30% versus v3 40% on ten positive images. V3 remains active to honor the no-recall-loss requirement. V5 is experimental.','status':'completed','full_images':32,'positive_images':16,'hard_negative_images':16,'paired_images':16,'full_evaluation':{n:compact(v) for n,v in all_results.items()},'paired_evaluation':{n:compact(v) for n,v in paired.items()},'gemini_difference':ci,'v3_difference':oldci,'gemini_valid_outputs':valid,'gemini_cost_usd':cost,'accounted_total_usd':budget.summary()['accounted_usd'],'deployment_gate':decision,'note':'New unused publisher-test images; chicken/duck proxy. Other foods lack balanced coverage, so this does not establish overall four-food superiority. Earlier 110/32-image sets are regression evidence for v5. Public pretraining overlap and real leftover performance remain unknown.'};(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(ROOT/'docs/chicken-final-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
