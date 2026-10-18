"""Score all prespecified new model responses; retain existing entries and Gemini archive."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from score_benchmark import cloud,presence_metrics,mask_metrics
from backend.app import budget
OUT=ROOT/'artifacts/benchmark'
def main():
 summary=json.loads((OUT/'summary.json').read_text());detail=json.loads((OUT/'detailed-results.json').read_text());protocol=json.loads((OUT/'expansion-protocol.json').read_text())
 for name in protocol['models']:
  p,s=cloud(name,normalized=True);pm=presence_metrics(p);sm=mask_metrics(s);lat=[r['latency_seconds'] for r in p if r.get('latency_seconds') is not None];cost=sum(r.get('cost_usd') or 0 for r in p+s);completep=sum(r['status']=='completed' for r in p);completes=sum(r['status']=='completed' for r in s)
  entry={'model':name,'presence_f1':pm['macro_f1'],'mask_iou':sm['macro_iou'],'rice':f'{pm["per_class"]["rice"]["precision"]:.0%} / {pm["per_class"]["rice"]["recall"]:.0%}','chicken':f'{pm["per_class"]["chicken"]["precision"]:.0%} / {pm["per_class"]["chicken"]["recall"]:.0%}','cost_usd':cost,'latency_seconds':float(np.median(lat)) if lat else None,'coverage':f'{completep}/38 presence; {completes}/12 parsed masks','presence_ci':None}
  summary['models']=[m for m in summary['models'] if m['model']!=name]+[entry];detail[name]={'presence':pm,'normalized_segmentation':sm,'presence_rows':p,'segmentation_rows':s,'cost_usd':cost,'note':'Same fixed images/prompts; low reasoning and 2400 total token cap; failed outputs count as failures.'};print(entry)
 v4=detail['Local semantic v4']['segmentation_rows'];opus=cloud('anthropic/claude-opus-5.5',normalized=True)[1];byid={r['id']:r for r in opus};opus=[byid[r['id']] for r in v4];rng=np.random.default_rng(20261018);diff=[]
 for _ in range(2000):
  ids=rng.integers(0,len(v4),len(v4));diff.append(mask_metrics([v4[i] for i in ids])['macro_iou']-mask_metrics([opus[i] for i in ids])['macro_iou'])
 comparison={'local':'Local semantic v4','cloud':'anthropic/claude-opus-5.5','images':len(v4),'difference':mask_metrics(v4)['macro_iou']-mask_metrics(opus)['macro_iou'],'paired_bootstrap_95_percentile_interval':np.percentile(diff,[2.5,97.5]).tolist(),'claim':'Point estimate on a small previously inspected pilot; not general superiority.'}
 summary.update(opus_comparison=comparison,accounted_total_usd=budget.summary()['accounted_usd'],display_exclusions=['google/gemini-4-argon']);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'detailed-results.json').write_text(json.dumps(detail,indent=2));(OUT/'opus-comparison.json').write_text(json.dumps(comparison,indent=2));print(comparison)
if __name__=='__main__':main()
