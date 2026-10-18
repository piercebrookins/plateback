"""Produce a comparison report and transparent examples after the frozen test run."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.mask_filter import keep_mask,LABELS
OUT=ROOT/'artifacts/evaluation-v2'
def read(name):return json.loads((OUT/(name+'.json')).read_text())
b=read('baseline-fresh');n=read('candidate-fresh');bv=read('baseline-validation');nv=read('candidate-validation')
# Paired resampling over test images; not a deployment confidence guarantee.
rng=np.random.default_rng(77);boot=[]
for _ in range(1000):
    ids=rng.integers(0,len(n['per_image']),len(n['per_image']));scores=[]
    for r in [b,n]:
        cls=[]
        for l in LABELS:
            inter=sum(r['per_image'][i]['classes'][l]['intersection'] for i in ids);union=sum(r['per_image'][i]['classes'][l]['union'] for i in ids);cls.append(inter/max(1,union))
        scores.append(np.mean(cls))
    boot.append(scores[1]-scores[0])
ci=np.percentile(boot,[2.5,97.5]).tolist();(OUT/'bootstrap.json').write_text(json.dumps({'seed':77,'replicates':1000,'paired_macro_iou_difference_95_percentile_interval':ci,'unit':'image; source grouping unavailable'},indent=2))
lines=['# Local model v2 evaluation','', 'Both versions are evaluated on the same 188 newly reserved publisher test images. The v2 checkpoint and confidence/area policy were selected using validation only and frozen before test inference. The original 88-image test is reported separately as a regression set.','', '| Food | v1 IoU | v2 IoU | v1 precision | v2 precision | v1 recall | v2 recall | v1 false positives | v2 false positives |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for l in LABELS:
    x=b['per_class'][l];y=n['per_class'][l];name='Chicken/duck proxy' if l=='chicken' else l.title();lines.append(f'| {name} | {x["iou"]:.1%} | {y["iou"]:.1%} | {x["precision"]:.1%} | {y["precision"]:.1%} | {x["recall"]:.1%} | {y["recall"]:.1%} | {x["fp"]} | {y["fp"]} |')
lines+=['',f'Macro category IoU: **{b["macro_iou"]:.1%} → {n["macro_iou"]:.1%}**. Other-food negative images with any false target: **{b["negative_food_false_positive_images"]}/{b["negative_food_images"]} → {n["negative_food_false_positive_images"]}/{n["negative_food_images"]}**.',f'Paired image bootstrap (1,000 resamples, seed 77) gives a 95% percentile interval of {ci[0]*100:.1f} to {ci[1]*100:.1f} percentage points for the macro-IoU change. This reflects this finite image sample, not cafeteria deployment or source-group uncertainty.','', '## Protocol and interpretation','', '- IoU is intersection/union of all semantic pixels per category, pooled across the set. Misses and false masks both count; absent/absent categories do not inflate the score.', '- Precision and recall are image/category presence metrics, not object-detection AP. False-positive counts are images with a predicted class absent from the reference.', '- Ground truth is independent publisher annotation. Chicken means the source chicken/duck proxy; chicken-only accuracy remains unknown. Other-food negatives do not establish empty-plate performance.', '- Policy search: confidence .15/.2/.3/.4/.5/.6/.7/.8 and minimum whole-image mask area 0/.002/.005/.01. Per class, maximize .55 × pooled IoU + .45 × presence F0.5 preferring validation recall at least .50 when feasible. The searched v2 chicken policies could not reach that target; the fallback maximized the objective and selected 38.2% validation recall. This deliberately trades some sensitivity for fewer false detections.',f'- Expanded validation macro IoU: {bv["macro_iou"]:.1%} → {nv["macro_iou"]:.1%}. Validation results are selection evidence, not unbiased generalization estimates.', '- The model and API do not infer food weight. Unmeasured weight remains null/unknown.','', '## Frozen policy','', '```json',json.dumps(n['settings'],indent=2),'```','', '## Runtime','',f'CPU prediction-only latency (single run, 4 PyTorch threads): v1 median {b["latency_median_seconds"]*1000:.1f} ms / p95 {b["latency_p95_seconds"]*1000:.1f} ms; v2 median {n["latency_median_seconds"]*1000:.1f} ms / p95 {n["latency_p95_seconds"]*1000:.1f} ms. These exclude app contour conversion, network, and cold import costs.','', '## Artifacts and reproduction','', '`artifacts/evaluation-v2/` contains validation, calibration candidates, both fresh-test comparisons, both regression comparisons, cache identities, per-image metrics, and bootstrap results. `artifacts/models/baseline-v1/` preserves rollback weights and settings. Candidate training metadata and SHA-256 are beside `artifacts/models/candidate-v2/best.pt`.','', '```sh','.venv/bin/python ml/download.py','.venv/bin/python ml/prepare_v2.py','.venv/bin/python ml/train_v2.py','.venv/bin/python ml/evaluate_v2.py','.venv/bin/python ml/report_v2.py','```','', 'Evaluation caches are bound to checkpoint hash/input size/split; a changed checkpoint requires a fresh cache directory. Reusing this test set for another model iteration makes it a regression benchmark. A future independent cafeteria set, including look-alike grains, bread, potatoes, fish and verified empty plates, is still needed.','']
(ROOT/'docs/EVALUATION_V2.md').write_text('\n'.join(lines))
portable={name:{k:v for k,v in read(name).items() if k!='per_image'} for name in ['baseline-validation','candidate-validation','baseline-fresh','candidate-fresh','baseline-regression','candidate-regression']};(ROOT/'docs/evaluation-v2-summary.json').write_text(json.dumps(portable,indent=2))
# Explicitly show a corrected negative, a remaining negative failure, and two fixed-ID positives.
rows=n['per_image'];br={r['id']:r for r in b['per_image']}
def neg(r):return not any(c['present'] for c in r['classes'].values())
def predicted(r):return any(c['predicted'] for c in r['classes'].values())
chosen=[]
for title,predicate in [('Corrected other-food false positive',lambda r:neg(r) and not predicted(r) and predicted(br[r['id']])),('Remaining other-food false positive',lambda r:neg(r) and predicted(r)),('First rice-positive test ID',lambda r:r['classes']['rice']['present']),('First poultry-positive test ID',lambda r:r['classes']['chicken']['present'])]:
    matches=[r for r in rows if predicate(r)]
    if matches:chosen.append((title,matches[0]))
colors=[(220,171,71),(216,121,72),(88,165,130),(172,129,192)];canvas=Image.new('RGB',(1280,310*len(chosen)+65),'#f8f8f2');draw=ImageDraw.Draw(canvas)
for j,label in enumerate(['Photo','Publisher reference','v1','v2 frozen policy']):draw.text((j*320+12,15),label,fill='#24261f')
for i,(title,row) in enumerate(chosen):
    id=row['id'];im=Image.open(ROOT/f'data/datasets/plateback-v2/images/test/{id}.jpg').convert('RGB');gt=np.array(Image.open(ROOT/f'data/datasets/plateback-v2/masks/test/{id}.png'));sets=[[gt==c+1 for c in range(4)]]
    for tag,settings in [('baseline',b['settings']),('candidate',n['settings'])]:
        cache=np.load(OUT/f'cache/{tag}-fresh/{id}.npz');masks=[np.zeros_like(gt,bool) for _ in LABELS]
        for c,s,m in zip(cache['cls'],cache['score'],cache['masks']):
            if keep_mask(int(c),float(s),m,settings):masks[int(c)]|=m
        sets.append(masks)
    images=[im.copy()]
    for masks in sets:
        a=np.array(im).astype(float)
        for mask,color in zip(masks,colors):a[mask]=a[mask]*.55+np.array(color)*.45
        images.append(Image.fromarray(a.astype('uint8')))
    for j,img in enumerate(images):img.thumbnail((300,245));canvas.paste(img,(j*320+12,i*310+50))
    draw.text((12,i*310+300),id+' | '+title,fill='#24261f')
canvas.save(OUT/'comparison.jpg');print('Wrote report, portable summary, bootstrap and comparison image.')
