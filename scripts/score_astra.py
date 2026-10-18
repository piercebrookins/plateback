"""Same-mask, same-prompt diagnostic; all raw responses retained."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/astra-comparison'
manifest=json.loads((OUT/'manifest.json').read_text());labels=['rice','pasta','broccoli','chicken'];scores={}
def masks(regions,shape):
 h,w=shape;out=[np.zeros(shape,bool) for _ in labels]
 for r in regions:
  if r['label'] not in labels:continue
  m=Image.new('L',(w,h));d=ImageDraw.Draw(m);d.polygon([(x*w,y*h) for x,y in r['polygon']],fill=1)
  for hole in r.get('holes',[]):d.polygon([(x*w,y*h) for x,y in hole],fill=0)
  out[labels.index(r['label'])]|=np.array(m)>0
 return out
for model in ['luna','astra']:
 counts={l:dict(intersection=0,union=0,tp=0,fp=0,fn=0) for l in labels};rows=[]
 for row in manifest['segmentation']:
  file=(ROOT/'artifacts/benchmark-openai-v7-normalized' if model=='luna' else OUT)/(row['id']+'.json')
  result=json.loads(file.read_text());gt=np.array(Image.open(ROOT/row['truth_path']));ps=masks(result.get('regions',[]),gt.shape)
  for i,l in enumerate(labels):
   t=gt==i+1;p=ps[i];c=counts[l];c['intersection']+=int((t&p).sum());c['union']+=int((t|p).sum());c['tp']+=int(t.any() and p.any());c['fp']+=int(not t.any() and p.any());c['fn']+=int(t.any() and not p.any())
  rows.append({'id':row['id'],'labels':sorted(set(r['label'] for r in result.get('regions',[]))),'status':result.get('status')})
 for c in counts.values():c['iou']=c['intersection']/max(1,c['union']);c['recall']=c['tp']/max(1,c['tp']+c['fn'])
 scores[model]={'per_class':counts,'macro_iou':sum(c['iou'] for c in counts.values())/4,'rows':rows}
scores['scope']='Six reused regression images, 3 chicken-positive and 3 negative. Same scanner prompt and 512px image encoding. Luna uses no reasoning and 2000 output tokens; Astra low reasoning and 4000. Not the 12/38 main benchmark.'
(OUT/'scores.json').write_text(json.dumps(scores,indent=2));print(json.dumps(scores,indent=2))
