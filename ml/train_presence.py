"""Local transfer learning using ingredient labels, without inventing segmentation masks."""
import json,time,sys,hashlib,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/models/presence-v1';OUT.mkdir(parents=True,exist_ok=True);LABELS=['rice','pasta','broccoli','chicken']
def main():
 torch.set_num_threads(4);torch.manual_seed(93);np.random.seed(93);random.seed(93);start=time.perf_counter()
 source=json.loads((ROOT/'ml/manifest-v2.json').read_text());nutrition=json.loads((ROOT/'ml/nutrition-manifest.json').read_text())
 rows=[{'id':r['id'],'split':r['split'],'path':f'data/datasets/plateback-v2/images/{r["split"]}/{r["id"]}.jpg','labels':[LABELS[c] for c in r['classes']]} for r in source if r['split'] in ['train','val']]
 rows += [r for r in nutrition if r['split']=='train']
 weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1;backbone=mobilenet_v3_small(weights=weights).eval();transform=weights.transforms()
 features=[]
 with torch.no_grad():
  for offset in range(0,len(rows),16):
   batch=torch.stack([transform(Image.open(ROOT/r['path']).convert('RGB')) for r in rows[offset:offset+16]])
   features.append(backbone.avgpool(backbone.features(batch)).flatten(1))
 x=torch.cat(features);y=torch.tensor([[float(l in r['labels']) for l in LABELS] for r in rows]);tr=torch.tensor([r['split']=='train' for r in rows]);va=~tr
 head=nn.Sequential(nn.Dropout(.2),nn.Linear(x.shape[1],4));optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.02);loss=nn.BCEWithLogitsLoss();best=float('inf');best_state=None;history=[];stale=0
 for epoch in range(1,201):
  head.train();order=torch.randperm(int(tr.sum()));tx=x[tr];ty=y[tr]
  for batch in order.split(64):optimizer.zero_grad();value=loss(head(tx[batch]),ty[batch]);value.backward();optimizer.step()
  head.eval()
  with torch.no_grad():vl=float(loss(head(x[va]),y[va]));history.append({'epoch':epoch,'validation_loss':vl})
  if vl<best-1e-5:best=vl;best_state={k:v.clone() for k,v in head.state_dict().items()};best_epoch=epoch;stale=0
  else:stale+=1
  if stale>=25:break
 head.load_state_dict(best_state);head.eval()
 with torch.no_grad():probs=head(x[va]).sigmoid().numpy()
 truth=y[va].numpy().astype(bool);thresholds=[];calibration={}
 for c,l in enumerate(LABELS):
  choices=[]
  for threshold in np.arange(.1,.91,.05):
   p=probs[:,c]>=threshold;g=truth[:,c];tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum());recall=tp/max(1,tp+fn);f=1.25*tp/max(1,1.25*tp+fp+.25*fn);choices.append({'threshold':float(threshold),'tp':tp,'fp':fp,'fn':fn,'recall':recall,'f0_5':f})
  chosen=max([a for a in choices if a['recall']>=.5] or choices,key=lambda a:a['f0_5']);thresholds.append(chosen['threshold']);calibration[l]={'selected':chosen,'candidates':choices}
 torch.save({'backbone':backbone.state_dict(),'head':head.state_dict(),'thresholds':thresholds},OUT/'best.pt')
 meta={'architecture':'MobileNetV3 Small frozen ImageNet features + locally trained multilabel linear head','training_images':int(tr.sum()),'nutrition5k_train_images':sum(r['split']=='train' for r in nutrition),'validation_images':int(va.sum()),'device':'local CPU','seconds':time.perf_counter()-start,'best_epoch':best_epoch,'epochs_completed':epoch,'checkpoint_selection':'minimum validation BCE','threshold_selection':'validation presence F0.5 with recall >= .5','classes':LABELS,'thresholds':thresholds,'weights_sha256':hashlib.sha256((OUT/'best.pt').read_bytes()).hexdigest(),'base_weights_url':weights.url,'limitation':'Presence only; does not produce masks. FoodSeg training poultry label includes duck. No test images used.'}
 (OUT/'metadata.json').write_text(json.dumps(meta,indent=2));(OUT/'calibration.json').write_text(json.dumps(calibration,indent=2));(OUT/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
