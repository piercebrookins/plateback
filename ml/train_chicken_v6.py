"""Local linear/MLP presence and weak-label MIL dense-head experiments. No test access."""
import json,sys,time,hashlib,random
from pathlib import Path
import torch,numpy as np
from PIL import Image
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.local_semantic import load
from backend.app.semantic_model import normalize
from train_semantic import loss_fn
OUT=ROOT/'artifacts/models/chicken-v6';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(161);random.seed(161);start=time.time()
model,_=load(ROOT/'artifacts/models/semantic-v5');model.eval();manifest=json.loads((ROOT/'ml/nutrition-v6-manifest.json').read_text());food=json.loads((ROOT/'ml/manifest-v4.json').read_text());data={}
def pooled(x):return torch.cat([x.mean((2,3)),x.amax((2,3))],1)
for split in ['train','val']:
 rows=[r for r in manifest if r['split']==split];cache=OUT/('nutrition-'+split+'.npy');stamp=OUT/('nutrition-'+split+'.json');identity={'ids':[r['id'] for r in rows],'sha':hashlib.sha256((ROOT/'ml/nutrition-v6-manifest.json').read_bytes()).hexdigest(),'size':336}
 if not stamp.exists() or json.loads(stamp.read_text())!=identity:
  features=[]
  with torch.inference_mode():
   for i in range(0,len(rows),8):
    ims=[np.array(Image.open(ROOT/r['path']).convert('RGB').resize((336,336))).transpose(2,0,1).copy() for r in rows[i:i+8]];x=torch.from_numpy(np.array(ims)).float()/255
    f=model.backbone.forward_features(normalize(x))['x_norm_patchtokens'].transpose(1,2).reshape(len(x),384,24,24);features.append(f.numpy().astype('float16'))
    if i%128==0:print('nutrition features',split,i,len(rows),flush=True)
  np.save(cache,np.concatenate(features));stamp.write_text(json.dumps(identity))
 nx=torch.from_numpy(np.load(cache).astype('float32'));ny=torch.tensor([float('chicken' in r['labels']) for r in rows]);fr=[r for r in food if r['split']==split];fx=torch.from_numpy(np.load(ROOT/'artifacts/dino-features'/(split+'.npy')).astype('float32'));fy=torch.tensor([float(3 in r['classes']) for r in fr]);mask=torch.from_numpy(np.stack([np.array(Image.open(ROOT/'data/datasets/plateback-v4/masks'/split/(r['id']+'.png')).resize((96,96),Image.Resampling.NEAREST),dtype=np.int64) for r in fr]));data[split]=(nx,ny,fx,fy,mask)
# Balanced class loss, each source equally weighted, validation-only checkpoint selection.
for method in ['linear','mlp']:
 torch.manual_seed(161);clf=nn.Linear(768,1) if method=='linear' else nn.Sequential(nn.Linear(768,128),nn.GELU(),nn.Dropout(.3),nn.Linear(128,1));opt=torch.optim.AdamW(clf.parameters(),lr=.0003,weight_decay=.1);best=1e9;history=[]
 nx,ny,fx,fy,_=data['train'];x=torch.cat([pooled(nx),pooled(fx)]);y=torch.cat([ny,fy]);mean=x.mean(0);std=x.std(0).clamp_min(.1);x=(x-mean)/std
 weights=torch.cat([torch.full((len(ny),),1/len(ny)),torch.full((len(fy),),1/len(fy))]);weights*=torch.where(y>0,.5/y.mean(),.5/(1-y.mean()))
 vx,vy,vfx,vfy,_=data['val'];vsets=[((pooled(a)-mean)/std,b) for a,b in [(vx,vy),(vfx,vfy)]]
 for epoch in range(40):
  clf.train();order=torch.multinomial(weights,len(y),replacement=True)
  for ids in order.split(64):
   opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(clf(x[ids]).flatten(),y[ids]);loss.backward();opt.step()
  clf.eval()
  with torch.inference_mode():score=sum(nn.functional.binary_cross_entropy_with_logits(clf(a).flatten(),b).item() for a,b in vsets)/2
  history.append({'epoch':epoch+1,'validation_bce':score})
  if score<best:best=score;torch.save({'method':method,'state_dict':clf.state_dict(),'mean':mean,'std':std,'epoch':epoch+1},OUT/(method+'.pt'))
 (OUT/(method+'-history.json')).write_text(json.dumps(history,indent=2));print(method,'best validation BCE',best,flush=True)
# MIL: top 5% chicken-vs-other patch logits, ingredient-only weak loss plus native pixel replay.
head=model.head;opt=torch.optim.AdamW(head.parameters(),lr=.0001,weight_decay=.02);nx,ny,fx,fy,mask=data['train'];vn,vny,vf,vfy,vm=data['val'];best=1e9;history=[]
def mil(logits):
 z=logits[:,4]-torch.logsumexp(logits[:,:4],dim=1);return z.flatten(1).topk(max(1,z[0].numel()//20),dim=1).values.mean(1)
for epoch in range(12):
 head.train();order=torch.randperm(len(fx))
 for ids in order.split(16):
  ni=torch.randint(len(nx),(len(ids),));a=fx[ids];g=mask[ids];b=nx[ni]
  if random.random()<.5:a=a.flip(-1);g=g.flip(-1);b=b.flip(-1)
  opt.zero_grad();dense=head(a);loss=loss_fn(nn.functional.interpolate(dense,size=(96,96),mode='bilinear',align_corners=False),g)+.2*nn.functional.binary_cross_entropy_with_logits(mil(head(b)),ny[ni]);loss.backward();opt.step()
 head.eval()
 with torch.inference_mode():
  pixel=sum(loss_fn(nn.functional.interpolate(head(vf[i:i+16]),size=(96,96),mode='bilinear',align_corners=False),vm[i:i+16]).item()*len(vf[i:i+16]) for i in range(0,len(vf),16))/len(vf)
  weak=sum(nn.functional.binary_cross_entropy_with_logits(mil(head(vn[i:i+16])),vny[i:i+16]).item()*len(vn[i:i+16]) for i in range(0,len(vn),16))/len(vn)
 score=pixel+.2*weak;history.append({'epoch':epoch+1,'pixel_loss':pixel,'weak_loss':weak,'score':score});print('MIL',history[-1],flush=True)
 if score<best:
  best=score;folder=ROOT/'artifacts/models/mil-v6';folder.mkdir(exist_ok=True);torch.save({'architecture':'dinov2-head','state_dict':model.state_dict(),'size':448,'epoch':epoch+1,'labels':['background','rice','pasta','broccoli','chicken']},folder/'best.pt')
 (OUT/'mil-history.json').write_text(json.dumps(history,indent=2))
(OUT/'metadata.json').write_text(json.dumps({'device':'cpu','seconds':time.time()-start,'seed':161,'nutrition_training':len(nx),'food_training':len(fx),'methods':['linear pooled DINO','MLP pooled DINO','MIL dense head + pixel replay'],'backbone':'frozen DINOv2 ViT-S/14','nutrition_supervision':'ingredient presence only; no masks or mass','manifest_sha256':hashlib.sha256((ROOT/'ml/nutrition-v6-manifest.json').read_bytes()).hexdigest()},indent=2))
