"""Local partial DINO fine-tuning: final two transformer blocks, norm and dense head."""
import json,sys,time,random,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.local_semantic import load
from backend.app.semantic_model import normalize
from train_semantic import loss_fn
OUT=ROOT/'artifacts/models/backbone-v7';OUT.mkdir(parents=True,exist_ok=True);CACHE=ROOT/'artifacts/v7-prefix';CACHE.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(171);random.seed(171);np.random.seed(171);device='mps' if torch.backends.mps.is_available() else 'cpu';start=time.time();model,_=load(ROOT/'artifacts/models/mil-v6');model.eval();food=json.loads((ROOT/'ml/manifest-v4.json').read_text());nutrition=json.loads((ROOT/'ml/nutrition-v6-manifest.json').read_text());data={};SIZE=336
for split in ['train','val']:
 fr=[r for r in food if r['split']==split];nr=[r for r in nutrition if r['split']==split];rows=[{'id':r['id'],'path':str(ROOT/'data/datasets/plateback-v4/images'/split/(r['id']+'.jpg'))} for r in fr]+[{'id':r['id'],'path':str(ROOT/r['path'])} for r in nr];identity={'food_manifest_sha256':hashlib.sha256((ROOT/'ml/manifest-v4.json').read_bytes()).hexdigest(),'nutrition_manifest_sha256':hashlib.sha256((ROOT/'ml/nutrition-v6-manifest.json').read_bytes()).hexdigest(),'ids':[r['id'] for r in rows],'size':SIZE,'initial_sha':hashlib.sha256((ROOT/'artifacts/models/mil-v6/best.pt').read_bytes()).hexdigest()};stamp=CACHE/(split+'.json');saved=CACHE/(split+'.npy')
 if not stamp.exists() or json.loads(stamp.read_text())!=identity:
  cache=np.lib.format.open_memmap(saved,mode='w+',dtype=np.float16,shape=(len(rows),577,384))
  with torch.inference_mode():
   for i in range(0,len(rows),8):
    ims=[np.array(Image.open(r['path']).convert('RGB').resize((SIZE,SIZE),Image.Resampling.BILINEAR)).transpose(2,0,1).copy() for r in rows[i:i+8]];x=normalize(torch.from_numpy(np.array(ims)).float()/255);x=model.backbone.prepare_tokens_with_masks(x)
    for block in model.backbone.blocks[:-2]:x=block(x)
    cache[i:i+len(x)]=x.numpy().astype('float16')
    if i%256==0:print('prefix',split,i,len(rows),flush=True)
  cache.flush();stamp.write_text(json.dumps(identity,indent=2))
 tokens=np.load(saved,mmap_mode='r');masks=torch.from_numpy(np.stack([np.array(Image.open(ROOT/'data/datasets/plateback-v4/masks'/split/(r['id']+'.png')).resize((96,96),Image.Resampling.NEAREST),dtype=np.int64) for r in fr]));ny=torch.tensor([float('chicken' in r['labels']) for r in nr]);data[split]=(tokens,masks,ny,fr)
for p in model.parameters():p.requires_grad=False
for module in [*model.backbone.blocks[-2:],model.backbone.norm,model.head]:
 for p in module.parameters():p.requires_grad=True
model.to(device);opt=torch.optim.AdamW([{'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':.00001},{'params':model.head.parameters(),'lr':.0001}],weight_decay=.02);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6,eta_min=.000001)
def forward(t):
 for b in model.backbone.blocks[-2:]:t=b(t)
 f=model.backbone.norm(t)[:,1:].transpose(1,2).reshape(len(t),384,24,24);return model.head(f)
def tensor(cache,ids):return torch.from_numpy(np.array(cache[ids],dtype=np.float32)).to(device)
def mil(z):
 x=z[:,4]-torch.logsumexp(z[:,:4],1);return x.flatten(1).topk(28,1).values.mean(1)
def dense_loss(z,y):
 z=nn.functional.interpolate(z,size=(96,96),mode='bilinear',align_corners=False);loss=loss_fn(z,y)
 # Extra supervised boundary penalty discourages spillover, without invented weak-label masks.
 edge=torch.zeros_like(y,dtype=torch.bool);edge[:,1:]|=y[:,1:]!=y[:,:-1];edge[:,:,1:]|=y[:,:,1:]!=y[:,:,:-1];edge=nn.functional.max_pool2d(edge[:,None].float(),3,1,1)[:,0]
 ce=nn.functional.cross_entropy(z,y,reduction='none');return loss+.15*(ce*edge).sum()/edge.sum().clamp_min(1)
tr,ty,ny,fr=data['train'];va,vy,vny,vfr=data['val'];history=[];best=float('inf')
print('trainable parameters',sum(p.numel() for p in model.parameters() if p.requires_grad),'device',device,flush=True)
for epoch in range(6):
 t=time.time();model.train();weights=torch.tensor([1.6 if 3 in r['classes'] else (1.4 if not r['classes'] else 1) for r in fr]);order=torch.multinomial(weights,len(fr),replacement=True)
 for ids in order.split(8):
  ni=torch.randint(len(ny),(4,));opt.zero_grad(set_to_none=True);z=forward(tensor(tr,ids.numpy()));loss=dense_loss(z,ty[ids].to(device))+.2*nn.functional.binary_cross_entropy_with_logits(mil(forward(tensor(tr,(len(fr)+ni).numpy()))),ny[ni].to(device));loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step()
 model.eval();total=0;weak=0
 with torch.inference_mode():
  for i in range(0,len(vfr),8):total+=dense_loss(forward(tensor(va,slice(i,min(i+8,len(vfr))))),vy[i:i+8].to(device)).item()*len(vy[i:i+8])
  for i in range(0,len(vny),8):weak+=nn.functional.binary_cross_entropy_with_logits(mil(forward(tensor(va,slice(len(vfr)+i,len(vfr)+min(i+8,len(vny)))))),vny[i:i+8].to(device)).item()*len(vny[i:i+8])
 score=total/len(vfr)+.2*weak/len(vny);row={'epoch':epoch+1,'validation_loss':score,'seconds':time.time()-t};history.append(row);print(row,flush=True)
 if score<best:
  best=score;torch.save({'architecture':'dinov2-head','state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'size':448,'epoch':epoch+1,'labels':['background','rice','pasta','broccoli','chicken']},OUT/'best.pt')
 scheduler.step();(OUT/'history.json').write_text(json.dumps(history,indent=2))
(OUT/'metadata.json').write_text(json.dumps({'device':device,'seconds':time.time()-start,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'unfrozen':'last 2 transformer blocks, final norm, dense head','epochs':6,'seed':171,'training_food':len(fr),'training_nutrition':len(ny),'selection':'validation supervised boundary + CE/Tversky + .2 weak MIL loss','sha256':hashlib.sha256((OUT/'best.pt').read_bytes()).hexdigest(),'license':'DINOv2 Apache-2.0; FoodSeg103 declared Apache-2.0; Nutrition5k CC-BY-4.0'},indent=2))
