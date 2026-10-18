"""Frozen DINO features; hard-pixel CE plus v6 distillation. Train/validation only."""
import json,sys,time,hashlib,copy
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.dino_model import head
from train_semantic import loss_fn

def main():
 torch.set_num_threads(2);torch.manual_seed(181);start=time.time();out=ROOT/'artifacts/models/hardpixels-v8';out.mkdir(parents=True,exist_ok=True)
 manifest=ROOT/'ml/manifest-v4.json';rows=json.loads(manifest.read_text());data={}
 for split in ['train','val']:
  rs=[r for r in rows if r['split']==split];identity=json.loads((ROOT/f'artifacts/dino-features/{split}.json').read_text());assert identity['ids']==[r['id'] for r in rs] and identity['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
  x=torch.from_numpy(np.load(ROOT/f'artifacts/dino-features/{split}.npy').astype('float32'));y=torch.from_numpy(np.stack([np.array(Image.open(ROOT/f'data/datasets/plateback-v4/masks/{split}'/(r['id']+'.png')).resize((96,96),Image.Resampling.NEAREST),dtype=np.int64) for r in rs]));data[split]=(x,y,rs)
 state=torch.load(ROOT/'artifacts/models/mil-v6/best.pt',map_location='cpu',weights_only=True);teacher=head().eval();teacher.load_state_dict({k[5:]:v for k,v in state['state_dict'].items() if k.startswith('head.')});decoder=copy.deepcopy(teacher)
 for p in teacher.parameters():p.requires_grad=False
 opt=torch.optim.AdamW(decoder.parameters(),lr=.00015,weight_decay=.02);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,16);best=float('inf');history=[]
 x,y,rs=data['train'];vx,vy,_=data['val'];weights=torch.tensor([1.6 if 3 in r['classes'] else 1.4 if not r['classes'] else 1 for r in rs],dtype=torch.double)
 def loss(a,b,training):
  raw=decoder(a);logits=nn.functional.interpolate(raw,size=b.shape[-2:],mode='bilinear',align_corners=False)
  ce=nn.functional.cross_entropy(logits,b,reduction='none').flatten(1);hard=ce.topk(max(1,ce.shape[1]//5),dim=1).values.mean()
  with torch.no_grad():target=teacher(a).softmax(1)
  distill=nn.functional.kl_div(raw.log_softmax(1),target,reduction='none').sum(1).mean()
  return loss_fn(logits,b)+.35*hard+.5*distill
 for epoch in range(16):
  decoder.train();order=torch.multinomial(weights,len(rs),replacement=True);total=0
  for i in range(0,len(order),16):
   ids=order[i:i+16];a=x[ids];b=y[ids]
   if torch.rand(())<.5:a=a.flip(-1);b=b.flip(-1)
   opt.zero_grad(set_to_none=True);v=loss(a,b,True);v.backward();nn.utils.clip_grad_norm_(decoder.parameters(),1);opt.step();total+=v.item()*len(ids)
  decoder.eval()
  with torch.no_grad():valid=sum(loss(vx[i:i+16],vy[i:i+16],False).item()*len(vx[i:i+16]) for i in range(0,len(vx),16))/len(vx)
  row={'epoch':epoch+1,'train_loss':total/len(x),'validation_loss':valid};history.append(row)
  if valid<best:
   best=valid;saved=copy.deepcopy(state);saved['state_dict'].update({'head.'+k:v for k,v in decoder.state_dict().items()});saved['size']=448;saved['epoch']=epoch+1;torch.save(saved,out/'best.pt')
  scheduler.step();(out/'history.json').write_text(json.dumps(history,indent=2));print(row,flush=True)
 (out/'metadata.json').write_text(json.dumps({'method':'Frozen DINO backbone; top-20% hard-pixel CE + Tversky + v6 soft-target distillation','seed':181,'epochs':16,'training_images':len(x),'validation_images':len(vx),'device':'cpu','seconds':time.time()-start,'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'selection':'minimum validation loss, then native-resolution recall-constrained calibration','test_used_for_training':False,'new_api_cost_usd':0},indent=2))
if __name__=='__main__':main()
