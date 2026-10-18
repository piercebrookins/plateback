"""Local MPS semantic fine-tuning with weighted CE, recall-weighted Tversky and chicken oversampling."""
import os,time,json,random,hashlib,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from semantic_model import make_model,normalize
ROOT=Path(__file__).resolve().parents[1]
class Plates(Dataset):
    def __init__(self,split,size=320):
        self.split=split;self.size=size;self.rows=[r for r in json.loads((ROOT/'ml/manifest-v4.json').read_text()) if r['split']==split]
        self.base=ROOT/'data/datasets/plateback-v4'
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];im=Image.open(self.base/'images'/self.split/(r['id']+'.jpg')).convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR);mask=Image.open(self.base/'masks'/self.split/(r['id']+'.png')).resize((self.size,self.size),Image.Resampling.NEAREST)
        if self.split=='train':
            if random.random()<.5:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT);mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            im=ImageEnhance.Brightness(im).enhance(random.uniform(.8,1.2));im=ImageEnhance.Color(im).enhance(random.uniform(.8,1.2))
        return torch.from_numpy(np.array(im).transpose(2,0,1).copy()).float()/255,torch.from_numpy(np.array(mask).astype('int64'))
def loss_fn(logits,y):
    ce=nn.functional.cross_entropy(logits,y,weight=logits.new_tensor([.45,1.5,1.5,1.5,2.0]))
    prob=logits.softmax(1);target=nn.functional.one_hot(y,5).permute(0,3,1,2);dims=(0,2,3)
    tp=(prob*target).sum(dims);fp=(prob*(1-target)).sum(dims);fn=((1-prob)*target).sum(dims)
    tv=(tp+1)/(tp+.3*fp+.7*fn+1)
    return ce+.5*(1-tv[1:]).mean()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=24);ap.add_argument('--stop-after-epochs',type=int);ap.add_argument('--size',type=int,default=320);ap.add_argument('--name',default='semantic-v4');args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(141);np.random.seed(141);random.seed(141)
    device='mps' if torch.backends.mps.is_available() else 'cpu';out=ROOT/'artifacts/models'/args.name;out.mkdir(parents=True,exist_ok=True)
    train=Plates('train',args.size);val=Plates('val',args.size)
    sampler=WeightedRandomSampler([1.6 if 3 in r['classes'] else 1 for r in train.rows],len(train),replacement=True,generator=torch.Generator().manual_seed(141))
    dl=DataLoader(train,batch_size=8,sampler=sampler,num_workers=0);vl=DataLoader(val,batch_size=8,num_workers=0)
    model=make_model(True).to(device)
    initial=Path(torch.hub.get_dir())/'checkpoints/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    (out/'initialization.json').write_text(json.dumps({'source':'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth','sha256':hashlib.sha256(initial.read_bytes()).hexdigest()},indent=2))
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':.00015},{'params':model.classifier.parameters(),'lr':.0015},{'params':model.aux_classifier.parameters(),'lr':.0015}],weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=.00001);best=-1;history=[];start=time.time()
    print(json.dumps({'device':device,'train':len(train),'val':len(val),'epochs':args.epochs}),flush=True)
    for epoch in range(min(args.epochs,args.stop_after_epochs or args.epochs)):
        t=time.time();model.train()
        # Keep pretrained running stats stable on small, correlated food batches.
        for m in model.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        total=0
        for x,y in dl:
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True);p=model(normalize(x));loss=loss_fn(p['out'],y)+.3*loss_fn(p['aux'],y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(x)
        model.eval();matrix=np.zeros((5,5),dtype=np.int64)
        with torch.inference_mode():
            for x,y in vl:
                pred=model(normalize(x.to(device)))['out'].argmax(1).cpu().numpy();g=y.numpy();matrix+=np.bincount((g*5+pred).ravel(),minlength=25).reshape(5,5)
        inter=np.diag(matrix);iou=inter/np.maximum(1,matrix.sum(0)+matrix.sum(1)-inter);score=float(iou[1:].mean());row={'epoch':epoch+1,'loss':total/len(train),'iou':iou.tolist(),'macro_iou':score,'seconds':time.time()-t};history.append(row)
        if score>best:
            best=score;torch.save({'state_dict':model.cpu().state_dict(),'size':args.size,'epoch':epoch+1,'labels':['background','rice','pasta','broccoli','chicken']},out/'best.pt');model.to(device)
        sched.step();(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    metadata={'architecture':'torchvision DeepLabV3 MobileNetV3 Large','initialization':'torchvision COCO_WITH_VOC_LABELS_V1 (no FoodSeg training)','device':device,'training_seconds':time.time()-start,'epochs':len(history),'epochs_requested':args.epochs,'seed':141,'selection':'maximum four-class pooled validation IoU; recall-constrained calibration follows, before test','training_images':len(train),'validation_images':len(val),'sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'data_manifest_sha256':hashlib.sha256((ROOT/'ml/manifest-v4.json').read_bytes()).hexdigest(),'license':'torchvision BSD-3-Clause; FoodSeg103 declared Apache-2.0','poultry_limitation':'FoodSeg103 labels chicken/duck jointly.'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
