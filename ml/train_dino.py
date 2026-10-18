"""CPU-only frozen DINOv2 feature extraction and dense-head transfer learning; no test access."""
import sys,json,time,hashlib,random,subprocess,shutil
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.dino_model import backbone,head,REVISION
from backend.app.semantic_model import normalize
from train_semantic import loss_fn
OUT=ROOT/'artifacts/models/dino-v4';CACHE=ROOT/'artifacts/dino-features';SIZE=336

def main():
    repo=ROOT/'data/source/dinov2'
    if not repo.exists():
        subprocess.run(['git','clone','https://github.com/facebookresearch/dinov2.git',str(repo)],check=True)
        subprocess.run(['git','-C',str(repo),'checkout',REVISION],check=True)
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==REVISION,'DINOv2 code revision changed'
    shutil.copy(repo/'LICENSE',ROOT/'docs/licenses/dinov2-Apache-2.0.txt')
    shutil.copy(repo/'MODEL_CARD.md',ROOT/'docs/licenses/dinov2-MODEL_CARD.md')
    torch.set_num_threads(2);torch.manual_seed(142);np.random.seed(142);random.seed(142);OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'ml/manifest-v4.json').read_text());model=backbone(True).eval();start=time.time();data={}
    initial=Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth'
    (OUT/'initialization.json').write_text(json.dumps({'source':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth','sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'code_revision':REVISION},indent=2))
    for split in ['train','val']:
        rs=[r for r in rows if r['split']==split];saved=CACHE/(split+'.npy');stamp=CACHE/(split+'.json');identity={'ids':[r['id'] for r in rs],'manifest_sha256':hashlib.sha256((ROOT/'ml/manifest-v4.json').read_bytes()).hexdigest(),'backbone_revision':REVISION,'size':SIZE}
        if not stamp.exists() or json.loads(stamp.read_text())!=identity:
            features=np.lib.format.open_memmap(saved,mode='w+',dtype=np.float16,shape=(len(rs),384,24,24))
            with torch.inference_mode():
                for i in range(0,len(rs),8):
                    ims=[]
                    for r in rs[i:i+8]:
                        im=Image.open(ROOT/'data/datasets/plateback-v4/images'/split/(r['id']+'.jpg')).convert('RGB').resize((SIZE,SIZE),Image.Resampling.BILINEAR);ims.append(np.array(im).transpose(2,0,1).copy())
                    x=torch.from_numpy(np.array(ims)).float()/255;f=model.forward_features(normalize(x))['x_norm_patchtokens'].transpose(1,2).reshape(len(x),384,24,24).numpy();features[i:i+len(x)]=f
                    if i%128==0:print('features',split,i,'/',len(rs),flush=True)
            features.flush();stamp.write_text(json.dumps(identity,indent=2))
        features=torch.from_numpy(np.load(saved).astype(np.float32));masks=torch.from_numpy(np.stack([np.array(Image.open(ROOT/'data/datasets/plateback-v4/masks'/split/(r['id']+'.png')).resize((96,96),Image.Resampling.NEAREST),dtype=np.int64) for r in rs]));data[split]=(features,masks,rs)
    print('Feature extraction seconds',time.time()-start,flush=True);decoder=head();opt=torch.optim.AdamW(decoder.parameters(),lr=.001,weight_decay=.02);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40,eta_min=.00003);history=[];best=-1
    x,y,rs=data['train'];vx,vy,_=data['val'];weights=torch.tensor([3 if 1 in r['classes'] else (1.6 if 3 in r['classes'] else 1) for r in rs],dtype=torch.double);rng=torch.Generator().manual_seed(142)
    for epoch in range(40):
        t=time.time();decoder.train();order=torch.multinomial(weights,len(rs),replacement=True,generator=rng);total=0
        for i in range(0,len(order),16):
            ids=order[i:i+16];a=x[ids];b=y[ids]
            if random.random()<.5:a=a.flip(-1);b=b.flip(-1)
            opt.zero_grad(set_to_none=True);logits=nn.functional.interpolate(decoder(a),size=(96,96),mode='bilinear',align_corners=False);loss=loss_fn(logits,b);loss.backward();opt.step();total+=loss.item()*len(ids)
        decoder.eval();matrix=np.zeros((5,5),np.int64)
        with torch.inference_mode():
            for i in range(0,len(vx),16):
                pred=nn.functional.interpolate(decoder(vx[i:i+16]),size=(96,96),mode='bilinear',align_corners=False).argmax(1).numpy();g=vy[i:i+16].numpy();matrix+=np.bincount((g*5+pred).ravel(),minlength=25).reshape(5,5)
        inter=np.diag(matrix);iou=inter/np.maximum(1,matrix.sum(0)+matrix.sum(1)-inter);score=float(iou[1:].mean());row={'epoch':epoch+1,'loss':total/len(rs),'iou':iou.tolist(),'macro_iou':score,'seconds':time.time()-t};history.append(row)
        if score>best:
            best=score;state={**{'backbone.'+k:v for k,v in model.state_dict().items()},**{'head.'+k:v for k,v in decoder.state_dict().items()}};torch.save({'architecture':'dinov2-head','state_dict':state,'size':SIZE,'epoch':epoch+1,'labels':['background','rice','pasta','broccoli','chicken']},OUT/'best.pt')
        scheduler.step();(OUT/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    meta={'architecture':'DINOv2 ViT-S/14 frozen backbone + trained 1x1 dense MLP head','revision':REVISION,'device':'cpu','training_seconds':time.time()-start,'epochs':40,'seed':142,'training_images':len(x),'validation_images':len(vx),'selection':'maximum 96px four-class pooled validation IoU; native resolution recall-constrained calibration before test','sha256':hashlib.sha256((OUT/'best.pt').read_bytes()).hexdigest(),'license':'Apache-2.0 DINOv2 code and weights; FoodSeg103 declared Apache-2.0','poultry_limitation':'Chicken/duck proxy','pretraining_overlap':'DINOv2 LVD-142M pretraining image identities are not audited against this public benchmark.'};(OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
