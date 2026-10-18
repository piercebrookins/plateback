"""Render fixed benchmark examples from raw masks, without choosing attractive outputs."""
import sys,json,argparse
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.local_semantic import predict
from benchmark_geometry import rasterize,LABELS
OUT=ROOT/'output/evaluation-v4';OUT.mkdir(parents=True,exist_ok=True)
COLORS=np.array([[219,168,40],[220,111,65],[58,140,78],[142,80,155]],np.uint8)
def overlay(image,masks):
    a=np.array(image).astype(float)
    for c,m in enumerate(masks):a[m]=.55*a[m]+.45*COLORS[c]
    return Image.fromarray(a.astype('uint8'))
def row_panel(row,folder,policy,benchmark):
    im=Image.open(ROOT/row['path']).convert('RGB');gt=np.array(Image.open(ROOT/row['truth_path']));local=[np.zeros_like(gt,bool) for _ in LABELS]
    for c,s,m in predict(ROOT/row['path'],folder,policy):local[c]|=m
    report=json.loads((benchmark/'responses/google__gemini-4-argon'/('segmentation-'+row['id']+'.json')).read_text());cloud=[np.zeros_like(gt,bool) for _ in LABELS];status='Gemini'
    try:
        content=report['response']['choices'][0]['message']['content'].strip()
        if content.startswith('```'):content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
        value=json.loads(content);assert isinstance(value['regions'],list)
        for region in value['regions']:
            assert region['label'] in LABELS
            for ring in [region['polygon']]+region.get('holes',[]):assert 3<=len(ring)<=4096 and all(len(p)==2 and all(isinstance(v,(int,float)) and 0<=v<=1000 for v in p) for p in ring)
        cloud=rasterize(value['regions'],gt.shape)
    except (ValueError,KeyError,TypeError,IndexError,AssertionError):status='Gemini: invalid output'
    panel=Image.new('RGB',(990,280),'#faf9f5');draw=ImageDraw.Draw(panel);draw.text((12,8),row['id']+' | public held-out food image; chicken includes duck',fill='#25271f')
    for i,(name,masks) in enumerate([('Ground truth',[gt==c+1 for c in range(4)]),('Local '+policy['version'],local),(status,cloud)]):
        pic=overlay(im,masks);pic.thumbnail((310,228));panel.paste(pic,(i*330+10+(310-pic.width)//2,43+(228-pic.height)//2));draw.text((i*330+10,28),name,fill='#25271f')
    return panel,gt
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='semantic-v4');ap.add_argument('--benchmark',default='benchmark-v4');ap.add_argument('--output',default='evaluation-v4');args=ap.parse_args();OUT=ROOT/'output'/args.output;OUT.mkdir(parents=True,exist_ok=True);benchmark=ROOT/'artifacts'/args.benchmark
    torch.set_num_threads(4);folder=ROOT/'artifacts/models'/args.model;policy=json.loads((folder/'settings.json').read_text());manifest=json.loads((benchmark/'manifest.json').read_text());chicken=[];negative=[];html=['<!doctype html><meta charset="utf-8"><title>PlateBack v4 benchmark examples</title><h1>All fixed paired benchmark images</h1><p>Left: ground truth. Center: local model. Right: Gemini. Rice: gold; pasta: orange; broccoli: green; chicken/duck: purple. No food weight is inferred.</p>']
    for r in manifest['segmentation']:
        panel,gt=row_panel(r,folder,policy,benchmark);panel.save(OUT/(r['id']+'.png'));html.append(f'<img style="max-width:100%" src="{r["id"]}.png" alt="{r["id"]} ground truth, local and Gemini masks">')
        if (gt==4).any() and len(chicken)<4:chicken.append(panel)
        if not gt.any() and len(negative)<4:negative.append(panel)
    for name,panels in [('chicken-comparison',chicken),('negative-comparison',negative)]:
        canvas=Image.new('RGB',(990,30+280*len(panels)),'#faf9f5');ImageDraw.Draw(canvas).text((12,8),'Rice: gold | Pasta: orange | Broccoli: green | Chicken/duck: purple',fill='#25271f')
        for i,p in enumerate(panels):canvas.paste(p,(0,30+i*280))
        canvas.save(OUT/(name+'.png'))
    (OUT/'index.html').write_text('\n'.join(html));print('Rendered all paired examples and fixed first-four chicken / negative contact sheets')
