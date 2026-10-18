"""Audit exact and perceptual near-duplicates without using labels/predictions."""
import json
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.fft import dctn
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'ml/manifest.json').read_text())
hashes=[]
for r in rows:
    im=Image.open(ROOT/'data/datasets/plateback/images'/r['split']/(r['id']+'.jpg')).convert('L').resize((32,32))
    d=dctn(np.asarray(im,dtype=float),norm='ortho')[:8,:8].flatten()[1:];hashes.append(d>np.median(d))
pairs=[]
for i in range(len(rows)):
    for j in range(i):
        dist=int(np.count_nonzero(hashes[i]!=hashes[j]))
        if rows[i]['split']!=rows[j]['split'] and dist<=6:pairs.append({'a':rows[i]['id'],'a_split':rows[i]['split'],'b':rows[j]['id'],'b_split':rows[j]['split'],'phash_distance':dist})
report={'images':len(rows),'exact_unique':len(set(r['sha256'] for r in rows)),'phash_bits':63,'threshold':6,'cross_split_candidates':pairs,'limitation':'Heuristic duplicate screening; source does not provide physical-serving or capture-session IDs. Scene-level leakage cannot be ruled out.'}
(ROOT/'ml/split_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
