"""Check split disjointness and preserve an inspectable v2 audit."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fft import dctn
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'ml/manifest-v2.json').read_text());old=json.loads((ROOT/'ml/manifest.json').read_text());seen={};hashes=[];collisions=[]
for r in rows:
    assert r['id'] not in seen;seen[r['id']]=r['split']
    assert (r['split']=='test')==(r['source_file']=='test.parquet')
    im=Image.open(ROOT/'data/datasets/plateback-v2/images'/r['split']/(r['id']+'.jpg'))
    v=dctn(np.array(im.convert('L').resize((32,32)),float),norm='ortho')[:8,:8].flatten()[1:];ph=v>np.median(v)
    for id,split,sha,other in hashes:
        if split!=r['split'] and (sha==r['sha256'] or np.count_nonzero(ph!=other)<=6):collisions.append([r['id'],id,int(np.count_nonzero(ph!=other))])
    hashes.append((r['id'],r['split'],r['sha256'],ph))
assert not set(r['id'] for r in old if r['split']=='test') & set(r['id'] for r in rows)
report={'images':len(rows),'splits':{s:sum(r['split']==s for r in rows) for s in ['train','val','test']},'exact_or_near_cross_split_collisions_after_saved_JPEG_resize':collisions,'old_test_IDs_excluded':True,'method':'63-bit DCT perceptual hash; Hamming <=6. This heuristic does not prove independent source groups.'}
(ROOT/'ml/split_audit-v2.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
