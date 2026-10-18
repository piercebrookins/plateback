"""A fixed 448px inference ablation; same trained parameters, no extra fitting."""
import json,hashlib,shutil
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def prepare():
    src=ROOT/'artifacts/models/dino-v4';out=ROOT/'artifacts/models/dino448-v4'
    meta=json.loads((src/'metadata.json').read_text());out.mkdir(parents=True,exist_ok=True)
    for name in ['history.json','initialization.json']:
        if (src/name).exists():shutil.copy(src/name,out/name)
    if (out/'metadata.json').exists():
        assert json.loads((out/'metadata.json').read_text())['parent_sha256']==meta['sha256'];return out
    state=torch.load(src/'best.pt',map_location='cpu',weights_only=True);state['size']=448;torch.save(state,out/'best.pt')
    meta.update(parent_sha256=meta['sha256'],sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),training_input_size=336,inference_size=448,inference_ablation='Same trained parameters; 448px inference instead of 336px; validation-only selection')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));return out
if __name__=='__main__':print(prepare())
