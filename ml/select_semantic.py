"""Freeze one of two local approaches using validation only, before any new test scoring."""
import json,shutil,subprocess,sys,os,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    original=ROOT/'artifacts/models/semantic-v4';deep=ROOT/'artifacts/models/deeplab-v4';dino=ROOT/'artifacts/models/dino-v4'
    assert (original/'metadata.json').exists() and (dino/'metadata.json').exists(),'Both training jobs must finish first'
    if not deep.exists():shutil.copytree(original,deep)
    from prepare_dino_scale import prepare
    dino_high=prepare()
    candidates=[]
    for folder in [deep,dino,dino_high]:
        ev=ROOT/('artifacts/evaluation-'+folder.name)
        if not (ev/'calibration.json').exists():subprocess.run([sys.executable,str(ROOT/'ml/evaluate_semantic.py'),'--folder',folder.name,'--calibrate-only'],check=True)
        identity=json.loads((ev/'probabilities'/folder.name/'identity.json').read_text());assert identity['sha256']==hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),'Validation cache checkpoint changed'
        calibration=json.loads((ev/'calibration.json').read_text());m=json.loads((ev/'semantic-validation.json').read_text());score=.5*m['per_class']['chicken']['iou']+.5*m['macro_iou'];candidates.append({'folder':folder.name,'validation':{k:v for k,v in m.items() if k!='per_image'},'all_constraints_satisfied':all(r['constraints_satisfied'] for r in calibration),'objective':score,'weights_sha256':hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()})
    eligible=[r for r in candidates if r['all_constraints_satisfied']]
    chosen=max(eligible or candidates,key=lambda r:r['objective']);decision={'selection':'Validation-only: maximize 0.5 chicken IoU + 0.5 four-class macro IoU among candidates passing ALL class recall and false-positive constraints. If none pass, retain explicit experimental status and block promotion. Defined before test access.','candidates':candidates,'selected':chosen['folder'],'promotion_eligible':bool(eligible)}
    source=ROOT/'artifacts/models'/chosen['folder']
    for name in ['best.pt','baseline.pt','settings.json','metadata.json','history.json','initialization.json']:
        temp=original/(name+'.selected.tmp');shutil.copy(source/name,temp);os.replace(temp,original/name)
    meta=json.loads((original/'metadata.json').read_text());meta.update(version='v4',selection_decision=decision);(original/'metadata.json').write_text(json.dumps(meta,indent=2))
    ev=ROOT/'artifacts/evaluation-v4';ev.mkdir(parents=True,exist_ok=True)
    for name in ['calibration.json','semantic-validation.json','baseline-validation.json']:shutil.copy(ROOT/('artifacts/evaluation-'+chosen['folder'])/name,ev/name)
    (ev/'model-selection.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
