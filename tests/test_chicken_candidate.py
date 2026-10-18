import numpy as np
from backend.app.chicken_candidate import combine
from backend.app.local_semantic import class_mask

def test_rescue_preserves_base_and_rejected_gate_adds_nothing():
 p=np.zeros((40,40),np.float32);p[2:12,2:12]=.9;q=np.zeros_like(p);q[24:32,24:32]=.8
 base_rule={'confidence':.75,'min_area_fraction':.04,'grow_confidence':.1};rule={'confidence':.5,'min_area_fraction':.002,'grow_confidence':.1};base,_=class_mask(p,p.shape,base_rule)
 rejected=combine(p,q,p.shape,base_rule,rule,.3,.5);accepted=combine(p,q,p.shape,base_rule,rule,.7,.5)
 assert np.array_equal(rejected,base)
 assert not (base&~accepted).any()
 assert accepted.sum()>base.sum()

def test_positive_classifier_cannot_invent_region_without_semantic_support():
 p=np.zeros((40,40),np.float32);rule={'confidence':.5,'min_area_fraction':.002,'grow_confidence':.1}
 assert not combine(p,p,p.shape,rule,rule,1.,.5).any()

def test_refinement_cannot_add_false_positive_image_and_fallback_preserves_detection():
 from backend.app.chicken_refinement import refine
 p=np.ones((40,40),np.float32);empty=np.zeros_like(p,dtype=bool);base=empty.copy();base[2:12,2:12]=True;rule={'confidence':.5,'min_area_fraction':0}
 assert not refine(empty,p,rule,'replace',False).any()
 assert np.array_equal(refine(base,p*0,rule,'replace',True),base)
 assert not refine(base,p*0,rule,'replace',False).any()

def test_selected_v6_local_mode_uses_candidate_not_yolo(tmp_path,monkeypatch):
 import json
 from backend.app import vision
 weights=tmp_path/'mlp.pt';weights.write_bytes(b'test checkpoint')
 (tmp_path/'settings.json').write_text(json.dumps({'version':'v6-research'}))
 monkeypatch.setenv('LOCAL_MODEL_PATH',str(weights))
 expected=([],{'model':'chicken-v6 research candidate','experimental':True})
 monkeypatch.setattr(vision,'research_segment',lambda path:expected)
 assert vision.local_segment(tmp_path/'plate.jpg')==expected

def test_redistribution_preserves_area_presence_and_is_deterministic():
 from backend.app.chicken_refinement import refine
 base=np.zeros((40,40),bool);base[10:20,10:20]=True
 p=np.zeros((40,40),np.float32);p[10:20,12:22]=1
 rule={'radius_fraction':.05,'base_prior':0}
 result=refine(base,p,rule,'redistribute',False)
 assert result.sum()==base.sum()
 assert np.array_equal(result,refine(base,p,rule,'redistribute',False))
 assert result[:,20:22].any()
 assert not refine(base&False,p,rule,'redistribute',False).any()
