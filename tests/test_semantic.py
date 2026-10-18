import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ml'))
from evaluate_semantic import eligible,one,metrics
from backend.app.local_semantic import class_mask

def test_recall_constraints_have_no_infeasible_fallback():
    reference={'recall':.8,'pixel_recall':.7,'fp':5}
    assert eligible(dict(reference),reference)
    assert not eligible(dict(reference,recall=.79),reference)
    assert not eligible(dict(reference,pixel_recall=.69),reference)
    assert not eligible(dict(reference,fp=6),reference)
    assert eligible({'recall':.9,'pixel_recall':.8,'fp':3},reference)

def test_tiny_false_positive_removed_without_filling_holes():
    p=np.zeros((40,40),np.float32);p[5:30,5:30]=.9;p[10:20,10:20]=0;p[35:37,35:37]=.9
    mask,_=class_mask(p,p.shape,{'confidence':.5,'min_area_fraction':.005})
    assert mask[6,6] and not mask[15,15] and not mask[35,35]

def test_pixel_recall_detects_tiny_mask_even_when_presence_recall_is_perfect():
    g=np.ones((10,10),bool);p=np.zeros_like(g);p[0,0]=True
    m=metrics([{'id':'x','classes':{'chicken':one(g,p)}}])['per_class']['chicken']
    assert m['recall']==1 and m['pixel_recall']==.01 and m['iou']==.01


def test_polygon_hole_does_not_erase_another_region_of_the_same_food():
    from benchmark_geometry import rasterize
    regions=[{'label':'chicken','polygon':[[300,300],[700,300],[700,700],[300,700]],'holes':[]},
             {'label':'chicken','polygon':[[100,100],[900,100],[900,900],[100,900]],'holes':[[[200,200],[800,200],[800,800],[200,800]]]}]
    mask=rasterize(regions,(100,100))[3]
    assert mask[50,50] and mask[15,15] and not mask[25,25]


def test_fast_component_calibration_matches_serving_masks_exactly():
    from evaluate_semantic import component_choices
    rng=np.random.default_rng(141);p=rng.random((12,15)).astype(np.float32);g=rng.random((40,50))>.7;areas=[0,.005,.02,.05]
    for threshold in [.2,.7,.95]:
        fast=component_choices(p,g,threshold,areas)
        for area in areas:
            mask,_=class_mask(p,g.shape,{'confidence':threshold,'min_area_fraction':area})
            assert fast[area]==one(g,mask)

def test_boundary_growth_preserves_core_and_does_not_create_food_presence():
    from refine_chicken_boundary import choices
    p=np.zeros((40,50),np.float32);p[5:30,5:30]=.4;p[10:20,10:20]=.95;p[32:38,32:45]=.4
    g=np.zeros_like(p,bool);g[5:30,5:30]=True
    rule={'confidence':.9,'min_area_fraction':.002,'grow_confidence':.3}
    mask,_=class_mask(p,p.shape,rule);core,_=class_mask(p,p.shape,{k:v for k,v in rule.items() if k!='grow_confidence'})
    assert np.all(mask[core]) and mask[5,5] and not mask[35,35]
    assert choices(p,g,[.9],[.002],[.3])[(.9,.002,.3)]==one(g,mask)
    empty,_=class_mask(np.full_like(p,.4),p.shape,rule)
    assert not empty.any()
