"""Identical confidence/area policy for serving and evaluation."""
LABELS=['rice','pasta','broccoli','chicken']
def keep_mask(cls,score,mask,settings):
    rule=settings.get('per_class',{}).get(LABELS[cls],{})
    return score >= rule.get('confidence',settings.get('confidence',.25)) and float(mask.mean()) >= rule.get('min_area_fraction',0)
