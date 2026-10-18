"""Rasterize each region independently, then union masks of the same food."""
import numpy as np
from PIL import Image,ImageDraw,ImageChops
LABELS=['rice','pasta','broccoli','chicken']
def rasterize(regions,shape):
    h,w=shape;masks=[Image.new('1',(w,h)) for _ in LABELS]
    for region in regions:
        layer=Image.new('1',(w,h));draw=ImageDraw.Draw(layer)
        draw.polygon([(x/1000*w,y/1000*h) for x,y in region['polygon']],fill=1)
        for hole in region.get('holes',[]):draw.polygon([(x/1000*w,y/1000*h) for x,y in hole],fill=0)
        c=LABELS.index(region['label']);masks[c]=ImageChops.logical_or(masks[c],layer)
    return [np.array(m,bool) for m in masks]
