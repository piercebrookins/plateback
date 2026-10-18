from typing import Literal
import math
from shapely.geometry import Polygon
from pydantic import BaseModel,Field,model_validator
Label=Literal['rice','pasta','broccoli','chicken','unknown']
class Region(BaseModel):
    id:str=Field(min_length=1,max_length=80)
    label:Label
    food_name:str|None=Field(default=None,max_length=60)
    review_status:Literal['pending','confirmed']='pending'
    polygon:list[tuple[float,float]]=Field(min_length=3,max_length=2000)
    holes:list[list[tuple[float,float]]]=Field(default_factory=list,max_length=100)
    score:float|None=Field(default=None,ge=0,le=1)
    @model_validator(mode='after')
    def geometry(self):
        for ring in [self.polygon,*self.holes]:
            if len(ring)<3 or any(not math.isfinite(v) or not 0<=v<=1 for pt in ring for v in pt):raise ValueError('Mask coordinates must be finite and between 0 and 1')
            area=abs(sum(ring[i][0]*ring[(i+1)%len(ring)][1]-ring[(i+1)%len(ring)][0]*ring[i][1] for i in range(len(ring))))
            if area<1e-7:raise ValueError('Mask must have nonzero area')
        shape=Polygon(self.polygon,self.holes)
        if not shape.is_valid:raise ValueError('Mask rings must not self-intersect; holes must be inside the region')
        return self
class Review(BaseModel):
    regions:list[Region]=Field(max_length=100)
    net_grams:float|None=Field(default=None,ge=0,le=10000,allow_inf_nan=False)
    weight_source:Literal['none','manual_scale']='none'
    manual:bool=False
    @model_validator(mode='after')
    def weight(self):
        if (self.net_grams is None)!=(self.weight_source=='none'):raise ValueError('Measured weight requires manual_scale; missing weight requires none')
        if len({r.id for r in self.regions})!=len(self.regions):raise ValueError('Region IDs must be unique')
        return self
class Analyze(BaseModel):
    openai_model:Literal['gpt-6-luna','gpt-6-astra']|None=None
    mode:Literal['local','local_openai','local_gemini','gemini','openai','research_openai']='local'
class Session(BaseModel):
    name:str=Field(min_length=1,max_length=100)
    meal_period:str=Field(default='lunch',max_length=50)
    meals_served:int|None=Field(default=None,ge=0)
    coverage_notes:str=Field(default='',max_length=2000)
class Experiment(BaseModel):
    hypothesis:str=Field(min_length=1,max_length=2000)
    intervention:str=Field(min_length=1,max_length=2000)
    baseline_session_id:str|None=None
    intervention_session_id:str|None=None
    metric:str='rice_presence_rate'
    status:Literal['proposed','running','complete']='proposed'
    notes:str=Field(default='',max_length=2000)
class Suggestion(BaseModel):
    region_id:str
    label:Label
    needs_review:bool
    explanation:str=Field(max_length=500)
class Suggestions(BaseModel):
    suggestions:list[Suggestion]=Field(max_length=100)
