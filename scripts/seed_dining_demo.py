"""Reproducible fictional return observations; public photos are illustrative only."""
import json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];rng=random.Random(103)
manifest=json.loads((ROOT/'ml/nutrition-manifest.json').read_text());photos=[r for r in manifest if r['split']=='train']
sessions=[];plates=[]
specs=[('2026-10-05','O-Hill','Chicken, rice & broccoli','rice-bowl',40,30,12,8),('2026-10-07','O-Hill','Chicken, rice & broccoli','rice-bowl',40,28,11,9),('2026-10-09','Runk','Chicken, rice & broccoli','rice-bowl',40,31,10,7),('2026-10-06','Newcomb','Chicken, rice & broccoli','rice-bowl',40,27,9,8),('2026-10-08','Newcomb','Pasta & broccoli','pasta-bowl',40,0,0,12),('2026-10-10','Runk','Pasta & broccoli','pasta-bowl',40,0,0,10)]
for n,(day,hall,meal,family,count,rice,chicken,broccoli) in enumerate(specs):
    sid=f'demo-meal-{n+1}';sessions.append({'id':sid,'date':day,'hall':hall,'meal':meal,'family':family,'meal_period':'Lunch','sampled_plates':count,'meals_served':200,'portion':'Standard scoop','serving_weight_grams':None})
    order=list(range(count));rng.shuffle(order);sets={'rice':set(order[:rice])};rng.shuffle(order);sets['chicken']=set(order[:chicken]);rng.shuffle(order);sets['broccoli']=set(order[:broccoli]);rng.shuffle(order);sets['pasta']=set(order[:16 if n==4 else 14 if n==5 else 0])
    for i in range(count):
        foods=[l for l,ids in sets.items() if i in ids];target=[['rice','pasta','broccoli','chicken'].index(l) for l in foods]
        pool=[r for r in photos if ({'rice','chicken'} if family=='rice-bowl' else {'pasta'}).issubset(r['labels'])]
        source=pool[(n*count+i)%len(pool)];id=f'PB-DEMO-{n*count+i+1:03d}'
        plates.append({'id':id,'session_id':sid,'date':day,'hall':hall,'meal':meal,'family':family,'foods':foods,'rice_extent':'substantial' if 'rice' in foods and i%3 else 'light' if 'rice' in foods else 'none','reviewed':True,'net_grams':None,'image_path':source['path'],'image_source_id':source['id'],'image_url':f'/api/demo-dining/plates/{id}/image?source={source["id"]}','observation_source':'synthetic scenario','photo_role':'Illustrative Nutrition5k served-food photograph; not the observed student return','photo_license':'CC-BY-4.0 (Nutrition5k; Thames et al., CVPR 2021)'})
data={'version':1,'classification':'fictional demo','title':'Campus dining demo','disclosure':'240 simulated plate returns. Public food photos are illustrative and may repeat. This is not UVA data or a measured waste study.','period':'October 5–10, 2026 (fictional)','sessions':sessions,'plates':plates,'methodology':{'presence':'Number of sampled return records with a food / sampled return records from meals serving that food.','rice_extent':'Synthetic reviewer category, not a volume or weight measurement.','coverage':'40 fictional sampled returns / 200 fictional meals per session; convenience sample.','weight':'No scale readings. All leftover weights are unknown.','recommendation':'Repeated rice leftovers motivate a controlled portion trial; taste, menu, selection and service timing are alternative explanations.'}}
(ROOT/'data/demo-dining.json').write_text(json.dumps(data,indent=2));print('Created',len(plates),'isolated demo plates across',len(sessions),'sessions.')
