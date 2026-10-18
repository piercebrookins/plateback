import io,json,concurrent.futures
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from backend.app import db,budget,vision,main
@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('PLATEBACK_DB',str(tmp_path/'test.db'));monkeypatch.setattr(main,'UPLOADS',tmp_path);monkeypatch.setenv('BUILD_API_BUDGET_USD','5')
    with TestClient(main.app) as c:yield c

def upload(c,key='one',demo=False):
    b=io.BytesIO();Image.new('RGB',(120,80),'white').save(b,'PNG')
    return c.post('/api/scans',files={'image':('x.png',b.getvalue(),'image/png')},data={'idempotency_key':key,'demo':str(demo).lower()})
def region():return dict(id='a',label='rice',polygon=[[.1,.1],[.7,.1],[.4,.8]],holes=[],score=None)
def test_roundtrip_idempotence_unknown_weight(client,monkeypatch):
    s=upload(client).json();assert upload(client).json()['id']==s['id'];id=s['id']
    monkeypatch.setattr(vision,'analyze',lambda *a:{'regions':[region()],'metadata':{'provider':'test'},'cloud_review':None})
    assert client.post(f'/api/scans/{id}/analyze',json={'mode':'local'}).status_code==200
    edited=region();edited['label']='broccoli'
    assert client.put(f'/api/scans/{id}/review',json={'regions':[edited]}).status_code==200
    for _ in range(2):assert client.post(f'/api/scans/{id}/confirm').status_code==200
    restored=client.get(f'/api/scans/{id}').json();assert restored['predictions'][0]['label']=='rice';assert restored['regions'][0]['label']=='broccoli';assert restored['net_grams'] is None
    dash=client.get('/api/dashboard').json();assert dash['scans']==1;assert dash['categories']['broccoli']==1;assert dash['measured_total_grams'] is None;assert dash['correction_rate']==1
    assert client.post(f'/api/scans/{id}/analyze',json={'mode':'local'}).status_code==409

def test_demo_and_zero_are_distinct(client):
    s=upload(client,demo=True).json();id=s['id']
    assert client.post(f'/api/scans/{id}/confirm').status_code==409
    assert client.put(f'/api/scans/{id}/review',json={'regions':[],'manual':True,'net_grams':0,'weight_source':'manual_scale'}).status_code==200
    client.post(f'/api/scans/{id}/confirm');assert client.get('/api/dashboard').json()['scans']==0
    d=client.get('/api/dashboard?include_demo=true').json();assert d['measured_total_grams']==0;assert d['weighed_scans']==1

def test_fail_retains_image_manual_recovery(client,monkeypatch):
    s=upload(client).json();id=s['id']
    def fail(*a):raise RuntimeError('secret must not be returned')
    monkeypatch.setattr(vision,'analyze',fail);r=client.post(f'/api/scans/{id}/analyze',json={'mode':'local'});assert r.status_code==502;assert 'secret' not in r.text
    assert client.get(f'/api/scans/{id}').json()['status']=='failed';assert client.get(f'/api/scans/{id}/image').status_code==200
    assert client.put(f'/api/scans/{id}/review',json={'regions':[]}).status_code==422
    assert client.put(f'/api/scans/{id}/review',json={'regions':[region()],'manual':True}).status_code==200

@pytest.mark.parametrize('body',[{'net_grams':1,'weight_source':'none'},{'net_grams':-1,'weight_source':'manual_scale'},{'weight_source':'manual_scale'},{'regions':[dict(id='a',label='rice',polygon=[[0,0],[2,0],[0,1]])]},{'regions':[dict(id='a',label='rice',polygon=[[0,0],[.5,.5],[1,1]])]}])
def test_invalid_reviews(client,body):
    id=upload(client).json()['id'];assert client.put(f'/api/scans/{id}/review',json={'regions':[],'manual':True,**body}).status_code==422

def test_budget_concurrent_fail_closed(client,monkeypatch):
    monkeypatch.setenv('BUILD_API_BUDGET_USD','0.2')
    def reserve(_):
        try:return budget.reserve('openai')
        except ValueError:return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:r=list(pool.map(reserve,range(10)))
    assert len([x for x in r if x])==2;assert budget.summary()['accounted_usd']==pytest.approx(.2)

def test_coordinate_holes():
    import numpy as np
    m=np.zeros((80,160),dtype='uint8');m[10:60,90:150]=1;m[25:40,110:130]=0
    p,h=vision.rings(m)[0];assert min(x for x,y in p)>.5;assert max(y for x,y in p)<.8;assert len(h)==1

def test_sessions_and_filter(client):
    s=client.post('/api/sessions',json={'name':'Lunch','coverage_notes':'First 10 plates'}).json()
    assert client.get('/api/dashboard?session_id='+s['id']).json()['scans']==0
    e=client.post('/api/experiments',json={'hypothesis':'less rice','intervention':'smaller portions','baseline_session_id':s['id']});assert e.status_code==200
    assert client.post('/api/experiments',json={'hypothesis':'x','intervention':'y','baseline_session_id':'missing'}).status_code==422

def test_orientation_normalized(client):
    im=Image.new('RGB',(40,80),'white');exif=im.getexif();exif[274]=6
    b=io.BytesIO();im.save(b,'JPEG',exif=exif)
    s=client.post('/api/scans',files={'image':('rotated.jpg',b.getvalue(),'image/jpeg')},data={'idempotency_key':'rotate'}).json()
    assert (s['width'],s['height'])==(80,40)
    stored=Image.open(io.BytesIO(client.get(s['image_url']).content));assert stored.size==(80,40);assert stored.getexif().get(274) is None

def test_key_reuse_conflict_and_invalid_upload(client):
    upload(client);b=io.BytesIO();Image.new('RGB',(100,100),'red').save(b,'PNG')
    r=client.post('/api/scans',files={'image':('other.png',b.getvalue(),'image/png')},data={'idempotency_key':'one'});assert r.status_code==409
    assert client.post('/api/scans',files={'image':('bad.jpg',b'not an image','image/jpeg')},data={'idempotency_key':'bad'}).status_code==422

def test_self_intersection_rejected(client):
    id=upload(client).json()['id'];r=region();r['polygon']=[[0,0],[1,1],[0,.9],[1,0]]
    assert client.put(f'/api/scans/{id}/review',json={'regions':[r],'manual':True}).status_code==422

def test_disabled_gemini_is_not_called(client,monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    with pytest.raises(ValueError,match='GEMINI_API_KEY'):vision.gemini_request('test',[],100)
    assert budget.summary()['calls']==[]

def test_pinched_contour_splits_validly():
    import numpy as np
    from backend.app.schemas import Region
    m=np.zeros((40,60),dtype=np.uint8);m[4:15,4:15]=1;m[15:30,15:30]=1
    for p,h in vision.rings(m):Region(id='pinch',label='rice',polygon=p,holes=h)

def test_luna_cached_usage_pricing():
    usage={'input_tokens':1000,'output_tokens':100,'input_tokens_details':{'cached_tokens':400,'cache_write_tokens':200}}
    assert budget.openai_cost('gpt-6-luna',usage)==pytest.approx((400*.1+400*.01+200*.125+100*.5)/1e6)
    assert budget.openai_cost('gpt-6-luna',{'input_tokens':562,'output_tokens':68})==pytest.approx(.0000902)

def test_class_filter_rejects_low_confidence_and_tiny_false_positives():
    import numpy as np
    from backend.app.mask_filter import keep_mask
    mask=np.zeros((100,100),bool);mask[:5,:5]=True
    policy={'confidence':.2,'per_class':{'rice':{'confidence':.6,'min_area_fraction':.005}}}
    assert not keep_mask(0,.9,mask,policy)
    mask[:20,:20]=True
    assert not keep_mask(0,.59,mask,policy)
    assert keep_mask(0,.6,mask,policy)
    assert keep_mask(1,.3,mask,policy)

def test_luna_request_and_budget_settlement(client,monkeypatch,tmp_path):
    from types import SimpleNamespace
    import openai
    calls=[]
    usage={'input_tokens':562,'output_tokens':68,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}}
    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda:usage),output_text=json.dumps({'suggestions':[{'region_id':'a','label':'rice','needs_review':True,'explanation':'Verify grains'}]}))
    def fake(**kw):
        assert kw['max_retries']==0
        return SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(openai,'OpenAI',fake);monkeypatch.setenv('OPENAI_API_KEY','test-only');monkeypatch.setenv('OPENAI_MODEL','gpt-6-luna')
    path=tmp_path/'image.png';Image.new('RGB',(100,100),'white').save(path)
    result=vision.review(path,[region()],'openai')
    assert result['model']=='gpt-6-luna';assert calls[0]['reasoning']=={'effort':'none'};assert calls[0]['service_tier']=='default'
    assert budget.summary()['accounted_usd']==pytest.approx(.0000902)

def test_dining_demo_isolated_and_denominators(client):
    from backend.app.demo_dining import snapshot
    data=snapshot();assert data['classification']=='fictional demo';assert len(data['plates'])==240
    assert len({p['id'] for p in data['plates']})==240
    rice=[p for p in data['plates'] if p['family']=='rice-bowl']
    assert len(rice)==160;assert sum('rice' in p['foods'] for p in rice)==116
    assert all(p['net_grams'] is None for p in data['plates'])
    assert client.get('/api/dashboard').json()['scans']==0
    assert client.get('/api/demo-dining/plates/PB-DEMO-001/image').status_code==200
    assert client.get('/api/demo-dining/plates/not-a-plate/image').status_code==404

def test_openrouter_uses_shared_cap(client,monkeypatch):
    monkeypatch.setenv('BUILD_API_BUDGET_USD','.15')
    budget.reserve('openrouter')
    with pytest.raises(ValueError,match='budget'):budget.reserve('openai')
    assert budget.summary()['accounted_usd']==pytest.approx(.1)

def test_sample_library_is_read_only_and_opt_in(client):
    data=client.get('/api/sample-scans').json()
    assert data['read_only'] is True
    assert len(data['scans'])==1200
    assert len({s['id'] for s in data['scans']})==1200
    assert len(data['sessions'])==30
    assert all(s['net_grams'] is None and s['regions']==[] and s['predictions']==[] for s in data['scans'])
    assert client.get('/api/scans').json()==[]
    assert client.get('/api/dashboard').json()['scans']==0
    assert client.get('/api/dashboard?include_samples=true').json()['scans']==0
    included=client.get('/api/dashboard?include_demo=true&include_samples=true').json()
    assert included['scans']==1200
    assert included['categories']['rice']==580
    assert included['weighed_scans']==0 and included['measured_total_grams'] is None
    sid=data['sessions'][0]['id']
    assert client.get('/api/dashboard',params={'include_demo':'true','include_samples':'true','session_id':sid}).json()['scans']==40
    assert client.get('/api/scans/'+data['scans'][0]['id']).status_code==404
    assert client.post('/api/scans/'+data['scans'][0]['id']+'/confirm').status_code==404
    assert client.get('/api/dashboard').json()['scans']==0

def test_openai_model_override_is_validated_and_routed(client,monkeypatch):
 s=upload(client).json();seen=[]
 def analyze(path,mode,openai_model=None):
  seen.append((mode,openai_model));return {'regions':[region()],'metadata':{'provider':'openai','model':openai_model},'cloud_review':None}
 monkeypatch.setattr(vision,'analyze',analyze)
 path=f'/api/scans/{s["id"]}/analyze'
 assert client.post(path,json={'mode':'openai','openai_model':'unpriced'}).status_code==422
 result=client.post(path,json={'mode':'openai','openai_model':'gpt-6-astra'})
 assert result.status_code==200
 assert seen==[('openai','gpt-6-astra')]
 assert result.json()['metadata']['model']=='gpt-6-astra'
 assert result.json()['net_grams'] is None

def test_uva_sessions_available(client):
 sessions=client.get('/api/sessions').json()
 assert {'O-Hill','Newcomb','Runk'} <= {s['name'] for s in sessions}
 data=client.get('/api/demo-dining').json()
 assert {s['hall'] for s in data['sessions']}=={'O-Hill','Newcomb','Runk'}

def test_named_food_review_roundtrip(client,monkeypatch):
 s=upload(client).json();ident=s['id'];a={**region(),'label':'unknown','food_name':'carrots','review_status':'pending'};b={**region(),'id':'b','label':'chicken'}
 monkeypatch.setattr(vision,'analyze',lambda *args:{'regions':[a,b],'metadata':{'provider':'openai'},'cloud_review':None})
 assert client.post(f'/api/scans/{ident}/analyze',json={'mode':'openai'}).status_code==200
 # Rename the other-food guess and reject chicken by omitting its region.
 a.update(food_name='sweet potato',review_status='confirmed')
 assert client.put(f'/api/scans/{ident}/review',json={'regions':[a]}).status_code==200
 assert client.post(f'/api/scans/{ident}/confirm').status_code==200
 saved=client.get(f'/api/scans/{ident}').json()
 assert len(saved['regions'])==1 and saved['regions'][0]['food_name']=='sweet potato'
 assert saved['regions'][0]['review_status']=='confirmed'
 assert len(saved['predictions'])==2 and saved['predictions'][0]['food_name']=='carrots'
 assert saved['net_grams'] is None
