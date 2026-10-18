import json,concurrent.futures
from types import SimpleNamespace
import pytest
from PIL import Image
from backend.app import budget,vision,db

@pytest.fixture
def ledger(tmp_path,monkeypatch):
 monkeypatch.setenv('PLATEBACK_DB',str(tmp_path/'budget.db'));monkeypatch.setenv('BUILD_API_BUDGET_USD','5');monkeypatch.delenv('PLATEBACK_BUDGET_RUN',raising=False);db.init()

def test_run_cap_is_atomic_and_unknown_calls_stay_charged(ledger,monkeypatch):
 budget.start_run('test',.3);monkeypatch.setenv('PLATEBACK_BUDGET_RUN','test')
 def reserve(i):
  try:return budget.reserve('openai')
  except ValueError:return None
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:ids=[x for x in pool.map(reserve,range(10)) if x]
 assert len(ids)==3
 assert budget.run_summary('test')['accounted_usd']==pytest.approx(.3)
 budget.settle(ids[0],.001,{})
 assert budget.run_summary('test')['accounted_usd']==pytest.approx(.201)
 with pytest.raises(ValueError):budget.start_run('test',1)

def test_cloud_parser_rejects_invented_mass_and_bad_geometry():
 valid={'label':'chicken','polygon':[[100,100],[900,100],[900,900],[100,900]],'holes':[[[300,300],[400,300],[400,400],[300,400]]]}
 r=vision.parse_cloud_regions(json.dumps({'regions':[valid],'grams':200}))[0]
 assert 'grams' not in r and r['score'] is None and len(r['holes'])==1
 for polygon in [[[0,0],[1001,0],[0,10]],[[0,0],[100,100],[0,100],[100,0]],[[0,0],[1,1],[2,2]]]:
  with pytest.raises(ValueError):vision.parse_cloud_regions(json.dumps({'regions':[{**valid,'polygon':polygon,'holes':[]}]}))
 assert vision.parse_cloud_regions('{"regions":[]}')==[]

def test_direct_openai_does_not_require_local_masks(ledger,monkeypatch,tmp_path):
 import openai
 usage={'input_tokens':1000,'output_tokens':200};calls=[]
 def create(**kw):
  calls.append(kw);return SimpleNamespace(status='completed',usage=SimpleNamespace(model_dump=lambda:usage),output_text='{"regions":[{"label":"chicken","polygon":[[0.1,0.1],[0.8,0.1],[0.4,0.8]],"holes":[]}]}')
 monkeypatch.setattr(openai,'OpenAI',lambda **kw:SimpleNamespace(responses=SimpleNamespace(create=create)))
 monkeypatch.setenv('OPENAI_API_KEY','test');monkeypatch.setenv('OPENAI_MODEL','gpt-6-luna');monkeypatch.setattr(vision,'local_segment',lambda *a:pytest.fail('Direct cloud mode called local model'))
 p=tmp_path/'plate.jpg';Image.new('RGB',(100,100),'white').save(p);result=vision.analyze(p,'openai')
 assert result['metadata']['provider']=='openai' and result['cloud_review'] is None
 assert result['regions'][0]['label']=='chicken' and result['regions'][0]['score'] is None
 assert calls[0]['max_output_tokens']==2000 and calls[0]['store'] is False
 assert budget.summary()['accounted_usd']==pytest.approx(.0002)

def test_fragmented_plate_keeps_unreviewed_regions_without_extra_calls(ledger,monkeypatch,tmp_path):
 import openai
 calls=[]
 def create(**kw):
  calls.append(kw);text=kw['input'][0]['content'][0]['text'];regions=json.loads(text.split('Regions: ')[1]);assert len(regions)==8
  value={'suggestions':[{'region_id':r['id'],'label':r['label'],'needs_review':True,'explanation':'Check food'} for r in regions]}
  return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda:{'input_tokens':1000,'output_tokens':100}),output_text=json.dumps(value))
 monkeypatch.setattr(openai,'OpenAI',lambda **kw:SimpleNamespace(responses=SimpleNamespace(create=create)));monkeypatch.setenv('OPENAI_API_KEY','test');monkeypatch.setenv('OPENAI_MODEL','gpt-6-luna');path=tmp_path/'plate.png';Image.new('RGB',(100,100),'white').save(path)
 regions=[{'id':str(i),'label':'chicken','polygon':[[.1,.1],[.8,.1],[.4,.8]],'holes':[]} for i in range(9)];result=vision.review(path,regions,'openai')
 assert len(calls)==1 and len(result['suggestions'])==9 and len(result['reviewed_region_ids'])==8
 assert 'Not sent to the cloud' in result['suggestions'][-1]['explanation']

def test_astra_cost_and_reservation(ledger,monkeypatch):
 from backend.app import budget
 ticket=budget.reserve('openai',model='gpt-6-astra')
 row=next(r for r in budget.summary()['calls'] if r['id']==ticket)
 assert row['reserved']==.5
 assert budget.openai_cost('gpt-6-astra',{'input_tokens':1000,'output_tokens':4000})==.21
 monkeypatch.setenv('BUILD_API_BUDGET_USD','.99')
 with pytest.raises(ValueError,match='budget'):
  budget.reserve('openai',model='gpt-6-astra')

def test_other_food_names_are_opt_in_and_bounded():
 row={'label':'unknown','food_name':'Carrots','polygon':[[.1,.1],[.8,.1],[.4,.8]],'holes':[]}
 raw=json.dumps({'regions':[row]})
 with pytest.raises(ValueError):vision.parse_cloud_regions(raw,scale=1)
 result=vision.parse_cloud_regions(raw,scale=1,allow_other=True)
 assert result[0]['food_name']=='carrots' and result[0]['review_status']=='pending'
 for bad in ['', 'x'*61, None]:
  with pytest.raises(ValueError):vision.parse_cloud_regions(json.dumps({'regions':[{**row,'food_name':bad}]}),scale=1,allow_other=True)
