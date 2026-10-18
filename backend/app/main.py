import os,io,json,uuid,hashlib,threading
from pathlib import Path
from datetime import datetime,timezone
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image,ImageOps,UnidentifiedImageError
# Keep upload decoding independent of optional image plugins installed by ML libraries.
_UPLOAD_IMAGE_OPEN=Image.open
from . import db,budget,vision
from .schemas import Review,Analyze,Session,Experiment
ROOT=Path(__file__).resolve().parents[2];load_dotenv(ROOT/'.env')
UPLOADS=Path(os.getenv('UPLOAD_DIR') or ROOT/'data/uploads');UPLOADS.mkdir(parents=True,exist_ok=True)
Image.MAX_IMAGE_PIXELS=25_000_000
locks={};guard=threading.Lock()
def scan_lock(id):
    with guard:return locks.setdefault(id,threading.Lock())
def now():return datetime.now(timezone.utc).isoformat()
@asynccontextmanager
async def lifespan(app):
    db.init();yield
app=FastAPI(title='PlateBack',lifespan=lifespan)
def get(id,c):
    s=db.get_scan(c,id)
    if not s:raise HTTPException(404,'Scan not found')
    return s
@app.get('/api/health')
def health():
    weights=Path(os.getenv('LOCAL_MODEL_PATH') or ROOT/'artifacts/models/plateback/best.pt')
    selection_path=ROOT/'artifacts/models/local-selection.json'
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    return {'local_model_label':selection.get('display_name','Local model'),'research_ready':(ROOT/'artifacts/models/chicken-v6/settings.json').exists(),'local_ready':weights.exists(),'openai_ready':bool(os.getenv('OPENAI_API_KEY')),'openai_model':os.getenv('OPENAI_MODEL') or 'gpt-6-luna','gemini_ready':bool(os.getenv('GEMINI_API_KEY') and os.getenv('GEMINI_MODEL')),'budget':budget.summary(),'poultry_note':'Chicken predictions use a chicken/duck source class. Verify manually.'}
@app.get('/api/demo-dining')
def demo_dining():
    from .demo_dining import snapshot
    return snapshot()
@app.get('/api/demo-dining/plates/{id}/image')
def demo_image(id:str):
    from .demo_dining import image_path
    path=image_path(id)
    if path is None:raise HTTPException(404,'Demo plate not found')
    return FileResponse(path,media_type='image/jpeg')
@app.get('/api/benchmark')
def benchmark_report():
    path=ROOT/'artifacts/benchmark/summary.json'
    report=json.loads(path.read_text()) if path.exists() else {'status':'running','models':[],'note':'Benchmark in progress. No performance claim yet.'}
    latest=ROOT/'artifacts/benchmark-v4/summary.json'
    if latest.exists():report['latest_comparison']=json.loads(latest.read_text())
    chicken=ROOT/'artifacts/benchmark-chicken-final/summary.json'
    if chicken.exists():report['chicken_comparison']=json.loads(chicken.read_text())
    selection_path=ROOT/'artifacts/models/local-selection.json'
    if selection_path.exists():report['selected_local_model']=json.loads(selection_path.read_text()).get('benchmark_model','Local chicken v6')
    diagnostic=ROOT/'artifacts/astra-comparison/scores.json'
    if diagnostic.exists():report['scanner_comparison']=json.loads(diagnostic.read_text())
    report['accounted_total_usd']=budget.summary()['accounted_usd']
    return report
@app.get('/api/sample-scans')
def sample_scans():
    from .sample_scans import snapshot_scans
    return snapshot_scans()
@app.get('/api/scans')
def list_scans():
    with db.connect() as c:return [json.loads(r[0]) for r in c.execute('SELECT body FROM scans ORDER BY rowid DESC LIMIT 200')]
@app.post('/api/scans')
async def create_scan(image:UploadFile=File(...),idempotency_key:str=Form(...),session_id:str=Form('default'),demo:bool=Form(False)):
    if not 1<=len(idempotency_key)<=100:raise HTTPException(422,'Invalid request key')
    raw=await image.read(12*1024*1024+1)
    if len(raw)>12*1024*1024:raise HTTPException(413,'Images must be under 12 MB')
    digest=hashlib.sha256(raw).hexdigest()
    try:
        im=ImageOps.exif_transpose(_UPLOAD_IMAGE_OPEN(io.BytesIO(raw),formats=['JPEG','PNG','WEBP'])).convert('RGB');im.load()
        if im.width<16 or im.height<16:raise ValueError('Image is too small')
    except (UnidentifiedImageError,ValueError,OSError,Image.DecompressionBombError):raise HTTPException(422,'Upload a valid JPEG, PNG or WebP image under 25 megapixels')
    with db.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        previous=c.execute('SELECT body FROM scans WHERE idempotency_key=?',(idempotency_key,)).fetchone()
        if previous:
            s=json.loads(previous[0])
            if (s['image_hash'],s['session_id'],s['demo'])!=(digest,session_id,demo):raise HTTPException(409,'Request key was already used for a different upload')
            return s
        if session_id!='default' and not c.execute('SELECT 1 FROM sessions WHERE id=?',(session_id,)).fetchone():raise HTTPException(422,'Unknown meal session')
        id=str(uuid.uuid4());path=UPLOADS/f'{id}.jpg';im.save(path,quality=95)
        s=dict(id=id,session_id=session_id,created_at=now(),image_url=f'/api/scans/{id}/image',image_hash=digest,width=im.width,height=im.height,status='draft',demo=demo,predictions=[],regions=[],metadata=None,cloud_review=None,net_grams=None,weight_source='none',reviewed=False,annotation_source=None,error=None)
        c.execute('INSERT INTO scans VALUES(?,?,?)',(id,idempotency_key,json.dumps(s)))
    return s
@app.get('/api/scans/{id}')
def read_scan(id:str):
    with db.connect() as c:return get(id,c)
@app.get('/api/scans/{id}/image')
def image(id:str):
    with db.connect() as c:get(id,c)
    return FileResponse(UPLOADS/f'{id}.jpg',media_type='image/jpeg')
@app.post('/api/scans/{id}/analyze')
def analyze(id:str,body:Analyze):
    lock=scan_lock(id)
    if not lock.acquire(blocking=False):raise HTTPException(409,'This scan is busy')
    try:
        with db.connect() as c:
            s=get(id,c)
            if s['status']=='confirmed' or s['reviewed']:raise HTTPException(409,'Reviewed scans cannot be re-analyzed; upload a new scan to compare models')
        try:
            if body.openai_model and body.mode!='openai':raise ValueError('OpenAI model selection requires OpenAI mode')
            result=vision.analyze(UPLOADS/f'{id}.jpg',body.mode,openai_model=body.openai_model) if body.openai_model else vision.analyze(UPLOADS/f'{id}.jpg',body.mode)
        except Exception as e:
            with db.connect() as c:
                s=get(id,c);s.update(status='failed',error=f'{type(e).__name__}: inference failed. Check provider configuration, quota and local weights. Retry or draw masks manually.');db.put_scan(c,s)
            # Do not return upstream errors that may include credentials or image payloads.
            raise HTTPException(502,s['error'])
        with db.connect() as c:
            s=get(id,c);s.update(status='analyzed',predictions=result['regions'],regions=result['regions'],metadata=result['metadata'],cloud_review=result['cloud_review'],annotation_source='model',error=None);db.put_scan(c,s)
        return s
    finally:lock.release()
@app.put('/api/scans/{id}/review')
def save_review(id:str,body:Review):
    with scan_lock(id),db.connect() as c:
        c.execute('BEGIN IMMEDIATE');s=get(id,c)
        if s['status'] in ['draft','failed'] and not body.manual:raise HTTPException(422,'Explicit manual annotation is required when inference has not succeeded')
        s.update(regions=[r.model_dump() for r in body.regions],net_grams=body.net_grams,weight_source=body.weight_source,reviewed=True,annotation_source='manual' if body.manual else 'reviewed_model')
        if s['status']!='confirmed':s['status']='analyzed'
        db.put_scan(c,s)
    return s
@app.post('/api/scans/{id}/confirm')
def confirm(id:str):
    with scan_lock(id),db.connect() as c:
        c.execute('BEGIN IMMEDIATE');s=get(id,c)
        if not s['reviewed']:raise HTTPException(409,'Review and save the masks before confirming')
        s.update(status='confirmed',confirmed_at=s.get('confirmed_at') or now());db.put_scan(c,s)
    return s
@app.get('/api/sessions')
def sessions():
    with db.connect() as c:return [dict(id='default',name='General collection',meal_period='unspecified',meals_served=None,coverage_notes='Convenience sample; coverage unknown')]+[json.loads(r[0]) for r in c.execute('SELECT body FROM sessions')]
@app.post('/api/sessions')
def create_session(body:Session):
    s={'id':str(uuid.uuid4()),'created_at':now(),**body.model_dump()}
    with db.connect() as c:c.execute('INSERT INTO sessions VALUES(?,?)',(s['id'],json.dumps(s)))
    return s
@app.get('/api/experiments')
def experiments():
    with db.connect() as c:return [json.loads(r[0]) for r in c.execute('SELECT body FROM experiments')]
@app.post('/api/experiments')
def create_experiment(body:Experiment):
    e={'id':str(uuid.uuid4()),'created_at':now(),**body.model_dump()}
    with db.connect() as c:
        for sid in [body.baseline_session_id,body.intervention_session_id]:
            if sid and sid!='default' and not c.execute('SELECT 1 FROM sessions WHERE id=?',(sid,)).fetchone():raise HTTPException(422,'Unknown experiment session')
        c.execute('INSERT INTO experiments VALUES(?,?)',(e['id'],json.dumps(e)))
    return e
@app.get('/api/dashboard')
def dashboard(session_id:str|None=None,include_demo:bool=False,include_samples:bool=False):
    with db.connect() as c:scans=[json.loads(r[0]) for r in c.execute('SELECT body FROM scans')]
    if include_samples and include_demo:
        from .sample_scans import snapshot_scans
        scans+=snapshot_scans()['scans']
    scans=[s for s in scans if s['status']=='confirmed' and (include_demo or not s['demo']) and (not session_id or s['session_id']==session_id)]
    n=len(scans);categories={l:sum(l in s.get('sample_foods',[r['label'] for r in s['regions']]) for s in scans) for l in [*vision.LABELS,'unknown']}
    weighed=[s for s in scans if s['net_grams'] is not None];by_session={}
    for s in scans:
        key=s['session_id'];by_session[key]=by_session.get(key,0)+1
    def canonical(rs):return [(r['id'],r['label'],r['polygon'],r.get('holes',[]),r.get('food_name')) for r in rs]
    corrected=sum(canonical(s['regions'])!=canonical(s['predictions']) for s in scans)
    return dict(scans=n,categories=categories,presence_rates={k:v/n if n else None for k,v in categories.items()},corrected_scans=corrected,correction_rate=corrected/n if n else None,manual_scans=sum(s['annotation_source']=='manual' for s in scans),weighed_scans=len(weighed),measured_total_grams=sum(s['net_grams'] for s in weighed) if weighed else None,by_session=by_session,includes_demo=include_demo,coverage='Confirmed sampled plates only; food presence is not mass or population waste.')
if (ROOT/'frontend/dist').exists():app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True),name='frontend')
