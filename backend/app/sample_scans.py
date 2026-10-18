"""Read-only fictional records; never inserted into the actual scans table."""
from datetime import datetime,timedelta
from .demo_dining import snapshot

def snapshot_scans():
    data=snapshot();scans=[];sessions=[]
    for week in range(5):
        for s in data['sessions']:
            date=(datetime.fromisoformat(s['date'])-timedelta(days=7*week)).date().isoformat()
            sessions.append({'id':f'sample-{week}-{s["id"]}','name':f'{s["hall"]} · {date} · {s["meal"]}','meal_period':'Lunch','meals_served':None,'coverage_notes':'Fictional sample session'})
        for i,p in enumerate(data['plates']):
            date=datetime.fromisoformat(p['date'])-timedelta(days=7*week)+timedelta(hours=12,minutes=i%90)
            scans.append(dict(id=f'PB-SAMPLE-{week*len(data["plates"])+i+1:04d}',image_url=p['image_url'],status='confirmed',regions=[],predictions=[],net_grams=None,weight_source='none',reviewed=True,demo=True,sample=True,sample_foods=p['foods'],meal_name=p['meal'],session_id=f'sample-{week}-{p["session_id"]}',created_at=date.isoformat(),error=None,metadata=None,annotation_source='sample',cloud_review=None))
    return {'scans':sorted(scans,key=lambda s:s['created_at'],reverse=True),'sessions':sessions,'classification':'fictional sample data','read_only':True}
