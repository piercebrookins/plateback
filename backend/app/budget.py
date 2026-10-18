"""Durable pessimistic reservation: unknown outcomes retain their full charge ceiling."""
import os,uuid,json
from datetime import datetime,timezone
from .db import connect
CAP=5.0
# Each request is restricted to <=9 512px images, <=8k prompt chars and <=2k output tokens.
# Luna/mini reserve $0.10. Astra reserves $0.50 for <=9 512px images,
# <=8k prompt chars and <=4k output tokens at standard short-context rates.
def reserve(provider,model=None):
    ceiling=0.50 if provider=='openai' and model=='gpt-6-astra' else 0.10 if provider in ['openai','openrouter'] else 0.50
    cap=min(CAP,float(os.getenv('BUILD_API_BUDGET_USD') or CAP))
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        spent=c.execute('SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM budget').fetchone()[0]
        if spent+ceiling>cap:raise ValueError('API budget exhausted; use local mode. Pending/failed calls retain their reservation.')
        run_id=os.getenv('PLATEBACK_BUDGET_RUN')
        if run_id:
            # Missing registration fails closed; the scope cannot silently reset.
            run=c.execute('SELECT cap FROM budget_runs WHERE id=?',(run_id,)).fetchone()
            if not run:raise ValueError('API run budget is not registered')
            used=c.execute('SELECT COALESCE(SUM(COALESCE(b.actual,b.reserved)),0) FROM budget b JOIN budget_run_calls r ON r.call_id=b.id WHERE r.run_id=?',(run_id,)).fetchone()[0]
            if used+ceiling>run[0]+1e-12:raise ValueError('API run budget exhausted; use local mode.')
        id=str(uuid.uuid4());c.execute('INSERT INTO budget VALUES(?,?,?,?,?,?,?)',(id,provider,ceiling,None,'reserved',datetime.now(timezone.utc).isoformat(),'{}'))
        if run_id:c.execute('INSERT INTO budget_run_calls VALUES(?,?)',(id,run_id))
    return id
def settle(id,actual,usage):
    with connect() as c:c.execute('UPDATE budget SET actual=?,status=?,usage=? WHERE id=?',(actual,'completed',json.dumps(usage),id))
def summary():
    with connect() as c:
        rows=[dict(r) for r in c.execute('SELECT * FROM budget')]
    return {'cap_usd':min(CAP,float(os.getenv('BUILD_API_BUDGET_USD') or CAP)),'accounted_usd':sum(r['actual'] if r['actual'] is not None else r['reserved'] for r in rows),'calls':rows,'scope':'PlateBack calls only; unknown outcomes retain reservations. Estimates, not provider billing.'}

# Standard-tier short-context rates, verified 2026-10-17. Historical ledger entries stay unchanged.
OPENAI_PRICES = {
    'gpt-6-astra': (10.0, 1.0, 12.50, 50.0),
    'gpt-6-luna': (0.10, 0.01, 0.125, 0.50),
    'gpt-4.1-mini': (0.40, 0.10, 0.40, 1.60),
    'gpt-4.1-mini-2025-04-14': (0.40, 0.10, 0.40, 1.60),
}
def openai_cost(model, usage):
    normal, cached, writes, output = OPENAI_PRICES[model]
    detail = usage.get('input_tokens_details') or {}
    hit = detail.get('cached_tokens') or 0
    write = detail.get('cache_write_tokens') or 0
    regular = max(0, usage['input_tokens'] - hit - write)
    return (regular*normal + hit*cached + write*writes + usage['output_tokens']*output)/1e6

def start_run(run_id,limit_usd):
    """Create an immutable per-run ceiling; requests opt in with PLATEBACK_BUDGET_RUN."""
    if not run_id or not 0<limit_usd<=CAP:raise ValueError('Invalid run budget')
    with connect() as c:
        c.execute('CREATE TABLE IF NOT EXISTS budget_runs(id TEXT PRIMARY KEY,cap REAL NOT NULL)')
        c.execute('CREATE TABLE IF NOT EXISTS budget_run_calls(call_id TEXT PRIMARY KEY,run_id TEXT NOT NULL)')
        c.execute('INSERT OR IGNORE INTO budget_runs VALUES(?,?)',(run_id,limit_usd))
        if c.execute('SELECT cap FROM budget_runs WHERE id=?',(run_id,)).fetchone()[0]!=limit_usd:raise ValueError('Existing run ceiling cannot change')

def run_summary(run_id):
    with connect() as c:
        row=c.execute('SELECT cap FROM budget_runs WHERE id=?',(run_id,)).fetchone()
        spent=c.execute('SELECT COALESCE(SUM(COALESCE(b.actual,b.reserved)),0) FROM budget b JOIN budget_run_calls r ON r.call_id=b.id WHERE r.run_id=?',(run_id,)).fetchone()[0]
    return {'id':run_id,'cap_usd':row[0],'accounted_usd':spent}
