import os,sqlite3,json
from pathlib import Path
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[2]
def db_path():return Path(os.getenv('PLATEBACK_DB',str(ROOT/'data/plateback.db')))
@contextmanager
def connect():
    path=db_path(); path.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(path,timeout=30);con.row_factory=sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA foreign_keys=ON')
    try:
        yield con;con.commit()
    except Exception:
        con.rollback();raise
    finally:con.close()
def init():
    with connect() as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scans(id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY,body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS budget(id TEXT PRIMARY KEY,provider TEXT, reserved REAL,actual REAL,status TEXT,created TEXT,usage TEXT);
''')
        for ident,name in [('uva-ohill','O-Hill'),('uva-newcomb','Newcomb'),('uva-runk','Runk')]:
            body={'id':ident,'name':name,'meal_period':'lunch','meals_served':None,'coverage_notes':'UVA dining hall'}
            c.execute('INSERT OR IGNORE INTO sessions VALUES(?,?)',(ident,json.dumps(body)))
def get_scan(c,id):
    r=c.execute('SELECT body FROM scans WHERE id=?',(id,)).fetchone()
    return json.loads(r[0]) if r else None
def put_scan(c,s):c.execute('UPDATE scans SET body=? WHERE id=?',(json.dumps(s),s['id']))
