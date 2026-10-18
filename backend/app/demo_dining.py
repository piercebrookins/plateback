"""Isolated fictional dining-hall scenario. Never mixed into actual scan analytics."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def snapshot():
    p=ROOT/'data/demo-dining.json'
    if not p.exists():raise ValueError('Run scripts/seed_dining_demo.py first')
    return json.loads(p.read_text())
def image_path(id):
    row=next((r for r in snapshot()['plates'] if r['id']==id),None)
    if row is None:return None
    return ROOT/row['image_path']
