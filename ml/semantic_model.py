"""Shared training and serving architecture."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.app.semantic_model import make_model,normalize,LABELS
