"""Install bundled fixtures without overwriting local files."""
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
FILES = {"demo-dining.json": "data/demo-dining.json", "benchmark.json": "artifacts/benchmark/summary.json", "local-selection.json": "artifacts/models/local-selection.json", "semantic-v4-settings.json": "artifacts/models/semantic-v4/settings.json"}
for source, target in FILES.items():
    dest = ROOT / target
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(ROOT / "assets" / source, dest)
print("Sample records and benchmark files are ready.")
