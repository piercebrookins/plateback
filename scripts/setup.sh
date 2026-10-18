#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements.lock
(cd frontend && npm ci && npm run build)
.venv/bin/python scripts/prepare_demo.py
echo 'Setup complete. Run scripts/start.sh and open http://127.0.0.1:8000'
