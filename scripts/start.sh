#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then echo 'Run scripts/setup.sh first.'; exit 1; fi
if [ ! -d frontend/dist ]; then (cd frontend && npm run build); fi
exec .venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
