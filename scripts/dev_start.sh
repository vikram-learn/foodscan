#!/usr/bin/env bash
set -euo pipefail

# repo root -> backend contains the Python package "app"
VENV="./backend/.venv"
PY="$VENV/bin/python"
APPDIR="./backend"
UVICORN_CMD="$PY -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir $APPDIR"

if [ ! -x "$PY" ]; then
  echo "ERROR: venv python not found at $PY"
  echo "Run: cd backend && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

echo "Using python: $PY"
echo "Starting uvicorn (dev) with app-dir=$APPDIR ..."
pkill -f uvicorn || true
exec $UVICORN_CMD
