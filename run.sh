#!/usr/bin/env bash
# Dancing with the SARs — one-command launcher (Linux / macOS)
set -e
cd "$(dirname "$0")/backend"
PORT="${PORT:-8000}"
python3 -m pip install -q -r requirements.txt
(sleep 2; (xdg-open "http://localhost:$PORT" || open "http://localhost:$PORT") >/dev/null 2>&1) &
exec python3 -m uvicorn main:app --host 127.0.0.1 --port "$PORT"
