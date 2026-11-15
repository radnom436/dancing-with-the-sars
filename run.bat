@echo off
cd /d "%~dp0backend"
if "%PORT%"=="" set PORT=8000
python -m pip install -q -r requirements.txt
start "" http://localhost:%PORT%
python -m uvicorn main:app --host 127.0.0.1 --port %PORT%
