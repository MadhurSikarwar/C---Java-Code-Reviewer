@echo off
REM Starts IntelliReview (backend + frontend) at http://localhost:8000
cd /d "%~dp0backend"
set PYTHONUTF8=1
set TF_CPP_MIN_LOG_LEVEL=3
"%~dp0.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
