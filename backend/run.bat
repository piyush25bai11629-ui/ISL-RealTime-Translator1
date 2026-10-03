@echo off
REM Starts the API on http://localhost:8000
cd /d "%~dp0"
python -m uvicorn app:app --reload --port 8000
