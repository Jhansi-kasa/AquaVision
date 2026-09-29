@echo off
REM =========================================================================
REM Aqua Vision SIH Backend Startup Script
REM Fixes Windows Smart App Control issues by using Python 3.11 environment
REM Disables SQLAlchemy Cython C-extension to avoid DLL load blocks
REM =========================================================================

setlocal
set DISABLE_SQLALCHEMY_CEXT=1
cd /d "%~dp0"

echo [Aqua Vision] Starting Backend Server from backend directory...
echo [Aqua Vision] Model: app/model/aqua_vision_100ep_best.pt

REM Run using verified Python 3.11 virtual environment
"%~dp0venv311\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
