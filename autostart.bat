@echo off
REM ─────────────────────────────────────────────────────────────
REM  EliteOmni AutoStart — runs the server silently on Windows
REM  To make this run on startup:
REM    1. Press Win+R, type: shell:startup
REM    2. Copy THIS file into that folder
REM  The server will start automatically when you log in.
REM ─────────────────────────────────────────────────────────────

set SCRIPT_DIR=%~dp0
start "EliteOmni" /MIN cmd /c "cd /d %SCRIPT_DIR% && python -m uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1 >> eliteomni.log 2>&1"

REM Open browser automatically after 5 seconds
timeout /t 5 /nobreak >nul
start http://localhost:8080
