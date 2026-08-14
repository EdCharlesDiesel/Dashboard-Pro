@echo off
REM ---------------------------------------------------------------------------
REM  Task Scheduler entry point for deploy/mt5_trade_import.py.
REM
REM  Thin wrapper for the same reason as mt5_sync.cmd: batch eats %s tokens in
REM  an inlined `python -c`, so the logic stays in the .py where it can be read,
REM  tested and version-controlled.
REM
REM  Registered by:
REM    schtasks /create /tn "Dashboard-Pro MT5 Trade Import" /sc weekly ^
REM             /d SUN /st 20:00 ^
REM             /tr "C:\x\Dashboard-Pro\deploy\mt5_trade_import.cmd"
REM
REM  Needs the MT5 terminal running (Windows-only package, session-resident
REM  link) — the same precondition as the sync loop.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
set "PYTHONIOENCODING=utf-8"

".venv\Scripts\python.exe" "deploy\mt5_trade_import.py" >> "logs\mt5_trade_import.log" 2>&1
exit /b %ERRORLEVEL%
