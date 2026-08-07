@echo off
REM ---------------------------------------------------------------------------
REM  Task Scheduler entry point for deploy/mt5_sync.py.
REM
REM  Deliberately a thin wrapper: an earlier version inlined the Python with
REM  `python -c "...%s..."` and batch silently ate every %s as a variable
REM  reference, so the task "ran" and logged a TypeError forever. Keep the logic
REM  in the .py file where it can be read, tested and version-controlled.
REM
REM  Registered by:
REM    schtasks /create /tn "Dashboard-Pro MT5 Sync" /sc minute /mo 5 ^
REM             /tr "C:\x\Dashboard-Pro\deploy\mt5_sync.cmd"
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
set "PYTHONIOENCODING=utf-8"

".venv\Scripts\python.exe" "deploy\mt5_sync.py" >> "logs\mt5_sync.log" 2>&1
exit /b %ERRORLEVEL%
