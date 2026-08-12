@echo off
REM ---------------------------------------------------------------------------
REM  Start the MT5 -> database sync loop in the logged-on desktop session.
REM
REM  Installed into the user's Startup folder. NOT a scheduled task:
REM  mt5.initialize() attaches to the MetaTrader terminal, which needs the
REM  interactive desktop session. Task Scheduler launches in a different window
REM  station, so the call never returns and the task wedges in "Queued" forever
REM  while reporting Last Result: 0. The scheduled-task version is registered but
REM  DISABLED for exactly this reason -- do not re-enable it.
REM
REM  pythonw.exe (not python.exe) so no console window appears at login. Because
REM  pythonw has no stdout, mt5_sync.py writes logs\mt5_sync.log itself rather
REM  than relying on shell redirection.
REM
REM  Safe to run twice: mt5_sync.py takes an exclusive OS lock on
REM  logs\mt5_sync.lock, so a second launch logs one line and exits.
REM
REM  Stop it:  taskkill /F /IM pythonw.exe        (or Task Manager)
REM  Run once: .venv\Scripts\python.exe deploy\mt5_sync.py
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
set "PYTHONIOENCODING=utf-8"

REM No tasklist pre-check here any more. It only warned, and it could not work:
REM on 2026-08-07 two loops launched in the same second, each looked first and
REM saw nothing, and both started. One wedged in mt5.initialize() at 0 CPU for
REM two and a half days while still appearing alive in Task Manager. The lock in
REM mt5_sync.py is arbitrated by the kernel, so it has no such race.
start "" /B ".venv\Scripts\pythonw.exe" "deploy\mt5_sync.py" --loop 300
endlocal
