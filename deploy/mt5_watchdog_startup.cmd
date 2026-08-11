@echo off
REM ---------------------------------------------------------------------------
REM  Start the MT5 sync watchdog in the logged-on desktop session.
REM
REM  Installed into the user's Startup folder as a SHORTCUT, never a copy: this
REM  script resolves the repo with %~dp0.., so a copy sitting in the Startup
REM  folder resolves to ...\Start Menu\Programs instead, silently launches
REM  nothing, and leaves only an empty logs\ folder there as evidence. That is
REM  exactly what happened to the sync loop's own startup item between
REM  2026-08-07 and 2026-08-12.
REM
REM  NOT a scheduled task, for the same reason as mt5_sync_startup.cmd: the
REM  loop it supervises needs the interactive desktop session.
REM
REM  Safe to run twice: mt5_watchdog.py takes an exclusive OS lock on
REM  logs\mt5_watchdog.lock, so a second launch logs one line and exits.
REM
REM  Stop it:  taskkill /F /IM pythonw.exe        (stops the sync loop too)
REM  Run once: .venv\Scripts\python.exe deploy\mt5_watchdog.py --once
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
set "PYTHONIOENCODING=utf-8"

start "" /B ".venv\Scripts\pythonw.exe" "deploy\mt5_watchdog.py" --loop 120
endlocal
