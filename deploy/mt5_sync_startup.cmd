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
REM  Stop it:  taskkill /F /IM pythonw.exe        (or Task Manager)
REM  Run once: .venv\Scripts\python.exe deploy\mt5_sync.py
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
set "PYTHONIOENCODING=utf-8"

REM Warn if a pythonw is already up -- two loops would double every write.
REM find.exe is fully qualified: a Git Bash shell puts Unix find on PATH, which
REM does not understand /I and makes this check fail noisily.
tasklist /FI "IMAGENAME eq pythonw.exe" /FO CSV 2>nul | "%SystemRoot%\System32\find.exe" /I "pythonw.exe" >nul
if not errorlevel 1 echo [startup] pythonw.exe already running - check it is not a second sync loop >> "logs\mt5_sync.log"

start "" /B ".venv\Scripts\pythonw.exe" "deploy\mt5_sync.py" --loop 300
endlocal
