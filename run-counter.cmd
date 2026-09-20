@echo off
REM D4 counter launcher.
REM Keep this file pure ASCII -- cmd.exe mis-parses batch files that contain
REM multibyte characters under most console codepages.
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"

"%~dp0.venv\Scripts\python.exe" -m d4tracker.monitor %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo ---- counter exited with code %RC% ----
  pause
)
exit /b %RC%
