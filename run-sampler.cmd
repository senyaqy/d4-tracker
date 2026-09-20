@echo off
REM D4 sample collector launcher.
REM Keep this file pure ASCII -- cmd.exe mis-parses batch files that contain
REM multibyte characters under most console codepages (it desyncs the parser).
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"

"%~dp0.venv\Scripts\python.exe" -m d4tracker.sampler %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo ---- sampler exited with code %RC% ----
  pause
)
exit /b %RC%
