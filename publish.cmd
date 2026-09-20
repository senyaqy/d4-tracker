@echo off
REM Commit local changes and push them to GitHub.
REM
REM The repository already exists and the first push is done:
REM     https://github.com/senyaqy/d4-tracker
REM
REM Keep this file pure ASCII: cmd.exe mis-parses batch files containing
REM multibyte characters under most console codepages.
REM
REM Networking note:
REM   github.com resolves to 127.0.0.1 on this machine on purpose -- that is how
REM   Watt Toolkit (Steam++) accelerates it. Do NOT delete those hosts entries.
REM   What makes git work through that local proxy is:
REM       git config --global http.schannelCheckRevoke false
REM       git config --global http.sslBackend schannel
REM   Both are already set. Round trips take 13-29 seconds, so be patient.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "GITDIR=D:\deepseek\PortableGit\cmd"
set "GHDIR=D:\deepseek\gh\bin"
set "PATH=%GITDIR%;%GHDIR%;%PATH%"

where git >nul 2>&1
if errorlevel 1 (
  echo [ERROR] git not found at %GITDIR%
  pause
  exit /b 1
)

echo === changes to be committed ===
git status --short
echo.

git add -A || goto :fail

set "MSG=%~1"
if "%MSG%"=="" set "MSG=Update"

echo === committing: %MSG% ===
git commit -m "%MSG%" || echo (nothing to commit)
echo.

echo === pushing to origin/main (may take 15-30s) ===
git push || goto :fail

echo.
echo === done ===
git log --oneline -3
git status --short --branch
echo.
pause
exit /b 0

:fail
echo.
echo [ERROR] command failed, see above. If it timed out, just run this again --
echo         the accelerated route is slow but the push is resumable.
pause
exit /b 1
