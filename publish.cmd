@echo off
REM Helper to publish this project to GitHub.
REM
REM Keep this file pure ASCII: cmd.exe mis-parses batch files containing
REM multibyte characters under most console codepages.
REM
REM Networking note -- READ THIS:
REM   github.com resolves to 127.0.0.1 on this machine on purpose. That is how
REM   Watt Toolkit (Steam++) accelerates it: its Steam++.Accelerator process
REM   listens on 80/443 and forwards the traffic. Do NOT delete those hosts
REM   entries -- that would break the accelerator, not fix anything.
REM
REM   The only setting needed is to stop schannel from doing a certificate
REM   revocation check, which fails behind that local proxy:
REM       git config --global http.schannelCheckRevoke false
REM       git config --global http.sslBackend schannel
REM   Both are already set. Verified working: connecting to GitHub takes
REM   13-29 seconds, so be patient; retry once if an operation times out.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "REPO=d4-tracker"
set "USER=senyaqy"

where git >nul 2>&1
if errorlevel 1 (
  echo.
  echo [ERROR] git not found. Install it first:
  echo         winget install --id Git.Git -e
  echo     or  https://git-scm.com/download/win
  echo.
  pause
  exit /b 1
)

if not exist ".git" (
  echo [1/5] git init
  git init -b main || goto :fail
) else (
  echo [1/5] already a git repository
)

echo [2/5] ensure the proxy-friendly TLS settings are present
git config --global http.schannelCheckRevoke false
git config --global http.sslBackend schannel

echo [3/5] git add -A
git add -A || goto :fail

echo [4/5] staged files -- expect 40 small files,
echo       NO .venv / dist / data / samples:
echo.
git status --short

echo.
echo [5/5] commit
git commit -m "Update" || echo (nothing to commit)

echo.
echo ============================================================
echo  Next, by hand:
echo.
echo  1) Create an EMPTY repo named "%REPO%" on https://github.com
echo     Do NOT let it add README / .gitignore / license.
echo.
echo  2) Create a Personal Access Token (classic, scope: repo):
echo     https://github.com/settings/tokens
echo     A password will NOT work -- GitHub disabled password auth for git
echo     in 2021. Use the token as the password when git prompts you.
echo.
echo  3) Push. Do NOT embed the token in the URL -- that would store it
echo     in plaintext in .git/config.
echo.
echo       git remote add origin https://github.com/%USER%/%REPO%.git
echo       git push -u origin main
echo.
echo     Username: %USER%     Password: paste the token
echo.
echo  4) Expect it to be SLOW (the accelerator path takes 13-29s per
echo     round trip). If it times out, just run the push again.
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [ERROR] command failed, see above.
pause
exit /b 1
