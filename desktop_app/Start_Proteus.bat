@echo off
setlocal
cd /d "%~dp0"
title Project Proteus v3
if exist ".venv\Scripts\python.exe" goto ready
py -3 -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
if errorlevel 1 goto nopy
py -3 -m venv .venv
if errorlevel 1 goto failed
:ready
if exist ".venv\.proteus_ready" goto launch
".venv\Scripts\python.exe" -u bootstrap.py
if errorlevel 1 goto failed
:launch
".venv\Scripts\python.exe" -u application\app.py
if errorlevel 1 goto failed
echo Project Proteus v3 stopped.
goto end
:nopy
echo Install 64-bit Python 3.10 or newer with the Windows py launcher, then retry.
echo Python and drivers are not installed automatically.
goto end
:failed
echo STARTUP FAILED. Read the error above. See README.md for troubleshooting.
:end
pause
