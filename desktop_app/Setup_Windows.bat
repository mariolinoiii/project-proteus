@echo off
setlocal
cd /d "%~dp0"
echo Project Proteus - Windows setup
echo Internet is required to install Python libraries.
set "PYTHON_CMD="
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3.12"
)
if not defined PYTHON_CMD (
    python -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,12)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo Python 3.11 or 3.12 was not found.
    echo Install 64-bit Python 3.11 or 3.12 from python.org, including the Python launcher.
    echo Then run this setup again.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import torch,rasterio,scipy,pyproj,matplotlib,gradio; print('Dependency imports passed.')"
if errorlevel 1 goto failed
echo SETUP COMPLETE. Double-click Start_Proteus.bat.
pause
exit /b 0
:failed
echo SETUP FAILED. Keep this window open to inspect the error above.
pause
exit /b 1
