@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run Setup_Windows.bat first.
    pause
    exit /b 1
)
echo Starting Project Proteus. Keep this window open while using the app.
echo The browser should open at http://127.0.0.1:7860
echo Press Ctrl+C in this window to stop the application.
".venv\Scripts\python.exe" app.py
if errorlevel 1 echo The application stopped with an error. Read the messages above.
pause
