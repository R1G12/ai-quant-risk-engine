@echo off
REM One double-click / one-command launcher for Windows (non-technical users).
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run_windows.ps1" %*
exit /b %ERRORLEVEL%
