@echo off
rem Start CardDex with the project venv (has torch + transformers for CLIP / ResNet / hybrids).
rem Double-click this file, or run it from a terminal. Extra arguments are passed on, e.g.:
rem     run_app.bat --method hybrid-clip+rot
cd /d "%~dp0"
"%USERPROFILE%\.venvs\cardproject\Scripts\python.exe" -m cardrec.app.server %*
pause
