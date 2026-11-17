@echo off
cd /d "%~dp0"
set PY=%USERPROFILE%\chronos2\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" -m streamlit run app.py
pause
