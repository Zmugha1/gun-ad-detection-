@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Create venv first: py -m venv .venv && .venv\Scripts\activate
  echo Then: pip install -r requirements.txt
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app.py
pause
