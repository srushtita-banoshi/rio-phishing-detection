@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting RIO Phishing Detection & Analysis System...
echo ========================================================
echo Opening Streamlit UI at http://localhost:8501 ...
.\.venv\Scripts\python.exe -m streamlit run app.py --browser.gatherUsageStats=false
pause
