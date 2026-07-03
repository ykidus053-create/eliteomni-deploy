@echo off
title EliteOmni - Install Dependencies
color 0B

echo.
echo  ========================================
echo   EliteOmni v16 - Installing
echo  ========================================
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install all required packages
pip install fastapi==0.111.0
pip install "uvicorn[standard]==0.29.0"
pip install llama-cpp-python==0.3.22
pip install faiss-cpu
pip install numpy
pip install requests

echo.
echo  ========================================
echo   Installation complete!
echo   Now double-click start.bat to run.
echo  ========================================
echo.
pause
