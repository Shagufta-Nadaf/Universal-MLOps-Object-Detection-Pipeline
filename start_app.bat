@echo off
echo =======================================================
echo   AutoTrain MLOps Web App
echo =======================================================
echo.
echo Starting FastAPI server...
echo.
echo Please wait roughly 3-5 seconds, then open:
echo http://localhost:8000
echo.
echo ^(Press Ctrl+C to stop the server^)
echo =======================================================
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
