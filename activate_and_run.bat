@echo off
REM ============================================================
REM activate_and_run.bat
REM Double-click this file to activate the conda environment
REM and open a ready-to-use command prompt for the MLOps project.
REM ============================================================

title Road Scene MLOps Pipeline

echo.
echo ============================================================
echo   Road Scene Object Detection - MLOps Pipeline
echo ============================================================
echo.

REM ── Try to find conda ────────────────────────────────────────
set CONDA_EXE=
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat
    goto :found_conda
)
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat
    goto :found_conda
)
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    set CONDA_ACTIVATE=C:\ProgramData\miniconda3\Scripts\activate.bat
    goto :found_conda
)
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    set CONDA_ACTIVATE=C:\ProgramData\anaconda3\Scripts\activate.bat
    goto :found_conda
)

echo [ERROR] Conda not found!
echo Please install Miniconda from:
echo   https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
echo.
pause
exit /b 1

:found_conda
echo [OK] Found conda at: %CONDA_ACTIVATE%

REM ── Activate mlops environment ────────────────────────────────
call "%CONDA_ACTIVATE%" mlops

if errorlevel 1 (
    echo.
    echo [ERROR] Could not activate 'mlops' environment.
    echo Run setup_windows.ps1 first to create it.
    echo.
    pause
    exit /b 1
)

echo [OK] Conda environment 'mlops' activated.
echo.
echo ============================================================
echo   Available commands:
echo.
echo   -- Run pipeline stages:
echo   python scripts\run_pipeline.py --smoke-test   (quick 1-epoch test)
echo   python scripts\run_pipeline.py                (full training run)
echo   python scripts\run_pipeline.py --start-from train  (skip download)
echo.
echo   -- Run individual stages:
echo   python src\data\download_data.py
echo   python src\data\preprocess.py
echo   python src\data\split_data.py
echo   python src\training\train.py
echo   python src\training\evaluate.py
echo.
echo   -- View MLflow results:
echo   mlflow ui
echo   (then open http://127.0.0.1:5000 in your browser)
echo.
echo   -- Run inference on images:
echo   python scripts\run_inference.py --source data\processed\images\val\
echo.
echo   -- DVC pipeline:
echo   dvc repro          (run all stages via DVC)
echo   dvc status         (see what changed)
echo   dvc dag            (view pipeline graph)
echo   dvc metrics show   (compare experiment metrics)
echo ============================================================
echo.

REM ── Stay in project directory ─────────────────────────────────
cd /d D:\ML_PROJECTS\MLOPS

REM ── Open interactive shell ────────────────────────────────────
cmd /k
