# setup_windows.ps1 - Automated Windows Setup for the MLOps Pipeline
# Run in PowerShell as Administrator from D:\ML_PROJECTS\MLOPS
# Usage: .\setup_windows.ps1

Write-Host ""
Write-Host "========================================================"
Write-Host "  Road Scene Object Detection - Windows Setup"
Write-Host "========================================================"
Write-Host ""

# Step 1: Check if conda is installed
Write-Host "[1/5] Checking for Conda..."

$condaPath = $null
$possibleConda = @(
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "C:\ProgramData\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\anaconda3\Scripts\conda.exe",
    "C:\miniconda3\Scripts\conda.exe"
)

foreach ($p in $possibleConda) {
    if (Test-Path $p) {
        $condaPath = $p
        break
    }
}

if ($condaPath) {
    Write-Host "  OK - Conda found at: $condaPath"
} else {
    Write-Host "  ERROR - Conda NOT found."
    Write-Host ""
    Write-Host "  Please install Miniconda first:"
    Write-Host "  1. Download from:"
    Write-Host "     https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    Write-Host "  2. Run installer (check Add to PATH during install)"
    Write-Host "  3. Restart PowerShell and run this script again."
    exit 1
}

# Step 2: Create conda environment
Write-Host ""
Write-Host "[2/5] Creating conda environment 'mlops' with Python 3.11..."

& $condaPath create -n mlops python=3.11 -y 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK - Conda environment 'mlops' created."
} else {
    Write-Host "  Environment may already exist - continuing..."
}

# Step 3: Find pip in the new environment
$condaDir  = Split-Path (Split-Path $condaPath)
$envPip    = "$condaDir\envs\mlops\Scripts\pip.exe"
$envPython = "$condaDir\envs\mlops\python.exe"

if (-not (Test-Path $envPip)) {
    Write-Host "  ERROR - Could not find pip at $envPip"
    Write-Host "  Try running manually: conda activate mlops && pip install -r requirements.txt"
    exit 1
}

Write-Host "  OK - Python at: $envPython"

# Step 4: Install dependencies
Write-Host ""
Write-Host "[3/5] Installing Python dependencies (this may take 5-10 minutes)..."

& $envPip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK - All dependencies installed."
} else {
    Write-Host "  WARNING - Some packages may have failed. Check output above."
}

# Step 5: Set up .env file
Write-Host ""
Write-Host "[4/5] Setting up API key configuration..."

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  OK - Created .env file."
    Write-Host ""
    Write-Host "  ACTION REQUIRED: Open .env and add your Roboflow API key."
    Write-Host "  Get key at: https://app.roboflow.com/settings/api"
    notepad.exe .env
} else {
    Write-Host "  OK - .env file already exists."
}

# Step 6: Initialize Git and DVC
Write-Host ""
Write-Host "[5/5] Initializing Git and DVC..."

$dvcExe = "$condaDir\envs\mlops\Scripts\dvc.exe"

if (-not (Test-Path ".git")) {
    git init
    Write-Host "  OK - Git initialized."
} else {
    Write-Host "  OK - Git already initialized."
}

if (Test-Path $dvcExe) {
    if (-not (Test-Path ".dvc")) {
        & $dvcExe init
        Write-Host "  OK - DVC initialized."
    } else {
        Write-Host "  OK - DVC already initialized."
    }
} else {
    Write-Host "  WARNING - dvc.exe not found. Run 'dvc init' after activating conda."
}

git add . 2>&1 | Out-Null
git commit -m "Initial project scaffold" 2>&1 | Out-Null
Write-Host "  OK - Initial git commit done."

$envPython | Out-File -FilePath ".python_path.txt" -Encoding ascii

# Final summary
Write-Host ""
Write-Host "========================================================"
Write-Host "  SETUP COMPLETE!"
Write-Host "========================================================"
Write-Host ""
Write-Host "  HOW TO RUN:"
Write-Host ""
Write-Host "  1. Double-click activate_and_run.bat"
Write-Host "     OR open Anaconda Prompt and run:"
Write-Host "     conda activate mlops"
Write-Host "     cd D:\ML_PROJECTS\MLOPS"
Write-Host ""
Write-Host "  2. Smoke test (1 epoch, verify it works):"
Write-Host "     python scripts\run_pipeline.py --smoke-test"
Write-Host ""
Write-Host "  3. Full training run:"
Write-Host "     python scripts\run_pipeline.py"
Write-Host ""
Write-Host "  4. View results:"
Write-Host "     mlflow ui   ->  open http://127.0.0.1:5000"
Write-Host ""
Write-Host "  REMEMBER: .env file must have your ROBOFLOW_API_KEY set!"
Write-Host ""
