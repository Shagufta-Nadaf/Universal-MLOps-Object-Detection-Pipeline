import subprocess
import os
import json
from pathlib import Path

# Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
LOGS_DIR = APP_DIR / "logs"
STATUS_FILE = APP_DIR / "status.json"
LOG_FILE = LOGS_DIR / "current_run.log"

os.makedirs(LOGS_DIR, exist_ok=True)

def update_status(status: str, progress: int = 0, error: str = None):
    data = {"status": status, "progress": progress, "error": error}
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

def start_training(run_name: str):
    """
    Runs the pipeline script in the background and streams logs to current_run.log
    """
    update_status("running", progress=5)
    
    # Initialize log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"--- Starting MLOps Pipeline: {run_name} ---\n")
    
    # Command to run the pipeline
    cmd = [
        "python", str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--run-name", run_name
    ]
    
    # Run process and capture output directly to log file
    try:
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as log_file:
            # Force UTF-8 encoding in Python's sys.stdout
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env
            )
            
            # Save PID to allow stopping the process
            pid_file = APP_DIR / "current_pid.txt"
            with open(pid_file, "w") as f:
                f.write(str(process.pid))
            
            # Wait for process to complete
            process.wait()
            
            if pid_file.exists():
                pid_file.unlink()

            if process.returncode == 0:
                update_status("done", progress=100)
                log_file.write(f"\n✅ Pipeline {run_name} completed successfully!\n")
            else:
                update_status("failed", progress=0, error="Pipeline failed. Check logs.")
                log_file.write(f"\n❌ Pipeline {run_name} failed with exit code {process.returncode}.\n")
                
    except Exception as e:
        update_status("failed", progress=0, error=str(e))
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n❌ Exception during training: {str(e)}\n")
