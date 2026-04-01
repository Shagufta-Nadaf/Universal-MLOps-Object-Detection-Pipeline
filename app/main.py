import os
import json
import asyncio
from pathlib import Path
from typing import Optional
import yaml
from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
import signal

from app.trainer import start_training, update_status, STATUS_FILE, LOG_FILE
from src.utils.helpers import load_params, setup_logger, get_dataset_paths

logger = setup_logger("webapp")

# Application paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_FILE = PROJECT_ROOT / "eval_metrics.json"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
ENV_FILE = PROJECT_ROOT / ".env"
PID_FILE = APP_DIR / "current_pid.txt"

app = FastAPI(title="Shagufta's AI Studio")

# Serve static files and HTML templates
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Ensure required directories exist
os.makedirs(APP_DIR / "logs", exist_ok=True)
os.makedirs(APP_DIR / "static", exist_ok=True)
os.makedirs(APP_DIR / "templates", exist_ok=True)

# Helper: Read YAML safely
def read_params():
    try:
        with open(PARAMS_FILE, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

# Helper: Write YAML safely
def write_params(data):
    with open(PARAMS_FILE, "w") as f:
        yaml.dump(data, f, sort_keys=False)

@app.on_event("startup")
def startup_event():
    # Reset status file on boot
    update_status("idle", progress=0)
    
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main frontend UI."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/status")
def get_status():
    """Poll endpoint to get current training state."""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        else:
            return {"status": "idle", "progress": 0}
    except Exception:
        return {"status": "idle", "progress": 0}

@app.post("/train")
async def train(
    background_tasks: BackgroundTasks,
    api_key: str = Form(""),
    workspace: str = Form(...),
    project: str = Form(...),
    version: int = Form(...),
    max_images: int = Form(400),
    epochs: int = Form(30),
    image_size: int = Form(320)
):
    """
    Accepts form data, overwrites config, and starts training.
    """
    # 1. Check if already running
    current_status = get_status()["status"]
    if current_status == "running":
        return {"error": "Training already in progress."}

    # 2. Handle API Key
    env_content = ""
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            env_content = f.read()
            
    if "ROBOFLOW_API_KEY=" not in env_content:
        if not api_key:
            return {"error": "Roboflow API Key is required for the first run! Please enter it."}
        else:
            with open(ENV_FILE, "a") as f:
                f.write(f"\nROBOFLOW_API_KEY={api_key}\n")
    
    # If the user provides a new key, update it
    elif api_key and f"ROBOFLOW_API_KEY={api_key}" not in env_content:
        # For simplicity, we just append it (dotenv takes the last one)
        with open(ENV_FILE, "a") as f:
            f.write(f"\nROBOFLOW_API_KEY={api_key}\n")
            
    # 3. Update params.yaml
    params = read_params()
    if "data" not in params: params["data"] = {}
    if "split" not in params: params["split"] = {}
    if "train" not in params: params["train"] = {}

    params["data"]["workspace"] = workspace
    params["data"]["project"] = project
    params["data"]["roboflow_version"] = version
    
    # Increment tracking version 
    current_v_str = str(params["data"].get("version", "v1")).replace("v", "")
    try:
        current_v = int(current_v_str)
    except ValueError:
        current_v = 1
        
    next_v = current_v + 1
    run_name = f"{project}-v{next_v}"
    params["data"]["version"] = f"v{next_v}"
    
    params["split"]["max_images"] = max_images
    params["train"]["epochs"] = epochs
    params["train"]["image_size"] = image_size
    
    write_params(params)

    # 4. Clear old metrics if any
    if METRICS_FILE.exists():
        METRICS_FILE.unlink()
        
    # Clear log file
    open(LOG_FILE, 'w').close()
    
    # 5. Start background task
    background_tasks.add_task(start_training, run_name)
    return {"message": "Training started", "run_name": run_name}

@app.post("/stop")
async def stop_training_job():
    """Kills the active training background task."""
    current_status = get_status()["status"]
    if current_status != "running":
        return {"error": "Training is not currently running."}

    if PID_FILE.exists():
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
                
            os.kill(pid, signal.SIGTERM)
            
            # Additional cleanup
            update_status("failed", progress=0, error="Job cancelled by user.")
            with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
                f.write("\n\n❌ [Shagufta AI] Training cancelled by user via UI.\n")
                
            PID_FILE.unlink()
            return {"message": "Training stop signal sent"}
            
        except Exception as e:
            return {"error": f"Failed to stop training: {e}"}
    
    return {"error": "Could not find process ID."}

@app.get("/download/confusion_matrix")
def download_cm():
    import shutil
    params = load_params()
    ds_paths = get_dataset_paths(params)
    
    # 1. Primary path: Look for latest folder in outputs/[workspace]/[project]/
    output_base = ds_paths["outputs"]
    if output_base.exists():
        subdirs = [os.path.join(output_base, d) for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
        if subdirs:
            latest_run = max(subdirs, key=os.path.getmtime)
            val_dir = Path(latest_run)
            logger.info(f"  Found latest output dir for download: {val_dir}")
        else:
            val_dir = PROJECT_ROOT / "runs" / "pipeline_evaluate"
    else:
        val_dir = PROJECT_ROOT / "runs" / "pipeline_evaluate"
    
    # 2. Strict Check: If no output dir found, error out. No more searching in runs/ folder!
    if not val_dir.exists():
        return {"error": "Confusion matrix not found for this project. Did you run a training session yet?"}
    
    zip_path = APP_DIR / "static" / "confusion_matrix.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), 'zip', val_dir)
    return FileResponse(
        path=zip_path,
        filename="confusion_matrices.zip",
        media_type="application/zip"
    )

@app.get("/download/val_predictions")
def download_predictions():
    import shutil
    params = load_params()
    ds_paths = get_dataset_paths(params)

    # 1. Primary path: Look for latest folder in outputs/[workspace]/[project]/
    output_base = ds_paths["outputs"]
    if output_base.exists():
        subdirs = [os.path.join(output_base, d) for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
        if subdirs:
            latest_run = max(subdirs, key=os.path.getmtime)
            infer_dir = Path(latest_run) / "predictions"
            if not infer_dir.exists(): infer_dir = Path(latest_run) # Fallback to parent
        else:
            infer_dir = PROJECT_ROOT / "runs" / "pipeline_inference" / "detect"
    else:
        infer_dir = PROJECT_ROOT / "runs" / "pipeline_inference" / "detect"
    
    # 2. Strict Check: No more searching in runs/ folder for old pothole data!
    if not infer_dir.exists():
        return {"error": "Inference predictions not found for this project. Did you run inference yet?"}
        
    zip_path = APP_DIR / "static" / "val_predictions.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), 'zip', infer_dir)
    return FileResponse(
        path=zip_path,
        filename="val_predictions.zip",
        media_type="application/zip"
    )

@app.get("/logs/stream")
async def stream_logs(request: Request):
    """Server-Sent Events endpoint to stream pipeline logs."""
    
    async def log_reader():
        if not LOG_FILE.exists():
            open(LOG_FILE, 'a').close()
            
        pos = 0 # Start reading from the beginning
        
        while True:
            if await request.is_disconnected():
                break

            try:
                with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    lines = f.readlines()
                    pos = f.tell()
                    
                if lines:
                    for line in lines:
                        # Only send non-empty lines, properly formatted for SSE
                        clean_line = line.strip().replace("\n", " ").replace("\r", "")
                        if clean_line:
                            yield dict(data=clean_line)
                            
                status = get_status()["status"]
                if status in ["done", "failed"] and not lines:
                    yield dict(data=f"--- PIPELINE FINISHED: {status.upper()} ---")
                    break
                    
            except Exception as e:
                yield dict(data=f"Error reading logs: {str(e)}")
                
            await asyncio.sleep(0.5)
            
    return EventSourceResponse(log_reader())

@app.get("/download/model")
def download_model():
    """Finds the most recent model in models/ and returns it."""
    # Find newest .pt file
    model_files = list(MODELS_DIR.glob("*.pt"))
    if not model_files:
        return {"error": "No trained model found."}
        
    latest_model = max(model_files, key=os.path.getmtime)
    return FileResponse(
        path=latest_model,
        filename=latest_model.name,
        media_type="application/octet-stream"
    )

@app.get("/download/metrics")
def download_metrics():
    """Returns the evaluation metrics JSON."""
    if not METRICS_FILE.exists():
        return {"error": "No metrics available. Did training finish?"}
        
    return FileResponse(
        path=METRICS_FILE,
        filename="eval_metrics.json",
        media_type="application/json"
    )
    
@app.get("/metrics")
def get_metrics_json():
    """Return metrics as plain JSON for UI parsing."""
    try:
        if METRICS_FILE.exists():
            with open(METRICS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}
