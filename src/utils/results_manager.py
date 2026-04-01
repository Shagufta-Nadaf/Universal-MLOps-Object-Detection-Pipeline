import os
import shutil
import json
from pathlib import Path
from datetime import datetime

from src.utils.helpers import get_dataset_paths, load_params, setup_logger

logger = setup_logger("results_manager")

def package_run_results(run_name: str):
    """
    Collects all key artifacts from a run into a single well-named folder.
    Format: outputs/[Dataset]_[Workspace]_[Epochs]_[Precision]
    """
    params = load_params()
    # ── Resolve dataset-specific paths ───────────────────────────────────────
    ds_paths = get_dataset_paths(params)
    
    project = params["data"].get("project", "unknown")
    workspace = params["data"].get("workspace", "unknown")
    epochs = params["train"].get("epochs", 0)
    
    # 1. Determine precision from eval_metrics.json
    metrics_path = Path("eval_metrics.json")
    precision = 0.0
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                precision = float(metrics.get("precision", 0.0))
        except Exception:
            pass

    # 2. Create target folder name
    clean_project = project.replace("-", "_")
    clean_workspace = workspace.replace("-", "_")
    folder_name = f"{run_name}_{epochs}_{precision:.3f}"
    
    export_dir = ds_paths["outputs"] / folder_name
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(export_dir / "predictions", exist_ok=True)

    logger.info(f"  Packaging results into: {export_dir}")

    # 3. Collect Model
    # Look for the latest saved model to models/
    models_dir = Path("models")
    pt_files = sorted(models_dir.glob("*.pt"), key=os.path.getmtime)
    if pt_files:
        latest_model = pt_files[-1]
        shutil.copy2(latest_model, export_dir / f"model_{folder_name}.pt")
        logger.info(f"    ✓ Model copied")

    # 4. Collect Confusion Matrix & Evaluation Plots
    # Look in runs/pipeline_evaluate
    eval_dir = Path("runs") / "pipeline_evaluate"
    if eval_dir.exists():
        for plot in eval_dir.glob("*.png"):
            shutil.copy2(plot, export_dir / plot.name)
        logger.info(f"    ✓ Eval plots copied")

    # 5. Collect Inference/Predictions
    # Look in runs/pipeline_inference/detect
    infer_dir = Path("runs") / "pipeline_inference" / "detect"
    if infer_dir.exists():
        # Copy up to 50 sample images to keep it manageable
        sample_count = 0
        for img in infer_dir.glob("*.jpg"):
            if sample_count >= 50: break
            shutil.copy2(img, export_dir / "predictions" / img.name)
            sample_count += 1
        logger.info(f"    ✓ {sample_count} prediction samples copied")

    # 6. Copy metrics and params snapshot
    if metrics_path.exists():
        shutil.copy2(metrics_path, export_dir / "eval_metrics.json")
    if Path("params.yaml").exists():
        shutil.copy2("params.yaml", export_dir / "params.yaml")

    logger.info(f"  ✅ Packaging COMPLETE: {export_dir}")
    return export_dir
