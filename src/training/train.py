"""
src/training/train.py
YOLOv8 training script with MLflow experiment tracking.

What this does:
  1. Loads hyperparameters from params.yaml
  2. Starts an MLflow experiment run
  3. Logs all params to MLflow
  4. Trains YOLOv8 using Ultralytics API
  5. Logs all training metrics (epoch-by-epoch) to MLflow
  6. Saves best model to models/yolov8_road_vN.pt
  7. Logs model artifact to MLflow

Usage:
    python src/training/train.py
    python src/training/train.py --run-name "yolov8s-experiment"
    python src/training/train.py --epochs 5 --smoke-test
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tracking.mlflow_logger import MLflowLogger
from src.utils.helpers import (
    ensure_dirs,
    get_next_model_version,
    load_params,
    save_json,
    setup_logger,
)

logger = setup_logger("train")


def run_training(
    params: dict,
    run_name: str | None = None,
    smoke_test: bool = False,
    epochs_override: int | None = None,
) -> dict:
    """
    Execute YOLOv8 training with MLflow tracking.

    Args:
        params:          Full params.yaml dict.
        run_name:        Human-readable MLflow run name.
        smoke_test:      If True, trains 1 epoch on tiny subset.
        epochs_override: Override epoch count from CLI.

    Returns:
        dict of final metrics logged.
    """
    train_cfg  = params["train"]
    mlflow_cfg = params["mlflow"]
    paths_cfg  = params["paths"]
    data_cfg   = params["data"]

    # ── Resolve paths ─────────────────────────────────────────────────────────
    dataset_yaml = Path(paths_cfg["dataset_yaml"])
    models_dir   = Path(paths_cfg["models_dir"])
    ensure_dirs(models_dir)

    if not dataset_yaml.exists():
        logger.error(f"  dataset.yaml not found: {dataset_yaml}")
        logger.error("  Make sure you ran preprocess.py and split_data.py first.")
        sys.exit(1)

    # ── Determine training hyperparameters ────────────────────────────────────
    model_name  = train_cfg["model"]
    epochs      = epochs_override if epochs_override else int(train_cfg["epochs"])
    batch       = int(train_cfg["batch_size"])
    imgsz       = int(train_cfg["image_size"])
    lr          = float(train_cfg["learning_rate"])
    patience    = int(train_cfg["patience"])
    device_raw  = train_cfg.get("device", "cpu")
    workers     = int(train_cfg.get("workers", 4))
    data_version = data_cfg.get("version", "v1")

    # Safe device resolution — "auto" crashes on Ultralytics when no CUDA
    try:
        import torch
        if device_raw == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        else:
            device = device_raw
    except ImportError:
        device = "cpu"
    logger.info(f"  Device resolved: {device_raw} → {device}")

    if smoke_test:
        epochs = 1
        batch  = 4
        logger.info("  [SMOKE TEST] — 1 epoch, batch=4")

    # ── Build run name ────────────────────────────────────────────────────────
    if run_name is None:
        run_name = f"{model_name.replace('.pt', '')}_{data_version}"

    # ── Determine model version for saving ───────────────────────────────────
    model_version = get_next_model_version(models_dir)
    final_model_name = f"yolov8_road_{model_version}.pt"

    logger.info("=" * 60)
    logger.info("  YOLOv8 Training")
    logger.info("=" * 60)
    logger.info(f"  Model         : {model_name}")
    logger.info(f"  Dataset YAML  : {dataset_yaml}")
    logger.info(f"  Epochs        : {epochs}")
    logger.info(f"  Batch size    : {batch}")
    logger.info(f"  Image size    : {imgsz}")
    logger.info(f"  Learning rate : {lr}")
    logger.info(f"  Device        : {device}")
    logger.info(f"  MLflow run    : {run_name}")
    logger.info(f"  Save as       : {models_dir / final_model_name}")

    # ── All params to log ─────────────────────────────────────────────────────
    logged_params = {
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch,
        "image_size": imgsz,
        "learning_rate": lr,
        "patience": patience,
        "device": device,
        "data_version": data_version,
        "dataset_yaml": str(dataset_yaml),
    }

    # ── Start training inside MLflow context ──────────────────────────────────
    with MLflowLogger(
        experiment_name=mlflow_cfg["experiment_name"],
        run_name=run_name,
        tracking_uri=mlflow_cfg["tracking_uri"],
        tags={
            "model": model_name,
            "data_version": data_version,
            "pipeline_stage": "training",
        },
    ) as tracker:

        tracker.log_params(logged_params)

        # ── Import and initialize YOLO ────────────────────────────────────────
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            sys.exit(1)

        yolo = YOLO(model_name)
        logger.info(f"  Model loaded: {model_name}")

        # ── Run training ──────────────────────────────────────────────────────
        logger.info("  Starting training...")
        results = yolo.train(
            data=str(dataset_yaml.resolve()),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr,
            patience=patience,
            device=device,
            workers=workers,
            project="runs/train",
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        # ── Find training run output dir ──────────────────────────────────────
        # Ultralytics saves to runs/detect/{name} for detection tasks
        # even if project="runs/train" is passed — search multiple locations
        runs_dir = None
        search_paths = [
            Path("runs") / "detect" / run_name,   # Ultralytics default
            Path("runs") / "train" / run_name,
            Path("runs") / "detect" / "runs" / "train" / run_name,  # nested case
        ]
        # Also search with numbered suffixes (e.g. yolov8n_v12, yolov8n_v13)
        for candidate in search_paths:
            if candidate.exists():
                runs_dir = candidate
                break
        if runs_dir is None:
            # Glob search as last resort
            candidates = sorted(Path("runs").rglob(f"{run_name}"))
            if candidates:
                runs_dir = candidates[-1]
        if runs_dir is None:
            runs_dir = Path("runs") / "train" / run_name  # fallback

        logger.info(f"  Training output dir: {runs_dir}")

        # ── Log epoch-by-epoch metrics from results.csv ───────────────────────
        tracker.log_yolo_results(runs_dir)

        # ── Log final summary metrics ─────────────────────────────────────────
        final_metrics = {}
        try:
            # Extract from Ultralytics results object
            if hasattr(results, 'results_dict'):
                rd = results.results_dict
                final_metrics = {
                    "mAP50":       float(rd.get("metrics/mAP50(B)", 0)),
                    "mAP50_95":    float(rd.get("metrics/mAP50-95(B)", 0)),
                    "precision":   float(rd.get("metrics/precision(B)", 0)),
                    "recall":      float(rd.get("metrics/recall(B)", 0)),
                    "box_loss":    float(rd.get("train/box_loss", 0)),
                    "cls_loss":    float(rd.get("train/cls_loss", 0)),
                }
                tracker.log_metrics(final_metrics)
        except Exception as e:
            logger.warning(f"  Could not extract final metrics from results object: {e}")

        # ── Save best model checkpoint ────────────────────────────────────────
        best_pt = runs_dir / "weights" / "best.pt"
        if best_pt.exists():
            dest = models_dir / final_model_name
            shutil.copy2(best_pt, dest)
            logger.info(f"  ✓ Saved best model → {dest}")
            tracker.log_artifact(str(dest))
            tracker.set_tag("saved_model", str(dest))
        else:
            logger.warning(f"  best.pt not found at {best_pt}")

        # ── Log training plots ────────────────────────────────────────────────
        for plot_file in runs_dir.glob("*.png"):
            tracker.log_artifact(str(plot_file))

        # ── Log params.yaml snapshot ──────────────────────────────────────────
        tracker.log_artifact("params.yaml")

        # ── Write metrics.json for DVC ────────────────────────────────────────
        metrics_out = {**final_metrics, "run_name": run_name, "model_version": model_version}
        save_json(metrics_out, "metrics.json")
        logger.info(f"  ✓ metrics.json written for DVC")

        logger.info("")
        logger.info("  ✅  Training complete!")
        logger.info(f"     Model saved : {models_dir / final_model_name}")
        logger.info(f"     MLflow run  : {tracker.run_id}")
        logger.info("")
        logger.info("  View results:")
        logger.info("    mlflow ui              → http://127.0.0.1:5000")
        logger.info("    python src/training/evaluate.py")
        logger.info("    python scripts/run_inference.py --model-version " + model_version)

    return final_metrics


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on road scene dataset")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--smoke-test", action="store_true",
                        help="1 epoch, small batch — just verify the pipeline works")
    args = parser.parse_args()

    params = load_params()
    run_training(
        params=params,
        run_name=args.run_name,
        smoke_test=args.smoke_test,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
