"""
src/training/evaluate.py
Evaluate a saved YOLOv8 model checkpoint on the validation set.

Outputs:
  - Per-class precision, recall, mAP50, mAP50-95
  - eval_metrics.json (for DVC metrics tracking)
  - Confusion matrix and sample prediction images (optional)

Usage:
    python src/training/evaluate.py                     # auto-finds latest model
    python src/training/evaluate.py --model-version v1
    python src/training/evaluate.py --model-path models/yolov8_road_v1.pt
    python src/training/evaluate.py --save-plots        # save visual outputs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import load_params, save_json, setup_logger

logger = setup_logger("evaluate")


def find_latest_model(models_dir: Path) -> Path | None:
    """Find the most recently created .pt file in models/."""
    candidates = sorted(models_dir.glob("yolov8_road_v*.pt"))
    if not candidates:
        # Fallback: any .pt file
        candidates = sorted(models_dir.glob("*.pt"))
    return candidates[-1] if candidates else None


def evaluate_model(
    model_path: Path,
    dataset_yaml: Path,
    save_plots: bool = False,
) -> dict:
    """
    Run YOLO validation and return metrics dict.

    Args:
        model_path:   Path to .pt checkpoint.
        dataset_yaml: Path to YOLO dataset config.
        save_plots:   If True, save confusion matrix + val samples.

    Returns:
        dict of evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info("  Model Evaluator")
    logger.info("=" * 60)
    logger.info(f"  Model      : {model_path}")
    logger.info(f"  Dataset    : {dataset_yaml}")
    logger.info(f"  Save plots : {save_plots}")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(str(model_path))

    # Run validation
    results = model.val(
        data=str(dataset_yaml.resolve()),
        split="val",
        save=save_plots,
        plots=save_plots,
        verbose=True,
    )

    # ── Extract metrics ───────────────────────────────────────────────────────
    metrics = {}
    try:
        box = results.box
        metrics["mAP50"]     = float(box.map50)
        metrics["mAP50_95"]  = float(box.map)
        metrics["precision"] = float(box.mp)     # mean precision
        metrics["recall"]    = float(box.mr)     # mean recall

        # Per-class (if available)
        class_names = results.names
        if hasattr(box, "ap_class_index") and hasattr(box, "ap"):
            per_class = {}
            for idx, cls_idx in enumerate(box.ap_class_index):
                cls_name = class_names.get(int(cls_idx), f"class_{cls_idx}")
                per_class[cls_name] = {
                    "AP50": float(box.ap50[idx]) if hasattr(box, "ap50") else None,
                }
            metrics["per_class"] = per_class

    except Exception as e:
        logger.warning(f"  Partial metric extraction: {e}")

    return metrics


def print_metrics_table(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    logger.info("")
    logger.info("  ── Evaluation Results ──────────────────────────")
    logger.info(f"  mAP@50       : {metrics.get('mAP50', 'N/A'):.4f}")
    logger.info(f"  mAP@50-95    : {metrics.get('mAP50_95', 'N/A'):.4f}")
    logger.info(f"  Precision    : {metrics.get('precision', 'N/A'):.4f}")
    logger.info(f"  Recall       : {metrics.get('recall', 'N/A'):.4f}")

    per_class = metrics.get("per_class", {})
    if per_class:
        logger.info("  ── Per-Class AP@50 ─────────────────────────────")
        for cls_name, cls_metrics in per_class.items():
            ap = cls_metrics.get("AP50")
            if ap is not None:
                logger.info(f"    {cls_name:<20} {ap:.4f}")

    logger.info("  ────────────────────────────────────────────────")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model")
    parser.add_argument("--model-version", default=None,
                        help="e.g. v1 → loads models/yolov8_road_v1.pt")
    parser.add_argument("--model-path", default=None,
                        help="Direct path to .pt file (overrides --model-version)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save confusion matrix and prediction samples")
    args = parser.parse_args()

    params       = load_params()
    paths_cfg    = params["paths"]
    models_dir   = Path(paths_cfg["models_dir"])
    dataset_yaml = Path(paths_cfg["dataset_yaml"])

    # Resolve model path
    if args.model_path:
        model_path = Path(args.model_path)
    elif args.model_version:
        model_path = models_dir / f"yolov8_road_{args.model_version}.pt"
    else:
        model_path = find_latest_model(models_dir)
        if model_path is None:
            logger.error("  No model found in models/. Run train.py first.")
            sys.exit(1)
        logger.info(f"  Auto-selected model: {model_path.name}")

    if not model_path.exists():
        logger.error(f"  Model not found: {model_path}")
        sys.exit(1)

    # Evaluate
    metrics = evaluate_model(model_path, dataset_yaml, save_plots=args.save_plots)
    print_metrics_table(metrics)

    # Write eval_metrics.json for DVC
    eval_metrics_safe = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float, str))
    }
    eval_metrics_safe["model"] = model_path.name
    save_json(eval_metrics_safe, "eval_metrics.json")
    logger.info("  ✓ eval_metrics.json written")


if __name__ == "__main__":
    main()
