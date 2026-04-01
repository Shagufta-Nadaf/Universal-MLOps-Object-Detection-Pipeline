"""
scripts/run_inference.py
Run YOLOv8 object detection inference on images or video.

Draws bounding boxes for: car, pedestrian, cyclist, traffic_sign

Usage:
    # Single image
    python scripts/run_inference.py --source path/to/image.jpg

    # Folder of images
    python scripts/run_inference.py --source path/to/images/

    # Video file
    python scripts/run_inference.py --source path/to/video.mp4

    # Webcam (device 0)
    python scripts/run_inference.py --source 0

    # Use specific model version
    python scripts/run_inference.py --source images/ --model-version v1

    # Adjust confidence threshold
    python scripts/run_inference.py --source images/ --conf 0.4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.helpers import load_params, setup_logger

logger = setup_logger("run_inference")

# ─── Class display config ──────────────────────────────────────────────────────
CLASS_COLORS = {
    0: (0, 212, 255),     # car         → cyan
    1: (255, 100, 80),    # pedestrian  → red-orange
    2: (0, 255, 128),     # cyclist     → green
    3: (255, 200, 0),     # traffic_sign → yellow
}

CLASS_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist", 3: "traffic_sign"}


def find_model(models_dir: Path, version: str | None) -> Path:
    """Resolve model path from version tag or auto-pick latest."""
    if version:
        path = models_dir / f"yolov8_road_{version}.pt"
        if not path.exists():
            logger.error(f"Model not found: {path}")
            logger.error(f"Available models: {list(models_dir.glob('*.pt'))}")
            sys.exit(1)
        return path

    import re
    candidates = list(models_dir.glob("yolov8_road_v*.pt"))
    if not candidates:
        logger.error(f"No trained models found in {models_dir}")
        logger.error("Run `python scripts/run_pipeline.py` or `python src/training/train.py` first.")
        sys.exit(1)
    
    # Numeric Sort: extract the 'vX' number and sort by it
    candidates.sort(key=lambda x: int(re.search(r"v(\d+)", x.name).group(1)) if re.search(r"v(\d+)", x.name) else 0)
    latest = candidates[-1]
    logger.info(f"  Auto-selected latest model (numeric sort): {latest.name}")
    return latest


def run_inference(
    source: str,
    model_path: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    save_output: bool = True,
    output_dir: Path | None = None,
    show: bool = False,
) -> None:
    """
    Run detection on source (image/folder/video/webcam).

    Args:
        source:     Path to image, folder, video, or webcam index.
        model_path: Path to .pt checkpoint.
        conf:       Confidence threshold (0-1).
        iou:        NMS IoU threshold.
        save_output: If True, save annotated results.
        output_dir: Where to save results (default: runs/inference/).
        show:       Show live preview window (requires display).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  Road Scene Object Detection — Inference")
    logger.info("=" * 60)
    logger.info(f"  Model      : {model_path}")
    logger.info(f"  Source     : {source}")
    logger.info(f"  Confidence : {conf}")
    logger.info(f"  IoU thresh : {iou}")

    model = YOLO(str(model_path))

    out_path = str(output_dir) if output_dir else "runs/inference"

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=save_output,
        project=str(Path(out_path).resolve()),
        name="detect",
        exist_ok=True,
        show=show,
        verbose=True,
        line_width=2,
    )

    # ── Print detection summary ───────────────────────────────────────────────
    from collections import Counter
    total_detections = 0
    all_class_counts = Counter()

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for cls_id in boxes.cls.tolist():
                cls_name = CLASS_NAMES.get(int(cls_id), f"class_{int(cls_id)}")
                all_class_counts[cls_name] += 1
                total_detections += 1

    logger.info("")
    logger.info("  ── Detection Summary ──────────────────────────")
    logger.info(f"  Images processed : {len(results)}")
    logger.info(f"  Total detections : {total_detections}")
    for cls_name, count in sorted(all_class_counts.items()):
        logger.info(f"    {cls_name:<20} {count:>4} detections")

    if save_output:
        logger.info(f"  ✅  Annotated images saved → {out_path}/detect/")
    logger.info("=" * 60)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run road scene object detection inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_inference.py --source data/processed/images/val/
  python scripts/run_inference.py --source my_photo.jpg --conf 0.5
  python scripts/run_inference.py --source 0 --show            # webcam
        """,
    )
    parser.add_argument("--source", required=True,
                        help="Input: image path, folder, video, or camera index (0)")
    parser.add_argument("--model-version", default=None,
                        help="Model version to use, e.g. v1 (auto=latest)")
    parser.add_argument("--model-path", default=None,
                        help="Direct path to .pt file (overrides --model-version)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save annotated results")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output images")
    parser.add_argument("--show", action="store_true",
                        help="Show live detection window")
    args = parser.parse_args()

    params     = load_params()
    models_dir = Path(params["paths"]["models_dir"])

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = find_model(models_dir, args.model_version)

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_inference(
        source=args.source,
        model_path=model_path,
        conf=args.conf,
        iou=args.iou,
        save_output=not args.no_save,
        output_dir=output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
