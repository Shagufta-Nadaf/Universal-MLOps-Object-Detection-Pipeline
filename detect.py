"""
detect.py — Pothole Detection on Test Images
=============================================
Drop your images into the  test_images/  folder, then run:

    python detect.py

Results (annotated images with bounding boxes) are saved to:

    results/

You can also point it at any image, folder, or video:

    python detect.py --source path/to/my_photo.jpg
    python detect.py --source path/to/folder/
    python detect.py --source path/to/video.mp4
    python detect.py --conf 0.3          # stricter threshold
"""

import argparse
import sys
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = "results"          # all annotated images saved here
DEFAULT_CONF = 0.05               # confidence threshold (5% — low for small dataset)
DEFAULT_IOU  = 0.45               # NMS IoU overlap threshold

# ─── Find the best trained model automatically ─────────────────────────────────

def find_model() -> Path:
    """Auto-pick the latest trained model from models/ or runs/."""
    search_dirs = [
        Path("models"),
        Path("runs"),
    ]
    candidates = []
    for d in search_dirs:
        candidates.extend(d.rglob("best.pt"))
        candidates.extend(d.rglob("yolov8_*.pt"))

    if not candidates:
        print("\n❌  No trained model found!")
        print("    Run the pipeline first:")
        print("    python scripts\\run_pipeline.py --run-name 'potholes-v5-fast'")
        sys.exit(1)

    # Pick the most recently modified .pt file
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"  Model → {latest}")
    return latest


# ─── Detection ─────────────────────────────────────────────────────────────────

def detect(source: str, conf: float, iou: float, model_path: Path | None):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌  ultralytics not installed. Run:  pip install ultralytics")
        sys.exit(1)

    if model_path is None:
        model_path = find_model()

    print("\n" + "=" * 55)
    print("  Pothole Detector")
    print("=" * 55)
    print(f"  Source     : {source}")
    print(f"  Model      : {model_path}")
    print(f"  Confidence : {conf}")
    print(f"  Saving to  : {OUTPUT_DIR}/")
    print("=" * 55 + "\n")

    model   = YOLO(str(model_path))
    results = model.predict(
        source   = source,
        conf     = conf,
        iou      = iou,
        save     = True,         # Save annotated images
        project  = OUTPUT_DIR,
        name     = "detect",
        exist_ok = True,
        verbose  = True,
        line_width = 3,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    from collections import Counter
    total = 0
    counts: Counter = Counter()

    for r in results:
        if r.boxes is not None and len(r.boxes):
            for cls_id in r.boxes.cls.tolist():
                label = model.names[int(cls_id)]
                counts[label] += 1
                total += 1

    print("\n" + "=" * 55)
    print(f"  Images processed : {len(results)}")
    print(f"  Total detections : {total}")
    if counts:
        for label, n in counts.most_common():
            print(f"    {label:<20} {n:>4} detections")
    else:
        print(f"\n  ⚠  No detections above conf={conf}.")
        print(f"     Try a lower threshold:  python detect.py --conf 0.01")
    print(f"\n  ✅  Results saved → {OUTPUT_DIR}\\detect\\")
    print("=" * 55 + "\n")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run pothole detection on images/video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect.py                                  # runs on test_images/ folder
  python detect.py --source my_photo.jpg            # single image
  python detect.py --source data/processed/images/val/
  python detect.py --source road_clip.mp4           # video file
  python detect.py --conf 0.01                      # very low threshold to see all boxes
        """
    )
    parser.add_argument(
        "--source", default="test_images",
        help="Image, folder, or video to run detection on (default: test_images/)"
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Path to a specific .pt model file (auto-detects latest if not set)"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Confidence threshold 0–1 (default: {DEFAULT_CONF})"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"NMS IoU threshold 0–1 (default: {DEFAULT_IOU})"
    )
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else None

    # Create default test_images/ folder if it doesn't exist yet
    if args.source == "test_images":
        Path("test_images").mkdir(exist_ok=True)
        if not list(Path("test_images").glob("*.*")):
            print("\n  📁  Drop your test images into the  test_images/  folder,")
            print("      then re-run:  python detect.py\n")
            sys.exit(0)

    detect(
        source     = args.source,
        conf       = args.conf,
        iou        = args.iou,
        model_path = model_path,
    )


if __name__ == "__main__":
    main()
