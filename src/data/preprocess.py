"""
src/data/preprocess.py
Converts downloaded dataset (YOLO format from Roboflow) into a
clean, standardized YOLO directory structure:

    data/processed/
    ├── images/
    │   └── all/         ← all images before split
    └── labels/
        └── all/         ← all labels before split

Class mapping used:
    pothole → 0

Usage:
    python src/data/preprocess.py
    python src/data/preprocess.py --dry-run
    python src/data/preprocess.py --verbose
"""
import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import (
    ensure_dirs,
    get_dataset_paths,
    load_params,
    setup_logger
)

logger = setup_logger("preprocess")

# Dynamic class mapping will be loaded from data.yaml
TARGET_CLASSES = {} 

def resolve_class_name(raw_name: str, src_class_map: dict) -> str:
    """
    Pass-through for class names from the source dataset.
    """
    return raw_name.strip()


# ─── Label File Parsers ────────────────────────────────────────────────────────

def parse_roboflow_yaml(raw_dir: Path) -> dict | None:
    """
    Try to read Roboflow's data.yaml to get original class names.
    Returns {class_index: class_name} or None if not found.
    """
    import yaml

    for yaml_path in raw_dir.rglob("data.yaml"):
        try:
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names", [])
            if names:
                return {i: name for i, name in enumerate(names)}
        except Exception:
            pass
    return None


def convert_label_file(
    src_label: Path,
    src_class_map: dict | None,
) -> list[str]:
    """
    Read a YOLO-format .txt label file. We now keep all classes from Roboflow.
    """
    converted_lines = []

    if not src_label.exists():
        return converted_lines

    with open(src_label) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        # For YOLO, we just pass through the index as-is if we want 1:1 mapping
        converted_lines.append(line.strip())

    return converted_lines


# ─── Main Preprocessing Logic ──────────────────────────────────────────────────

def find_raw_images_and_labels(raw_dir: Path):
    """
    Discover all image/label pairs in the raw directory.
    Supports nested Roboflow structure: train/, valid/, test/ subfolders.
    """
    pairs = []
    img_extensions = {".jpg", ".jpeg", ".png"}

    for img_path in sorted(raw_dir.rglob("*")):
        if img_path.suffix.lower() not in img_extensions:
            continue
        # Find companion label (same name, .txt, in labels/ instead of images/)
        # Try same directory
        lbl_path = img_path.with_suffix(".txt")
        if not lbl_path.exists():
            # Try sibling labels/ folder
            lbl_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            # Try labels/ at same level
            lbl_path = img_path.parent / "labels" / img_path.with_suffix(".txt").name
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    return pairs


def preprocess(
    raw_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Main preprocessing function.

    Returns:
        Summary statistics dict.
    """
    logger.info("=" * 60)
    logger.info("  Data Preprocessor")
    logger.info("=" * 60)
    logger.info(f"  Input  : {raw_dir}")
    logger.info(f"  Output : {output_dir}")

    # ── Discover source class mapping ────────────────────────────────────────
    src_class_map = parse_roboflow_yaml(raw_dir)
    if src_class_map:
        logger.info(f"  Source classes: {src_class_map}")
    else:
        logger.warning("  No data.yaml found — will try to preserve class indices as-is")

    # ── Discover image/label pairs ───────────────────────────────────────────
    pairs = find_raw_images_and_labels(raw_dir)
    if not pairs:
        logger.error(f"  No image/label pairs found in {raw_dir}")
        logger.error("  Have you run download_data.py first?")
        sys.exit(1)

    logger.info(f"  Found {len(pairs)} image/label pairs")

    # ── Setup output dirs ────────────────────────────────────────────────────
    out_images = output_dir / "images" / "all"
    out_labels = output_dir / "labels" / "all"

    if not dry_run:
        # CRITICAL: Wipe existing processed data to prevent mixing projects
        if out_images.exists():
            shutil.rmtree(out_images)
        if out_labels.exists():
            shutil.rmtree(out_labels)
        ensure_dirs(out_images, out_labels)
        logger.info("  ✓ Cleaned processed data folders (no mixing!)")

    # ── Process each pair ────────────────────────────────────────────────────
    stats = Counter()
    class_counts = defaultdict(int)
    skipped_no_labels = 0

    for img_path, lbl_path in pairs:
        # CRITICAL FILTER: If this is a face project, skip anything containing 'pothole'
        if "face" in str(raw_dir).lower() or "human" in str(raw_dir).lower():
            if "pothole" in img_path.name.lower():
                logger.warning(f"  🚨  Strict Filter: Skipping '{img_path.name}' (pothole detected in face project)")
                skipped_no_labels += 1
                continue

        # Convert label
        converted = convert_label_file(lbl_path, src_class_map)

        if not converted:
            skipped_no_labels += 1
            continue

        stats["processed"] += 1
        for line in converted:
            class_idx = int(line.split()[0])
            # Use original names if available
            class_name = src_class_map.get(class_idx, f"class_{class_idx}") if src_class_map else f"class_{class_idx}"
            class_counts[class_name] += 1

        if not dry_run:
            # Copy image
            shutil.copy2(img_path, out_images / img_path.name)
            # Write cleaned label
            with open(out_labels / img_path.with_suffix(".txt").name, "w") as f:
                f.write("\n".join(converted) + "\n")

        if verbose:
            logger.info(f"  ✓ {img_path.name} → {len(converted)} boxes")

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("  ── Preprocessing Summary ──")
    logger.info(f"  Total pairs found  : {len(pairs)}")
    logger.info(f"  Processed          : {stats['processed']}")
    logger.info(f"  Skipped (no labels): {skipped_no_labels}")
    logger.info("  Class distribution :")
    total_boxes = sum(class_counts.values())
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (cnt / total_boxes * 100) if total_boxes > 0 else 0
        logger.info(f"    {cls:<15} {cnt:>5} boxes  ({pct:.1f}%)")

    if dry_run:
        logger.info("  [DRY RUN] — no files written.")
    else:
        logger.info(f"  ✅  Output → {output_dir}")
        logger.info("  Next step → python src/data/split_data.py")

    # ── Write YOLO dataset.yaml ──────────────────────────────────────────────
    if not dry_run:
        dataset_yaml_path = Path("configs/dataset.yaml")
        dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort classes by index to ensure correct mapping
        sorted_classes = [name for idx, name in sorted(src_class_map.items())] if src_class_map else ["object"]
        
        yaml_content = {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(sorted_classes)}
        }
        
        import yaml
        with open(dataset_yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        logger.info(f"  ✓ Dynamic dataset.yaml written → {dataset_yaml_path}")

    return {
        "total_pairs": len(pairs),
        "processed": stats["processed"],
        "skipped": skipped_no_labels,
        "class_distribution": dict(class_counts),
        "total_boxes": total_boxes,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def resolve_raw_dir(raw_dir: Path) -> Path:
    """
    Roboflow downloads into a subfolder like data/raw/project-version/.
    This function checks for a saved dataset_location.txt to find the real path.
    Falls back to scanning raw_dir for any subfolder containing a data.yaml.
    """
    # Check for saved location from download step
    location_file = raw_dir / "dataset_location.txt"
    if location_file.exists():
        saved_path = Path(location_file.read_text().strip())
        if saved_path.exists():
            logger.info(f"  Using saved dataset location: {saved_path}")
            return saved_path

    # Fallback: look for data.yaml in subdirectories
    for yaml_path in raw_dir.rglob("data.yaml"):
        logger.info(f"  Found data.yaml at: {yaml_path.parent}")
        return yaml_path.parent

    # Fallback: use raw_dir itself
    logger.info(f"  Using raw_dir directly: {raw_dir}")
    return raw_dir


def main():
    parser = argparse.ArgumentParser(description="Preprocess downloaded dataset -> YOLO format")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print every file processed")
    args = parser.parse_args()

    params = load_params()
    # ── Resolve dataset-specific paths ───────────────────────────────────────
    ds_paths   = get_dataset_paths(params)
    raw_dir    = ds_paths["raw"]
    output_dir = ds_paths["processed"]

    # Resolve the actual dataset subfolder (Roboflow creates a subfolder)
    actual_raw_dir = resolve_raw_dir(raw_dir)

    preprocess(actual_raw_dir, output_dir, dry_run=args.dry_run, verbose=args.verbose)



if __name__ == "__main__":
    main()
