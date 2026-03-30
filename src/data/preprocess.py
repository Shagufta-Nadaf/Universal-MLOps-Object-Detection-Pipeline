"""
src/data/preprocess.py
Converts downloaded dataset (KITTI or YOLO format from Roboflow) into a
clean, standardized YOLO directory structure:

    data/processed/
    ├── images/
    │   ├── all/         ← all images before split
    └── labels/
        └── all/         ← all labels before split

If the Roboflow download already provides YOLO format, this script:
  1. Validates and fixes label class indices to match our 4-class mapping
  2. Filters out any bounding boxes for classes we don't care about
  3. Removes image-label pairs with no valid boxes
  4. Copies everything into data/processed/ with clean names

Class mapping used:
    car          → 0
    pedestrian   → 1
    cyclist      → 2
    traffic_sign → 3  (maps from: 'traffic sign', 'sign', 'trafficlight', etc.)

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

from src.utils.helpers import ensure_dirs, load_params, setup_logger

logger = setup_logger("preprocess")

# ─── Class Mapping ─────────────────────────────────────────────────────────────

# Canonical 4-class mapping (output class indices)
TARGET_CLASSES = {
    "car": 0,
    "pedestrian": 1,
    "cyclist": 2,
    "traffic_sign": 3,
}

# Aliases: map various source class names → target class name
CLASS_ALIASES = {
    # Car variants
    "car": "car",
    "vehicle": "car",
    "truck": "car",
    "bus": "car",
    "van": "car",
    "misc": "car",
    "tram": "car",
    # Pedestrian variants
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "people": "pedestrian",
    "person-sitting": "pedestrian",
    # Cyclist variants
    "cyclist": "cyclist",
    "bicycle": "cyclist",
    "rider": "cyclist",
    # Traffic sign variants
    "traffic_sign": "traffic_sign",
    "traffic sign": "traffic_sign",
    "sign": "traffic_sign",
    "trafficsign": "traffic_sign",
    "traffic_light": "traffic_sign",
    "trafficlight": "traffic_sign",
    "stop sign": "traffic_sign",
    "stopsign": "traffic_sign",
}


def resolve_class_name(raw_name: str) -> str | None:
    """
    Map a raw class name to our canonical 4-class taxonomy.
    Returns None if the class should be discarded.
    """
    normalized = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
    # Check direct match first
    if normalized in CLASS_ALIASES:
        return CLASS_ALIASES[normalized]
    # Partial match
    for alias, canonical in CLASS_ALIASES.items():
        if alias in normalized:
            return canonical
    return None  # discard


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
    target_index_map: dict,
) -> list[str]:
    """
    Read a YOLO-format .txt label file and remap class indices to our 4-class scheme.

    Args:
        src_label:       Path to source .txt file.
        src_class_map:   {int → class_name} from source dataset (or None).
        target_index_map: {canonical_class_name → target_index}.

    Returns:
        List of converted label lines (may be empty if all classes filtered out).
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

        src_class_idx = int(parts[0])
        bbox = parts[1:]  # cx cy w h

        # Resolve class name
        if src_class_map and src_class_idx in src_class_map:
            raw_name = src_class_map[src_class_idx]
        else:
            # If no map available, try to preserve if it's already 0-3
            if src_class_idx < len(TARGET_CLASSES):
                converted_lines.append(line.strip())
            continue

        canonical = resolve_class_name(raw_name)
        if canonical is None:
            continue  # skip this class

        target_idx = target_index_map[canonical]
        converted_lines.append(f"{target_idx} {' '.join(bbox)}")

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
        ensure_dirs(out_images, out_labels)

    # ── Process each pair ────────────────────────────────────────────────────
    target_index_map = {name: idx for name, idx in TARGET_CLASSES.items()}
    stats = Counter()
    class_counts = defaultdict(int)
    skipped_no_labels = 0

    for img_path, lbl_path in pairs:
        # Convert label
        converted = convert_label_file(lbl_path, src_class_map, target_index_map)

        if not converted:
            skipped_no_labels += 1
            if verbose:
                logger.debug(f"  Skip (no valid boxes): {img_path.name}")
            continue

        stats["processed"] += 1
        for line in converted:
            class_idx = int(line.split()[0])
            class_name = list(TARGET_CLASSES.keys())[class_idx]
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

    return {
        "total_pairs": len(pairs),
        "processed": stats["processed"],
        "skipped": skipped_no_labels,
        "class_distribution": dict(class_counts),
        "total_boxes": total_boxes,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess downloaded dataset → YOLO format")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print every file processed")
    args = parser.parse_args()

    params = load_params()
    raw_dir    = Path(params["paths"]["raw_data"])
    output_dir = Path(params["paths"]["processed_data"])

    preprocess(raw_dir, output_dir, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
