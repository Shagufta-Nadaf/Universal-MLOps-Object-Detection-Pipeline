"""
src/data/split_data.py
Creates a reproducible 80/20 train/validation split.

MLOps key: The split is saved as text index files so:
  - The SAME images always go to train/val (fixed seed)
  - You can compare experiments fairly
  - The split is version-tagged (v1, v2, ...)

Output files:
    data/splits/train_v1.txt   ← list of image filenames in training set
    data/splits/val_v1.txt     ← list of image filenames in validation set
    data/splits/split_v1_manifest.json ← metadata about the split

It also creates the physical train/val directory structure expected by YOLO:
    data/processed/images/train/
    data/processed/images/val/
    data/processed/labels/train/
    data/processed/labels/val/

Usage:
    python src/data/split_data.py
    python src/data/split_data.py --verify    # re-run and confirm same split
    python src/data/split_data.py --version v2 --seed 99   # new experiment version
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import (
    ensure_dirs,
    get_dataset_paths,
    load_params,
    setup_logger
)

logger = setup_logger("split_data")


# ─── Core Split Function ───────────────────────────────────────────────────────

def create_split(
    processed_dir: Path,
    splits_dir: Path,
    version: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    max_images: int = None,
    dry_run: bool = False,
    verify: bool = False,
) -> dict:
    """
    Create or verify a reproducible train/val split.

    Returns:
        dict with split statistics.
    """
    logger.info("=" * 60)
    logger.info("  Train/Val Splitter")
    logger.info("=" * 60)
    logger.info(f"  Processed dir : {processed_dir}")
    logger.info(f"  Splits dir    : {splits_dir}")
    logger.info(f"  Version       : {version}")
    logger.info(f"  Seed          : {seed}")
    logger.info(f"  Train/Val     : {train_ratio:.0%} / {val_ratio:.0%}")

    # ── Discover all processed images ────────────────────────────────────────
    all_images_dir = processed_dir / "images" / "all"
    all_labels_dir = processed_dir / "labels" / "all"

    if not all_images_dir.exists():
        logger.error(f"  Processed images dir not found: {all_images_dir}")
        logger.error("  Run preprocess.py first.")
        sys.exit(1)

    all_images = sorted([
        f.name for f in all_images_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not all_images:
        logger.error(f"  No images found in {all_images_dir}")
        sys.exit(1)

    logger.info(f"  Total images  : {len(all_images)}")

    # ── Verify that labels exist for all images ───────────────────────────────
    missing_labels = []
    for img_name in all_images:
        lbl_name = Path(img_name).with_suffix(".txt").name
        if not (all_labels_dir / lbl_name).exists():
            missing_labels.append(img_name)

    if missing_labels:
        logger.warning(f"  {len(missing_labels)} images have no label file — they will be skipped")
        all_images = [img for img in all_images if img not in set(missing_labels)]

    if max_images and len(all_images) > max_images:
        logger.info(f"  Limiting to {max_images} images as requested (for fast CPU training)")
        np.random.seed(seed)
        all_images = np.random.choice(all_images, max_images, replace=False).tolist()

    # ── Create split ─────────────────────────────────────────────────────────
    train_images, val_images = train_test_split(
        all_images,
        train_size=train_ratio,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    logger.info(f"  Train images  : {len(train_images)}")
    logger.info(f"  Val images    : {len(val_images)}")

    train_path = splits_dir / f"train_{version}.txt"
    val_path   = splits_dir / f"val_{version}.txt"

    # ── Verify mode: compare against existing split ───────────────────────────
    if verify:
        if not train_path.exists() or not val_path.exists():
            logger.error("  No existing split files to verify against.")
            sys.exit(1)

        existing_train = set(train_path.read_text().strip().splitlines())
        existing_val   = set(val_path.read_text().strip().splitlines())
        new_train      = set(train_images)
        new_val        = set(val_images)

        if existing_train == new_train and existing_val == new_val:
            logger.info("  ✅  VERIFIED — split is identical (reproducible)")
        else:
            diff_train = len(existing_train.symmetric_difference(new_train))
            diff_val   = len(existing_val.symmetric_difference(new_val))
            logger.error(f"  ❌  MISMATCH — {diff_train} train diffs, {diff_val} val diffs")
            logger.error("  Check: seed, train_ratio, or processed images may have changed.")
            sys.exit(1)
        return {}

    if dry_run:
        logger.info("  [DRY RUN] — not writing split files.")
        return {"train": len(train_images), "val": len(val_images)}

    # ── Write split index files ───────────────────────────────────────────────
    ensure_dirs(splits_dir)
    train_path.write_text("\n".join(train_images) + "\n")
    val_path.write_text("\n".join(val_images) + "\n")
    logger.info(f"  ✓ Wrote: {train_path}")
    logger.info(f"  ✓ Wrote: {val_path}")

    # ── Write split manifest ──────────────────────────────────────────────────
    manifest = {
        "version": version,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "total_images": len(all_images),
        "train_count": len(train_images),
        "val_count": len(val_images),
        "created_at": datetime.datetime.now().isoformat(),
        "notes": "Fixed-seed split for reproducible experiment comparison",
    }
    manifest_path = splits_dir / f"split_{version}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  ✓ Manifest: {manifest_path}")

    # ── Build physical YOLO directory structure ───────────────────────────────
    _build_yolo_dirs(
        all_images_dir, all_labels_dir,
        processed_dir,
        train_images, val_images,
    )

    logger.info("")
    logger.info("  ✅  Split complete!")
    logger.info("  Next step → python src/training/train.py")

    return manifest


def _build_yolo_dirs(
    all_images_dir: Path,
    all_labels_dir: Path,
    processed_dir: Path,
    train_images: list,
    val_images: list,
) -> None:
    """
    Populate data/processed/images/{train,val}/ and labels/{train,val}/
    by symlinking or copying from the 'all' directories.
    """
    for split_name, file_list in [("train", train_images), ("val", val_images)]:
        img_out = processed_dir / "images" / split_name
        lbl_out = processed_dir / "labels" / split_name

        # CRITICAL: Clean PREVIOUS split images/labels to avoid data mixing
        if img_out.exists():
            shutil.rmtree(img_out)
        if lbl_out.exists():
            shutil.rmtree(lbl_out)

        ensure_dirs(img_out, lbl_out)
        logger.info(f"  ✓ Initialized clean {split_name} fold")

        for img_name in file_list:
            lbl_name = Path(img_name).with_suffix(".txt").name
            shutil.copy2(all_images_dir / img_name, img_out / img_name)
            lbl_src = all_labels_dir / lbl_name
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_out / lbl_name)

    logger.info(f"  ✓ Built YOLO directory structure in {processed_dir}")
    logger.info(f"      images/train/ | images/val/")
    logger.info(f"      labels/train/ | labels/val/")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create reproducible train/val split")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing split is still identical (reproducibility check)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show split stats without writing files")
    parser.add_argument("--version", default=None,
                        help="Override dataset version tag (e.g. v2)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    args = parser.parse_args()

    params = load_params()
    split_cfg = params["split"]
    data_cfg  = params["data"]
    paths_cfg = params["paths"]

    version      = args.version or data_cfg.get("version", "v1")
    seed         = args.seed if args.seed is not None else int(split_cfg["seed"])
    train_ratio  = float(split_cfg["train_ratio"])
    val_ratio    = float(split_cfg["val_ratio"])
    max_images   = split_cfg.get("max_images")
    if max_images: max_images = int(max_images)

    # ── Resolve dataset-specific paths ───────────────────────────────────────
    ds_paths      = get_dataset_paths(params)
    processed_dir = ds_paths["processed"]
    splits_dir    = ds_paths["splits"]

    create_split(
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        version=version,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_images=max_images,
        dry_run=args.dry_run,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
