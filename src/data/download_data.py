"""
src/data/download_data.py
Downloads a road scene dataset from Roboflow Universe.

Dataset used:
  Roboflow Universe — KITTI Object Detection dataset
  (Cars, Pedestrians, Cyclists, Traffic Signs — YOLO format)

Before running:
  1. Sign up FREE at https://app.roboflow.com/
  2. Get your API key: https://app.roboflow.com/settings/api
  3. Either:
     a) Set environment variable:   set ROBOFLOW_API_KEY=your_key_here
     b) Create a .env file:         ROBOFLOW_API_KEY=your_key_here
     c) Pass on CLI:                python src/data/download_data.py --api-key YOUR_KEY

Usage:
    python src/data/download_data.py
    python src/data/download_data.py --api-key YOUR_KEY
    python src/data/download_data.py --dry-run   # just show what would be downloaded
"""
import argparse
import datetime
import hashlib
import os
import sys
from pathlib import Path

# Allow running from project root:  python src/data/download_data.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

from src.utils.helpers import ensure_dirs, load_params, setup_logger, write_dataset_manifest

load_dotenv()  # load .env if present
logger = setup_logger("download_data")


# ─── Constants ─────────────────────────────────────────────────────────────────

# Roboflow public KITTI-style road detection dataset
# Browse more at: https://universe.roboflow.com/
ROBOFLOW_WORKSPACE = "kitti-bounding-boxes"
ROBOFLOW_PROJECT   = "kitti-dataset-object-detect"
ROBOFLOW_VERSION   = 1

# Fallback: if you change workspace/project in params.yaml, those take priority


# ─── Core Download Function ────────────────────────────────────────────────────

def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    fmt: str,
    output_dir: Path,
    dry_run: bool = False,
) -> Path:
    """
    Download a dataset from Roboflow Universe.

    Args:
        api_key:    Roboflow API key.
        workspace:  Roboflow workspace slug.
        project:    Roboflow project slug.
        version:    Dataset version number.
        fmt:        Export format (e.g. 'yolov8').
        output_dir: Where to save the downloaded data.
        dry_run:    If True, just print what would be downloaded.

    Returns:
        Path to the downloaded dataset directory.
    """
    logger.info("=" * 60)
    logger.info("  Roboflow Dataset Downloader")
    logger.info("=" * 60)
    logger.info(f"  Workspace : {workspace}")
    logger.info(f"  Project   : {project}")
    logger.info(f"  Version   : {version}")
    logger.info(f"  Format    : {fmt}")
    logger.info(f"  Output    : {output_dir}")

    if dry_run:
        logger.info("  [DRY RUN] — skipping actual download.")
        return output_dir

    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    ensure_dirs(output_dir)

    logger.info("  Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    logger.info("  Fetching project...")
    project_obj = rf.workspace(workspace).project(project)

    logger.info(f"  Downloading version {version} in {fmt} format...")
    dataset = project_obj.version(version).download(
        model_format=fmt,
        location=str(output_dir),
        overwrite=False,  # skip if already downloaded
    )

    dataset_path = Path(dataset.location)
    logger.info(f"  ✓ Dataset saved to: {dataset_path}")
    return dataset_path


def build_manifest(
    dataset_path: Path,
    workspace: str,
    project: str,
    version: int,
    data_version_tag: str,
) -> dict:
    """Build a metadata manifest describing the downloaded dataset."""
    # Count images
    image_count = len(list(dataset_path.rglob("*.jpg"))) + \
                  len(list(dataset_path.rglob("*.png")))
    label_count = len(list(dataset_path.rglob("*.txt")))

    # Compute a lightweight fingerprint (hash of all label filenames sorted)
    label_files = sorted([f.name for f in dataset_path.rglob("*.txt")])
    fingerprint = hashlib.md5("\n".join(label_files).encode()).hexdigest()[:12]

    return {
        "dataset_version": data_version_tag,
        "source": "roboflow",
        "workspace": workspace,
        "project": project,
        "roboflow_version": version,
        "download_timestamp": datetime.datetime.now().isoformat(),
        "image_count": image_count,
        "label_count": label_count,
        "fingerprint": fingerprint,
        "notes": "KITTI road scene dataset — cars, pedestrians, cyclists, traffic signs",
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download road scene dataset from Roboflow Universe"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var / .env file)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan, skip download")
    args = parser.parse_args()

    # ── Load params ──────────────────────────────────────────────────────────
    params = load_params()
    data_cfg      = params["data"]
    paths_cfg     = params["paths"]

    workspace  = data_cfg.get("workspace", ROBOFLOW_WORKSPACE)
    project    = data_cfg.get("project", ROBOFLOW_PROJECT)
    version    = int(data_cfg.get("roboflow_version", ROBOFLOW_VERSION))
    fmt        = data_cfg.get("format", "yolov8")
    ver_tag    = data_cfg.get("version", "v1")
    output_dir = Path(paths_cfg["raw_data"])

    # ── Resolve API key (priority: CLI > env > .env) ─────────────────────────
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key and not args.dry_run:
        logger.error(
            "\n"
            "  ❌  Roboflow API key not found!\n\n"
            "  To get your FREE key:\n"
            "    1. Sign up at https://app.roboflow.com/\n"
            "    2. Go to https://app.roboflow.com/settings/api\n"
            "    3. Copy your Private API Key\n\n"
            "  Then either:\n"
            "    A) Create a .env file in the project root:\n"
            "         ROBOFLOW_API_KEY=your_key_here\n"
            "    B) Set environment variable:\n"
            "         set ROBOFLOW_API_KEY=your_key_here    (Windows)\n"
            "    C) Pass on command line:\n"
            "         python src/data/download_data.py --api-key YOUR_KEY\n"
        )
        sys.exit(1)

    # ── Download ─────────────────────────────────────────────────────────────
    dataset_path = download_dataset(
        api_key=api_key or "dry_run",
        workspace=workspace,
        project=project,
        version=version,
        fmt=fmt,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        # ── Write manifest ───────────────────────────────────────────────────
        manifest = build_manifest(dataset_path, workspace, project, version, ver_tag)
        manifest_path = output_dir / "dataset_manifest.json"
        write_dataset_manifest(manifest, manifest_path)

        logger.info("")
        logger.info("  ✅  Download complete!")
        logger.info(f"     Images : {manifest['image_count']}")
        logger.info(f"     Labels : {manifest['label_count']}")
        logger.info(f"     Version: {ver_tag}")
        logger.info("")
        logger.info("  Next step → python src/data/preprocess.py")


if __name__ == "__main__":
    main()
