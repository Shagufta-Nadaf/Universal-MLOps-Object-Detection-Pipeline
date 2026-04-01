"""
src/utils/helpers.py
Shared utility functions used across the pipeline.
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


# ─── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently formatted logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


# ─── Config Loading ─────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_params(params_path: str | Path = "params.yaml") -> Dict[str, Any]:
    """Load the central params.yaml file."""
    return load_yaml(params_path)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a dict to a JSON file, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved JSON → {path}")


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ─── Path Utilities ─────────────────────────────────────────────────────────────

def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (and parents) if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Return the project root directory (where params.yaml lives)."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "params.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (params.yaml not found)")


def get_dataset_paths(params: Dict[str, Any]) -> Dict[str, Path]:
    """
    Build dataset-specific paths based on workspace and project name.
    Ensures isolation between different projects (e.g. Faces vs Potholes).
    """
    workspace = params["data"].get("workspace", "default")
    project = params["data"].get("project", "default")
    paths_cfg = params["paths"]
    
    root = get_project_root()
    subpath = Path(workspace) / project
    
    return {
        "raw": root / paths_cfg["raw_data"] / subpath,
        "processed": root / paths_cfg["processed_data"] / subpath,
        "splits": root / paths_cfg["splits_dir"] / subpath,
        "outputs": root / "outputs" / subpath,
        "dataset_yaml": root / "configs" / workspace / f"{project}_data.yaml"
    }


# ─── Dataset Utilities ──────────────────────────────────────────────────────────

def count_files(directory: str | Path, extension: str = ".jpg") -> int:
    """Count files with a given extension in a directory tree."""
    return len(list(Path(directory).rglob(f"*{extension}")))


def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> list:
    """
    Return a list of (image_path, label_path) tuples where both files exist.
    Supports .jpg, .jpeg, .png images and .txt labels.
    """
    pairs = []
    for img_ext in ["*.jpg", "*.jpeg", "*.png"]:
        for img_path in sorted(images_dir.rglob(img_ext)):
            label_path = labels_dir / img_path.with_suffix(".txt").name
            if label_path.exists():
                pairs.append((img_path, label_path))
    return pairs


def copy_file_pair(
    img_src: Path,
    lbl_src: Path,
    img_dst_dir: Path,
    lbl_dst_dir: Path,
) -> None:
    """Copy an image+label pair to destination directories."""
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, img_dst_dir / img_src.name)
    shutil.copy2(lbl_src, lbl_dst_dir / lbl_src.name)


# ─── Metadata / Versioning ──────────────────────────────────────────────────────

def write_dataset_manifest(
    manifest: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """Write dataset version metadata to a JSON manifest file."""
    save_json(manifest, output_path)
    print(f"  ✓ Dataset manifest written → {output_path}")


def get_next_model_version(models_dir: str | Path) -> str:
    """
    Scan models/ directory and return the next version tag.
    e.g. if yolov8_road_v3.pt exists, returns 'v4'.
    """
    models_dir = Path(models_dir)
    existing = list(models_dir.glob("yolov8_road_v*.pt"))
    if not existing:
        return "v1"
    versions = []
    for f in existing:
        try:
            v = int(f.stem.split("_v")[-1])
            versions.append(v)
        except ValueError:
            pass
    return f"v{max(versions) + 1}" if versions else "v1"
