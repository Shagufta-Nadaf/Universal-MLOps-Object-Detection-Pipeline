# Road Scene Object Detection MLOps Pipeline

## Overview

A complete, beginner-friendly yet realistic MLOps pipeline for road scene object detection. The system will detect **cars, pedestrians, cyclists, and traffic signs** using **YOLOv8** on the **KITTI** dataset subset. Every MLOps stage is explicitly tracked: data versioning, reproducible splits, experiment logging (MLflow), and versioned model saving.

---

## Project Architecture

```
MLOPS/
├── data/
│   ├── raw/                    # Downloaded raw KITTI images + labels
│   ├── processed/              # Converted to YOLO format
│   └── splits/                 # Saved train/val split index files (versioned)
│
├── configs/
│   ├── dataset.yaml            # YOLO dataset config
│   └── train_config.yaml       # Hyperparameter configs (lr, batch, epochs, etc.)
│
├── src/
│   ├── data/
│   │   ├── download_data.py    # Downloads/extracts KITTI subset
│   │   ├── preprocess.py       # Converts KITTI → YOLO label format
│   │   └── split_data.py       # Creates reproducible train/val splits
│   │
│   ├── training/
│   │   ├── train.py            # Main training script
│   │   └── evaluate.py         # Evaluate saved model checkpoint
│   │
│   ├── tracking/
│   │   └── mlflow_logger.py    # MLflow experiment logging helpers
│   │
│   └── utils/
│       └── helpers.py          # Shared utility functions
│
├── models/
│   └── .gitkeep               # Versioned model checkpoints saved here
│
├── mlruns/                    # MLflow tracking directory (auto-generated)
│
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis
│
├── scripts/
│   ├── run_pipeline.py         # End-to-end pipeline orchestrator
│   └── run_inference.py        # Run inference on new images
│
├── requirements.txt
├── .dvcignore                  # DVC ignore file
├── dvc.yaml                    # DVC pipeline stages
├── params.yaml                 # Versioned parameters for DVC
├── .gitignore
└── README.md
```

---

## Proposed Changes

### Stage 1 — Project Scaffold & Configuration

#### [NEW] `requirements.txt`
Python dependencies: `ultralytics`, `mlflow`, `dvc`, `opencv-python`, `pyyaml`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`

#### [NEW] `params.yaml`
DVC-tracked parameters (split seed, split ratio, training hyperparameters)

#### [NEW] `dvc.yaml`
Defines the full DVC pipeline with stages: `download → preprocess → split → train`

#### [NEW] `configs/dataset.yaml`
YOLO-compatible dataset config with class names and paths

#### [NEW] `configs/train_config.yaml`
Training hyperparameters config

---

### Stage 2 — Data Layer

#### [NEW] `src/data/download_data.py`
- Downloads a **KITTI sample subset** (≈1000 images) from a public mirror
- Saves to `data/raw/`
- Records dataset version metadata (hash, source URL, date)

#### [NEW] `src/data/preprocess.py`
- Converts KITTI `.txt` labels → YOLO format (`class cx cy w h` normalized)
- Maps KITTI classes → 4 target classes: `car`, `pedestrian`, `cyclist`, `traffic_sign`
- Outputs images + labels to `data/processed/`
- Logs a data manifest CSV

#### [NEW] `src/data/split_data.py`
- Creates a reproducible 80/20 train/val split using a fixed seed from `params.yaml`
- Saves split index files: `data/splits/train_v1.txt` and `data/splits/val_v1.txt`
- Versioned naming supports future dataset iterations (v2, v3)

---

### Stage 3 — Training

#### [NEW] `src/training/train.py`
- Loads split files to determine train/val images
- Trains YOLOv8n (nano, fast) using Ultralytics API
- Logs all hyperparameters + metrics to **MLflow**
- Saves best model to `models/yolov8_road_v{N}.pt`

#### [NEW] `src/training/evaluate.py`
- Loads a saved checkpoint and runs evaluation on the val set
- Prints mAP50, mAP50-95, per-class precision/recall

---

### Stage 4 — Experiment Tracking

#### [NEW] `src/tracking/mlflow_logger.py`
- Helper class wrapping MLflow: `log_params()`, `log_metrics()`, `log_artifact()`
- Structured to create named experiments (e.g., `"road-detection-v1"`)

---

### Stage 5 — Pipeline Orchestrator & Inference

#### [NEW] `scripts/run_pipeline.py`
- Single script to run full pipeline end-to-end:
  `download → preprocess → split → train → evaluate`
- Accepts CLI args for experiment name, version tag

#### [NEW] `scripts/run_inference.py`
- Load a saved model checkpoint
- Run inference on a folder of images
- Draw bounding boxes and save output images

---

### Stage 6 — Documentation & Notebooks

#### [NEW] `README.md`
Complete setup and usage guide

#### [NEW] `notebooks/01_eda.ipynb`
Basic EDA: class distribution, image sizes, sample visualizations

---

## Technology Choices

| Component | Tool | Reason |
|---|---|---|
| Model | YOLOv8n (Ultralytics) | Fast, modern, great ecosystem |
| Dataset | KITTI subset | Public, realistic road scenes |
| Experiment tracking | MLflow | Local, free, professional |
| Data versioning | DVC | Industry standard, git-compatible |
| Label format | YOLO `.txt` | Native to Ultralytics |
| Split tracking | Seed + index files | Simple, reproducible |

---

## Open Questions

> [!IMPORTANT]
> **Dataset Strategy**: KITTI requires agreement to terms and manual download. Instead, I propose using a **pre-formatted KITTI subset from Kaggle or Roboflow** (publicly accessible, ~500-1000 images) to keep this beginner-friendly. Alternatively, we can use **COCO's road subset** or **BDD100K sample**. Which do you prefer?
> - Option A: Small KITTI subset via Roboflow API (recommended, ~800 images, ready YOLO format)
> - Option B: Full KITTI download with manual preprocessing
> - Option C: BDD100K sample subset

> [!NOTE]
> **Scope of training**: YOLOv8n training on ~800 images will run on CPU in ~30-60 minutes. If you have a GPU (CUDA), training will be much faster. Should I include automatic GPU detection?

> [!NOTE]
> **MLflow UI**: MLflow runs a local web server (`mlflow ui`) to view experiment results visually. I'll include setup instructions. This requires no account or cloud service.

---

## Verification Plan

### Automated Checks
- `python src/data/preprocess.py --dry-run` → verify label conversion
- `python src/data/split_data.py --verify` → confirm split reproducibility
- `python scripts/run_pipeline.py --smoke-test` → train for 1 epoch on 10 images

### Manual Verification
- Run `mlflow ui` → confirm metrics and params are logged and visible
- Check `models/` directory for saved `.pt` checkpoint
- Run `scripts/run_inference.py` on sample images → visually verify bounding boxes
- Re-run pipeline with same params → confirm identical split files are produced (reproducibility test)
