# 🚗 Road Scene Object Detection — MLOps Pipeline

> A beginner-friendly **ADAS object detection** pipeline built with real MLOps practices.  
> Detects **cars, pedestrians, cyclists, and traffic signs** using YOLOv8 on a KITTI-style road scene dataset.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [MLOps Concepts Covered](#mlops-concepts-covered)  
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [Running Experiments](#running-experiments)
7. [Viewing Results in MLflow](#viewing-results-in-mlflow)
8. [Running Inference](#running-inference)
9. [Dataset Versioning with DVC](#dataset-versioning-with-dvc)
10. [Configuration Reference](#configuration-reference)

---

## Project Overview

This project simulates a real-world ML engineering workflow for ADAS (Advanced Driver Assistance Systems). Instead of just training a model once, it treats the entire ML process as a **reproducible, versioned pipeline**:

```
Download Dataset → Preprocess → Split → Train (YOLOv8) → Evaluate
                                  ↓              ↓
                            Versioned splits   MLflow tracking
                         (data/splits/v1.txt) (localhost:5000)
```

**Detected classes:**
| Index | Class | Description |
|-------|-------|-------------|
| 0 | `car` | Cars, trucks, buses, vans |
| 1 | `pedestrian` | People walking |
| 2 | `cyclist` | Bicycle/motorcycle riders |
| 3 | `traffic_sign` | Signs, traffic lights |

---

## MLOps Concepts Covered

| Concept | Implementation |
|---------|---------------|
| **Data versioning** | Dataset version tags (`v1`, `v2`) + manifest JSON with fingerprints |
| **Pipeline as code** | `dvc.yaml` defines all stages with inputs/outputs |
| **Reproducible splits** | Fixed-seed train/val split saved as index files |
| **Experiment tracking** | MLflow logs all params + epoch-by-epoch metrics |
| **Versioned models** | `models/yolov8_road_v1.pt`, `v2.pt`, ... |
| **Config management** | `params.yaml` = single source of truth |
| **Artifact logging** | Model weights + training plots stored in MLflow |

---

## Project Structure

```
MLOPS/
├── 📁 configs/
│   ├── dataset.yaml          # YOLO dataset config (class names, paths)
│   └── train_config.yaml     # Hyperparameter documentation + ablation guide
│
├── 📁 data/
│   ├── raw/                  # Downloaded raw data (DVC tracked)
│   ├── processed/            # YOLO-format images + labels (DVC tracked)
│   │   ├── images/{all,train,val}/
│   │   └── labels/{all,train,val}/
│   └── splits/               # ✅ Git tracked: train_v1.txt, val_v1.txt
│
├── 📁 models/                # Saved model checkpoints (DVC tracked)
│   └── yolov8_road_v1.pt
│
├── 📁 mlruns/                # MLflow tracking (auto-generated)
│
├── 📁 notebooks/
│   └── 01_eda.ipynb          # Exploratory Data Analysis
│
├── 📁 scripts/
│   ├── run_pipeline.py       # 🚀 One command: full pipeline
│   └── run_inference.py      # 🎯 Run detection on new images
│
├── 📁 src/
│   ├── data/
│   │   ├── download_data.py  # Roboflow downloader
│   │   ├── preprocess.py     # KITTI → YOLO format converter
│   │   └── split_data.py     # Reproducible train/val splitter
│   ├── training/
│   │   ├── train.py          # YOLOv8 training + MLflow logging
│   │   └── evaluate.py       # Validation metrics + per-class breakdown
│   ├── tracking/
│   │   └── mlflow_logger.py  # MLflow helper wrapper
│   └── utils/
│       └── helpers.py        # Shared utilities
│
├── dvc.yaml                  # DVC pipeline stages
├── params.yaml               # ⚙️ Central config (edit to run experiments)
├── requirements.txt
└── .env.example              # API key template
```

---

## Quick Start

### 1. Install dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install all packages
pip install -r requirements.txt
```

### 2. Set up Roboflow API key

```powershell
# Option A: Copy .env.example to .env and fill in your key
copy .env.example .env
# Edit .env → set ROBOFLOW_API_KEY=your_key_here

# Option B: Set environment variable directly
$env:ROBOFLOW_API_KEY = "your_key_here"
```

> **Get your free key:** Sign up at [app.roboflow.com](https://app.roboflow.com) → Settings → API → Copy Private Key

### 3. Run the full pipeline

```powershell
python scripts/run_pipeline.py
```

That's it! The pipeline will:
- ✅ Download the dataset from Roboflow
- ✅ Preprocess and remap class labels
- ✅ Create a versioned train/val split
- ✅ Train YOLOv8n for 30 epochs
- ✅ Log everything to MLflow
- ✅ Save the best model to `models/yolov8_road_v1.pt`

---

## Step-by-Step Guide

You can also run each stage individually:

```powershell
# Stage 1: Download dataset (requires ROBOFLOW_API_KEY)
python src/data/download_data.py

# Stage 2: Convert labels to YOLO format
python src/data/preprocess.py

# Stage 3: Create reproducible train/val split
python src/data/split_data.py

# Stage 4: Train YOLOv8
python src/training/train.py

# Stage 5: Evaluate on validation set
python src/training/evaluate.py
```

---

## Running Experiments

### Changing hyperparameters

Edit `params.yaml` before training:

```yaml
train:
  model: "yolov8s.pt"   # ← try a larger model
  epochs: 50            # ← more epochs
  batch_size: 8
  learning_rate: 0.005  # ← lower LR
```

Then run again:

```powershell
python scripts/run_pipeline.py --start-from train --run-name "yolov8s-lr0005"
```

### Smoke test (verify pipeline works without full training)

```powershell
python scripts/run_pipeline.py --smoke-test
```

### Running a new dataset version

1. Update `params.yaml`:
   ```yaml
   data:
     version: "v2"
   ```
2. Create a new split (different seed or more images):
   ```powershell
   python src/data/split_data.py --version v2
   ```
3. Train again:
   ```powershell
   python scripts/run_pipeline.py --start-from train --run-name "v2-dataset"
   ```

---

## Viewing Results in MLflow

Start the MLflow UI:

```powershell
mlflow ui
```

Then open: **http://127.0.0.1:5000**

You'll see:
- 📊 All experiments and runs side-by-side
- 📈 Training curves (loss, mAP per epoch)
- ⚙️ Exact hyperparameters used per run
- 📦 Saved model artifacts
- 🔖 Run tags (model version, data version, status)

### Comparing runs

In the MLflow UI:
1. Select multiple runs using checkboxes
2. Click **"Compare"**
3. See metrics plotted together on one chart

---

## Running Inference

```powershell
# Run on a folder of images (auto-selects latest model)
python scripts/run_inference.py --source data/processed/images/val/

# Run on a single image
python scripts/run_inference.py --source my_road_photo.jpg

# Use a specific model version
python scripts/run_inference.py --source images/ --model-version v1

# Adjust confidence threshold
python scripts/run_inference.py --source images/ --conf 0.4

# Run on webcam
python scripts/run_inference.py --source 0 --show
```

Annotated output images are saved to `runs/inference/detect/`.

---

## Dataset Versioning with DVC

```powershell
# Initialize DVC in the project (one-time setup)
dvc init

# Run the full pipeline through DVC (with caching)
dvc repro

# Check which stages are up to date vs. changed
dvc status

# View the pipeline as a DAG
dvc dag

# Compare metrics between versions
dvc metrics show
dvc metrics diff
```

**How DVC versioning works:**
- Large files (`data/`, `models/`) are stored in DVC cache (`.dvc/cache/`)
- Git only tracks tiny `.dvc` pointer files + `dvc.yaml` + `params.yaml`
- Run `dvc repro` after changing `params.yaml` → only changed stages re-run

---

## Configuration Reference

All pipeline behavior is controlled through `params.yaml`:

```yaml
data:
  version: "v1"              # Bump to v2, v3 as dataset evolves
  workspace: "..."           # Roboflow workspace
  project: "..."             # Roboflow project

split:
  seed: 42                   # NEVER change this mid-experiment!
  train_ratio: 0.80

train:
  model: "yolov8n.pt"       # n=fast, s=better, m=even better, l/x=best
  epochs: 30
  batch_size: 16             # Reduce to 8 if out of memory
  image_size: 640
  learning_rate: 0.01
  device: "auto"             # auto=GPU if available, else CPU

mlflow:
  experiment_name: "road-detection"  # Groups all runs together
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ROBOFLOW_API_KEY not found` | Create `.env` file from `.env.example` |
| `CUDA out of memory` | Set `batch_size: 8` or `device: "cpu"` in params.yaml |
| `No image/label pairs found` | Run `download_data.py` first |
| `dataset.yaml not found` | Run `preprocess.py` and `split_data.py` first |
| MLflow not showing runs | Make sure you're in the project root when running `mlflow ui` |

---

## Next Steps

Once you've completed this project, consider:

- 🔄 **Data augmentation**: Add Albumentations transforms to improve robustness
- 📦 **Model export**: Export to ONNX or TensorRT for deployment  
  ```powershell
  yolo export model=models/yolov8_road_v1.pt format=onnx
  ```
- ☁️ **Remote DVC storage**: Push data to S3/GCS for team collaboration
- 📡 **Remote MLflow**: Deploy MLflow server for team experiment sharing
- 🔁 **CI/CD**: Automate retraining on new data with GitHub Actions

---

*Built as a beginner MLOps learning project for ADAS object detection.*
