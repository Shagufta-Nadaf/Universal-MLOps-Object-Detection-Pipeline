# Pothole Detection MLOps Pipeline — Project Report

## 🎯 Project Overview

This project builds a **production-grade pothole detection system** using MLOps principles on a Windows laptop (Intel Core i5-8265U, CPU-only). The system detects potholes in road images using YOLOv8, with every step of the machine learning lifecycle automated, versioned, and tracked.

**Final model performance (v3):**
| Metric | Value |
|---|---|
| mAP@50 | **59.1%** |
| Precision | **63.6%** |
| Recall | **55.5%** |
| mAP@50-95 | **34.3%** |

---

## 🏗️ What is MLOps and Why We Used It

Traditional ML: a single Jupyter Notebook, train once, forget the settings, can't reproduce results.

**MLOps (Machine Learning Operations)** treats an ML project like a software system:

| MLOps Principle | How we applied it |
|---|---|
| **Pipeline as Code** | One command runs all 6 stages from download to inference |
| **Configuration Management** | `params.yaml` is the single source of truth for all settings |
| **Experiment Tracking** | MLflow logs every run: params, metrics, loss curves, model weights |
| **Reproducibility** | Fixed random seed=42 always creates the same train/val split |
| **Data Versioning** | Each dataset version gets a unique tag (v1 → v8) tracked through the pipeline |
| **Model Versioning** | Every trained model is saved as `yolov8_road_v{N}.pt` |
| **Automated Evaluation** | mAP, Precision, Recall computed automatically after every training run |

---

## 🔧 Technology Stack

| Tool | Role |
|---|---|
| **Python 3.11** | Core language |
| **Conda (mlops env)** | Isolated, reproducible environment |
| **YOLOv8n (Ultralytics)** | Object detection model architecture |
| **PyTorch (CPU)** | Deep learning backend |
| **Roboflow** | Dataset source and download API |
| **MLflow** | Experiment tracking, metric logging, artifact storage |
| **DVC** | Pipeline orchestration and data versioning |
| **params.yaml** | Single config file driving all pipeline stages |

---

## 📐 Project Structure

```
D:\ML_PROJECTS\MLOPS\
│
├── params.yaml                  ← THE BRAIN — all config lives here
├── cleanup.bat                  ← Wipes old data for fresh start
├── detect.py                    ← Standalone detection script
│
├── scripts/
│   ├── run_pipeline.py          ← Master orchestrator (6 stages)
│   └── run_inference.py         ← Flexible inference runner
│
├── src/
│   ├── data/
│   │   ├── download_data.py     ← Stage 1: Roboflow downloader
│   │   ├── preprocess.py        ← Stage 2: Label conversion & class mapping
│   │   └── split_data.py        ← Stage 3: Reproducible 80/20 split
│   ├── training/
│   │   ├── train.py             ← Stage 4: YOLOv8 training + MLflow logging
│   │   └── evaluate.py          ← Stage 5: mAP/Precision/Recall evaluation
│   └── tracking/
│       └── mlflow_logger.py     ← MLflow wrapper (params, metrics, artifacts)
│
├── configs/
│   └── dataset.yaml             ← YOLO dataset config (nc, class names, paths)
│
├── data/
│   ├── raw/                     ← Downloaded Roboflow dataset
│   ├── processed/               ← Cleaned, normalised images & labels
│   │   ├── images/{train,val}/
│   │   └── labels/{train,val}/
│   └── splits/                  ← train_v8.txt, val_v8.txt (reproducible index)
│
├── models/
│   └── yolov8_road_v3.pt        ← Best trained model (59.1% mAP)
│
└── runs/
    ├── pipeline_inference/detect/ ← Auto-saved annotated val images
    └── detect/runs/train/         ← Ultralytics training outputs
```

---

## 🔄 The 6-Stage Automated Pipeline

When you run:
```batch
python scripts\run_pipeline.py --run-name "potholes-v8-lowram"
```

These 6 stages execute **automatically in sequence**:

### Stage 1 — DOWNLOAD 📥
**Script:** `src/data/download_data.py`

- Reads `workspace`, `project`, `roboflow_version` from `params.yaml`
- Connects to Roboflow API using `ROBOFLOW_API_KEY` from `.env`
- Downloads the dataset in YOLOv8 format to `data/raw/`
- Saves the actual download path to `dataset_location.txt` (handles Roboflow's nested folder structure)
- **MLOps value:** Dataset is pulled by version number — changing the version in `params.yaml` downloads a completely different dataset automatically

### Stage 2 — PREPROCESS 🔧
**Script:** `src/data/preprocess.py`

- Reads Roboflow's `data.yaml` to discover source class names
- Applies `CLASS_ALIASES` mapping — handles all naming variants:
  - `"pot-hole"`, `"pot_hole"`, `"hole"`, `"road damage"` → all map to `"pothole"` (class 0)
- Filters out images with no valid labels
- Copies clean images + normalised labels to `data/processed/images/all/` and `labels/all/`
- Prints class distribution summary
- **MLOps value:** Class mapping is code — changing task (e.g. from cars to potholes) only requires updating `TARGET_CLASSES` and `CLASS_ALIASES` in one file

### Stage 3 — SPLIT ✂️
**Script:** `src/data/split_data.py`

- Reads `max_images` from `params.yaml` — limits dataset size for fast CPU training
- Applies `np.random.seed(42)` before sampling → always picks the **same subset**
- Splits 80% train / 20% val using `sklearn.train_test_split(random_state=42)`
- Writes `data/splits/train_v8.txt` and `val_v8.txt` — the ground truth index files
- Physically copies images into `data/processed/images/train/` and `val/`
- **MLOps value:** Reproducibility. Running the pipeline again next month picks the exact same 400 images in the exact same split. Fair experiment comparison is guaranteed.

### Stage 4 — TRAIN 🏋️
**Script:** `src/training/train.py`

- Loads all hyperparameters from `params.yaml` (epochs, batch size, image size, patience, lr)
- Resolves device automatically (`"auto"` → CPU when no CUDA GPU found)
- Opens an MLflow run, logs all 9 parameters before training starts
- Calls `YOLO.train()` with:
  - `data=configs/dataset.yaml` (our 1-class pothole config)
  - `patience=5` → **early stopping** (stops if mAP doesn't improve for 5 epochs)
  - `project="runs/train"`, `name={run_name}`
- After training, searches multiple possible output directories (handles Ultralytics' quirky nesting)
- Copies `best.pt` → `models/yolov8_road_v{N}.pt`
- Logs to MLflow: epoch-by-epoch loss curves, confusion matrix PNG, PR curve PNG, F1 curve PNG, `params.yaml` snapshot
- **MLOps value:** Every experiment run is permanently recorded. You can compare v6 vs v8 side by side in MLflow UI

### Stage 5 — EVALUATE 📊
**Script:** `src/training/evaluate.py`

- Auto-picks the latest model from `models/`
- Runs `YOLO.val()` on `data/processed/images/val/`
- Extracts and prints:
  - **mAP@50** (main accuracy metric)
  - **mAP@50-95** (stricter IoU threshold)
  - **Precision** (of detected potholes, how many are real)
  - **Recall** (of all real potholes, how many were found)
  - **Per-class AP** breakdown
- Saves `eval_metrics.json` (machine-readable for DVC comparison)
- **MLOps value:** Standardised evaluation means every model is tested identically. No cherry-picking test sets.

### Stage 6 — INFERENCE 🖼️
**Script:** `scripts/run_pipeline.py` → calls `run_inference()`

- Auto-picks the latest model from `models/`
- Runs detection on all images in `data/processed/images/val/` at `conf=0.05`
- Saves annotated images (with bounding boxes drawn) to `runs/pipeline_inference/detect/`
- Prints detection summary: images processed, total potholes found
- **MLOps value:** Every pipeline run produces visual evidence. You can open the folder and see exactly what the model detected, without running any extra commands.

---

## 📊 Experiment History — How We Got to 59% mAP

| Version | Images | Epochs | img_size | batch | mAP@50 | Precision | Recall | Issue |
|---|---|---|---|---|---|---|---|---|
| **v5** | 100 | 5 | 640 | 16 | ~12% | <1% | ~80% | Too few epochs, low confidence |
| **v6** | 500 | 30 | 640 | 16 | 12.8% | 0.5% | 79.6% | OOM crash on epoch 5 |
| **v7** | 800 | 50 | 640 | 8 | ❌ OOM | — | — | RAM exhausted |
| **v8** | 400 | 30 | **320** | **4** | **59.1%** | **63.6%** | **55.5%** | ✅ Success! |

**Key insight:** Reducing `image_size: 640 → 320` saves **4x memory** (quadratic relationship).
Combined with `workers: 0` and `batch_size: 4`, training completed without any memory errors.

---

## 🐛 Problems We Solved Along the Way

### 1. Roboflow Downloads to a Nested Folder
**Problem:** `data/raw/car-detection-fj7cr-3/train/images/` — pipeline couldn't find files at expected path
**Fix:** `resolve_raw_dir()` in `preprocess.py` reads `dataset_location.txt` saved during download

### 2. CUDA Device Error on CPU-only Machine
**Problem:** `device="auto"` crashed with `ValueError: Invalid CUDA 'device=auto'`
**Fix:** Added safe device resolution in `train.py` — checks `torch.cuda.is_available()` and falls back to `"cpu"`

### 3. MLflow Rejects Metric Names with Parentheses
**Problem:** `metrics/precision(B)` caused `MlflowException` because `()` are not allowed
**Fix:** Added `sanitize()` function in `mlflow_logger.py` — replaces `(` → `_`, `)` → `` before logging

### 4. max_images Was Ignored by Pipeline
**Problem:** `params.yaml` had `max_images: 100` but pipeline used all 10,000 images
**Root cause:** `run_pipeline.py` called `create_split()` without passing `max_images`
**Fix:** Added `max_images=int(split_cfg.get("max_images"))` to the split stage call

### 5. Out of Memory (OOM) During Training
**Problem:** `RuntimeError: DefaultCPUAllocator: not enough memory` on epoch 5
**Fix:** Changed `image_size: 640 → 320` (4x reduction), `batch_size: 8 → 4`, `workers: 4 → 0`

### 6. Class Mapping for New Dataset
**Problem:** Switching from car detection to pothole detection — old classes filtered out all pothole labels
**Fix:** Updated `TARGET_CLASSES` and `CLASS_ALIASES` in `preprocess.py` to map all pothole name variants

---

## 🔮 MLOps vs Regular ML

```
Regular ML (Jupyter Notebook):
  1. Run cell to download              ← Manual
  2. Manually fix paths                ← Error-prone
  3. Run training cell                 ← Manual
  4. Copy accuracy to a notepad        ← Not tracked
  5. Save model file manually          ← No versioning
  6. Can't reproduce which settings    ← Not reproducible
     gave which result

This MLOps Pipeline:
  python scripts\run_pipeline.py       ← ONE command does everything

  ✅ Auto-downloads correct dataset
  ✅ Auto-maps class names
  ✅ Same split every run (seed=42)
  ✅ Stops early if overfitting (patience)
  ✅ All metrics logged to MLflow
  ✅ Model versioned automatically
  ✅ Val images annotated & saved
```

---

## 🚀 How to Use This System Going Forward

### Try a new dataset:
Edit `params.yaml`:
```yaml
data:
  version: "v9"
  workspace: "new-workspace"
  project: "new-project-name"
  roboflow_version: 1
```
Then run:
```batch
python scripts\run_pipeline.py --run-name "v9-new-data"
```

### Tune for better accuracy:
```yaml
train:
  epochs: 50
  batch_size: 4
  image_size: 320
  patience: 5
```

### View all experiments side by side:
```batch
mlflow ui
# Open http://127.0.0.1:5000
```

### Test on your own pothole photos:
```batch
python detect.py --source path\to\your\photos\
```

---

## ✅ Final State of the Project

| Item | Status |
|---|---|
| Best model | `models/yolov8_road_v3.pt` — 59.1% mAP |
| Experiments logged | MLflow → `mlruns/` |
| Annotated val images | `runs/pipeline_inference/detect/` |
| Config | `params.yaml` — single file controls everything |
| Reproducible | seed=42 → same split every run |
| Extensible | Change dataset in `params.yaml` → pipeline handles the rest |



to run application on web
.\start_app.bat
