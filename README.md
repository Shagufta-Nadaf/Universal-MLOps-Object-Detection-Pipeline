# 🚀 Universal MLOps Object Detection Pipeline

![MLOps](https://img.shields.io/badge/MLOps-Ready-brightgreen)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Fast-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Built-blueviolet)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)

Transform any Roboflow-hosted object detection dataset into a production-ready model with **zero code changes**. This system is a dynamic, end-to-end MLOps pipeline designed for the rapid training and deployment of vision models.

---

## 🌟 Key Features

*   **⚡ 100% Universal Dataset Support:** Switch between Face Detection, Potholes, Car Detection, or any custom project by simply providing the Roboflow project name.
*   **🛡️ Multi-Dataset Isolation (MDIS):** A project-specific directory hierarchy prevents dataset mixing and ensures clean, reproducible experiments.
*   **🤖 Dynamic Preprocessing:** Automatically discovers class names and generates YOLO `dataset.yaml` configurations on the fly.
*   **📈 MLflow Experiment Tracking:** Every run captures metrics, losses, and hyperparameters for deep performance analysis.
*   **📂 Smart Result Packaging:** Automatically packages the trained model, evaluation plots, and inference samples into a single versioned folder.
*   **🌐 Web Interface:** A FastAPI-powered browser UI for live log streaming, training orchestration, and artifact download.

---

## 🏗️ Technical Architecture & Pipeline

The pipeline follows a modular, process-oriented architecture:

1.  **Data Ingestion:** Securely downloads datasets from Roboflow into isolated workspace subdirectories.
2.  **Preprocessing:** Cleans raw labels, performs class mapping, and implements a `Strict Ghost Filter` to remove accidental image inclusions.
3.  **Splitting:** Managed Train/Val partitioning with manifest JSON generation for experiment lineage.
4.  **Training:** High-performance YOLOv8 training with custom-tuned hyperparameters for low-end hardware.
5.  **Evaluation:** Robust model testing using **Numeric Version Sorting** to ensure valid benchmark comparison.
6.  **Inference:** Automated validation results generation on hold-out data.
7.  **Packaging:** Collection of all training artifacts into an isolated "One-Folder" output.

---

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.10+
*   Conda (Recommended)
*   Roboflow Private API Key

### 2. Installation
```bash
git clone https://github.com/your-username/universal-mlops-pipeline.git
cd universal-mlops-pipeline
conda create -n mlops python=3.10
conda activate mlops
pip install -r requirements.txt
```

### 3. Launching the Web UI
Run the provided batch file to start the FastAPI server:
```bash
.\start_app.bat
```
Visit `http://localhost:8000` in your browser to start a training run.

---

## 🛠️ CLI Usage (Automated Mode)

Update `params.yaml` with your project metadata, then run:
```bash
python scripts/run_pipeline.py
```

To run individual stages:
```bash
python src/data/download_data.py   # Step 1: Isolated Ingestion
python src/data/preprocess.py      # Step 2: Dynamic Prep
python src/training/train.py       # Step 3: Train + MLflow
```

---

## 📂 Project Structure

```text
├── app/                  # FastAPI Web Interface & Frontend
├── configs/              # Dynamic YOLO dataset configurations
├── data/                 # Isolated Data Storage (Raw, Processed, Splits)
├── models/               # Trained .pt checkpoints (Versioned)
├── outputs/              # Smart-Packaged run results
├── scripts/              # Pipeline orchestrators & Inference
├── src/                  # Core Module Logic (Data, Tracking, Training)
├── params.yaml           # Central Configuration Hub
└── README.md             # You are here!
```

---

## 🏆 Acknowledgements

*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection engine.
*   [Roboflow](https://roboflow.com/) for the seamless dataset management.
*   [MLflow](https://mlflow.org/) for experiment lifecycle tracking.

---

**Developed with ❤️ for the MLOps Community.**
