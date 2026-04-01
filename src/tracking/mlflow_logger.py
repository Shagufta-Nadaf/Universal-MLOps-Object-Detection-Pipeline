"""
src/tracking/mlflow_logger.py
MLflow experiment logging helpers.

Usage:
    from src.tracking.mlflow_logger import MLflowLogger

    with MLflowLogger("road-detection", run_name="yolov8n-v1") as tracker:
        tracker.log_params({"epochs": 30, "lr": 0.01})
        tracker.log_metrics({"mAP50": 0.72, "mAP50-95": 0.45})
        tracker.log_artifact("models/yolov8_road_v1.pt")
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch

from src.utils.helpers import setup_logger

logger = setup_logger("MLflowLogger")


class MLflowLogger:
    """
    Context-manager wrapper around MLflow for clean experiment logging.

    Attributes:
        experiment_name (str): Name of the MLflow experiment.
        run_name (str): Human-readable name for this run.
        run_id (str): MLflow run ID (set after start).
    """

    def __init__(
        self,
        experiment_name: str = "road-detection",
        run_name: Optional[str] = None,
        tracking_uri: str = "mlruns",
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self.run_id: Optional[str] = None
        self._run = None

    # ── Context Manager ────────────────────────────────────────────────────────

    def __enter__(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        self.run_id = self._run.info.run_id
        logger.info(f"MLflow run started: {self.run_name or 'unnamed'} (id={self.run_id})")
        logger.info(f"  Experiment : {self.experiment_name}")
        logger.info(f"  Tracking UI: run `mlflow ui` then open http://127.0.0.1:5000")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(exc_val))
            logger.error(f"MLflow run FAILED: {exc_val}")
        else:
            mlflow.set_tag("status", "SUCCESS")
        mlflow.end_run()
        logger.info(f"MLflow run ended (id={self.run_id})")

    # ── Logging Methods ────────────────────────────────────────────────────────

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dict of hyperparameters."""
        mlflow.log_params(params)
        logger.info(f"  Logged {len(params)} params")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric (optionally at a training step for time-series)."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"  Logged metrics: { {k: f'{v:.4f}' for k, v in metrics.items()} }")

    def log_artifact(self, local_path: str | Path) -> None:
        """Upload a local file or directory as an MLflow artifact."""
        mlflow.log_artifact(str(local_path))
        logger.info(f"  Logged artifact: {local_path}")

    def log_artifacts_dir(self, local_dir: str | Path) -> None:
        """Upload all files in a directory as MLflow artifacts."""
        mlflow.log_artifacts(str(local_dir))
        logger.info(f"  Logged artifact dir: {local_dir}")

    def set_tag(self, key: str, value: str) -> None:
        """Set an MLflow tag on the current run."""
        mlflow.set_tag(key, value)

    def log_dict_as_artifact(self, data: Dict, filename: str) -> None:
        """Log a Python dict as a JSON artifact."""
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=filename
        ) as f:
            json.dump(data, f, indent=2)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="configs")
        os.unlink(tmp_path)

    # ── Convenience ───────────────────────────────────────────────────────────

    def log_yolo_results(self, results_dir: Path) -> None:
        """
        Parse YOLOv8 results/results.csv produced by Ultralytics and log
        final-epoch metrics to MLflow.
        """
        import pandas as pd

        results_csv = results_dir / "results.csv"
        if not results_csv.exists():
            logger.warning(f"results.csv not found at {results_csv}")
            return

        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # strip whitespace from column names

        def sanitize(name: str) -> str:
            """MLflow metric names can't contain parentheses — replace them."""
            return (name.replace("(", "_")
                        .replace(")", "")
                        .replace(" ", "_")
                        .strip("_"))

        # Log epoch-by-epoch metrics as time series
        metric_cols = [c for c in df.columns if any(
            k in c.lower() for k in ["map", "loss", "precision", "recall"]
        )]
        for _, row in df.iterrows():
            epoch = int(row.get("epoch", 0))
            for col in metric_cols:
                try:
                    self.log_metric(sanitize(col), float(row[col]), step=epoch)
                except (ValueError, KeyError):
                    pass

        # Log final epoch summary metrics
        last_row = df.iloc[-1]
        summary = {}
        for col in metric_cols:
            try:
                summary[f"final_{sanitize(col)}"] = float(last_row[col])
            except (ValueError, KeyError):
                pass
        if summary:
            self.log_metrics(summary)

        logger.info(f"  Parsed & logged YOLO results from {results_csv}")

    @property
    def active_run_url(self) -> str:
        """Return the MLflow UI URL for this run."""
        return f"http://127.0.0.1:5000/#/experiments/{self.experiment_name}/runs/{self.run_id}"
