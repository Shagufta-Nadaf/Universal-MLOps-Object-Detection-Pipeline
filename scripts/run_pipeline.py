"""
scripts/run_pipeline.py
End-to-end pipeline orchestrator.

Runs all stages in sequence:
  1. download   → data/raw/
  2. preprocess → data/processed/
  3. split      → data/splits/ + data/processed/{train,val}/
  4. train      → models/ + MLflow experiment
  5. evaluate   → eval_metrics.json

This is the "one command to rule them all" script.
You can also use `dvc repro` for the same effect with full caching.

Usage:
    # Full pipeline
    python scripts/run_pipeline.py

    # Skip already-done stages
    python scripts/run_pipeline.py --start-from split

    # Quick smoke test (1 epoch, verifies everything works end-to-end)
    python scripts/run_pipeline.py --smoke-test

    # Custom run name for MLflow
    python scripts/run_pipeline.py --run-name "experiment-lr-0.005"

    # Skip download if data already exists
    python scripts/run_pipeline.py --skip-download
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.helpers import load_params, setup_logger

logger = setup_logger("run_pipeline")

STAGES = ["download", "preprocess", "split", "train", "evaluate"]


def run_stage(name: str, fn, *args, **kwargs) -> bool:
    """Run a pipeline stage and return True on success."""
    logger.info("")
    logger.info("━" * 60)
    logger.info(f"  STAGE: {name.upper()}")
    logger.info("━" * 60)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info(f"  ✅  {name} completed in {elapsed:.1f}s")
        return True
    except SystemExit as e:
        if e.code == 0:
            return True
        logger.error(f"  ❌  {name} failed (exit code {e.code})")
        return False
    except Exception as e:
        logger.exception(f"  ❌  {name} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the full road detection MLOps pipeline")
    parser.add_argument(
        "--start-from",
        choices=STAGES,
        default="download",
        help="Start pipeline from this stage (skips earlier stages)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download stage (data already exists)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 epoch on small batch — verify pipeline works end-to-end",
    )
    parser.add_argument("--run-name", default=None, help="MLflow experiment run name")
    parser.add_argument("--api-key", default=None, help="Roboflow API key")
    args = parser.parse_args()

    params = load_params()
    active_stages = STAGES[STAGES.index(args.start_from):]
    if args.skip_download and "download" in active_stages:
        active_stages.remove("download")

    logger.info("=" * 60)
    logger.info("  Road Scene Object Detection — MLOps Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Stages to run : {' → '.join(active_stages)}")
    if args.smoke_test:
        logger.info("  Mode          : SMOKE TEST (1 epoch)")
    logger.info("")

    pipeline_start = time.time()
    failed_stage   = None

    # ── Stage: Download ───────────────────────────────────────────────────────
    if "download" in active_stages:
        from src.data.download_data import download_dataset, build_manifest, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION
        import os
        from dotenv import load_dotenv
        load_dotenv()

        api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
        data_cfg  = params["data"]
        paths_cfg = params["paths"]

        ok = run_stage(
            "download",
            download_dataset,
            api_key=api_key or "missing",
            workspace=data_cfg.get("workspace", ROBOFLOW_WORKSPACE),
            project=data_cfg.get("project", ROBOFLOW_PROJECT),
            version=int(data_cfg.get("roboflow_version", ROBOFLOW_VERSION)),
            fmt=data_cfg.get("format", "yolov8"),
            output_dir=Path(paths_cfg["raw_data"]),
        )
        if not ok:
            failed_stage = "download"

    # ── Stage: Preprocess ─────────────────────────────────────────────────────
    if "preprocess" in active_stages and not failed_stage:
        from src.data.preprocess import preprocess

        paths_cfg = params["paths"]
        ok = run_stage(
            "preprocess",
            preprocess,
            raw_dir=Path(paths_cfg["raw_data"]),
            output_dir=Path(paths_cfg["processed_data"]),
        )
        if not ok:
            failed_stage = "preprocess"

    # ── Stage: Split ──────────────────────────────────────────────────────────
    if "split" in active_stages and not failed_stage:
        from src.data.split_data import create_split

        split_cfg = params["split"]
        data_cfg  = params["data"]
        paths_cfg = params["paths"]

        ok = run_stage(
            "split",
            create_split,
            processed_dir=Path(paths_cfg["processed_data"]),
            splits_dir=Path(paths_cfg["splits_dir"]),
            version=data_cfg.get("version", "v1"),
            seed=int(split_cfg["seed"]),
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
        )
        if not ok:
            failed_stage = "split"

    # ── Stage: Train ──────────────────────────────────────────────────────────
    if "train" in active_stages and not failed_stage:
        from src.training.train import run_training

        ok = run_stage(
            "train",
            run_training,
            params=params,
            run_name=args.run_name,
            smoke_test=args.smoke_test,
        )
        if not ok:
            failed_stage = "train"

    # ── Stage: Evaluate ───────────────────────────────────────────────────────
    if "evaluate" in active_stages and not failed_stage:
        from src.training.evaluate import evaluate_model, find_latest_model, print_metrics_table
        from src.utils.helpers import save_json

        paths_cfg    = params["paths"]
        models_dir   = Path(paths_cfg["models_dir"])
        dataset_yaml = Path(paths_cfg["dataset_yaml"])
        model_path   = find_latest_model(models_dir)

        if model_path:
            def _eval():
                metrics = evaluate_model(model_path, dataset_yaml)
                print_metrics_table(metrics)
                safe_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
                safe_metrics["model"] = model_path.name
                save_json(safe_metrics, "eval_metrics.json")

            ok = run_stage("evaluate", _eval)
            if not ok:
                failed_stage = "evaluate"
        else:
            logger.warning("  No model found for evaluation — skipping.")

    # ── Final Summary ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 60)
    if failed_stage:
        logger.error(f"  ❌  Pipeline FAILED at stage: {failed_stage}")
        sys.exit(1)
    else:
        logger.info(f"  ✅  Pipeline COMPLETE! ({elapsed_total:.1f}s total)")
        logger.info("")
        logger.info("  What to do next:")
        logger.info("    1. View experiments: mlflow ui")
        logger.info("       → Open http://127.0.0.1:5000")
        logger.info("    2. Run inference:")
        logger.info("       → python scripts/run_inference.py --source path/to/images/")
        logger.info("    3. Try another experiment:")
        logger.info("       → Edit params.yaml → python scripts/run_pipeline.py --start-from train")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
