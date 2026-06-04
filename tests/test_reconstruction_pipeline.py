from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ai_aquatica_rs.pipelines import run_training_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [str(SRC_PATH)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _write_synthetic_pipeline_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    n_rows = 24
    feature_a = np.linspace(0.5, 4.5, n_rows)
    feature_b = np.cos(feature_a)
    feature_c = rng.normal(loc=0.0, scale=0.05, size=n_rows)
    target = 1.0 + 1.7 * feature_a - 0.8 * feature_b + 0.3 * feature_c

    dataset = pd.DataFrame(
        {
            "station_id": ["station-1"] * n_rows,
            "date": pd.date_range("2024-01-01", periods=n_rows, freq="D").astype(str),
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            "target": target,
        }
    )
    dataset.loc[2, "feature_c"] = np.nan
    dataset.loc[5, "target"] = np.nan

    dataset_path = tmp_path / "prepared_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "AI-Aquatica-RS"},
                "data": {
                    "dataset_path": str(dataset_path),
                    "date_column": "date",
                    "target_column": "target",
                    "station_id_column": "station_id",
                },
                "features": {
                    "feature_columns": ["feature_a", "feature_b", "feature_c"],
                },
                "runtime": {
                    "random_state": 13,
                    "output_dir": str(output_dir),
                    "nearest_days_tolerance": 3,
                    "models": ["median", "linear_regression", "random_forest"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path, output_dir


def test_run_training_pipeline_writes_metrics_and_predictions(tmp_path: Path) -> None:
    config_path, output_dir = _write_synthetic_pipeline_inputs(tmp_path)

    result = run_training_pipeline(config_path)

    assert result.metrics_path.exists()
    assert result.predictions_path.exists()
    assert result.metrics_path == output_dir / "reconstruction_metrics.csv"
    assert result.predictions_path == output_dir / "reconstruction_predictions.csv"

    metrics = pd.read_csv(result.metrics_path)
    predictions = pd.read_csv(result.predictions_path)
    assert {"method", "status", "rmse", "mae", "r2", "mape"}.issubset(metrics.columns)
    assert {
        "station_id",
        "date",
        "target",
        "y_true",
        "median",
        "linear_regression",
    }.issubset(predictions.columns)
    assert len(predictions) > 0


def test_cli_train_reconstruction_command_runs_end_to_end(tmp_path: Path) -> None:
    config_path, output_dir = _write_synthetic_pipeline_inputs(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_aquatica_rs.cli",
            "train-reconstruction",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_cli_env(),
        cwd=REPO_ROOT,
    )

    assert "Best method:" in result.stdout
    assert (output_dir / "reconstruction_metrics.csv").exists()
    assert (output_dir / "reconstruction_predictions.csv").exists()
