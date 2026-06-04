"""End-to-end reconstruction training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import AppConfig, load_config
from ..data.io import load_table
from ..data.validators import parse_datetime_column, require_columns
from ..models.reconstruction import benchmark_reconstruction_split


@dataclass(frozen=True, slots=True)
class ReconstructionPipelineResult:
    """Artifacts and summary values produced by the training pipeline."""

    metrics_path: Path
    predictions_path: Path
    metrics: pd.DataFrame
    predictions: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ReconstructionSplit:
    """Deterministic train/validation split for reconstruction modeling."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    validation_rows: pd.DataFrame


DEFAULT_VALIDATION_FRACTION = 0.25


def validate_reconstruction_columns(
    data: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
) -> list[str]:
    """Validate reconstruction target and feature selection against a dataset."""
    if not target_column:
        raise ValueError("target_column must be a non-empty string.")
    if not feature_columns:
        raise ValueError("feature_columns must contain at least one column name.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must not contain duplicates.")
    if target_column in feature_columns:
        raise ValueError("target_column must not also appear in feature_columns.")

    require_columns(data, [target_column, *feature_columns])
    return list(feature_columns)


def load_prepared_reconstruction_dataset(config: AppConfig) -> pd.DataFrame:
    """Load the prepared modeling dataset referenced by the application config."""
    data = load_table(config.dataset_path)
    required_columns = [
        config.station_id_column,
        config.date_column,
        config.target_column,
        *config.feature_columns,
    ]
    require_columns(data, required_columns)
    parsed = parse_datetime_column(data, config.date_column)
    return parsed.reset_index(drop=True)


def build_reconstruction_split(
    data: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    random_state: int,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> ReconstructionSplit:
    """Create a deterministic complete-case train/validation split."""
    validated_features = validate_reconstruction_columns(
        data,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    complete_cases = data.dropna(subset=[target_column, *validated_features]).copy()
    if complete_cases.empty:
        raise ValueError(
            "No complete cases are available after dropping rows with missing target/features."
        )
    if len(complete_cases) < 4:
        raise ValueError("At least four complete cases are required for training.")

    train_rows, valid_rows = train_test_split(
        complete_cases.reset_index(drop=True),
        test_size=validation_fraction,
        random_state=random_state,
        shuffle=True,
    )
    train_rows = train_rows.reset_index(drop=True)
    valid_rows = valid_rows.reset_index(drop=True)

    return ReconstructionSplit(
        X_train=train_rows[validated_features].copy(),
        y_train=train_rows[target_column].copy(),
        X_valid=valid_rows[validated_features].copy(),
        y_valid=valid_rows[target_column].copy(),
        validation_rows=valid_rows.copy(),
    )


def run_training_pipeline(config_path: str | Path) -> ReconstructionPipelineResult:
    """Run the reconstruction benchmark workflow and persist outputs to disk."""
    config = load_config(config_path)
    dataset = load_prepared_reconstruction_dataset(config)
    split = build_reconstruction_split(
        dataset,
        target_column=config.target_column,
        feature_columns=config.feature_columns,
        random_state=config.random_state,
    )

    metrics_df, predictions_df = benchmark_reconstruction_split(
        X_train=split.X_train,
        y_train=split.y_train,
        X_valid=split.X_valid,
        y_valid=split.y_valid,
        methods=config.models,
        random_state=config.random_state,
    )

    prediction_output = split.validation_rows[[config.station_id_column, config.date_column]].copy()
    prediction_output[config.target_column] = split.y_valid.reset_index(drop=True)
    for column in predictions_df.columns:
        prediction_output[column] = predictions_df[column].reset_index(drop=True)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "reconstruction_metrics.csv"
    predictions_path = output_dir / "reconstruction_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    prediction_output.to_csv(predictions_path, index=False)

    successful = metrics_df.loc[metrics_df["status"] == "ok"].sort_values("rmse")
    if successful.empty:
        best_summary = "No models completed successfully"
    else:
        best = successful.iloc[0]
        best_summary = (
            f"Best method: {best['method']} "
            f"(RMSE={best['rmse']:.4f}, MAE={best['mae']:.4f}, R2={best['r2']:.4f})"
        )

    print(
        "\n".join(
            [
                f"Loaded {len(dataset)} rows from {config.dataset_path}",
                (
                    f"Train rows: {len(split.X_train)} | "
                    f"Validation rows: {len(split.X_valid)} | "
                    f"Models: {len(metrics_df)}"
                ),
                best_summary,
                f"Metrics saved to: {metrics_path}",
                f"Predictions saved to: {predictions_path}",
            ]
        )
    )

    return ReconstructionPipelineResult(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        metrics=metrics_df,
        predictions=prediction_output,
    )
