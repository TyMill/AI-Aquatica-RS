"""Reconstruction model benchmark utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from .base import ConstantRegressor, OptionalDependencyUnavailable, ReconstructionMethodSpec
from .evaluation import regression_metrics


def _optional_xgboost(random_state: int):
    """Build an XGBoost regressor when xgboost is installed."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise OptionalDependencyUnavailable("xgboost is not installed.") from exc

    return XGBRegressor(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=random_state,
        verbosity=0,
        n_jobs=1,
    )


def _optional_lightgbm(random_state: int):
    """Build a LightGBM regressor when lightgbm is installed."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise OptionalDependencyUnavailable("lightgbm is not installed.") from exc

    return LGBMRegressor(
        n_estimators=50,
        learning_rate=0.05,
        random_state=random_state,
        verbosity=-1,
        n_jobs=1,
    )


def get_reconstruction_method_specs() -> dict[str, ReconstructionMethodSpec]:
    """Return the supported reconstruction benchmark methods."""
    return {
        "median": ReconstructionMethodSpec(
            name="median",
            family="baseline",
            factory=lambda _rs: ConstantRegressor(strategy="median"),
        ),
        "mean": ReconstructionMethodSpec(
            name="mean",
            family="baseline",
            factory=lambda _rs: ConstantRegressor(strategy="mean"),
        ),
        "linear_regression": ReconstructionMethodSpec(
            name="linear_regression",
            family="linear",
            factory=lambda _rs: LinearRegression(),
        ),
        "ridge": ReconstructionMethodSpec(
            name="ridge",
            family="linear",
            factory=lambda rs: Ridge(random_state=rs),
        ),
        "lasso": ReconstructionMethodSpec(
            name="lasso",
            family="linear",
            factory=lambda rs: Lasso(random_state=rs),
        ),
        "elastic_net": ReconstructionMethodSpec(
            name="elastic_net",
            family="linear",
            factory=lambda rs: ElasticNet(random_state=rs),
        ),
        "knn": ReconstructionMethodSpec(
            name="knn",
            family="distance",
            factory=lambda _rs: KNeighborsRegressor(n_neighbors=5),
        ),
        "decision_tree": ReconstructionMethodSpec(
            name="decision_tree",
            family="tree",
            factory=lambda rs: DecisionTreeRegressor(random_state=rs),
        ),
        "random_forest": ReconstructionMethodSpec(
            name="random_forest",
            family="ensemble",
            factory=lambda rs: RandomForestRegressor(n_estimators=50, random_state=rs, n_jobs=1),
        ),
        "extra_trees": ReconstructionMethodSpec(
            name="extra_trees",
            family="ensemble",
            factory=lambda rs: ExtraTreesRegressor(n_estimators=50, random_state=rs, n_jobs=1),
        ),
        "gradient_boosting": ReconstructionMethodSpec(
            name="gradient_boosting",
            family="ensemble",
            factory=lambda rs: GradientBoostingRegressor(random_state=rs),
        ),
        "hist_gradient_boosting": ReconstructionMethodSpec(
            name="hist_gradient_boosting",
            family="ensemble",
            factory=lambda rs: HistGradientBoostingRegressor(random_state=rs),
        ),
        "adaboost": ReconstructionMethodSpec(
            name="adaboost",
            family="ensemble",
            factory=lambda rs: AdaBoostRegressor(random_state=rs),
        ),
        "xgboost": ReconstructionMethodSpec(
            name="xgboost",
            family="ensemble",
            factory=_optional_xgboost,
            optional_dependency="xgboost",
        ),
        "lightgbm": ReconstructionMethodSpec(
            name="lightgbm",
            family="ensemble",
            factory=_optional_lightgbm,
            optional_dependency="lightgbm",
        ),
    }


def list_reconstruction_methods() -> list[str]:
    """Return all supported reconstruction method names."""
    return list(get_reconstruction_method_specs().keys())


def _resolve_methods(methods: Sequence[str] | None) -> list[str]:
    specs = get_reconstruction_method_specs()
    selected_methods = list(specs) if methods is None else list(methods)
    unknown_methods = sorted(set(selected_methods) - set(specs))
    if unknown_methods:
        raise ValueError(
            "Unknown reconstruction methods: " + ", ".join(unknown_methods) + "."
        )
    return selected_methods


def _validate_feature_selection(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str],
) -> list[str]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not target_column:
        raise ValueError("target_column must be a non-empty string.")
    if len(feature_columns) == 0:
        raise ValueError("feature_columns must contain at least one column name.")

    normalized_feature_columns = list(feature_columns)
    if len(set(normalized_feature_columns)) != len(normalized_feature_columns):
        raise ValueError("feature_columns must not contain duplicates.")
    if target_column in normalized_feature_columns:
        raise ValueError("target_column must not also appear in feature_columns.")

    required_columns = [target_column, *normalized_feature_columns]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns) + ".")

    return normalized_feature_columns


def _prepare_complete_cases(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    validated_features = _validate_feature_selection(data, target_column, feature_columns)
    complete_cases = data[[target_column, *validated_features]].dropna().copy()
    if complete_cases.empty:
        raise ValueError("No complete cases are available after dropping missing values.")
    if len(complete_cases) < 4:
        raise ValueError("At least four complete cases are required for benchmarking.")
    return complete_cases


def _coerce_split_inputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if X_train.empty or X_valid.empty:
        raise ValueError("X_train and X_valid must contain at least one row.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")
    if len(X_valid) != len(y_valid):
        raise ValueError("X_valid and y_valid must have the same number of rows.")

    if X_train.isna().any().any() or X_valid.isna().any().any():
        raise ValueError("Training and validation features must not contain missing values.")
    if y_train.isna().any() or y_valid.isna().any():
        raise ValueError("Training and validation targets must not contain missing values.")

    return (
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        X_valid.reset_index(drop=True),
        y_valid.reset_index(drop=True),
    )


def _benchmark_from_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    methods: Sequence[str] | None,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_prepared, y_train_prepared, X_valid_prepared, y_valid_prepared = _coerce_split_inputs(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )
    selected_methods = _resolve_methods(methods)
    specs = get_reconstruction_method_specs()

    results: list[dict[str, object]] = []
    predictions = pd.DataFrame({"y_true": y_valid_prepared})
    for method_name in selected_methods:
        spec = specs[method_name]
        try:
            estimator = spec.factory(random_state)
        except OptionalDependencyUnavailable as exc:
            results.append(
                {
                    "method": method_name,
                    "family": spec.family,
                    "status": "skipped",
                    "n_train": int(len(X_train_prepared)),
                    "n_valid": int(len(X_valid_prepared)),
                    "rmse": np.nan,
                    "mae": np.nan,
                    "r2": np.nan,
                    "mape": np.nan,
                    "skipped_reason": str(exc),
                }
            )
            continue

        estimator.fit(X_train_prepared, y_train_prepared)
        y_pred = np.asarray(estimator.predict(X_valid_prepared), dtype=float)
        metrics = regression_metrics(y_valid_prepared.to_numpy(), y_pred)
        predictions[method_name] = y_pred
        results.append(
            {
                "method": method_name,
                "family": spec.family,
                "status": "ok",
                "n_train": int(len(X_train_prepared)),
                "n_valid": int(len(X_valid_prepared)),
                **metrics,
                "skipped_reason": None,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df.sort_values(["status", "rmse", "method"], na_position="last").reset_index(
        drop=True
    ), predictions


def benchmark_reconstruction(
    data: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Sequence[str],
    methods: Sequence[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> pd.DataFrame:
    """Benchmark reconstruction methods on a deterministic train/validation split."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    complete_cases = _prepare_complete_cases(data, target_column, feature_columns)
    prepared_feature_columns = [column for column in complete_cases.columns if column != target_column]
    X_train, X_valid, y_train, y_valid = train_test_split(
        complete_cases[prepared_feature_columns],
        complete_cases[target_column],
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    results_df, _predictions_df = _benchmark_from_split(
        X_train,
        y_train,
        X_valid,
        y_valid,
        methods,
        random_state=random_state,
    )
    return results_df


def benchmark_reconstruction_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    methods: Sequence[str] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark reconstruction methods on a pre-defined split."""
    return _benchmark_from_split(
        X_train,
        y_train,
        X_valid,
        y_valid,
        methods,
        random_state=random_state,
    )
