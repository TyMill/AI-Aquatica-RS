"""Evaluation metrics for reconstruction benchmarks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> float:
    """Return MAPE while protecting against near-zero denominators."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero.")

    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true_array.size == 0:
        raise ValueError("y_true and y_pred must contain at least one value.")

    denominator = np.maximum(np.abs(y_true_array), epsilon)
    percentage_errors = np.abs(y_true_array - y_pred_array) / denominator
    return float(np.mean(percentage_errors) * 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, R2, and safe MAPE for regression predictions."""
    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true_array.size == 0:
        raise ValueError("y_true and y_pred must contain at least one value.")

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "r2": float(r2_score(y_true_array, y_pred_array)),
        "mape": safe_mean_absolute_percentage_error(y_true_array, y_pred_array),
    }
