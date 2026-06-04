"""Base utilities for reconstruction model benchmarking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class OptionalDependencyUnavailable(ImportError):
    """Raised when an optional modeling dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class ReconstructionMethodSpec:
    """Specification for a reconstruction benchmark method."""

    name: str
    family: str
    factory: Callable[[int], RegressorMixin]
    optional_dependency: str | None = None


class ConstantRegressor(BaseEstimator, RegressorMixin):
    """Deterministic baseline regressor using the mean or median target value."""

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def fit(self, X: object, y: object) -> "ConstantRegressor":
        """Fit the regressor by storing a single summary value from ``y``."""
        del X
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if y_array.size == 0:
            raise ValueError("y must contain at least one observation.")
        if self.strategy == "mean":
            self.constant_ = float(np.mean(y_array))
        elif self.strategy == "median":
            self.constant_ = float(np.median(y_array))
        else:
            raise ValueError("strategy must be either 'mean' or 'median'.")
        return self

    def predict(self, X: object) -> np.ndarray:
        """Predict a constant value for every row in ``X``."""
        if not hasattr(self, "constant_"):
            raise ValueError("ConstantRegressor must be fitted before predicting.")
        n_samples = len(X) if hasattr(X, "__len__") else 0
        return np.full(shape=n_samples, fill_value=self.constant_, dtype=float)
