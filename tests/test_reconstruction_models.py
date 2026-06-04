from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_aquatica_rs.models.base import OptionalDependencyUnavailable, ReconstructionMethodSpec
from ai_aquatica_rs.models.reconstruction import (
    benchmark_reconstruction,
    get_reconstruction_method_specs,
)


CORE_METHODS = {
    "median",
    "mean",
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "knn",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "adaboost",
}
OPTIONAL_METHODS = {"xgboost", "lightgbm"}


@pytest.fixture()
def synthetic_regression_frame() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n_samples = 120
    f1 = np.linspace(-2.0, 3.0, n_samples)
    f2 = np.sin(f1)
    f3 = rng.normal(loc=0.0, scale=0.25, size=n_samples)
    f4 = rng.normal(loc=1.0, scale=0.4, size=n_samples)
    noise = rng.normal(loc=0.0, scale=0.05, size=n_samples)
    target = 0.2 + 2.5 * f1 - 1.2 * f2 + 0.8 * f3 + 0.5 * f4 + noise

    frame = pd.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "target": target,
        }
    )
    frame.loc[[3, 17, 88], "target"] = np.nan
    frame.loc[[5, 9], "f3"] = np.nan
    frame.loc[0, "target"] = 0.0
    frame.loc[1, "target"] = 1e-12
    return frame


def test_benchmark_reconstruction_contains_all_core_methods(
    synthetic_regression_frame: pd.DataFrame,
) -> None:
    results = benchmark_reconstruction(
        synthetic_regression_frame,
        target_column="target",
        feature_columns=["f1", "f2", "f3", "f4"],
        random_state=11,
        test_size=0.2,
    )

    assert CORE_METHODS.issubset(set(results["method"]))
    assert OPTIONAL_METHODS.issubset(set(results["method"]))
    assert set(results["status"]).issubset({"ok", "skipped"})


def test_benchmark_reconstruction_metrics_are_numeric_for_evaluated_methods(
    synthetic_regression_frame: pd.DataFrame,
) -> None:
    results = benchmark_reconstruction(
        synthetic_regression_frame,
        target_column="target",
        feature_columns=["f1", "f2", "f3", "f4"],
        random_state=11,
        test_size=0.2,
    )

    evaluated = results.loc[results["status"] == "ok", ["rmse", "mae", "r2", "mape"]]
    assert not evaluated.empty
    assert all(np.issubdtype(dtype, np.number) for dtype in evaluated.dtypes)
    assert np.isfinite(evaluated.to_numpy()).all()


def test_benchmark_reconstruction_skips_missing_optional_dependencies_gracefully(
    synthetic_regression_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = get_reconstruction_method_specs()
    specs["xgboost"] = ReconstructionMethodSpec(
        name="xgboost",
        family="ensemble",
        factory=lambda _rs: (_ for _ in ()).throw(OptionalDependencyUnavailable("xgboost missing")),
        optional_dependency="xgboost",
    )

    monkeypatch.setattr(
        "ai_aquatica_rs.models.reconstruction.get_reconstruction_method_specs",
        lambda: specs,
    )

    results = benchmark_reconstruction(
        synthetic_regression_frame,
        target_column="target",
        feature_columns=["f1", "f2", "f3", "f4"],
        methods=["median", "xgboost"],
        random_state=11,
        test_size=0.2,
    )

    xgboost_row = results.loc[results["method"] == "xgboost"].iloc[0]
    assert xgboost_row["status"] == "skipped"
    assert xgboost_row["skipped_reason"] == "xgboost missing"
    assert pd.isna(xgboost_row["rmse"])
