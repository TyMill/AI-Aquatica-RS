"""Lag feature generation utilities."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype

DEFAULT_DATE_COLUMN = "date"


def _validate_common_inputs(
    df: pd.DataFrame,
    *,
    group_column: str,
    date_column: str,
    value_columns: Sequence[str],
) -> None:
    """Validate required input columns and dtypes for feature generation."""
    required_columns = [group_column, date_column, *value_columns]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

    non_numeric_columns = [column for column in value_columns if not is_numeric_dtype(df[column])]
    if non_numeric_columns:
        raise ValueError(f"Value columns must be numeric for lag features: {non_numeric_columns}")


def add_lag_features(
    df: pd.DataFrame,
    *,
    group_column: str,
    value_columns: Sequence[str],
    lags: Sequence[int],
    date_column: str = DEFAULT_DATE_COLUMN,
) -> pd.DataFrame:
    """Create group-wise lag features ordered by time.

    Lags are generated using only prior observations within each group, preventing
    target leakage from future rows. Input rows are deterministically sorted by
    ``group_column`` then ``date_column`` before lagging.

    Args:
        df: Source table.
        group_column: Grouping key (for example ``station_id``).
        value_columns: Numeric columns to lag.
        lags: Positive lag steps to compute.
        date_column: Datetime column used for chronological ordering.

    Returns:
        A new DataFrame with lag feature columns added.
    """
    if not lags:
        raise ValueError("lags must contain at least one positive integer")
    if any(lag <= 0 for lag in lags):
        raise ValueError("All lag values must be positive integers")

    _validate_common_inputs(
        df,
        group_column=group_column,
        date_column=date_column,
        value_columns=value_columns,
    )

    out = df.copy()
    try:
        out[date_column] = pd.to_datetime(out[date_column], errors="raise")
    except Exception as exc:  # pragma: no cover - pandas error text can vary
        raise ValueError(f"{date_column} must contain parseable datetime values") from exc

    out = out.sort_values([group_column, date_column]).reset_index(drop=True)
    grouped = out.groupby(group_column, sort=False)
    for column in value_columns:
        for lag in lags:
            out[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

    return out
