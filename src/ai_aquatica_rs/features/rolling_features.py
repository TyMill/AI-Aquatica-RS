"""Rolling feature generation utilities."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ai_aquatica_rs.features.lag_features import DEFAULT_DATE_COLUMN, _validate_common_inputs


def add_rolling_features(
    df: pd.DataFrame,
    *,
    group_column: str,
    value_columns: Sequence[str],
    windows: Sequence[int],
    date_column: str = DEFAULT_DATE_COLUMN,
    leakage_safe: bool = True,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Create group-wise rolling statistics using deterministic chronological order.

    By default, rows are shifted by one time step before rolling statistics are
    computed so that only historical values are used (leakage-safe behavior).

    Args:
        df: Source table.
        group_column: Grouping key (for example ``station_id``).
        value_columns: Numeric columns used for rolling calculations.
        windows: Positive rolling window sizes.
        date_column: Datetime column used for chronological ordering.
        leakage_safe: Whether to exclude current-row values by applying a shift(1).
        min_periods: Minimum observations required in each window. Defaults to the
            window size when ``None``.

    Returns:
        A new DataFrame with rolling statistic columns added.
    """
    if not windows:
        raise ValueError("windows must contain at least one positive integer")
    if any(window <= 0 for window in windows):
        raise ValueError("All window sizes must be positive integers")

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
        series = grouped[column].shift(1) if leakage_safe else out[column]
        for window in windows:
            window_min_periods = window if min_periods is None else min_periods
            rolling = series.groupby(out[group_column], sort=False).rolling(
                window=window,
                min_periods=window_min_periods,
            )
            out[f"{column}_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            out[f"{column}_roll_median_{window}"] = rolling.median().reset_index(level=0, drop=True)
            out[f"{column}_roll_std_{window}"] = rolling.std().reset_index(level=0, drop=True)
            out[f"{column}_roll_min_{window}"] = rolling.min().reset_index(level=0, drop=True)
            out[f"{column}_roll_max_{window}"] = rolling.max().reset_index(level=0, drop=True)

    return out
