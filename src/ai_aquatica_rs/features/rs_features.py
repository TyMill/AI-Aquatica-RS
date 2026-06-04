"""Calendar feature helpers."""

from __future__ import annotations

import pandas as pd

DEFAULT_DATE_COLUMN = "date"


def add_calendar_features(
    df: pd.DataFrame,
    *,
    date_column: str = DEFAULT_DATE_COLUMN,
    include_week_number: bool = False,
) -> pd.DataFrame:
    """Add deterministic seasonal calendar features from a datetime column.

    Args:
        df: Source table.
        date_column: Datetime column used to derive calendar features.
        include_week_number: Whether to add ISO week number as ``weekofyear``.

    Returns:
        A new DataFrame with calendar columns.
    """
    if date_column not in df.columns:
        raise ValueError(f"Input DataFrame is missing required column: {date_column}")

    out = df.copy()
    try:
        dt = pd.to_datetime(out[date_column], errors="raise")
    except Exception as exc:  # pragma: no cover - pandas error text can vary
        raise ValueError(f"{date_column} must contain parseable datetime values") from exc

    out["month"] = dt.dt.month
    out["dayofyear"] = dt.dt.dayofyear
    out["quarter"] = dt.dt.quarter
    if include_week_number:
        out["weekofyear"] = dt.dt.isocalendar().week.astype("int64")
    return out
