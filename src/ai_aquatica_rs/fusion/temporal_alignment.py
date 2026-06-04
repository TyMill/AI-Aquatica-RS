"""Temporal alignment utilities for in-situ and remote-sensing tabular data."""

from __future__ import annotations

from typing import Literal

import pandas as pd

DuplicatePolicy = Literal["error", "first", "last"]


def _validate_required_columns(df: pd.DataFrame, *, required: list[str], label: str) -> None:
    """Validate that required columns are present in a DataFrame."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _coerce_datetime_column(df: pd.DataFrame, date_column: str, *, label: str) -> pd.DataFrame:
    """Return a copy of DataFrame with a validated datetime column."""
    out = df.copy()
    try:
        out[date_column] = pd.to_datetime(out[date_column], errors="raise")
    except Exception as exc:  # pragma: no cover - pandas exception details vary by version
        raise ValueError(f"{label}.{date_column} must contain parseable datetime values") from exc
    return out


def _resolve_right_duplicates(
    right: pd.DataFrame,
    *,
    by: str,
    on: str,
    duplicate_policy: DuplicatePolicy,
) -> pd.DataFrame:
    """Apply explicit duplicate handling to right-side match keys."""
    duplicate_mask = right.duplicated(subset=[by, on], keep=False)
    if not duplicate_mask.any():
        return right

    if duplicate_policy == "error":
        duplicate_rows = right.loc[duplicate_mask, [by, on]].drop_duplicates().head(5)
        raise ValueError(
            "Duplicate right-side matches found for (group, date) keys. "
            f"Examples: {duplicate_rows.to_dict(orient='records')}"
        )
    if duplicate_policy == "first":
        return right.drop_duplicates(subset=[by, on], keep="first").copy()
    if duplicate_policy == "last":
        return right.drop_duplicates(subset=[by, on], keep="last").copy()

    raise ValueError(f"Unsupported duplicate_policy: {duplicate_policy}")


def align_exact(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    by: str,
    *,
    duplicate_policy: DuplicatePolicy = "error",
) -> pd.DataFrame:
    """Align rows using exact station/date matching.

    The function preserves left-side row order and adds traceability fields:
    - `left_row_id`: original left row index before alignment
    - `matched_rs_row_id`: original right row index used for the match
    - `matched_rs_date`: right-side date used for matching
    - `match_type`: always ``"exact"`` for this function
    - `time_delta_days`: signed date difference between right and left rows
    """
    _validate_required_columns(left, required=[by, on], label="left")
    _validate_required_columns(right, required=[by, on], label="right")

    left_prepared = _coerce_datetime_column(left, on, label="left").copy()
    right_prepared = _coerce_datetime_column(right, on, label="right").copy()

    left_prepared["left_row_id"] = left_prepared.index
    right_prepared["matched_rs_row_id"] = right_prepared.index
    right_prepared = _resolve_right_duplicates(
        right_prepared,
        by=by,
        on=on,
        duplicate_policy=duplicate_policy,
    )

    right_prepared = right_prepared.rename(columns={on: "matched_rs_date"})
    merged = left_prepared.merge(
        right_prepared,
        left_on=[by, on],
        right_on=[by, "matched_rs_date"],
        how="left",
        suffixes=("", "_rs"),
        sort=False,
    )
    merged["match_type"] = "exact"
    merged["time_delta_days"] = (merged["matched_rs_date"] - merged[on]).dt.total_seconds() / 86_400
    return merged


def align_nearest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    by: str,
    tolerance_days: int,
    *,
    duplicate_policy: DuplicatePolicy = "error",
) -> pd.DataFrame:
    """Align rows to the nearest right-side date within a tolerance window.

    Args:
        left: In-situ table.
        right: Remote-sensing feature table.
        on: Datetime column name shared by both tables.
        by: Grouping key (for example `station_id`).
        tolerance_days: Maximum absolute distance allowed for nearest matching.
        duplicate_policy: How duplicate right-side `(by, on)` keys are handled.

    Returns:
        Left-aligned DataFrame with right-side columns and traceability metadata.
    """
    if tolerance_days < 0:
        raise ValueError("tolerance_days must be non-negative")

    _validate_required_columns(left, required=[by, on], label="left")
    _validate_required_columns(right, required=[by, on], label="right")

    left_prepared = _coerce_datetime_column(left, on, label="left").copy()
    right_prepared = _coerce_datetime_column(right, on, label="right").copy()

    left_prepared["left_row_id"] = left_prepared.index
    right_prepared["matched_rs_row_id"] = right_prepared.index
    right_prepared = _resolve_right_duplicates(
        right_prepared,
        by=by,
        on=on,
        duplicate_policy=duplicate_policy,
    )
    right_prepared = right_prepared.rename(columns={on: "matched_rs_date"})

    left_sorted = left_prepared.sort_values([on, by, "left_row_id"]).reset_index(drop=True)
    right_sorted = right_prepared.sort_values(
        ["matched_rs_date", by, "matched_rs_row_id"]
    ).reset_index(drop=True)

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=on,
        right_on="matched_rs_date",
        by=by,
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
        suffixes=("", "_rs"),
    )

    merged["match_type"] = "nearest"
    merged["time_delta_days"] = (merged["matched_rs_date"] - merged[on]).dt.total_seconds() / 86_400
    return merged.sort_values("left_row_id").reset_index(drop=True)
