"""Assembly helpers for modeling datasets."""

from __future__ import annotations

import pandas as pd

from .temporal_alignment import DuplicatePolicy, align_exact, align_nearest


def assemble_modeling_table(
    insitu_df: pd.DataFrame,
    rs_df: pd.DataFrame,
    date_column: str,
    station_id_column: str,
    tolerance_days: int | None = None,
    *,
    duplicate_policy: DuplicatePolicy = "error",
) -> pd.DataFrame:
    """Build one modeling-ready DataFrame from in-situ and remote-sensing tables.

    Args:
        insitu_df: In-situ observations table.
        rs_df: Remote-sensing feature table.
        date_column: Shared datetime column name.
        station_id_column: Shared station/group identifier.
        tolerance_days: Optional nearest-match tolerance in days. If omitted,
            exact matching is used.
        duplicate_policy: Explicit duplicate handling strategy for right-side
            `(station_id, date)` collisions.

    Returns:
        A single left-aligned DataFrame ready for downstream modeling.
    """
    if tolerance_days is None:
        return align_exact(
            insitu_df,
            rs_df,
            on=date_column,
            by=station_id_column,
            duplicate_policy=duplicate_policy,
        )

    return align_nearest(
        insitu_df,
        rs_df,
        on=date_column,
        by=station_id_column,
        tolerance_days=tolerance_days,
        duplicate_policy=duplicate_policy,
    )
