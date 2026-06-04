"""High-level dataset loaders for environmental tabular data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import load_table
from .schemas import EnvironmentalObservationSchema
from .validators import drop_duplicates, parse_datetime_column, require_columns


def load_environmental_dataset(
    path: str | Path,
    schema: EnvironmentalObservationSchema,
    *,
    datetime_format: str | None = None,
    deduplicate_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Load and validate an environmental in-situ dataset.

    Args:
        path: Path to CSV or Parquet file.
        schema: Required tabular schema for validation.
        datetime_format: Optional format string for datetime parsing.
        deduplicate_subset: Optional columns used for duplicate detection. If
            omitted, full-row duplicate removal is applied.

    Returns:
        Cleaned and validated DataFrame.
    """
    df = load_table(path)
    require_columns(df, schema.required_columns)
    df = parse_datetime_column(df, schema.date_column, datetime_format=datetime_format)
    return drop_duplicates(df, subset=deduplicate_subset)
