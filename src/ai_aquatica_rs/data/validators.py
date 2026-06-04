"""Validation and transformation helpers for tabular datasets."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ai_aquatica_rs.exceptions import DataValidationError



def require_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Ensure a DataFrame contains all required columns.

    Args:
        df: Input DataFrame.
        required_columns: Column names that must exist in ``df``.

    Raises:
        DataValidationError: If any required columns are missing.
    """
    required = list(required_columns)
    missing = sorted([column for column in required if column not in df.columns])
    if missing:
        available = list(df.columns)
        raise DataValidationError(
            "Missing required columns: "
            f"{missing}. Available columns: {available}."
        )


def parse_datetime_column(
    df: pd.DataFrame,
    column: str,
    *,
    datetime_format: str | None = None,
) -> pd.DataFrame:
    """Parse a datetime column with explicit error reporting.

    Args:
        df: Input DataFrame.
        column: Name of the column to parse.
        datetime_format: Optional explicit datetime format passed to
            ``pandas.to_datetime``.

    Returns:
        A copy of ``df`` with ``column`` converted to datetime.

    Raises:
        DataValidationError: If the column does not exist or any values fail to
            parse.
    """
    if column not in df.columns:
        raise DataValidationError(
            f"Datetime column '{column}' not found. Available columns: {list(df.columns)}."
        )

    parsed = pd.to_datetime(df[column], format=datetime_format, errors="coerce")
    invalid_mask = parsed.isna() & df[column].notna()
    if invalid_mask.any():
        invalid_rows = df.index[invalid_mask].tolist()
        invalid_values = df.loc[invalid_mask, column].astype(str).tolist()
        preview = invalid_values[:5]
        raise DataValidationError(
            f"Failed to parse datetime column '{column}'. "
            f"Invalid values at rows {invalid_rows[:5]} (showing up to 5): {preview}."
        )

    result = df.copy()
    result[column] = parsed
    return result


def drop_duplicates(
    df: pd.DataFrame,
    subset: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Drop duplicate rows and reset index.

    Args:
        df: Input DataFrame.
        subset: Optional subset of columns to consider for duplicate detection.

    Returns:
        DataFrame with duplicates removed.

    Raises:
        DataValidationError: If any subset columns are missing.
    """
    subset_columns = list(subset) if subset is not None else None
    if subset_columns is not None:
        require_columns(df, subset_columns)

    return df.drop_duplicates(subset=subset_columns).reset_index(drop=True)


def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values per column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns ``column``, ``missing_count``, and
        ``missing_fraction`` sorted by descending missing count.
    """
    total_rows = len(df)
    missing_count = df.isna().sum()
    if total_rows == 0:
        missing_fraction = pd.Series(0.0, index=df.columns)
    else:
        missing_fraction = missing_count / float(total_rows)

    summary = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_fraction": missing_fraction.values,
        }
    )
    return summary.sort_values(by=["missing_count", "column"], ascending=[False, True]).reset_index(
        drop=True
    )
