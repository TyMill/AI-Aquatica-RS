"""File I/O utilities for tabular environmental datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_aquatica_rs.exceptions import DataValidationError


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a tabular file from disk.

    Supported formats are CSV (``.csv``) and Parquet (``.parquet``).

    Args:
        path: File path to a CSV or Parquet dataset.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        DataValidationError: If the path does not exist, format is unsupported,
            or the underlying pandas loader fails.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DataValidationError(f"Input file does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix == ".parquet":
            return pd.read_parquet(file_path)
    except Exception as exc:  # pragma: no cover - defensive passthrough
        raise DataValidationError(
            f"Failed to load '{file_path}' as {suffix or 'unknown'}: {exc}"
        ) from exc

    raise DataValidationError(
        f"Unsupported file format '{suffix}'. Supported formats are: .csv, .parquet"
    )
