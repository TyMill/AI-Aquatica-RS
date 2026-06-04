from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai_aquatica_rs.data import (
    EnvironmentalObservationSchema,
    drop_duplicates,
    load_environmental_dataset,
    null_summary,
    parse_datetime_column,
)
from ai_aquatica_rs.exceptions import DataValidationError


def _make_environmental_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["S1", "S1", "S1", "S2"],
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"],
            "latitude": [10.0, 10.0, 10.0, 11.0],
            "longitude": [20.0, 20.0, 20.0, 21.0],
            "chlorophyll_a": [3.2, 3.2, None, 5.0],
        }
    )


def test_load_environmental_dataset_from_csv(tmp_path: Path) -> None:
    df = _make_environmental_df()
    schema = EnvironmentalObservationSchema(target_columns=["chlorophyll_a"])

    csv_path = tmp_path / "observations.csv"
    df.to_csv(csv_path, index=False)

    csv_loaded = load_environmental_dataset(csv_path, schema)

    assert len(csv_loaded) == 3
    assert pd.api.types.is_datetime64_any_dtype(csv_loaded["date"])


def test_load_environmental_dataset_from_parquet_when_engine_available(tmp_path: Path) -> None:
    if pytest.importorskip("pyarrow", reason="Parquet test requires pyarrow") is None:
        return

    df = _make_environmental_df()
    schema = EnvironmentalObservationSchema(target_columns=["chlorophyll_a"])

    parquet_path = tmp_path / "observations.parquet"
    df.to_parquet(parquet_path, index=False)

    parquet_loaded = load_environmental_dataset(parquet_path, schema)

    assert len(parquet_loaded) == 3
    assert pd.api.types.is_datetime64_any_dtype(parquet_loaded["date"])


def test_missing_required_columns_raises_explicit_error(tmp_path: Path) -> None:
    invalid_df = pd.DataFrame(
        {
            "station_id": ["S1"],
            "date": ["2024-01-01"],
            "latitude": [10.0],
            "chlorophyll_a": [1.2],
        }
    )
    csv_path = tmp_path / "missing_cols.csv"
    invalid_df.to_csv(csv_path, index=False)

    schema = EnvironmentalObservationSchema(target_columns=["chlorophyll_a"])

    with pytest.raises(DataValidationError, match="Missing required columns"):
        load_environmental_dataset(csv_path, schema)


def test_drop_duplicates_with_subset() -> None:
    df = _make_environmental_df()
    deduplicated = drop_duplicates(df, subset=["station_id", "date"])

    assert len(deduplicated) == 3
    assert deduplicated["date"].tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_datetime_parsing_raises_on_invalid_values() -> None:
    df = pd.DataFrame({"date": ["2024-01-01", "not-a-date"], "value": [1, 2]})

    with pytest.raises(DataValidationError, match="Failed to parse datetime column 'date'"):
        parse_datetime_column(df, "date")


def test_null_summary_output_shape_and_values() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["S1", None, "S3"],
            "chlorophyll_a": [1.0, None, None],
            "temperature_c": [None, 12.0, 13.5],
        }
    )

    summary = null_summary(df)
    expected_columns = ["column", "missing_count", "missing_fraction"]

    assert summary.columns.tolist() == expected_columns
    chlorophyll_row = summary.loc[summary["column"] == "chlorophyll_a"].iloc[0]
    assert chlorophyll_row["missing_count"] == 2
    assert chlorophyll_row["missing_fraction"] == pytest.approx(2 / 3)
