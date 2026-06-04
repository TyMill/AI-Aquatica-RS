import pandas as pd
import pytest

from ai_aquatica_rs.fusion.temporal_alignment import align_exact, align_nearest


def test_align_exact_matches_by_station_and_date() -> None:
    left = pd.DataFrame(
        {
            "station_id": ["A", "A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-02"]),
            "target": [1.0, 2.0, 3.0],
        }
    )
    right = pd.DataFrame(
        {
            "station_id": ["A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "ndwi": [0.1, 0.2],
        }
    )

    result = align_exact(left, right, on="date", by="station_id")

    assert result.loc[0, "ndwi"] == 0.1
    assert pd.isna(result.loc[1, "ndwi"])
    assert pd.isna(result.loc[2, "ndwi"])
    assert result.loc[0, "matched_rs_date"] == pd.Timestamp("2024-01-01")
    assert pd.isna(result.loc[1, "matched_rs_date"])


def test_align_nearest_matches_within_tolerance() -> None:
    left = pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "date": pd.to_datetime(["2024-01-03", "2024-01-10"]),
            "target": [1.0, 2.0],
        }
    )
    right = pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "rs_value": [10.0, 20.0],
        }
    )

    result = align_nearest(left, right, on="date", by="station_id", tolerance_days=3)

    assert result["rs_value"].tolist() == [10.0, 20.0]
    assert result["time_delta_days"].tolist() == [-2.0, -2.0]


def test_align_nearest_rejects_matches_outside_tolerance() -> None:
    left = pd.DataFrame(
        {
            "station_id": ["A"],
            "date": pd.to_datetime(["2024-01-10"]),
            "target": [1.0],
        }
    )
    right = pd.DataFrame(
        {
            "station_id": ["A"],
            "date": pd.to_datetime(["2024-01-01"]),
            "rs_value": [10.0],
        }
    )

    result = align_nearest(left, right, on="date", by="station_id", tolerance_days=3)

    assert pd.isna(result.loc[0, "rs_value"])
    assert pd.isna(result.loc[0, "matched_rs_date"])


def test_align_exact_duplicate_handling_is_explicit_and_deterministic() -> None:
    left = pd.DataFrame(
        {
            "station_id": ["A"],
            "date": pd.to_datetime(["2024-01-01"]),
            "target": [1.0],
        }
    )
    right = pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "rs_value": [10.0, 99.0],
        }
    )

    with pytest.raises(ValueError, match="Duplicate right-side matches"):
        align_exact(left, right, on="date", by="station_id", duplicate_policy="error")

    first_result = align_exact(left, right, on="date", by="station_id", duplicate_policy="first")
    last_result = align_exact(left, right, on="date", by="station_id", duplicate_policy="last")

    assert first_result.loc[0, "rs_value"] == 10.0
    assert last_result.loc[0, "rs_value"] == 99.0


def test_align_nearest_handles_interleaved_station_dates() -> None:
    left = pd.DataFrame(
        {
            "station_id": ["A", "B", "A", "B"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-09", "2024-01-09"]),
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    right = pd.DataFrame(
        {
            "station_id": ["A", "B", "A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-10", "2024-01-08"]),
            "rs_value": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = align_nearest(left, right, on="date", by="station_id", tolerance_days=2)

    assert result["rs_value"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert result["left_row_id"].tolist() == [0, 1, 2, 3]
