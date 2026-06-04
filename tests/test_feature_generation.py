import pandas as pd
import pytest

from ai_aquatica_rs.features import add_calendar_features, add_lag_features, add_rolling_features


def make_feature_input() -> pd.DataFrame:
    """Create deterministic synthetic feature input with unsorted timestamps."""
    return pd.DataFrame(
        {
            "station_id": ["B", "A", "A", "B", "A"],
            "date": pd.to_datetime(
                ["2024-01-03", "2024-01-02", "2024-01-01", "2024-01-01", "2024-01-03"]
            ),
            "value": [10.0, 2.0, 1.0, 5.0, 3.0],
        }
    )


def test_add_lag_features_grouped_and_ordered_by_station_and_date() -> None:
    result = add_lag_features(
        make_feature_input(),
        group_column="station_id",
        value_columns=["value"],
        lags=[1, 2],
    )

    expected_order = [
        ("A", pd.Timestamp("2024-01-01")),
        ("A", pd.Timestamp("2024-01-02")),
        ("A", pd.Timestamp("2024-01-03")),
        ("B", pd.Timestamp("2024-01-01")),
        ("B", pd.Timestamp("2024-01-03")),
    ]
    assert list(result[["station_id", "date"]].itertuples(index=False, name=None)) == expected_order

    assert pd.isna(result.loc[0, "value_lag_1"])
    assert result.loc[1, "value_lag_1"] == 1.0
    assert result.loc[2, "value_lag_1"] == 2.0
    assert pd.isna(result.loc[3, "value_lag_1"])
    assert result.loc[4, "value_lag_1"] == 5.0

    assert pd.isna(result.loc[0, "value_lag_2"])
    assert pd.isna(result.loc[1, "value_lag_2"])
    assert result.loc[2, "value_lag_2"] == 1.0
    assert pd.isna(result.loc[3, "value_lag_2"])
    assert pd.isna(result.loc[4, "value_lag_2"])


def test_add_rolling_features_is_leakage_safe_and_computes_all_statistics() -> None:
    result = add_rolling_features(
        make_feature_input(),
        group_column="station_id",
        value_columns=["value"],
        windows=[2],
    )

    # For station A values [1, 2, 3], leakage-safe rolling uses prior rows only.
    station_a = result[result["station_id"] == "A"].reset_index(drop=True)
    assert pd.isna(station_a.loc[0, "value_roll_mean_2"])
    assert pd.isna(station_a.loc[1, "value_roll_mean_2"])
    assert station_a.loc[2, "value_roll_mean_2"] == 1.5

    assert station_a.loc[2, "value_roll_median_2"] == 1.5
    assert station_a.loc[2, "value_roll_min_2"] == 1.0
    assert station_a.loc[2, "value_roll_max_2"] == 2.0
    assert station_a.loc[2, "value_roll_std_2"] == pytest.approx(0.70710678)


def test_add_calendar_features_generates_seasonal_columns_and_week_number() -> None:
    source = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-03-31", "2024-10-10"])})

    result = add_calendar_features(source, include_week_number=True)

    assert result["month"].tolist() == [1, 3, 10]
    assert result["dayofyear"].tolist() == [1, 91, 284]
    assert result["quarter"].tolist() == [1, 1, 4]
    assert result["weekofyear"].tolist() == [1, 13, 41]


def test_add_lag_features_rejects_non_positive_lag_values() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        add_lag_features(
            make_feature_input(),
            group_column="station_id",
            value_columns=["value"],
            lags=[0],
        )
