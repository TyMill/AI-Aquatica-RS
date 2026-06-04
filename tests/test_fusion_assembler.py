import pandas as pd

from ai_aquatica_rs.fusion.assemblers import assemble_modeling_table


def test_assemble_modeling_table_exact_columns() -> None:
    insitu_df = pd.DataFrame(
        {
            "station_id": ["A", "B"],
            "date": pd.to_datetime(["2024-02-01", "2024-02-02"]),
            "chlorophyll": [2.4, 3.1],
        }
    )
    rs_df = pd.DataFrame(
        {
            "station_id": ["A", "B"],
            "date": pd.to_datetime(["2024-02-01", "2024-02-03"]),
            "ndvi": [0.45, 0.52],
            "ndwi": [0.11, 0.09],
        }
    )

    assembled = assemble_modeling_table(
        insitu_df=insitu_df,
        rs_df=rs_df,
        date_column="date",
        station_id_column="station_id",
    )

    expected_columns = {
        "station_id",
        "date",
        "chlorophyll",
        "left_row_id",
        "matched_rs_row_id",
        "matched_rs_date",
        "ndvi",
        "ndwi",
        "match_type",
        "time_delta_days",
    }

    assert expected_columns.issubset(set(assembled.columns))
    assert assembled["match_type"].tolist() == ["exact", "exact"]
    assert assembled.loc[0, "ndvi"] == 0.45
    assert pd.isna(assembled.loc[1, "ndvi"])


def test_assemble_modeling_table_nearest_mode() -> None:
    insitu_df = pd.DataFrame(
        {
            "station_id": ["A"],
            "date": pd.to_datetime(["2024-02-03"]),
            "chlorophyll": [2.4],
        }
    )
    rs_df = pd.DataFrame(
        {
            "station_id": ["A"],
            "date": pd.to_datetime(["2024-02-01"]),
            "ndvi": [0.45],
        }
    )

    assembled = assemble_modeling_table(
        insitu_df=insitu_df,
        rs_df=rs_df,
        date_column="date",
        station_id_column="station_id",
        tolerance_days=2,
    )

    assert assembled.loc[0, "ndvi"] == 0.45
    assert assembled.loc[0, "match_type"] == "nearest"
    assert assembled.loc[0, "time_delta_days"] == -2.0
