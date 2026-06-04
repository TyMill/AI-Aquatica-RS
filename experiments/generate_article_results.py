"""Generate deterministic demonstration results for the AI-Aquatica-RS manuscript.

The experiment uses a controlled synthetic aquatic-monitoring dataset. It is not
intended to replace validation on a real satellite/field dataset, but it provides
reproducible evidence that the package components work together: spectral-index
calculation, temporal alignment, feature engineering, and reconstruction benchmarking.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_aquatica_rs.features.rs_features import add_calendar_features
from ai_aquatica_rs.fusion.assemblers import assemble_modeling_table
from ai_aquatica_rs.models.reconstruction import benchmark_reconstruction_split
from ai_aquatica_rs.remote_sensing.indices import compute_indices

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "article_results"
DATA_OUT = ROOT / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)


def generate_synthetic_observations() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    stations = [f"station-{i}" for i in range(1, 7)]
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    station_effect = dict(zip(stations, np.linspace(-2.5, 2.5, len(stations))))

    insitu_rows = []
    rs_rows = []
    for s_idx, station in enumerate(stations):
        phase = s_idx * 0.35
        for day_idx, date in enumerate(dates):
            doy = date.dayofyear
            seasonal = np.sin(2 * np.pi * doy / 365 + phase)
            storm = rng.gamma(shape=1.2, scale=3.0)
            storm = storm if rng.random() < 0.34 else 0.0
            rainfall_lag_proxy = 0.55 * storm + rng.normal(0, 0.6)
            water_temp = 12.0 + 9.0 * seasonal + rng.normal(0, 0.9)
            conductivity = 430 + 35 * s_idx + 18 * np.cos(2 * np.pi * doy / 120) + rng.normal(0, 7)
            ph = 7.55 + 0.18 * seasonal + rng.normal(0, 0.04)

            # Latent water optical state. Remote-sensing bands are generated from it.
            suspended_signal = 0.16 + 0.015 * station_effect[station] + 0.018 * rainfall_lag_proxy + 0.025 * np.cos(2 * np.pi * doy / 70) + rng.normal(0, 0.01)
            vegetation_signal = 0.35 + 0.08 * seasonal + rng.normal(0, 0.015)
            water_signal = 0.30 - 0.04 * seasonal - 0.012 * station_effect[station] + rng.normal(0, 0.015)

            red = np.clip(0.10 + suspended_signal + rng.normal(0, 0.006), 0.02, 0.90)
            green = np.clip(0.18 + water_signal - 0.35 * suspended_signal + rng.normal(0, 0.006), 0.02, 0.90)
            nir = np.clip(0.14 + vegetation_signal - 0.28 * water_signal + rng.normal(0, 0.006), 0.02, 0.90)
            swir = np.clip(0.08 + 0.45 * suspended_signal - 0.18 * water_signal + rng.normal(0, 0.006), 0.02, 0.90)

            indices = compute_indices({"red": np.array([red]), "green": np.array([green]), "nir": np.array([nir]), "swir": np.array([swir])})
            ndti = float(indices["ndti"][0])
            mndwi = float(indices["mndwi"][0])

            turbidity = (
                45.0
                + 70.0 * ndti
                - 14.0 * mndwi
                + 0.40 * rainfall_lag_proxy
                + 0.006 * conductivity
                + station_effect[station]
                + 1.2 * seasonal
                + rng.normal(0, 1.10)
            )

            insitu_rows.append(
                {
                    "station_id": station,
                    "date": date,
                    "water_temperature_c": water_temp,
                    "conductivity_us_cm": conductivity,
                    "ph": ph,
                    "rainfall_24h_mm": storm,
                    "turbidity_ntu": turbidity,
                }
            )

            # Simulate revisit interval and missing satellite observations due to cloud cover.
            if day_idx % 5 == 0 and rng.random() > 0.18:
                rs_rows.append(
                    {
                        "station_id": station,
                        "date": date,
                        "red": red,
                        "green": green,
                        "nir": nir,
                        "swir": swir,
                    }
                )

    return pd.DataFrame(insitu_rows), pd.DataFrame(rs_rows)


def add_remote_sensing_indices(rs: pd.DataFrame) -> pd.DataFrame:
    out = rs.copy()
    indices = compute_indices({
        "red": out["red"].to_numpy(),
        "green": out["green"].to_numpy(),
        "nir": out["nir"].to_numpy(),
        "swir": out["swir"].to_numpy(),
    })
    for name, values in indices.items():
        out[name] = values
    return out


def temporal_split(df: pd.DataFrame, feature_cols: list[str], target: str):
    complete = df.dropna(subset=feature_cols + [target]).copy()
    cutoff = pd.Timestamp("2024-05-01")
    train = complete[complete["date"] < cutoff].copy()
    valid = complete[complete["date"] >= cutoff].copy()
    if len(valid) < 20:
        # Fallback for very restrictive feature sets.
        train = complete.sample(frac=0.75, random_state=RANDOM_STATE)
        valid = complete.drop(train.index)
    return train, valid


def evaluate_feature_sets(modeling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    modeling = modeling.copy()
    modeling["abs_time_delta_days"] = modeling["time_delta_days"].abs()
    modeling = add_calendar_features(modeling, date_column="date")
    modeling["sin_dayofyear"] = np.sin(2 * np.pi * modeling["dayofyear"] / 365.25)
    modeling["cos_dayofyear"] = np.cos(2 * np.pi * modeling["dayofyear"] / 365.25)
    station_dummies = pd.get_dummies(modeling["station_id"], prefix="station", dtype=float)
    modeling = pd.concat([modeling, station_dummies], axis=1)
    station_cols = station_dummies.columns.tolist()

    methods = [
        "median", "mean", "linear_regression", "ridge", "knn",
        "decision_tree", "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting"
    ]
    feature_sets = {
        "in_situ_only": [
            "water_temperature_c", "conductivity_us_cm", "ph", "rainfall_24h_mm",
            "sin_dayofyear", "cos_dayofyear", *station_cols,
        ],
        "rs_indices_only": ["ndvi", "ndwi", "mndwi", "ndti", "abs_time_delta_days", *station_cols],
        "fused_in_situ_rs": [
            "water_temperature_c", "conductivity_us_cm", "ph", "rainfall_24h_mm",
            "sin_dayofyear", "cos_dayofyear", "ndvi", "ndwi", "mndwi", "ndti", "abs_time_delta_days", *station_cols,
        ],
    }

    all_metrics = []
    all_predictions = []
    for name, features in feature_sets.items():
        train, valid = temporal_split(modeling, features, "turbidity_ntu")
        metrics, preds = benchmark_reconstruction_split(
            train[features], train["turbidity_ntu"], valid[features], valid["turbidity_ntu"],
            methods=methods, random_state=RANDOM_STATE,
        )
        metrics.insert(0, "feature_set", name)
        all_metrics.append(metrics)
        best_method = metrics.query("status == 'ok'").sort_values("rmse").iloc[0]["method"]
        pred_out = valid[["station_id", "date", "turbidity_ntu"]].reset_index(drop=True).copy()
        pred_out["feature_set"] = name
        pred_out["best_method"] = best_method
        pred_out["prediction"] = preds[best_method].to_numpy()
        all_predictions.append(pred_out)

    return pd.concat(all_metrics, ignore_index=True), pd.concat(all_predictions, ignore_index=True)


def make_figures(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    best = metrics.query("status == 'ok'").sort_values("rmse").groupby("feature_set", as_index=False).first()
    best.to_csv(OUT / "best_models_by_feature_set.csv", index=False)

    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.gca()
    ax.bar(best["feature_set"], best["rmse"])
    ax.set_ylabel("RMSE [NTU]")
    ax.set_xlabel("Feature set")
    ax.set_title("Best reconstruction error by feature set")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / "figure_1_rmse_by_feature_set.png", dpi=300)
    plt.close(fig)

    fused = predictions[predictions["feature_set"] == "fused_in_situ_rs"].copy()
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.gca()
    ax.scatter(fused["turbidity_ntu"], fused["prediction"], alpha=0.7)
    mn = float(min(fused["turbidity_ntu"].min(), fused["prediction"].min()))
    mx = float(max(fused["turbidity_ntu"].max(), fused["prediction"].max()))
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlabel("Observed turbidity [NTU]")
    ax.set_ylabel("Predicted turbidity [NTU]")
    ax.set_title("Observed vs predicted values for the fused configuration")
    fig.tight_layout()
    fig.savefig(OUT / "figure_2_observed_vs_predicted_fused.png", dpi=300)
    plt.close(fig)


def main() -> None:
    insitu, rs = generate_synthetic_observations()
    rs = add_remote_sensing_indices(rs)
    insitu.to_csv(DATA_OUT / "synthetic_insitu_observations.csv", index=False)
    rs.to_csv(DATA_OUT / "synthetic_remote_sensing_observations.csv", index=False)

    coverage_rows = []
    aligned_by_tol = {}
    for tolerance in [0, 1, 3, 5, 7]:
        aligned = assemble_modeling_table(
            insitu, rs, date_column="date", station_id_column="station_id", tolerance_days=tolerance
        )
        aligned_by_tol[tolerance] = aligned
        coverage_rows.append(
            {
                "tolerance_days": tolerance,
                "insitu_rows": len(insitu),
                "matched_rows": int(aligned["matched_rs_row_id"].notna().sum()),
                "coverage_percent": float(aligned["matched_rs_row_id"].notna().mean() * 100),
                "mean_abs_time_delta_days": float(aligned["time_delta_days"].abs().mean()),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT / "temporal_alignment_coverage.csv", index=False)

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.gca()
    ax.plot(coverage["tolerance_days"], coverage["coverage_percent"], marker="o")
    ax.set_xlabel("Nearest-match tolerance [days]")
    ax.set_ylabel("Matched in-situ observations [%]")
    ax.set_title("Temporal-alignment coverage under simulated satellite revisit gaps")
    fig.tight_layout()
    fig.savefig(OUT / "figure_3_alignment_coverage.png", dpi=300)
    plt.close(fig)

    modeling = aligned_by_tol[3]
    modeling.to_csv(DATA_OUT / "synthetic_modeling_table_tolerance_3d.csv", index=False)
    rs[["ndvi", "ndwi", "mndwi", "ndti"]].describe().T.to_csv(OUT / "spectral_indices_summary.csv")

    metrics, predictions = evaluate_feature_sets(modeling)
    metrics.to_csv(OUT / "metrics_by_feature_set.csv", index=False)
    predictions.to_csv(OUT / "predictions_best_models.csv", index=False)
    make_figures(metrics, predictions)

    print("Generated article-ready demonstration results in", OUT)
    print(pd.read_csv(OUT / "best_models_by_feature_set.csv").to_string(index=False))


if __name__ == "__main__":
    main()
