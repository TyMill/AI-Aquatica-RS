# AI-Aquatica-RS article results package

These results were generated with `experiments/generate_article_results.py` using a deterministic synthetic multi-station aquatic-monitoring dataset (`RANDOM_STATE = 42`). The dataset is intended as a controlled demonstration benchmark for the software paper, not as a substitute for a real satellite/field validation dataset.

## Table 1. Temporal alignment coverage

|   tolerance_days |   insitu_rows |   matched_rows |   coverage_percent |   mean_abs_time_delta_days |
|-----------------:|--------------:|---------------:|-------------------:|---------------------------:|
|                0 |          1080 |            181 |             16.759 |                      0     |
|                1 |          1080 |            538 |             49.815 |                      0.664 |
|                3 |          1080 |            954 |             88.333 |                      1.308 |
|                5 |          1080 |           1045 |             96.759 |                      1.573 |
|                7 |          1080 |           1061 |             98.241 |                      1.648 |

## Table 2. Spectral index summary

| index   |   count |    mean |    std |     min |     25% |     50% |     75% |    max |
|:--------|--------:|--------:|-------:|--------:|--------:|--------:|--------:|-------:|
| ndvi    |     181 |  0.2331 | 0.1083 | -0.0564 |  0.1628 |  0.2462 |  0.3213 | 0.4506 |
| ndwi    |     181 | -0.0374 | 0.0935 | -0.2536 | -0.1065 | -0.0466 |  0.0318 | 0.2048 |
| mndwi   |     181 |  0.5839 | 0.0952 |  0.3147 |  0.5256 |  0.5861 |  0.652  | 0.8097 |
| ndti    |     181 | -0.1977 | 0.1074 | -0.4484 | -0.2804 | -0.2007 | -0.1315 | 0.0953 |

## Table 3. Best model per feature configuration

| feature_set      | method            | family   |   n_train |   n_valid |   rmse |    mae |     r2 |    mape |
|:-----------------|:------------------|:---------|----------:|----------:|-------:|-------:|-------:|--------:|
| fused_in_situ_rs | gradient_boosting | ensemble |       645 |       309 | 4.9059 | 3.9658 | 0.6923 | 24.2582 |
| in_situ_only     | ridge             | linear   |       726 |       354 | 5.3425 | 4.5218 | 0.6599 | 25.0236 |
| rs_indices_only  | ridge             | linear   |       645 |       309 | 5.6158 | 4.2899 | 0.5968 | 25.7609 |

## Main quantitative finding

Using the best model from each configuration, the fused in-situ + remote-sensing feature set achieved RMSE = 4.906 NTU. This corresponds to a 8.2% RMSE reduction compared with the in-situ-only configuration (RMSE = 5.342 NTU) and a 12.6% RMSE reduction compared with the remote-sensing-index-only configuration (RMSE = 5.616 NTU).

## Suggested manuscript figures

- Figure 1: `figure_1_rmse_by_feature_set.png` — best RMSE by feature set.
- Figure 2: `figure_2_observed_vs_predicted_fused.png` — observed vs predicted turbidity for the fused configuration.
- Figure 3: `figure_3_alignment_coverage.png` — temporal alignment coverage as a function of nearest-match tolerance.

## Suggested wording for limitations

Because the current experiment uses deterministic synthetic observations, the reported metrics should be interpreted as a reproducibility and integration benchmark rather than as evidence of environmental generalization. The software paper should state that real-scene validation with Sentinel/Landsat imagery and independent in-situ monitoring records is the next validation step.
