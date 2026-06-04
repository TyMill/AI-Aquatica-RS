# AI-Aquatica-RS

AI-Aquatica-RS is a scientific Python package for integrating remote-sensing features with in-situ aquatic observations. It is designed for reproducible environmental monitoring workflows that need:

- deterministic tabular data ingestion and validation,
- remote-sensing spectral index computation,
- temporal alignment between field and remote observations,
- feature engineering for time-aware modeling, and
- reconstruction benchmarks for missing environmental targets.

The repository is intentionally structured as a small, testable package that can evolve toward GitHub collaboration and future PyPI publication.

## Installation

### Local editable install

```bash
python -m pip install -e .
```

### Optional extras

Install development tooling:

```bash
python -m pip install -e .[dev]
```

Install optional gradient-boosting libraries used by the benchmark when available:

```bash
python -m pip install -e .[boosting]
```

## CLI usage

The package exposes both a module entry point and a console script.

Show package information:

```bash
python -m ai_aquatica_rs.cli info
```

Validate a YAML configuration file:

```bash
python -m ai_aquatica_rs.cli validate-config --config configs/default.yaml
python -m ai_aquatica_rs.cli train-reconstruction --config configs/default.yaml
```

Run the end-to-end reconstruction benchmark pipeline with the bundled example dataset:

```bash
python -m ai_aquatica_rs.cli train-reconstruction --config configs/default.yaml
```

After a successful training run, the pipeline writes:

- `reconstruction_metrics.csv`
- `reconstruction_predictions.csv`

into the configured `runtime.output_dir`.

You can also use the installed console script:

```bash
ai-aquatica-rs info
```

## Expected data format

The default configuration included in this repository points to `data/example.csv`. More generally, the training pipeline expects a prepared tabular dataset with:

- one row per observation,
- a station identifier column,
- a parseable date column,
- a numeric target column to reconstruct, and
- one or more numeric feature columns.

The exact required column names are controlled by the YAML config.

### Minimal CSV example

```csv
station_id,date,feature_a,feature_b,feature_c,target
station-1,2024-01-01,0.50,0.88,0.01,1.14
station-1,2024-01-02,0.67,0.78,-0.02,1.39
station-1,2024-01-03,0.84,0.66,0.03,1.71
```

### Matching configuration example

```yaml
project:
  name: AI-Aquatica-RS

data:
  dataset_path: data/example.csv
  date_column: date
  target_column: target
  station_id_column: station_id

features:
  feature_columns:
    - feature_a
    - feature_b
    - feature_c

runtime:
  random_state: 42
  output_dir: outputs
  nearest_days_tolerance: 3
  models:
    - median
    - mean
    - linear_regression
    - ridge
    - knn
    - random_forest
    - extra_trees
    - hist_gradient_boosting
```

## Modeling benchmark

The reconstruction benchmark currently supports deterministic baselines plus multiple regression families.

### Always-available methods

- `median`
- `mean`
- `linear_regression`
- `ridge`
- `lasso`
- `elastic_net`
- `knn`
- `decision_tree`
- `random_forest`
- `extra_trees`
- `gradient_boosting`
- `hist_gradient_boosting`
- `adaboost`

### Optional methods

These are only evaluated when their dependencies are installed. Otherwise, they are reported as skipped instead of failing the pipeline.

- `xgboost`
- `lightgbm`

### Reported metrics

Each benchmark row includes:

- RMSE
- MAE
- R²
- MAPE
- train/validation row counts
- evaluation status (`ok` or `skipped`)

## Package layout

- `src/ai_aquatica_rs/data/` — data loading, parsing, and validation
- `src/ai_aquatica_rs/remote_sensing/` — spectral index computation
- `src/ai_aquatica_rs/fusion/` — temporal alignment and dataset assembly
- `src/ai_aquatica_rs/features/` — lag, rolling, and calendar features
- `src/ai_aquatica_rs/models/` — reconstruction methods and metrics
- `src/ai_aquatica_rs/pipelines/` — executable workflows used by the CLI

## Reproducible local validation

```bash
python -m pip install -e .[dev]
pytest -q
python -m ai_aquatica_rs.cli --help
python -m ai_aquatica_rs.cli info
python -m ai_aquatica_rs.cli validate-config --config configs/default.yaml
python -m ai_aquatica_rs.cli train-reconstruction --config configs/default.yaml
```

For deterministic behavior on shared CI or HPC machines, limiting numerical backend threads is recommended:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Citation and archival release

Citation metadata is provided in `CITATION.cff`. For manuscript submission, create a tagged GitHub release and archive it on Zenodo so the paper can cite a persistent software DOI.

## Current scope

This repository focuses on prepared tabular modeling workflows that combine in-situ observations with remote-sensing-derived features. It includes a packaged example dataset in `data/example.csv`. Direct geospatial raster extraction is intentionally kept outside the v0.1.0 scope and should be described as a future extension unless implemented before submission.
