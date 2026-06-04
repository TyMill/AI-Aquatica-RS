# DOCUMENTATION.md

## Project status
Phase: bootstrap

## 2026-03-17 — Initial repository starter

### Status
- repository structure created
- execution rules defined
- working Python package scaffold added

### Decisions
- package name: `ai_aquatica_rs`
- repository name: `AI-Aquatica-RS`
- MVP focuses on tabular fusion of in-situ observations and remote-sensing-derived features
- model benchmark is intentionally broader than a minimal baseline and includes linear, distance-based, tree-based, and boosting-based regressors

### Validation
- repository tree created
- local package files added
- tests written for key modules

### Next step
- upload repository to GitHub
- run tests locally
- iterate milestone-by-milestone with Codex

## 2026-03-17 — M9 documentation and release-readiness update

### Date
- 2026-03-17

### Milestone
- M9 — Documentation and release readiness

### Files modified
- `README.md`
- `DOCUMENTATION.md`

### Decisions made
- Added an explicit reproducible local workflow section to standardize setup and validation for local development.
- Added a contribution workflow section for agentic coding that points contributors to `AGENTS.md`, `PLANS.md`, `IMPLEMENT.md`, and `DOCUMENTATION.md`.
- Kept scope limited to milestone M9 documentation requirements only.

### Validation performed
- `python -m ai_aquatica_rs.cli info`
- `pytest -q`

### Next follow-up
- Add clearer human contributor guidance (branching and PR conventions) if collaboration scope expands beyond agentic workflows.

## 2026-03-17 — M1 core package bootstrap implementation

### Date
- 2026-03-17

### Milestone
- M1 — Core package bootstrap

### Files modified
- `src/ai_aquatica_rs/__init__.py`
- `src/ai_aquatica_rs/config.py`
- `src/ai_aquatica_rs/logging_utils.py`
- `src/ai_aquatica_rs/exceptions.py`
- `src/ai_aquatica_rs/cli.py`
- `configs/default.yaml`
- `README.md`
- `tests/test_config.py`
- `tests/test_cli.py`
- `DOCUMENTATION.md`

### Decisions made
- Standardized bootstrap configuration around required top-level YAML sections (`project`, `data`, `features`, `runtime`) with explicit validation errors.
- Kept `AppConfig` compatibility properties so existing pipeline code can continue reading flat-style attributes.
- Updated CLI validation flow to use `validate-config --config ...` while preserving `info` and `train-reconstruction` subcommands.
- Added deterministic CLI/config tests that execute through the public command line interface.

### Validation performed
- `python -c "import ai_aquatica_rs"`
- `python -m ai_aquatica_rs.cli info`
- `python -m ai_aquatica_rs.cli validate-config --config configs/default.yaml`
- `pytest -q tests/test_config.py tests/test_cli.py`

### Next follow-up
- Align downstream modules to consume nested config sections directly where beneficial.

## 2026-03-17 — M2 data layer (in-situ) implementation

### Date
- 2026-03-17

### Milestone
- M2 — Data layer (in-situ)

### Files modified
- `src/ai_aquatica_rs/data/__init__.py`
- `src/ai_aquatica_rs/data/io.py`
- `src/ai_aquatica_rs/data/loaders.py`
- `src/ai_aquatica_rs/data/validators.py`
- `src/ai_aquatica_rs/data/schemas.py`
- `tests/test_data_layer.py`
- `DOCUMENTATION.md`

### Decisions made
- Standardized ingestion through `load_table` with explicit support for `.csv` and `.parquet` only, and actionable format/path errors.
- Added schema-driven dataset loading using `EnvironmentalObservationSchema` to keep required-column validation practical and lightweight.
- Added explicit datetime parsing failures (row/value preview) instead of silent coercion.
- Added reusable utilities for duplicate removal and null summaries suitable for exploratory QA and pipeline checks.
- Exposed a clean public API from `ai_aquatica_rs.data` to avoid importing internal modules directly.

### Validation performed
- `python -c "import ai_aquatica_rs; from ai_aquatica_rs.data import load_environmental_dataset"`
- `pytest -q tests/test_data_layer.py`

### Next follow-up
- Integrate the new schema-driven data loader into training pipeline entry points when milestone scope reaches end-to-end orchestration.

## 2026-03-18 — M3 remote sensing indices implementation

### Date
- 2026-03-18

### Milestone
- M3 — Remote sensing indices

### Files modified
- `src/ai_aquatica_rs/remote_sensing/indices.py`
- `src/ai_aquatica_rs/remote_sensing/__init__.py`
- `tests/test_indices.py`
- `DOCUMENTATION.md`

### Decisions made
- Implemented pure NumPy normalized-difference index functions (`ndvi`, `ndwi`, `mndwi`, `ndti`) with explicit input validation and concise scientific docstrings.
- Added safe division behavior that returns `0.0` where denominators are zero to keep outputs finite and stable.
- Added shape validation to produce explicit errors for non-conformant input arrays.
- Added `compute_indices` helper for batch computation from a required band dictionary (`nir`, `red`, `green`, `swir`).
- Exported the remote sensing public API from `remote_sensing.__init__`.

### Validation performed
- `python -c "import ai_aquatica_rs; from ai_aquatica_rs.remote_sensing import ndvi, compute_indices"`
- `pytest -q tests/test_indices.py`

### Next follow-up
- Integrate index helper outputs into downstream fusion/feature steps in later milestones.

## 2026-03-18 — M4 data fusion implementation

### Date
- 2026-03-18

### Milestone
- M4 — Data fusion

### Files modified
- `src/ai_aquatica_rs/fusion/__init__.py`
- `src/ai_aquatica_rs/fusion/temporal_alignment.py`
- `src/ai_aquatica_rs/fusion/assemblers.py`
- `tests/test_temporal_alignment.py`
- `tests/test_fusion_assembler.py`
- `DOCUMENTATION.md`

### Decisions made
- Implemented exact and nearest temporal alignment by `(station_id, date)` using pandas merge operations only.
- Added explicit duplicate handling policy for right-side keys with deterministic strategies (`error`, `first`, `last`).
- Added row-level traceability fields (`left_row_id`, `matched_rs_row_id`, `matched_rs_date`, `match_type`, `time_delta_days`) to support auditability in fused tables.
- Kept geospatial extraction out of scope for this milestone per roadmap constraints.

### Assumptions
- Input date columns are parseable by `pandas.to_datetime`; invalid values raise explicit errors.
- Duplicate handling applies to the remote-sensing/right table keys used for matching.
- Nearest matching uses absolute nearest timestamp within `tolerance_days`; out-of-window rows remain unmatched (`NaN` in RS fields).

### Validation performed
- `python -c "import ai_aquatica_rs; from ai_aquatica_rs.fusion import align_exact, align_nearest, assemble_modeling_table"`
- `pytest -q tests/test_temporal_alignment.py tests/test_fusion_assembler.py`

### Next follow-up
- Integrate fused output into downstream feature engineering and pipeline milestones.

## 2026-03-18 — M5 feature engineering implementation

### Date
- 2026-03-18

### Milestone
- M5 — Feature engineering

### Files modified
- `src/ai_aquatica_rs/features/__init__.py`
- `src/ai_aquatica_rs/features/lag_features.py`
- `src/ai_aquatica_rs/features/rolling_features.py`
- `src/ai_aquatica_rs/features/rs_features.py`
- `tests/test_feature_generation.py`
- `DOCUMENTATION.md`

### Decisions made
- Implemented explicit, typed public APIs for lag, rolling, and calendar feature generation under `ai_aquatica_rs.features`.
- Enforced deterministic ordering assumptions by sorting feature computation on `(station_id, date)` before grouped operations.
- Made rolling features leakage-aware by default via `shift(1)` before rolling-window statistics.
- Expanded rolling statistics to include mean, median, standard deviation, minimum, and maximum.
- Standardized seasonal outputs to `month`, `dayofyear`, and `quarter`, with optional ISO `weekofyear`.
- Added explicit validation for required columns, numeric inputs, positive lags/windows, and parseable datetime values.

### Validation performed
- `pytest -q tests/test_feature_generation.py`
- `PYTHONPATH=src python -c "import ai_aquatica_rs; from ai_aquatica_rs.features import add_lag_features, add_rolling_features, add_calendar_features"`

### Next follow-up
- Integrate engineered features directly into the end-to-end training workflow during pipeline milestones.


## 2026-03-18 — M6 reconstruction models implementation

### Date
- 2026-03-18

### Milestone
- M6 — Reconstruction models

### Files modified
- `src/ai_aquatica_rs/models/__init__.py`
- `src/ai_aquatica_rs/models/base.py`
- `src/ai_aquatica_rs/models/reconstruction.py`
- `src/ai_aquatica_rs/models/evaluation.py`
- `src/ai_aquatica_rs/pipelines/train_reconstruction.py`
- `tests/test_reconstruction_models.py`
- `tests/test_cli.py`
- `DOCUMENTATION.md`

### Decisions made
- Replaced the initial minimal benchmark stub with a typed reconstruction benchmarking API that accepts a modeling DataFrame, explicit target column, and explicit feature columns.
- Implemented deterministic baselines, linear models, distance-based models, and tree/ensemble regressors under one method registry with stable `random_state` handling where supported.
- Added safe optional dependency loading for XGBoost and LightGBM so missing packages produce skipped benchmark rows rather than import failures.
- Expanded evaluation metrics to include RMSE, MAE, R2, and a safe MAPE formulation that remains finite near zero-valued targets.
- Preserved a split-based helper for downstream pipeline compatibility while centering the milestone API on a unified benchmark results table.

### Validation performed
- `pytest -q tests/test_reconstruction_models.py`
- `PYTHONPATH=src python -c "from ai_aquatica_rs.models import benchmark_reconstruction, list_reconstruction_methods; print(len(list_reconstruction_methods()))"`

### Next follow-up
- Wire the richer benchmark table into the training pipeline output format during the end-to-end workflow milestone.

## 2026-03-18 — M7 training pipeline implementation

### Date
- 2026-03-18

### Milestone
- M7 — Training pipeline

### Files modified
- `src/ai_aquatica_rs/pipelines/__init__.py`
- `src/ai_aquatica_rs/pipelines/train_reconstruction.py`
- `src/ai_aquatica_rs/cli.py`
- `tests/test_reconstruction_pipeline.py`
- `DOCUMENTATION.md`

### Decisions made
- Implemented a typed end-to-end reconstruction workflow that loads a prepared tabular dataset, validates the configured target/features, parses the configured date column, and drops incomplete modeling rows before training.
- Standardized the pipeline split as a deterministic shuffled train/validation split with a fixed validation fraction and the configured `random_state`.
- Saved pipeline artifacts as `reconstruction_metrics.csv` and `reconstruction_predictions.csv`, with predictions including station/date traceability plus observed validation targets and per-model outputs.
- Updated the CLI so `train-reconstruction` accepts `--config` consistently with `validate-config`, and prints a concise summary showing row counts, best model, and artifact paths.
- Added deterministic end-to-end tests that write a synthetic dataset/config to a temporary directory and validate both the direct pipeline API and the CLI entry point.

### Validation performed
- `pytest -q tests/test_reconstruction_pipeline.py tests/test_cli.py tests/test_config.py`
- `pytest -q tests/test_reconstruction_models.py tests/test_reconstruction_pipeline.py`

### Next follow-up
- Consider making the validation split fraction configurable once roadmap scope expands beyond the fixed milestone requirements.

## 2026-03-18 — M8 repository stabilization for GitHub and publication prep

### Date
- 2026-03-18

### Milestone
- M8 — Tests and stability

### Files modified
- `README.md`
- `pyproject.toml`
- `.gitignore`
- `src/ai_aquatica_rs/__init__.py`
- `data/example.csv`
- `DOCUMENTATION.md`
- removed tracked `__pycache__` artifacts under `src/` and `tests/`

### Decisions made
- Rewrote the README so install, CLI, benchmark, and expected-data guidance match the current codebase and bundled example assets.
- Added a small deterministic example dataset so `configs/default.yaml` supports an actual end-to-end benchmark run from the documented CLI command.
- Tightened `pyproject.toml` metadata and tool configuration for packaging, pytest discovery, and Ruff lint/format behavior while keeping dependencies minimal.
- Added a focused `.gitignore` and removed committed bytecode caches to keep the repository publishable and review-friendly.
- Added `__version__` to the top-level package exports so consumers can access package metadata through the public API cleanly.

### Validation performed
- full test suite
- editable install
- CLI help output
- CLI info/config validation/training commands
- import path and top-level version export

### Next follow-up
- Consider packaging example config/data assets explicitly if the project is published to PyPI as installable runtime examples.

## 2026-06-04 — Submission-readiness hardening

### Date
- 2026-06-04

### Milestone
- M8/M9 — Tests, stability, and release readiness

### Files modified
- `src/ai_aquatica_rs/models/reconstruction.py`
- `tests/test_data_layer.py`
- `tests/test_reconstruction_pipeline.py`
- `.github/workflows/ci.yml`
- `CITATION.cff`
- `CONTRIBUTING.md`
- `RELEASE_CHECKLIST.md`
- `README.md`
- `DOCUMENTATION.md`

### Decisions made
- Made Parquet-specific tests conditional on the availability of a Parquet engine so the CSV/data layer test suite remains usable in minimal environments.
- Reduced ensemble sizes and fixed single-threaded behavior for heavier benchmark estimators to improve reproducibility and avoid unnecessary CI slowdowns.
- Added GitHub Actions CI for Python 3.11 and 3.12.
- Added citation metadata and a release checklist for manuscript/Zenodo submission.
- Clarified that v0.1.0 focuses on prepared tabular workflows; direct raster extraction remains a future extension unless implemented before submission.

### Validation performed
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q`
- `PYTHONPATH=src python -m ai_aquatica_rs.cli validate-config --config configs/default.yaml`
- `PYTHONPATH=src python -m ai_aquatica_rs.cli train-reconstruction --config configs/default.yaml`

### Next follow-up
- Upload to GitHub, create a `v0.1.0` release, archive on Zenodo, and add the DOI to the manuscript before final submission.
