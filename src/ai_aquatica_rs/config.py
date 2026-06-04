"""Configuration loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from .exceptions import ConfigError

_REQUIRED_TOP_LEVEL_SECTIONS: tuple[str, ...] = ("project", "data", "features", "runtime")


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str = "AI-Aquatica-RS"


class DataConfig(BaseModel):
    """Data source and schema configuration."""

    dataset_path: str
    date_column: str = "date"
    target_column: str = "target"
    station_id_column: str = "station_id"


class FeaturesConfig(BaseModel):
    """Feature selection configuration."""

    feature_columns: list[str] = Field(default_factory=list)


class RuntimeConfig(BaseModel):
    """Runtime options for pipelines and models."""

    random_state: int = 42
    output_dir: str = "outputs"
    nearest_days_tolerance: int = 3
    models: list[str] = Field(default_factory=lambda: ["median", "random_forest"])


class AppConfig(BaseModel):
    """Validated application configuration."""

    project: ProjectConfig
    data: DataConfig
    features: FeaturesConfig
    runtime: RuntimeConfig

    @property
    def project_name(self) -> str:
        """Return project display name."""
        return self.project.name

    @property
    def dataset_path(self) -> str:
        """Return configured dataset path."""
        return self.data.dataset_path

    @property
    def date_column(self) -> str:
        """Return configured date column."""
        return self.data.date_column

    @property
    def target_column(self) -> str:
        """Return configured target column."""
        return self.data.target_column

    @property
    def station_id_column(self) -> str:
        """Return configured station id column."""
        return self.data.station_id_column

    @property
    def feature_columns(self) -> list[str]:
        """Return configured feature column names."""
        return self.features.feature_columns

    @property
    def random_state(self) -> int:
        """Return runtime random seed."""
        return self.runtime.random_state

    @property
    def output_dir(self) -> str:
        """Return output directory for generated artifacts."""
        return self.runtime.output_dir

    @property
    def nearest_days_tolerance(self) -> int:
        """Return tolerance for nearest date alignment."""
        return self.runtime.nearest_days_tolerance

    @property
    def models(self) -> list[str]:
        """Return model names selected for benchmarking."""
        return self.runtime.models


def _validate_required_sections(raw_config: dict[str, Any]) -> None:
    """Validate required top-level sections are present and dictionary-like."""
    missing = [section for section in _REQUIRED_TOP_LEVEL_SECTIONS if section not in raw_config]
    if missing:
        raise ConfigError(f"Missing required top-level sections: {', '.join(missing)}")

    invalid = [
        section
        for section in _REQUIRED_TOP_LEVEL_SECTIONS
        if not isinstance(raw_config.get(section), dict)
    ]
    if invalid:
        raise ConfigError(
            f"Top-level sections must be mappings: {', '.join(invalid)}"
        )


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML: {exc}") from exc

    if raw is None:
        raise ConfigError("Configuration file is empty")
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")

    _validate_required_sections(raw)

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
