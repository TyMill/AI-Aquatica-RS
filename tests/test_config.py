from pathlib import Path

import pytest

from ai_aquatica_rs.config import load_config
from ai_aquatica_rs.exceptions import ConfigError


def test_load_config_success(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: demo",
                "data:",
                "  dataset_path: sample.csv",
                "features:",
                "  feature_columns:",
                "    - a",
                "    - b",
                "runtime:",
                "  random_state: 7",
            ]
        )
    )

    config = load_config(config_path)

    assert config.project_name == "demo"
    assert config.dataset_path == "sample.csv"
    assert config.feature_columns == ["a", "b"]
    assert config.random_state == 7


def test_load_config_missing_required_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("data:\n  dataset_path: sample.csv\n")

    with pytest.raises(ConfigError, match="Missing required top-level sections"):
        load_config(config_path)
