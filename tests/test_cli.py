from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [str(SRC_PATH)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def test_cli_info_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ai_aquatica_rs.cli", "info"],
        check=True,
        capture_output=True,
        text=True,
        env=_cli_env(),
        cwd=REPO_ROOT,
    )
    assert "AI-Aquatica-RS" in result.stdout


def test_cli_validate_config_command() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_aquatica_rs.cli",
            "validate-config",
            "--config",
            "configs/default.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_cli_env(),
        cwd=REPO_ROOT,
    )
    assert '"project"' in result.stdout
    assert '"runtime"' in result.stdout
