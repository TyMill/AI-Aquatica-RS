"""Command-line interface."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import load_config
from .logging_utils import get_logger
from .pipelines.train_reconstruction import run_training_pipeline

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI parser."""
    parser = argparse.ArgumentParser(prog="ai-aquatica-rs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="Show package information.")

    validate_parser = subparsers.add_parser("validate-config", help="Validate a YAML config file.")
    validate_parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML configuration file.",
    )

    train_parser = subparsers.add_parser(
        "train-reconstruction", help="Run reconstruction pipeline."
    )
    train_parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML config.",
    )

    return parser


def _package_version() -> str:
    """Resolve installed package version."""
    try:
        return version("ai-aquatica-rs")
    except PackageNotFoundError:
        return "0+unknown"


def main() -> None:
    """Execute CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "info":
        print(f"AI-Aquatica-RS | version {_package_version()}")
        return

    if args.command == "validate-config":
        config = load_config(Path(args.config))
        print(config.model_dump_json(indent=2))
        return

    if args.command == "train-reconstruction":
        run_training_pipeline(Path(args.config))
        return


if __name__ == "__main__":
    main()
