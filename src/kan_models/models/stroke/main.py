"""Dataset-specific main entrypoint for stroke experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kan_models.common.runtime import configure_matplotlib

configure_matplotlib()

from kan_models.common.tabular.experiment import run_experiment


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "stroke" / "pruning.toml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the stroke KAN model.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the TOML config file. Default: {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args(argv)
    run_experiment(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
