"""Main entrypoint for the default conic model."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kan_models.common.runtime import configure_matplotlib

configure_matplotlib()

from kan_models.models.conic.baseline import DEFAULT_CONFIG_PATH, main, run_baseline

__all__ = ["DEFAULT_CONFIG_PATH", "main", "run_baseline"]


if __name__ == "__main__":
    raise SystemExit(main())
