"""Main entrypoint for the continual conic model."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from kan_models.common.runtime import configure_matplotlib

configure_matplotlib()

from kan_models.models.conic.continual.config import DEFAULT_CONFIG_PATH
from kan_models.models.conic.continual.experiment import main, run_continual

__all__ = ["DEFAULT_CONFIG_PATH", "main", "run_continual"]


if __name__ == "__main__":
    raise SystemExit(main())
