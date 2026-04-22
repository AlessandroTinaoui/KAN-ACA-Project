"""Runtime helpers shared by command-line entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .paths import ARTIFACTS_DIR


def configure_matplotlib(headless: bool = True) -> None:
    """Configure matplotlib to work reliably in non-interactive runs."""
    mpl_config_dir = ARTIFACTS_DIR / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    if headless:
        matplotlib.use("Agg")


def detect_device(preferred: str | None = None) -> torch.device:
    """Resolve the device requested in the config."""
    if preferred is not None and preferred != "auto":
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
