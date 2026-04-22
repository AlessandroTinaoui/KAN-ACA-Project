"""Config helpers for the continual conic experiment."""

from __future__ import annotations

from pathlib import Path

from kan_models.models.conic.config import ContinualConfig, load_continual_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[5] / "configs" / "conic" / "continual.toml"
DEFAULT_REVERSED_CONFIG_PATH = Path(__file__).resolve().parents[5] / "configs" / "conic" / "continual_reversed.toml"

__all__ = [
    "ContinualConfig",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_REVERSED_CONFIG_PATH",
    "load_continual_config",
]
