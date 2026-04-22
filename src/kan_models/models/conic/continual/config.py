"""Config helpers for the continual conic experiment."""

from __future__ import annotations

from pathlib import Path

from kan_models.models.conic.config import ContinualConfig, load_continual_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[5] / "configs" / "conic" / "continual.toml"
DEFAULT_REVERSED_CONFIG_PATH = Path(__file__).resolve().parents[5] / "configs" / "conic" / "continual_reversed.toml"

CONTINUAL_CONFIG_VARIANTS = {
    "standard": DEFAULT_CONFIG_PATH,
    "normal": DEFAULT_CONFIG_PATH,
    "reversed": DEFAULT_REVERSED_CONFIG_PATH,
}

__all__ = [
    "ContinualConfig",
    "CONTINUAL_CONFIG_VARIANTS",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_REVERSED_CONFIG_PATH",
    "load_continual_config",
]
