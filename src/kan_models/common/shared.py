"""Generic helpers shared by multiple experiment families."""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from kan import KAN


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    """Resolve a potentially relative path against a base directory."""
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_toml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a TOML file and return both its resolved path and parsed content."""
    resolved_path = Path(path).resolve()
    with resolved_path.open("rb") as handle:
        return resolved_path, tomllib.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with a stable, human-readable format."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def clear_directory(path: Path) -> None:
    """Remove a directory tree if it exists."""
    if path.exists():
        shutil.rmtree(path)


def clear_matching_files(directory: Path, *patterns: str) -> None:
    """Remove matching files or folders inside a directory."""
    if not directory.exists():
        return

    for pattern in patterns:
        for candidate in directory.glob(pattern):
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()


def clone_state_dict(model: KAN) -> dict[str, Any]:
    """Clone a model state for later restoration."""
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def copy_kan_model(model: KAN) -> KAN:
    """Copy a KAN while temporarily clearing cached activations."""
    old_cache_data = model.cache_data
    old_acts = model.acts

    model.cache_data = None
    model.acts = None

    try:
        copied_model = model.copy()
    finally:
        model.cache_data = old_cache_data
        model.acts = old_acts

    copied_model.auto_save = False
    return copied_model


def serialize_width(width: list[Any]) -> list[Any]:
    """Convert pykan width metadata into JSON-friendly values."""
    serialized: list[Any] = []
    for layer in width:
        if isinstance(layer, (list, tuple)):
            serialized.append([int(value) for value in layer])
        else:
            serialized.append(int(layer))
    return serialized


def hidden_units(model: KAN) -> int:
    """Count the active hidden units inside a KAN."""
    total = 0
    for layer_width in model.width[1:-1]:
        if isinstance(layer_width, list):
            total += int(sum(layer_width))
        else:
            total += int(layer_width)
    return total


def json_number(value: object) -> float | int | bool | None:
    """Convert numpy-friendly values into plain JSON scalars."""
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    if float(number).is_integer():
        return int(number)
    return number
