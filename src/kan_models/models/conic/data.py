"""Shared data utilities for conic-section experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_feature_names(data_path: Path, target_column: str = "shape") -> list[str]:
    """Read numerical feature names from the CSV header."""
    columns = pd.read_csv(data_path, nrows=0).columns
    return [str(column) for column in columns if str(column) != target_column]


def load_conic_csv(
    data_path: Path,
    target_column: str = "shape",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the conic dataset and encode target labels as integers."""
    frame = pd.read_csv(data_path)
    features = frame.drop(columns=[target_column]).to_numpy(dtype=np.float32)
    labels, shape_names = pd.factorize(frame[target_column])
    return features, labels.astype(np.int64), list(shape_names)


def stratified_split(
    labels: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Split every class independently to keep a balanced test set."""
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    train_by_class: dict[int, np.ndarray] = {}
    test_by_class: dict[int, np.ndarray] = {}

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)

        n_test = max(1, int(round(len(class_indices) * test_ratio)))
        class_test = np.sort(class_indices[:n_test])
        class_train = np.sort(class_indices[n_test:])

        train_by_class[int(class_id)] = class_train
        test_by_class[int(class_id)] = class_test
        train_indices.append(class_train)
        test_indices.append(class_test)

    return (
        np.concatenate(train_indices),
        np.concatenate(test_indices),
        train_by_class,
        test_by_class,
    )


def standardization_stats(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std using training rows only."""
    mean = features[train_indices].mean(axis=0, keepdims=True)
    std = features[train_indices].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_from_train(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using only statistics from the training split."""
    mean, std = standardization_stats(features, train_indices)
    standardized = ((features - mean) / std).astype(np.float32)
    return standardized, mean, std


def make_kan_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create the dictionary format expected by pykan's `KAN.fit`."""
    return {
        "train_input": torch.tensor(features[train_indices], device=device),
        "train_label": torch.tensor(labels[train_indices], device=device),
        "test_input": torch.tensor(features[test_indices], device=device),
        "test_label": torch.tensor(labels[test_indices], device=device),
    }


def sample_indices(
    indices: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a reproducible subset without replacement."""
    if sample_size <= 0 or sample_size >= len(indices):
        return indices
    return np.sort(rng.choice(indices, size=sample_size, replace=False))


def count_by_class(indices: np.ndarray, labels: np.ndarray, shape_names: list[str]) -> dict[str, int]:
    """Count examples per class name for a group of indices."""
    return {
        shape_names[class_id]: int(np.sum(labels[indices] == class_id))
        for class_id in range(len(shape_names))
    }


def count_by_class_id(
    indices: np.ndarray,
    labels: np.ndarray,
    class_ids: list[int],
) -> dict[int, int]:
    """Count examples per numeric class id for a group of indices."""
    return {
        class_id: int(np.sum(labels[indices] == class_id))
        for class_id in class_ids
    }
