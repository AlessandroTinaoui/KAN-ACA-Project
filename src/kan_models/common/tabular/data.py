"""Data loading and preprocessing utilities for tabular experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kan_models.common.tabular.config import DataConfig, PreprocessingConfig, SplitConfig


@dataclass
class TaskSpec:
    label_names: list[str]
    is_binary: bool

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    @property
    def output_dim(self) -> int:
        return 1 if self.is_binary else self.num_classes


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_dataframe(config: DataConfig) -> pd.DataFrame:
    """Load the raw dataset according to the TOML settings."""
    frame = pd.read_csv(config.csv_path, header=config.header_row, sep=config.separator)
    frame.columns = [str(column).strip() for column in frame.columns]

    if config.drop_unnamed_columns:
        unnamed = [column for column in frame.columns if column.startswith("Unnamed:")]
        if unnamed:
            frame = frame.drop(columns=unnamed)
    return frame


def encode_target(target: pd.Series) -> tuple[np.ndarray, TaskSpec]:
    """Encode target labels as consecutive integers."""
    label_names = sorted(target.astype(str).unique().tolist())
    label_to_index = {label: index for index, label in enumerate(label_names)}
    encoded = target.astype(str).map(label_to_index).to_numpy(dtype=np.int64)
    return encoded, TaskSpec(label_names=label_names, is_binary=len(label_names) == 2)


def resolve_feature_groups(frame: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Split features into numeric and categorical groups."""
    missing = [column for column in [config.target_column, *config.drop_columns] if column not in frame.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {missing}")

    if config.feature_columns:
        feature_columns = config.feature_columns
    else:
        excluded = {config.target_column, *config.drop_columns}
        feature_columns = [column for column in frame.columns if column not in excluded]

    X = frame[feature_columns].copy()
    categorical_columns = [column for column in config.categorical_columns if column in X.columns]
    if config.numeric_columns:
        numeric_columns = [column for column in config.numeric_columns if column in X.columns]
    else:
        numeric_columns = [column for column in X.columns if column not in categorical_columns]

    overlap = set(categorical_columns).intersection(numeric_columns)
    if overlap:
        raise ValueError(f"Colonne presenti sia tra numeriche che categoriche: {sorted(overlap)}")

    return X, numeric_columns, categorical_columns


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    config: PreprocessingConfig,
) -> ColumnTransformer:
    """Build the sklearn preprocessing pipeline from the config."""
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = []
        if config.impute_numeric:
            numeric_steps.append(("imputer", SimpleImputer(strategy=config.numeric_imputation_strategy)))
        if config.standardize_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_transformer: Any = Pipeline(steps=numeric_steps) if numeric_steps else "passthrough"
        transformers.append(("numeric", numeric_transformer, numeric_columns))

    if categorical_columns:
        categorical_steps: list[tuple[str, Any]] = []
        if config.impute_categorical:
            imputer_kwargs: dict[str, Any] = {"strategy": config.categorical_imputation_strategy}
            if config.categorical_imputation_strategy == "constant":
                imputer_kwargs["fill_value"] = config.categorical_fill_value
            categorical_steps.append(("imputer", SimpleImputer(**imputer_kwargs)))
        if config.one_hot_encode_categorical:
            categorical_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)))

        categorical_transformer: Any = Pipeline(steps=categorical_steps) if categorical_steps else "passthrough"
        transformers.append(("categorical", categorical_transformer, categorical_columns))

    if not transformers:
        raise ValueError("Nessuna feature disponibile dopo la selezione delle colonne.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_dataframe(X: pd.DataFrame, y: np.ndarray, config: SplitConfig) -> DatasetSplit:
    """Create train/validation/test splits."""
    stratify_labels = y if config.stratify else None
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=stratify_labels,
    )

    val_fraction_over_remaining = config.val_size / (config.train_size + config.val_size)
    stratify_train = y_train_full if config.stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_fraction_over_remaining,
        random_state=config.random_seed,
        stratify=stratify_train,
    )

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=np.asarray(y_train, dtype=np.int64),
        y_val=np.asarray(y_val, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
    )


def transform_features(
    preprocessor: ColumnTransformer,
    split: DatasetSplit,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit and apply preprocessing, returning torch tensors."""
    X_train = preprocessor.fit_transform(split.X_train)
    X_val = preprocessor.transform(split.X_val)
    X_test = preprocessor.transform(split.X_test)
    return (
        torch.tensor(np.asarray(X_train, dtype=np.float32), dtype=torch.float32),
        torch.tensor(np.asarray(X_val, dtype=np.float32), dtype=torch.float32),
        torch.tensor(np.asarray(X_test, dtype=np.float32), dtype=torch.float32),
    )
