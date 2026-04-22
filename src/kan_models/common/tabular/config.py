"""Configuration models and TOML loader for tabular experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kan_models.common.shared import load_toml, resolve_path


@dataclass
class DataConfig:
    csv_path: Path
    target_column: str
    header_row: int = 0
    separator: str = ","
    drop_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    drop_unnamed_columns: bool = True


@dataclass
class SplitConfig:
    train_size: float
    val_size: float
    test_size: float
    random_seed: int = 42
    stratify: bool = True


@dataclass
class PreprocessingConfig:
    standardize_numeric: bool = True
    one_hot_encode_categorical: bool = True
    impute_numeric: bool = True
    numeric_imputation_strategy: str = "median"
    impute_categorical: bool = True
    categorical_imputation_strategy: str = "most_frequent"
    categorical_fill_value: str = "missing"


@dataclass
class ModelConfig:
    hidden_layers: list[int]
    grid: int
    k: int
    symbolic_enabled: bool = False
    auto_save: bool = False
    device: str = "cpu"


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grid_update_epochs: int
    patience: int
    log_every: int
    optimizer: str = "adam"
    class_weighting: str = "none"
    monitor: str = "roc_auc"


@dataclass
class ThresholdConfig:
    enabled: bool = True
    metric: str = "f1"
    min: float = 0.05
    max: float = 0.95
    steps: int = 181
    default: float = 0.5


@dataclass
class PruningConfig:
    enabled: bool = False
    probe_hidden: int = 24
    probe_epochs: int = 8
    keep_hidden_schedule: list[int] = field(default_factory=list)
    epochs_per_model: int | None = None


@dataclass
class OutputConfig:
    output_dir: Path
    model_filename: str = "model.pt"
    preprocessor_filename: str = "preprocessor.pkl"
    metrics_filename: str = "metrics.json"
    config_snapshot_filename: str = "config.toml"


@dataclass
class ExperimentConfig:
    data: DataConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    threshold: ThresholdConfig
    pruning: PruningConfig
    output: OutputConfig
    raw_config: dict[str, Any]
    config_path: Path


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load a tabular experiment config from TOML."""
    resolved_path, raw_config = load_toml(config_path)
    config_dir = resolved_path.parent

    data_section = raw_config["data"]
    split_section = raw_config["split"]
    preprocessing_section = raw_config.get("preprocessing", {})
    model_section = raw_config["model"]
    training_section = raw_config["training"]
    threshold_section = raw_config.get("threshold", {})
    pruning_section = raw_config.get("pruning", {})
    output_section = raw_config["output"]

    split = SplitConfig(**split_section)
    total_size = split.train_size + split.val_size + split.test_size
    if abs(total_size - 1.0) > 1e-8:
        raise ValueError(f"Le proporzioni di split devono sommare a 1.0, trovato {total_size:.6f}.")

    return ExperimentConfig(
        data=DataConfig(
            csv_path=resolve_path(config_dir, data_section["csv_path"]),
            target_column=data_section["target_column"],
            header_row=data_section.get("header_row", 0),
            separator=data_section.get("separator", ","),
            drop_columns=data_section.get("drop_columns", []),
            feature_columns=data_section.get("feature_columns", []),
            categorical_columns=data_section.get("categorical_columns", []),
            numeric_columns=data_section.get("numeric_columns", []),
            drop_unnamed_columns=data_section.get("drop_unnamed_columns", True),
        ),
        split=split,
        preprocessing=PreprocessingConfig(**preprocessing_section),
        model=ModelConfig(**model_section),
        training=TrainingConfig(**training_section),
        threshold=ThresholdConfig(**threshold_section),
        pruning=PruningConfig(**pruning_section),
        output=OutputConfig(
            output_dir=resolve_path(config_dir, output_section["output_dir"]),
            model_filename=output_section.get("model_filename", "model.pt"),
            preprocessor_filename=output_section.get("preprocessor_filename", "preprocessor.pkl"),
            metrics_filename=output_section.get("metrics_filename", "metrics.json"),
            config_snapshot_filename=output_section.get("config_snapshot_filename", "config.toml"),
        ),
        raw_config=raw_config,
        config_path=resolved_path,
    )
