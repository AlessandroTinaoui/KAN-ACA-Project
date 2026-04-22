"""Configuration models and TOML loaders for conic experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kan_models.common.shared import load_toml, resolve_path


@dataclass
class ConicDataConfig:
    csv_path: Path
    target_column: str = "shape"


@dataclass
class ConicSplitConfig:
    test_ratio: float
    seed: int


@dataclass
class ConicModelConfig:
    hidden: int
    grid: int
    spline_order: int
    auto_save: bool = False
    device: str = "auto"


@dataclass
class ConicOutputConfig:
    metrics_path: Path | None = None
    class_tests_path: Path | None = None
    high_loss_path: Path | None = None
    run_config_path: Path | None = None
    final_model_path: Path | None = None
    accuracy_config_path: Path | None = None
    plot_dir: Path | None = None


@dataclass
class BaselineTrainingConfig:
    optimizer: str
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    steps: int
    label_smoothing: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    grid_update_every: int
    stop_grid_update_step: int
    lr_scheduler: str = "ReduceLROnPlateau"


@dataclass
class BaselinePlotConfig:
    clear_old_plots: bool = True
    prediction_examples: int = 2
    high_loss_examples: int = 12


@dataclass
class BaselineConfig:
    data: ConicDataConfig
    split: ConicSplitConfig
    model: ConicModelConfig
    training: BaselineTrainingConfig
    standardize: bool
    plots: BaselinePlotConfig
    output: ConicOutputConfig
    raw_config: dict[str, Any]
    config_path: Path


@dataclass
class PruningTrainingConfig:
    optimizer: str
    learning_rate: float
    standardize: bool
    probe_hidden: int
    probe_steps: int
    start_hidden: int
    keep_hidden_schedule: list[int]
    train_steps_per_model: int


@dataclass
class PruningPlotConfig:
    clear_old_plots: bool = True
    prediction_examples: int = 2
    high_loss_examples: int = 12
    plot_edge_functions: bool = True


@dataclass
class PruningConfig:
    data: ConicDataConfig
    split: ConicSplitConfig
    model: ConicModelConfig
    training: PruningTrainingConfig
    plots: PruningPlotConfig
    output: ConicOutputConfig
    raw_config: dict[str, Any]
    config_path: Path


@dataclass
class ContinualTrainingConfig:
    epochs_per_task: list[int]
    batch_size: int
    optimizer: str
    learning_rate: float
    min_learning_rate: float
    mask_future_classes: bool
    label_smoothing: float
    use_balanced_sampler: bool
    use_lwf: bool
    initial_lambda_kd: float
    lambda_kd_decay: float
    distillation_temperature: float
    freeze_grid_after_first: bool


@dataclass
class ContinualEarlyStoppingConfig:
    enabled: bool
    check_every: int
    patience: int
    min_delta: float
    anti_forgetting_enabled: bool
    anti_forgetting_tolerance: float
    anti_forgetting_patience: int


@dataclass
class ContinualGridConfig:
    update_every: int
    stop_update_epoch: int


@dataclass
class ContinualPlotConfig:
    prediction_examples: int = 2
    enable_predictions: bool = True
    enable_loss: bool = True
    enable_error: bool = True
    enable_class_tests: bool = True
    enable_confusion_matrices: bool = True
    enable_edge_functions: bool = True


@dataclass
class ContinualConfig:
    data: ConicDataConfig
    split: ConicSplitConfig
    model: ConicModelConfig
    training: ContinualTrainingConfig
    early_stopping: ContinualEarlyStoppingConfig
    grid: ContinualGridConfig
    standardize: bool
    plots: ContinualPlotConfig
    schedule: list[dict[str, int]]
    output: ConicOutputConfig
    raw_config: dict[str, Any]
    config_path: Path


def _optional_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(base_dir, value)


def _load_output_config(base_dir: Path, section: dict[str, Any]) -> ConicOutputConfig:
    return ConicOutputConfig(
        metrics_path=_optional_path(base_dir, section.get("metrics_path")),
        class_tests_path=_optional_path(base_dir, section.get("class_tests_path")),
        high_loss_path=_optional_path(base_dir, section.get("high_loss_path")),
        run_config_path=_optional_path(base_dir, section.get("run_config_path")),
        final_model_path=_optional_path(base_dir, section.get("final_model_path")),
        accuracy_config_path=_optional_path(base_dir, section.get("accuracy_config_path")),
        plot_dir=_optional_path(base_dir, section.get("plot_dir")),
    )


def _load_base_sections(raw_config: dict[str, Any], config_dir: Path) -> tuple[
    ConicDataConfig,
    ConicSplitConfig,
    ConicModelConfig,
    ConicOutputConfig,
]:
    data_section = raw_config["data"]
    split_section = raw_config["split"]
    model_section = raw_config["model"]
    output_section = raw_config["output"]

    data = ConicDataConfig(
        csv_path=resolve_path(config_dir, data_section["csv_path"]),
        target_column=data_section.get("target_column", "shape"),
    )
    split = ConicSplitConfig(
        test_ratio=float(split_section["test_ratio"]),
        seed=int(split_section["seed"]),
    )
    model = ConicModelConfig(
        hidden=int(model_section["hidden"]),
        grid=int(model_section["grid"]),
        spline_order=int(model_section["spline_order"]),
        auto_save=bool(model_section.get("auto_save", False)),
        device=str(model_section.get("device", "auto")),
    )
    output = _load_output_config(config_dir, output_section)
    return data, split, model, output


def load_baseline_config(path: str | Path) -> BaselineConfig:
    config_path, raw_config = load_toml(path)
    config_dir = config_path.parent
    data, split, model, output = _load_base_sections(raw_config, config_dir)
    training_section = dict(raw_config["training"])
    standardize = bool(raw_config.get("standardize", training_section.pop("standardize", True)))
    training = BaselineTrainingConfig(**training_section)
    plots = BaselinePlotConfig(**raw_config.get("plots", {}))
    return BaselineConfig(
        data=data,
        split=split,
        model=model,
        training=training,
        standardize=standardize,
        plots=plots,
        output=output,
        raw_config=raw_config,
        config_path=config_path,
    )


def load_pruning_config(path: str | Path) -> PruningConfig:
    config_path, raw_config = load_toml(path)
    config_dir = config_path.parent
    data, split, model, output = _load_base_sections(raw_config, config_dir)
    training = PruningTrainingConfig(**raw_config["training"])
    plots = PruningPlotConfig(**raw_config.get("plots", {}))
    return PruningConfig(
        data=data,
        split=split,
        model=model,
        training=training,
        plots=plots,
        output=output,
        raw_config=raw_config,
        config_path=config_path,
    )


def load_continual_config(path: str | Path) -> ContinualConfig:
    config_path, raw_config = load_toml(path)
    config_dir = config_path.parent
    data, split, model, output = _load_base_sections(raw_config, config_dir)
    training = ContinualTrainingConfig(**raw_config["training"])
    early_stopping = ContinualEarlyStoppingConfig(**raw_config["early_stopping"])
    grid_section = dict(raw_config["grid"])
    standardize = bool(raw_config.get("standardize", grid_section.pop("standardize", True)))
    grid = ContinualGridConfig(**grid_section)
    plots = ContinualPlotConfig(**raw_config.get("plots", {}))
    schedule = raw_config["schedule"]
    return ContinualConfig(
        data=data,
        split=split,
        model=model,
        training=training,
        early_stopping=early_stopping,
        grid=grid,
        standardize=standardize,
        plots=plots,
        schedule=schedule,
        output=output,
        raw_config=raw_config,
        config_path=config_path,
    )
