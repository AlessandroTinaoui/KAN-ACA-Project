from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CONFIG_DIR = PROJECT_ROOT / "artifacts" / "mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "credit_default_kan.toml"

import torch
import torch.nn.functional as F
from kan import KAN
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)

    config_dir = config_path.parent

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
    if not np.isclose(total_size, 1.0):
        raise ValueError(
            f"Le proporzioni di split devono sommare a 1.0, trovato {total_size:.6f}."
        )

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
        config_path=config_path,
    )


def load_dataframe(config: DataConfig) -> pd.DataFrame:
    frame = pd.read_csv(
        config.csv_path,
        header=config.header_row,
        sep=config.separator,
    )
    frame.columns = [str(column).strip() for column in frame.columns]

    if config.drop_unnamed_columns:
        unnamed = [column for column in frame.columns if column.startswith("Unnamed:")]
        if unnamed:
            frame = frame.drop(columns=unnamed)

    return frame


def encode_target(target: pd.Series) -> tuple[np.ndarray, TaskSpec]:
    label_names = sorted(target.astype(str).unique().tolist())
    label_to_index = {label: index for index, label in enumerate(label_names)}
    encoded = target.astype(str).map(label_to_index).to_numpy(dtype=np.int64)
    return encoded, TaskSpec(label_names=label_names, is_binary=len(label_names) == 2)


def resolve_feature_groups(
    frame: pd.DataFrame,
    config: DataConfig,
) -> tuple[pd.DataFrame, list[str], list[str]]:
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
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = []
        if config.impute_numeric:
            numeric_steps.append(
                (
                    "imputer",
                    SimpleImputer(strategy=config.numeric_imputation_strategy),
                )
            )
        if config.standardize_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_transformer: Any = "passthrough"
        if numeric_steps:
            numeric_transformer = Pipeline(steps=numeric_steps)
        transformers.append(("numeric", numeric_transformer, numeric_columns))

    if categorical_columns:
        categorical_steps: list[tuple[str, Any]] = []
        if config.impute_categorical:
            imputer_kwargs: dict[str, Any] = {
                "strategy": config.categorical_imputation_strategy,
            }
            if config.categorical_imputation_strategy == "constant":
                imputer_kwargs["fill_value"] = config.categorical_fill_value
            categorical_steps.append(("imputer", SimpleImputer(**imputer_kwargs)))
        if config.one_hot_encode_categorical:
            categorical_steps.append(
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
            )

        categorical_transformer: Any = "passthrough"
        if categorical_steps:
            categorical_transformer = Pipeline(steps=categorical_steps)
        transformers.append(("categorical", categorical_transformer, categorical_columns))

    if not transformers:
        raise ValueError("Nessuna feature disponibile dopo la selezione delle colonne.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_dataframe(
    X: pd.DataFrame,
    y: np.ndarray,
    config: SplitConfig,
) -> DatasetSplit:
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
    X_train = preprocessor.fit_transform(split.X_train)
    X_val = preprocessor.transform(split.X_val)
    X_test = preprocessor.transform(split.X_test)

    return (
        torch.tensor(np.asarray(X_train, dtype=np.float32), dtype=torch.float32),
        torch.tensor(np.asarray(X_val, dtype=np.float32), dtype=torch.float32),
        torch.tensor(np.asarray(X_test, dtype=np.float32), dtype=torch.float32),
    )


def build_model(
    input_dim: int,
    task: TaskSpec,
    model_config: ModelConfig,
    seed: int,
    hidden_layers: list[int] | None = None,
) -> KAN:
    width = [input_dim, *(hidden_layers or model_config.hidden_layers), task.output_dim]
    return KAN(
        width=width,
        grid=model_config.grid,
        k=model_config.k,
        seed=seed,
        symbolic_enabled=model_config.symbolic_enabled,
        auto_save=model_config.auto_save,
        device=model_config.device,
    )


def build_optimizer(model: KAN, config: TrainingConfig) -> torch.optim.Optimizer:
    optimizer_name = config.optimizer.lower()
    if optimizer_name != "adam":
        raise ValueError(f"Ottimizzatore non supportato: {config.optimizer}")

    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def build_loss_kwargs(
    task: TaskSpec,
    y_train: np.ndarray,
    class_weighting: str,
) -> dict[str, torch.Tensor]:
    weighting = class_weighting.lower()
    if weighting == "none":
        return {}

    if weighting != "balanced":
        raise ValueError(f"class_weighting non supportato: {class_weighting}")

    counts = np.bincount(y_train, minlength=task.num_classes).astype(np.float64)
    if task.is_binary:
        negative_count, positive_count = counts[0], counts[1]
        pos_weight = negative_count / max(positive_count, 1.0)
        return {"pos_weight": torch.tensor([pos_weight], dtype=torch.float32)}

    weights = counts.sum() / (task.num_classes * np.maximum(counts, 1.0))
    return {"weight": torch.tensor(weights, dtype=torch.float32)}


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task: TaskSpec,
    loss_kwargs: dict[str, torch.Tensor],
) -> torch.Tensor:
    if task.is_binary:
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(1),
            targets.float(),
            **loss_kwargs,
        )

    return F.cross_entropy(logits, targets.long(), **loss_kwargs)


def compute_dataset_loss(
    model: KAN,
    features: torch.Tensor,
    targets: torch.Tensor,
    task: TaskSpec,
    loss_kwargs: dict[str, torch.Tensor],
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = compute_loss(logits, targets, task, loss_kwargs)
    return float(loss.detach().cpu().item())


def predict_probabilities(model: KAN, features: torch.Tensor, task: TaskSpec) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        if task.is_binary:
            return torch.sigmoid(logits.squeeze(1)).cpu().numpy()
        return torch.softmax(logits, dim=1).cpu().numpy()


def batch_iterator(
    size: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


def optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    config: ThresholdConfig,
) -> float:
    if not config.enabled:
        return config.default

    thresholds = np.linspace(config.min, config.max, config.steps)
    best_threshold = config.default
    best_score = -np.inf

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int64)
        if config.metric == "f1":
            score = f1_score(y_true, predictions, zero_division=0)
        elif config.metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, predictions)
        elif config.metric == "accuracy":
            score = accuracy_score(y_true, predictions)
        else:
            raise ValueError(f"Metric di threshold non supportata: {config.metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    task: TaskSpec,
    threshold: float,
) -> dict[str, Any]:
    if task.is_binary:
        predictions = (probabilities >= threshold).astype(np.int64)
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "threshold": float(threshold),
            "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        }

    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def monitor_direction(metric_name: str) -> str:
    minimized_metrics = {"loss", "cross_entropy", "log_loss"}
    return "min" if metric_name in minimized_metrics else "max"


def extract_monitor_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name not in metrics:
        raise ValueError(f"Metica di monitoraggio non trovata: {metric_name}")
    return float(metrics[metric_name])


def serialize_width(width: list[Any]) -> list[Any]:
    serialized: list[Any] = []
    for layer in width:
        if isinstance(layer, (list, tuple)):
            serialized.append([int(value) for value in layer])
        else:
            serialized.append(int(layer))
    return serialized


def hidden_units(model: KAN) -> int:
    total = 0
    for layer_width in model.width[1:-1]:
        if isinstance(layer_width, list):
            total += int(sum(layer_width))
        else:
            total += int(layer_width)
    return total


def top_nodes_from_scores(scores: np.ndarray, n_nodes: int) -> list[int]:
    n_nodes = min(n_nodes, len(scores))
    selected = np.argsort(scores)[-n_nodes:]
    return np.sort(selected).astype(int).tolist()


def score_hidden_nodes(model: KAN, features: torch.Tensor) -> np.ndarray:
    model.get_act(features)
    model.attribute(plot=False)
    return model.node_scores[1].detach().cpu().numpy()


def copy_kan_model(model: KAN) -> KAN:
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


def train_single_model(
    experiment: ExperimentConfig,
    task: TaskSpec,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    X_test: torch.Tensor,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    model: KAN | None = None,
    epochs: int | None = None,
    stage_name: str = "standard",
) -> tuple[KAN, dict[str, Any]]:
    seed = experiment.split.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model is None:
        model = build_model(X_train.shape[1], task, experiment.model, seed=seed)

    optimizer = build_optimizer(model, experiment.training)
    loss_kwargs = build_loss_kwargs(task, y_train, experiment.training.class_weighting)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32 if task.is_binary else torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32 if task.is_binary else torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32 if task.is_binary else torch.long)

    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_monitor_value = -np.inf if monitor_direction(experiment.training.monitor) == "max" else np.inf
    best_threshold = experiment.threshold.default
    patience_counter = 0
    history: list[dict[str, float]] = []
    max_epochs = experiment.training.epochs if epochs is None else epochs

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for batch_indices in batch_iterator(len(X_train), experiment.training.batch_size, rng):
            batch_X = X_train[batch_indices]
            batch_y = y_train_tensor[batch_indices]

            if epoch <= experiment.training.grid_update_epochs:
                model.update_grid(batch_X)

            logits = model(batch_X)
            loss = compute_loss(logits, batch_y, task, loss_kwargs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_probabilities = predict_probabilities(model, X_train, task)
        val_probabilities = predict_probabilities(model, X_val, task)

        if task.is_binary:
            epoch_threshold = optimize_threshold(y_val, val_probabilities, experiment.threshold)
        else:
            epoch_threshold = experiment.threshold.default

        train_metrics = compute_metrics(y_train, train_probabilities, task, epoch_threshold)
        val_metrics = compute_metrics(y_val, val_probabilities, task, epoch_threshold)

        train_loss = compute_dataset_loss(model, X_train, y_train_tensor, task, loss_kwargs)
        val_loss = compute_dataset_loss(model, X_val, y_val_tensor, task, loss_kwargs)

        train_metrics["loss"] = train_loss
        val_metrics["loss"] = val_loss

        monitor_value = extract_monitor_value(val_metrics, experiment.training.monitor)
        direction = monitor_direction(experiment.training.monitor)
        is_improved = (
            monitor_value > best_monitor_value if direction == "max" else monitor_value < best_monitor_value
        )

        if is_improved:
            best_monitor_value = monitor_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_threshold = epoch_threshold
            patience_counter = 0
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "batch_loss": float(np.mean(epoch_losses)),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "monitor_value": float(monitor_value),
            }
        )

        if epoch == 1 or epoch % experiment.training.log_every == 0:
            log_parts = [
                f"Epoch {epoch:03d}",
                f"batch_loss={np.mean(epoch_losses):.4f}",
                f"train_loss={train_loss:.4f}",
                f"val_loss={val_loss:.4f}",
                f"train_acc={train_metrics['accuracy']:.4f}",
                f"val_acc={val_metrics['accuracy']:.4f}",
            ]
            if task.is_binary and "f1" in val_metrics:
                log_parts.append(f"val_f1={val_metrics['f1']:.4f}")
                log_parts.append(f"val_auc={val_metrics['roc_auc']:.4f}")
                log_parts.append(f"thr={epoch_threshold:.3f}")
            log_parts.append(f"best_{experiment.training.monitor}={best_monitor_value:.4f}")
            print(" | ".join(log_parts))

        if patience_counter >= experiment.training.patience:
            print(f"Early stopping alla epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training fallito: nessuno stato migliore trovato.")

    model.load_state_dict(best_state)

    train_probabilities = predict_probabilities(model, X_train, task)
    val_probabilities = predict_probabilities(model, X_val, task)
    test_probabilities = predict_probabilities(model, X_test, task)

    train_metrics = compute_metrics(y_train, train_probabilities, task, best_threshold)
    val_metrics = compute_metrics(y_val, val_probabilities, task, best_threshold)
    test_metrics = compute_metrics(y_test, test_probabilities, task, best_threshold)

    train_metrics["loss"] = compute_dataset_loss(model, X_train, y_train_tensor, task, loss_kwargs)
    val_metrics["loss"] = compute_dataset_loss(model, X_val, y_val_tensor, task, loss_kwargs)
    test_metrics["loss"] = compute_dataset_loss(model, X_test, y_test_tensor, task, loss_kwargs)

    metrics_bundle = {
        "stage": stage_name,
        "history": history,
        "monitor": experiment.training.monitor,
        "best_monitor_value": float(best_monitor_value),
        "epochs_ran": len(history),
        "model_width": serialize_width(model.width),
        "hidden_units": hidden_units(model),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    return model, metrics_bundle


def train_standard(
    experiment: ExperimentConfig,
    task: TaskSpec,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    X_test: torch.Tensor,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[KAN, dict[str, Any]]:
    model, metrics = train_single_model(
        experiment=experiment,
        task=task,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        stage_name="standard",
    )
    metrics["mode"] = "standard"
    return model, metrics


def validate_pruning_config(experiment: ExperimentConfig) -> None:
    if len(experiment.model.hidden_layers) != 1:
        raise ValueError(
            "Il pruning-first supporta solo modelli con un solo hidden layer in model.hidden_layers."
        )

    if not experiment.pruning.keep_hidden_schedule:
        raise ValueError(
            "Con pruning.enabled = true devi specificare pruning.keep_hidden_schedule."
        )

    if any(hidden <= 0 for hidden in experiment.pruning.keep_hidden_schedule):
        raise ValueError("La keep_hidden_schedule deve contenere solo interi positivi.")

    if experiment.pruning.probe_hidden < max(experiment.pruning.keep_hidden_schedule):
        raise ValueError(
            "pruning.probe_hidden deve essere >= del massimo valore in keep_hidden_schedule."
        )


def train_with_pruning(
    experiment: ExperimentConfig,
    task: TaskSpec,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    X_test: torch.Tensor,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[KAN, dict[str, Any]]:
    validate_pruning_config(experiment)

    seed = experiment.split.random_seed
    probe_model = build_model(
        X_train.shape[1],
        task,
        experiment.model,
        seed=seed,
        hidden_layers=[experiment.pruning.probe_hidden],
    )
    probe_model.auto_save = False

    print(
        "Pruning-first mode"
        f" | probe_hidden={experiment.pruning.probe_hidden}"
        f" | probe_epochs={experiment.pruning.probe_epochs}"
        f" | keep_schedule={experiment.pruning.keep_hidden_schedule}"
    )

    probe_model, probe_metrics = train_single_model(
        experiment=experiment,
        task=task,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        model=probe_model,
        epochs=experiment.pruning.probe_epochs,
        stage_name="probe",
    )

    node_scores = score_hidden_nodes(probe_model, X_train)
    stages: list[dict[str, Any]] = [
        {
            **probe_metrics,
            "selected_nodes": [],
        }
    ]

    epochs_per_model = experiment.pruning.epochs_per_model or experiment.training.epochs
    final_model = probe_model
    final_metrics = probe_metrics

    for target_hidden in experiment.pruning.keep_hidden_schedule:
        selected_nodes = top_nodes_from_scores(node_scores, target_hidden)
        print(
            f"Training pruned model with top {target_hidden} probe nodes: {selected_nodes}"
        )

        probe_source = copy_kan_model(probe_model)
        probe_source.get_act(X_train)
        current_model = probe_source.prune_node(
            active_neurons_id=[selected_nodes],
            log_history=False,
        )
        current_model.auto_save = False
        current_model.get_act(X_train)

        current_model, stage_metrics = train_single_model(
            experiment=experiment,
            task=task,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            model=current_model,
            epochs=epochs_per_model,
            stage_name=f"keep_{target_hidden}",
        )
        stage_metrics["selected_nodes"] = selected_nodes
        stages.append(stage_metrics)
        final_model = current_model
        final_metrics = stage_metrics

    metrics_bundle = {
        "mode": "pruning",
        "monitor": final_metrics["monitor"],
        "best_monitor_value": final_metrics["best_monitor_value"],
        "train": final_metrics["train"],
        "val": final_metrics["val"],
        "test": final_metrics["test"],
        "pruning": {
            "probe_hidden": experiment.pruning.probe_hidden,
            "probe_epochs": experiment.pruning.probe_epochs,
            "keep_hidden_schedule": experiment.pruning.keep_hidden_schedule,
            "epochs_per_model": epochs_per_model,
            "node_scores": node_scores.tolist(),
        },
        "stages": stages,
    }
    return final_model, metrics_bundle


def save_artifacts(
    experiment: ExperimentConfig,
    model: KAN,
    preprocessor: ColumnTransformer,
    task: TaskSpec,
    metrics: dict[str, Any],
    feature_names_out: list[str],
) -> None:
    experiment.output.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = experiment.output.output_dir / experiment.output.model_filename
    preprocessor_path = experiment.output.output_dir / experiment.output.preprocessor_filename
    metrics_path = experiment.output.output_dir / experiment.output.metrics_filename
    config_snapshot_path = experiment.output.output_dir / experiment.output.config_snapshot_filename

    model_payload = {
        "model_state_dict": model.state_dict(),
        "model_width": serialize_width(model.width),
        "grid": experiment.model.grid,
        "k": experiment.model.k,
        "label_names": task.label_names,
        "feature_names_out": feature_names_out,
        "is_binary": task.is_binary,
        "best_threshold": metrics["val"].get("threshold", experiment.threshold.default),
        "config": experiment.raw_config,
    }
    torch.save(model_payload, model_path)

    with preprocessor_path.open("wb") as handle:
        pickle.dump(preprocessor, handle)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    shutil.copy2(experiment.config_path, config_snapshot_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allena una KAN tabellare con pykan usando una configurazione TOML."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Percorso al file TOML di configurazione. Default: {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args()

    experiment = load_config(args.config)
    frame = load_dataframe(experiment.data)

    if experiment.data.target_column not in frame.columns:
        raise ValueError(
            f"Colonna target '{experiment.data.target_column}' non trovata nel dataset."
        )

    X_frame, numeric_columns, categorical_columns = resolve_feature_groups(frame, experiment.data)
    y_encoded, task = encode_target(frame[experiment.data.target_column])
    split = split_dataframe(X_frame, y_encoded, experiment.split)
    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        config=experiment.preprocessing,
    )
    X_train, X_val, X_test = transform_features(preprocessor, split)

    if experiment.pruning.enabled:
        model, metrics = train_with_pruning(
            experiment=experiment,
            task=task,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=split.y_train,
            y_val=split.y_val,
            y_test=split.y_test,
        )
    else:
        model, metrics = train_standard(
            experiment=experiment,
            task=task,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=split.y_train,
            y_val=split.y_val,
            y_test=split.y_test,
        )

    feature_names_out = preprocessor.get_feature_names_out().tolist()
    save_artifacts(
        experiment=experiment,
        model=model,
        preprocessor=preprocessor,
        task=task,
        metrics=metrics,
        feature_names_out=feature_names_out,
    )

    print("\nRisultati finali")
    print(f"Train metrics: {json.dumps(metrics['train'], ensure_ascii=True)}")
    print(f"Validation metrics: {json.dumps(metrics['val'], ensure_ascii=True)}")
    print(f"Test metrics: {json.dumps(metrics['test'], ensure_ascii=True)}")
    print(f"Artefatti salvati in: {experiment.output.output_dir}")


if __name__ == "__main__":
    main()
