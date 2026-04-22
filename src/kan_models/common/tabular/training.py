"""Training and evaluation utilities for tabular KAN experiments."""

from __future__ import annotations

import json
import pickle
import shutil
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from kan import KAN
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from kan_models.common.shared import copy_kan_model, hidden_units, serialize_width
from kan_models.common.tabular.config import ExperimentConfig, ModelConfig, ThresholdConfig, TrainingConfig
from kan_models.common.tabular.data import TaskSpec


def build_model(
    input_dim: int,
    task: TaskSpec,
    model_config: ModelConfig,
    seed: int,
    hidden_layers: list[int] | None = None,
) -> KAN:
    """Build the tabular KAN model."""
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
    """Build the optimizer for the tabular experiment."""
    optimizer_name = config.optimizer.lower()
    if optimizer_name != "adam":
        raise ValueError(f"Ottimizzatore non supportato: {config.optimizer}")
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def build_loss_kwargs(task: TaskSpec, y_train: np.ndarray, class_weighting: str) -> dict[str, torch.Tensor]:
    """Build class weights or pos_weight depending on the task."""
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
    """Compute task-aware loss."""
    if task.is_binary:
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), targets.float(), **loss_kwargs)
    return F.cross_entropy(logits, targets.long(), **loss_kwargs)


def compute_dataset_loss(
    model: KAN,
    features: torch.Tensor,
    targets: torch.Tensor,
    task: TaskSpec,
    loss_kwargs: dict[str, torch.Tensor],
) -> float:
    """Compute loss on a full split."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = compute_loss(logits, targets, task, loss_kwargs)
    return float(loss.detach().cpu().item())


def predict_probabilities(model: KAN, features: torch.Tensor, task: TaskSpec) -> np.ndarray:
    """Return probabilities for binary or multiclass tasks."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        if task.is_binary:
            return torch.sigmoid(logits.squeeze(1)).cpu().numpy()
        return torch.softmax(logits, dim=1).cpu().numpy()


def batch_iterator(size: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle indices and split them into batches."""
    indices = np.arange(size, dtype=np.int64)
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


def optimize_threshold(y_true: np.ndarray, probabilities: np.ndarray, config: ThresholdConfig) -> float:
    """Select the best threshold according to the configured metric."""
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


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, task: TaskSpec, threshold: float) -> dict[str, Any]:
    """Compute classification metrics for binary or multiclass tasks."""
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
    """Tell whether the monitored metric should be minimized or maximized."""
    return "min" if metric_name in {"loss", "cross_entropy", "log_loss"} else "max"


def extract_monitor_value(metrics: dict[str, Any], metric_name: str) -> float:
    """Extract the monitored metric from the metrics bundle."""
    if metric_name not in metrics:
        raise ValueError(f"Metica di monitoraggio non trovata: {metric_name}")
    return float(metrics[metric_name])


def top_nodes_from_scores(scores: np.ndarray, n_nodes: int) -> list[int]:
    """Select the highest-scoring hidden nodes."""
    n_nodes = min(n_nodes, len(scores))
    selected = np.argsort(scores)[-n_nodes:]
    return np.sort(selected).astype(int).tolist()


def score_hidden_nodes(model: KAN, features: torch.Tensor) -> np.ndarray:
    """Score hidden nodes through pykan attribution."""
    model.get_act(features)
    model.attribute(plot=False)
    return model.node_scores[1].detach().cpu().numpy()


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
    """Train one tabular KAN model with early stopping."""
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
        epoch_threshold = optimize_threshold(y_val, val_probabilities, experiment.threshold) if task.is_binary else experiment.threshold.default

        train_metrics = compute_metrics(y_train, train_probabilities, task, epoch_threshold)
        val_metrics = compute_metrics(y_val, val_probabilities, task, epoch_threshold)
        train_loss = compute_dataset_loss(model, X_train, y_train_tensor, task, loss_kwargs)
        val_loss = compute_dataset_loss(model, X_val, y_val_tensor, task, loss_kwargs)
        train_metrics["loss"] = train_loss
        val_metrics["loss"] = val_loss

        monitor_value = extract_monitor_value(val_metrics, experiment.training.monitor)
        direction = monitor_direction(experiment.training.monitor)
        is_improved = monitor_value > best_monitor_value if direction == "max" else monitor_value < best_monitor_value

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
                log_parts.extend(
                    [
                        f"val_f1={val_metrics['f1']:.4f}",
                        f"val_auc={val_metrics['roc_auc']:.4f}",
                        f"thr={epoch_threshold:.3f}",
                    ]
                )
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
    """Train the standard, non-pruned tabular model."""
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
    """Validate the pruning-first configuration before training."""
    if len(experiment.model.hidden_layers) != 1:
        raise ValueError("Il pruning-first supporta solo modelli con un solo hidden layer in model.hidden_layers.")
    if not experiment.pruning.keep_hidden_schedule:
        raise ValueError("Con pruning.enabled = true devi specificare pruning.keep_hidden_schedule.")
    if any(hidden <= 0 for hidden in experiment.pruning.keep_hidden_schedule):
        raise ValueError("La keep_hidden_schedule deve contenere solo interi positivi.")
    if experiment.pruning.probe_hidden < max(experiment.pruning.keep_hidden_schedule):
        raise ValueError("pruning.probe_hidden deve essere >= del massimo valore in keep_hidden_schedule.")


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
    """Train the probe-and-prune schedule for the tabular model."""
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
    stages: list[dict[str, Any]] = [{**probe_metrics, "selected_nodes": []}]
    epochs_per_model = experiment.pruning.epochs_per_model or experiment.training.epochs
    final_model = probe_model
    final_metrics = probe_metrics

    for target_hidden in experiment.pruning.keep_hidden_schedule:
        selected_nodes = top_nodes_from_scores(node_scores, target_hidden)
        print(f"Training pruned model with top {target_hidden} probe nodes: {selected_nodes}")

        probe_source = copy_kan_model(probe_model)
        probe_source.get_act(X_train)
        current_model = probe_source.prune_node(active_neurons_id=[selected_nodes], log_history=False)
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
    """Persist the trained model, metrics, preprocessor, and config snapshot."""
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
        handle.write("\n")

    shutil.copy2(experiment.config_path, config_snapshot_path)
