"""Shared modeling utilities for conic experiments."""

from __future__ import annotations

import numpy as np
import torch
from kan import KAN

from kan_models.models.conic.config import ConicModelConfig


def build_model(
    input_dim: int,
    output_dim: int,
    config: ConicModelConfig,
    seed: int,
    device: torch.device,
    hidden: int | None = None,
) -> KAN:
    """Build a KAN with the common conic architecture settings."""
    return KAN(
        width=[input_dim, hidden or config.hidden, output_dim],
        grid=config.grid,
        k=config.spline_order,
        seed=seed,
        auto_save=config.auto_save,
        device=device,
    )


@torch.no_grad()
def predict_classes(
    model: KAN,
    features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    active_classes: list[int] | None = None,
) -> np.ndarray:
    """Return predicted class ids for the selected examples."""
    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    logits = model(inputs)
    if active_classes is not None:
        active = active_class_tensor(active_classes, device)
        logits = logits.index_select(dim=1, index=active)
        predicted_positions = logits.argmax(dim=1)
        return active[predicted_positions].cpu().numpy()
    return logits.argmax(dim=1).cpu().numpy()


@torch.no_grad()
def accuracy(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    active_classes: list[int] | None = None,
) -> float:
    """Compute accuracy on a fixed subset of examples."""
    if len(indices) == 0:
        return float("nan")
    predictions = predict_classes(model, features, indices, device, active_classes=active_classes)
    return float(np.mean(predictions == labels[indices]))


def active_class_tensor(active_classes: list[int] | None, device: torch.device) -> torch.Tensor | None:
    """Build the tensor used to mask future classes in continual learning."""
    if active_classes is None:
        return None
    return torch.tensor(active_classes, dtype=torch.long, device=device)


def masked_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_classes: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict logits to active classes and remap labels accordingly."""
    if active_classes is None:
        return logits, labels.long()

    label_map = torch.full((logits.shape[1],), -1, dtype=torch.long, device=logits.device)
    label_map[active_classes] = torch.arange(len(active_classes), device=logits.device)
    mapped_labels = label_map[labels.long()]
    if torch.any(mapped_labels < 0):
        raise ValueError("The batch contains labels outside the active class mask.")
    return logits.index_select(dim=1, index=active_classes), mapped_labels


def masked_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_classes: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    """Compute CrossEntropyLoss after masking future classes."""
    masked_logits, mapped_labels = masked_logits_and_labels(logits, labels, active_classes)
    return torch.nn.functional.cross_entropy(
        masked_logits,
        mapped_labels,
        label_smoothing=label_smoothing,
    )


def knowledge_distillation_loss(
    current_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    previous_classes: list[int],
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    """Apply Learning without Forgetting on previously seen classes only."""
    if len(previous_classes) < 2:
        return current_logits.new_tensor(0.0)

    old_class_tensor = active_class_tensor(previous_classes, device)
    current_old_logits = current_logits.index_select(dim=1, index=old_class_tensor)
    teacher_old_logits = teacher_logits.index_select(dim=1, index=old_class_tensor)
    student_log_probs = torch.nn.functional.log_softmax(current_old_logits / temperature, dim=1)
    teacher_probs = torch.nn.functional.softmax(teacher_old_logits / temperature, dim=1)
    return torch.nn.functional.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature**2)


@torch.no_grad()
def cross_entropy_loss(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    active_classes: list[int] | None = None,
) -> float:
    """Measure CrossEntropyLoss on selected examples."""
    if len(indices) == 0:
        return float("nan")

    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    targets = torch.tensor(labels[indices], device=device)
    logits = model(inputs)
    class_mask = active_class_tensor(active_classes, device)
    masked_logits, mapped_targets = masked_logits_and_labels(logits, targets, class_mask)
    loss = torch.nn.functional.cross_entropy(masked_logits, mapped_targets)
    return float(loss.cpu().item())


@torch.no_grad()
def sqrt_cross_entropy_loss(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    active_classes: list[int] | None = None,
) -> float:
    """Legacy helper that returns sqrt(CrossEntropyLoss)."""
    loss = cross_entropy_loss(
        model=model,
        features=features,
        labels=labels,
        indices=indices,
        device=device,
        active_classes=active_classes,
    )
    return float(np.sqrt(loss))


@torch.no_grad()
def compute_confusion_matrix(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    class_ids: list[int],
    active_classes: list[int] | None = None,
) -> np.ndarray:
    """Compute a raw confusion matrix for the selected classes."""
    if len(indices) == 0:
        return np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)

    predictions = predict_classes(model, features, indices, device, active_classes=active_classes)
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    class_to_position = {class_id: position for position, class_id in enumerate(class_ids)}

    for true_label, predicted_label in zip(labels[indices], predictions):
        true_position = class_to_position.get(int(true_label))
        predicted_position = class_to_position.get(int(predicted_label))
        if true_position is not None and predicted_position is not None:
            matrix[true_position, predicted_position] += 1

    return matrix
