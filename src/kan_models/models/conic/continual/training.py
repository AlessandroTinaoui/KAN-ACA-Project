"""Training primitives for the continual conic experiment."""

from __future__ import annotations

import numpy as np
import torch
from kan import KAN
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from kan_models.models.conic.config import ContinualConfig
from kan_models.models.conic.modeling import (
    accuracy,
    active_class_tensor,
    cross_entropy_loss,
    knowledge_distillation_loss,
    masked_classification_loss,
)
from kan_models.common.shared import copy_kan_model


class ConicStageDataset(Dataset):
    """Tiny Dataset wrapper for the rows selected in one continual stage."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        self.inputs = torch.tensor(features[indices], dtype=torch.float32)
        self.targets = torch.tensor(labels[indices], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def make_balanced_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    class_counts: dict[int, int],
    batch_size: int,
    seed: int,
    use_balanced_sampler: bool,
) -> DataLoader:
    """Create balanced mini-batches with a WeightedRandomSampler."""
    dataset = ConicStageDataset(features, labels, train_indices)
    if not use_balanced_sampler:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    selected_labels = labels[train_indices]
    present_counts = {class_id: count for class_id, count in class_counts.items() if count > 0}
    if not present_counts:
        raise ValueError("The current stage does not contain any training examples.")

    sample_weights = np.array([1.0 / present_counts[int(label)] for label in selected_labels], dtype=np.float64)
    samples_per_epoch = max(present_counts.values()) * len(present_counts)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def make_frozen_teacher_model(model: KAN) -> KAN:
    """Build the frozen teacher used by Learning without Forgetting."""
    teacher = copy_kan_model(model)
    teacher.eval()
    teacher.save_act = False
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def make_optimizer(model: KAN, optimizer_name: str, learning_rate: float) -> torch.optim.Optimizer:
    """Create the optimizer used by the manual mini-batch training loop."""
    params = model.get_params() if hasattr(model, "get_params") else model.parameters()
    normalized_name = optimizer_name.lower()
    if normalized_name == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    if normalized_name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate)
    raise ValueError("The continual loop supports only 'Adam' or 'AdamW'.")


def epochs_for_stage(epochs_per_task: list[int], stage: int) -> int:
    """Return how many epochs should be used for one continual stage."""
    if not epochs_per_task:
        raise ValueError("epochs_per_task must contain at least one value.")
    epochs = int(epochs_per_task[stage - 1]) if stage <= len(epochs_per_task) else int(epochs_per_task[-1])
    if epochs <= 0:
        raise ValueError(f"Invalid number of epochs for stage {stage}: {epochs}")
    return epochs


def lambda_kd_for_stage(initial_lambda_kd: float, decay: float, stage: int, has_previous_classes: bool) -> float:
    """Decay the distillation weight as the class-incremental task grows."""
    if not has_previous_classes:
        return 0.0
    incremental_index = max(0, stage - 2)
    return float(initial_lambda_kd * (decay**incremental_index))


def train_stage_with_early_stopping(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    previous_validation_indices: np.ndarray,
    device: torch.device,
    active_classes: list[int] | None,
    previous_classes: list[int],
    teacher_model: KAN | None,
    class_counts: dict[int, int],
    update_grid: bool,
    epochs: int,
    lambda_kd: float,
    config: ContinualConfig,
) -> tuple[KAN, dict[str, list], dict[str, float | int | bool | list[float]]]:
    """Train one continual stage with balanced batches, smoothing, and LwF."""
    train_history: list[float] = []
    test_history: list[float] = []
    reg_history: list[float] = []
    objective_history: list[float] = []
    supervised_history: list[float] = []
    distillation_history: list[float] = []
    old_accuracy_history: list[float] = []
    learning_rate_history: list[float] = []

    active_tensor = active_class_tensor(active_classes, device)
    train_loader = make_balanced_dataloader(
        features=features,
        labels=labels,
        train_indices=train_indices,
        class_counts=class_counts,
        batch_size=config.training.batch_size,
        seed=config.split.seed,
        use_balanced_sampler=config.training.use_balanced_sampler,
    )
    optimizer = make_optimizer(model, config.training.optimizer, config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=config.training.min_learning_rate,
    )

    best_loss = cross_entropy_loss(
        model=model,
        features=features,
        labels=labels,
        indices=validation_indices,
        device=device,
        active_classes=active_classes,
    )
    best_model = copy_kan_model(model)
    best_step = 0
    best_old_class_accuracy = float("-inf")
    best_old_class_accuracy_step = 0
    best_old_class_model = None
    steps_without_improvement = 0
    old_accuracy_bad_epochs = 0
    trained_steps = 0
    stopped_early = False
    anti_forgetting_stopped = False
    stop_reason = "max_epochs"
    full_stage_inputs = torch.tensor(features[train_indices], device=device)

    pbar = tqdm(range(1, epochs + 1), desc="training", ncols=100)
    for epoch in pbar:
        model.train()

        if (
            update_grid
            and config.grid.update_every > 0
            and epoch <= config.grid.stop_update_epoch
            and (epoch == 1 or (epoch - 1) % config.grid.update_every == 0)
        ):
            with torch.no_grad():
                model.update_grid(full_stage_inputs)

        epoch_examples = 0
        epoch_supervised = 0.0
        epoch_distillation = 0.0
        epoch_objective = 0.0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            batch_size = int(batch_inputs.shape[0])

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            supervised_loss = masked_classification_loss(
                logits=logits,
                labels=batch_labels,
                active_classes=active_tensor,
                label_smoothing=config.training.label_smoothing,
            )

            if config.training.use_lwf and teacher_model is not None and previous_classes:
                with torch.no_grad():
                    teacher_logits = teacher_model(batch_inputs)
                distillation_loss = knowledge_distillation_loss(
                    current_logits=logits,
                    teacher_logits=teacher_logits,
                    previous_classes=previous_classes,
                    device=device,
                    temperature=config.training.distillation_temperature,
                )
            else:
                distillation_loss = logits.new_tensor(0.0)

            objective = supervised_loss + lambda_kd * distillation_loss
            objective.backward()
            optimizer.step()

            epoch_examples += batch_size
            epoch_supervised += float(supervised_loss.detach().cpu()) * batch_size
            epoch_distillation += float(distillation_loss.detach().cpu()) * batch_size
            epoch_objective += float(objective.detach().cpu()) * batch_size

        trained_steps = epoch
        supervised_history.append(epoch_supervised / epoch_examples)
        distillation_history.append(epoch_distillation / epoch_examples)
        objective_history.append(epoch_objective / epoch_examples)
        current_lr = float(optimizer.param_groups[0]["lr"])
        learning_rate_history.append(current_lr)

        train_loss = cross_entropy_loss(model, features, labels, train_indices, device, active_classes=active_classes)
        validation_loss = cross_entropy_loss(model, features, labels, validation_indices, device, active_classes=active_classes)
        train_history.append(train_loss)
        test_history.append(validation_loss)
        reg_history.append(0.0)

        if len(previous_validation_indices) > 0:
            old_class_accuracy = accuracy(
                model=model,
                features=features,
                labels=labels,
                indices=previous_validation_indices,
                device=device,
                active_classes=active_classes,
            )
        else:
            old_class_accuracy = float("nan")
        old_accuracy_history.append(old_class_accuracy)

        if np.isfinite(old_class_accuracy):
            if old_class_accuracy >= best_old_class_accuracy:
                best_old_class_accuracy = old_class_accuracy
                best_old_class_accuracy_step = epoch
                best_old_class_model = copy_kan_model(model)
                old_accuracy_bad_epochs = 0
            elif best_old_class_accuracy - old_class_accuracy > config.early_stopping.anti_forgetting_tolerance:
                old_accuracy_bad_epochs += 1
            else:
                old_accuracy_bad_epochs = 0

        pbar.set_description(
            "| train_loss: %.3f | test_loss: %.3f | old_acc: %.3f | kd: %.3f | lr: %.2e | "
            % (train_loss, validation_loss, old_class_accuracy, distillation_history[-1], current_lr)
        )

        should_check = epoch % config.early_stopping.check_every == 0 or epoch == epochs
        if should_check:
            if validation_loss < best_loss - config.early_stopping.min_delta:
                best_loss = validation_loss
                best_model = copy_kan_model(model)
                best_step = trained_steps
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if config.early_stopping.enabled and steps_without_improvement >= config.early_stopping.patience:
                stopped_early = True
                stop_reason = "validation_loss"
                break

        scheduler.step()

        if (
            config.early_stopping.anti_forgetting_enabled
            and old_accuracy_bad_epochs >= config.early_stopping.anti_forgetting_patience
        ):
            stopped_early = True
            anti_forgetting_stopped = True
            stop_reason = "anti_forgetting"
            break

    model = best_old_class_model if anti_forgetting_stopped and best_old_class_model is not None else best_model
    final_train_loss = cross_entropy_loss(model, features, labels, train_indices, device, active_classes=active_classes)
    final_validation_loss = cross_entropy_loss(model, features, labels, validation_indices, device, active_classes=active_classes)
    if len(previous_validation_indices) > 0:
        final_old_class_accuracy = accuracy(
            model=model,
            features=features,
            labels=labels,
            indices=previous_validation_indices,
            device=device,
            active_classes=active_classes,
        )
    else:
        final_old_class_accuracy = float("nan")

    results = {
        "train_loss": train_history,
        "test_loss": test_history,
        "reg": reg_history,
        "objective_loss": objective_history,
        "supervised_loss": supervised_history,
        "distillation_loss": distillation_history,
        "old_class_accuracy": old_accuracy_history,
        "learning_rate": learning_rate_history,
    }
    summary = {
        "trained_steps": trained_steps,
        "best_step": best_step,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "anti_forgetting_stopped": anti_forgetting_stopped,
        "best_old_class_accuracy": float("nan") if best_old_class_accuracy == float("-inf") else best_old_class_accuracy,
        "best_old_class_accuracy_step": best_old_class_accuracy_step,
        "final_old_class_accuracy": final_old_class_accuracy,
        "best_test_loss": best_loss,
        "final_train_loss": final_train_loss,
        "final_test_loss": final_validation_loss,
        "final_learning_rate": learning_rate_history[-1] if learning_rate_history else config.training.learning_rate,
        "lambda_kd": lambda_kd,
        "epochs": epochs,
    }
    return model, results, summary


def parse_training_schedule(training_schedule: list[dict[str, int]], shape_names: list[str]) -> list[dict[int, int]]:
    """Convert the human-readable schedule into class ids and sample counts."""
    name_to_id = {name: index for index, name in enumerate(shape_names)}
    flat_names = [name for stage in training_schedule for name in stage]
    unknown = [name for name in flat_names if name not in name_to_id]
    if unknown:
        valid = ", ".join(shape_names)
        raise ValueError(f"Unknown tasks: {unknown}. Valid tasks: {valid}")

    parsed_schedule = []
    for stage_index, stage in enumerate(training_schedule, start=1):
        if not stage:
            raise ValueError(f"Stage {stage_index} does not contain any classes.")
        parsed_stage = {}
        for name, count in stage.items():
            if count <= 0:
                raise ValueError(f"Stage {stage_index} has an invalid count for {name}: {count}")
            parsed_stage[name_to_id[name]] = int(count)
        parsed_schedule.append(parsed_stage)
    return parsed_schedule
