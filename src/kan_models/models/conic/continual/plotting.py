"""Plotting utilities for continual conic experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kan import KAN

from kan_models.models.conic.plotting import plot_prediction_examples


def short_task_label(task_name: object) -> str:
    """Create compact task names so plot text does not overlap."""
    short_names = {"parabola": "par", "ellipse": "ell", "hyperbola": "hyp", "circle": "cir"}
    parts = str(task_name).split("+")
    return "+".join(short_names.get(part, part[:3]) for part in parts)


def stage_tick_label(stage: object, task_name: object) -> str:
    """Create a short two-line tick label for one continual stage."""
    return f"S{stage}\n{short_task_label(task_name)}"


def plot_stage_predictions(
    model: KAN,
    model_features: np.ndarray,
    plot_features: np.ndarray,
    labels: np.ndarray,
    test_by_class: dict[int, np.ndarray],
    seen_classes: list[int],
    shape_names: list[str],
    stage: int,
    trained_task: str,
    device: torch.device,
    output_dir: Path,
    examples_per_class: int,
    seed: int,
    active_classes: list[int] | None = None,
) -> None:
    """Plot a few test predictions after one continual learning stage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_prediction_examples(
        model=model,
        model_features=model_features,
        plot_features=plot_features,
        labels=labels,
        test_by_class=test_by_class,
        shape_names=shape_names,
        device=device,
        output_file=output_dir / f"stage_{stage:02d}_{trained_task}_predictions.png",
        examples_per_class=examples_per_class,
        seed=seed + stage,
        title=f"Stage {stage}: after training on {trained_task}",
        active_classes=active_classes,
        class_ids=seen_classes,
    )


def plot_continual_losses(loss_history: list[dict[str, object]], output_dir: Path) -> None:
    """Save the loss curves collected during continual learning."""
    if not loss_history:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=False)
    fig.subplots_adjust(bottom=0.24, wspace=0.28)

    for item in loss_history:
        stage = item["stage"]
        task = item["task"]
        train_loss = item["train_loss"]
        test_loss = item["test_loss"]
        steps = np.arange(1, len(train_loss) + 1)
        stage_label = f"S{stage} {short_task_label(task)}"
        axes[0].plot(steps, train_loss, label=f"{stage_label} train")
        axes[0].plot(steps, test_loss, linestyle="--", label=f"{stage_label} test")

    axes[0].set_title("Loss curves during continual learning")
    axes[0].set_xlabel("Training epoch")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[0].grid(True, linewidth=0.4, alpha=0.35)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)

    stages = [item["stage"] for item in loss_history]
    task_labels = [item["task"] for item in loss_history]
    final_train = [item.get("final_train_loss", item["train_loss"][-1]) for item in loss_history]
    final_test = [item.get("final_test_loss", item["test_loss"][-1]) for item in loss_history]
    x = np.arange(len(stages))
    width = 0.36

    axes[1].bar(x - width / 2, final_train, width, label="train")
    axes[1].bar(x + width / 2, final_test, width, label="test")
    axes[1].set_title("Final loss after each task")
    axes[1].set_xlabel("Continual task")
    axes[1].set_ylabel("Final loss")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([stage_tick_label(stage, task) for stage, task in zip(stages, task_labels)], fontsize=9)
    axes[1].grid(True, axis="y", linewidth=0.4, alpha=0.35)
    axes[1].legend(loc="upper right", frameon=False)

    fig.savefig(output_dir / "continual_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_continual_error(records: list[dict[str, object]], output_dir: Path) -> None:
    """Plot the mean error and accuracy after each learned task."""
    if not records:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    stages = [record["stage"] for record in records]
    task_labels = [record["trained_task"] for record in records]
    mean_accuracy = np.array([record["mean_seen_accuracy"] for record in records], dtype=float)
    error_rate = 1.0 - mean_accuracy

    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=False)
    fig.subplots_adjust(bottom=0.22)
    ax.plot(stages, error_rate, marker="o", label="mean error")
    ax.plot(stages, mean_accuracy, marker="o", label="mean accuracy")
    ax.set_title("Continual learning error and accuracy")
    ax.set_xlabel("Continual task")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(stages)
    ax.set_xticklabels([stage_tick_label(stage, task) for stage, task in zip(stages, task_labels)], fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.savefig(output_dir / "continual_error.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_test_metrics(class_test_frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot test loss and accuracy separately for each conic class."""
    if class_test_frame.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_frame = class_test_frame[class_test_frame["seen"]]
    if seen_frame.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for test_class, group in seen_frame.groupby("test_class", sort=False):
        group = group.sort_values("stage")
        axes[0].plot(group["stage"], group["test_loss"], marker="o", linewidth=2, label=test_class)
        axes[1].plot(group["stage"], group["test_accuracy"], marker="o", linewidth=2, label=test_class)

    axes[0].set_title("Separate test loss by class")
    axes[0].set_xlabel("Continual stage")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[1].set_title("Separate test accuracy by class")
    axes[1].set_xlabel("Continual stage")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)

    for ax in axes:
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend()

    fig.savefig(output_dir / "continual_class_tests.png", dpi=160)
    plt.close(fig)
