"""Single-task baseline for the conic-section dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kan import KAN
from tqdm import tqdm

from kan_models.models.conic.config import BaselineConfig, load_baseline_config
from kan_models.models.conic.data import (
    load_conic_csv,
    load_feature_names,
    make_kan_dataset,
    standardization_stats,
    stratified_split,
)
from kan_models.models.conic.modeling import accuracy, build_model, compute_confusion_matrix
from kan_models.models.conic.plotting import (
    high_loss_examples,
    plot_high_loss_examples,
    plot_kan_edge_functions,
    plot_prediction_examples,
    save_confusion_matrix,
    save_final_loss_summary,
)
from kan_models.common.runtime import detect_device
from kan_models.common.shared import clone_state_dict, clear_directory, write_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "conic" / "baseline.toml"


@torch.no_grad()
def cross_entropy_loss(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    """Compute CrossEntropyLoss on a fixed split."""
    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    targets = torch.tensor(labels[indices], device=device)
    logits = model(inputs)
    return float(torch.nn.functional.cross_entropy(logits, targets).cpu().item())


@torch.no_grad()
def tensor_cross_entropy_loss(model: KAN, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute CrossEntropyLoss on already-built tensors."""
    model.eval()
    logits = model(inputs)
    return torch.nn.functional.cross_entropy(logits, targets)


def build_optimizer(model: KAN, config: BaselineConfig) -> torch.optim.Optimizer:
    """Build the optimizer configured for the baseline run."""
    training = config.training
    params = model.get_params() if hasattr(model, "get_params") else model.parameters()
    optimizer_name = training.optimizer.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=training.learning_rate, weight_decay=training.weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=training.learning_rate, weight_decay=training.weight_decay)
    raise ValueError(f"Unsupported baseline optimizer: {training.optimizer}")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: BaselineConfig,
) -> tuple[object, str]:
    """Build the learning-rate scheduler and describe how it should be stepped."""
    training = config.training
    scheduler_name = training.lr_scheduler.lower()
    if scheduler_name == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training.lr_scheduler_factor,
            patience=training.lr_scheduler_patience,
            min_lr=training.min_learning_rate,
        )
        return scheduler, "metric"
    if scheduler_name == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, training.steps),
            eta_min=training.min_learning_rate,
        )
        return scheduler, "step"
    raise ValueError(f"Unsupported baseline lr_scheduler: {training.lr_scheduler}")


def train_with_early_stopping(
    model: KAN,
    dataset: dict[str, torch.Tensor],
    config: BaselineConfig,
) -> tuple[KAN, dict[str, list | float | int | bool | str]]:
    """Train the baseline KAN with configurable optimizer, scheduler, and strict stopping."""
    training = config.training
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=training.label_smoothing)
    optimizer = build_optimizer(model, config)
    scheduler, scheduler_step_mode = build_lr_scheduler(optimizer, config)

    train_input = dataset["train_input"]
    train_label = dataset["train_label"]
    test_input = dataset["test_input"]
    test_label = dataset["test_label"]

    best_test_ce = float("inf")
    best_state = clone_state_dict(model)
    best_step = 0
    steps_without_improvement = 0
    stopped_early = False

    results: dict[str, list | float | int | bool | str] = {
        "train_loss": [],
        "test_loss": [],
        "train_cross_entropy": [],
        "test_cross_entropy": [],
        "train_objective_loss": [],
        "learning_rate": [],
        "best_step": best_step,
        "best_test_cross_entropy": best_test_ce,
        "stopped_early": stopped_early,
        "stop_reason": "max_steps",
    }

    pbar = tqdm(range(1, training.steps + 1), desc="single-task training", ncols=100)
    for step in pbar:
        model.train()

        if (
            training.grid_update_every > 0
            and step <= training.stop_grid_update_step
            and (step == 1 or (step - 1) % training.grid_update_every == 0)
        ):
            with torch.no_grad():
                model.update_grid(train_input)

        optimizer.zero_grad(set_to_none=True)
        logits = model(train_input)
        objective_loss = loss_fn(logits, train_label)
        objective_loss.backward()
        optimizer.step()

        train_ce = tensor_cross_entropy_loss(model, train_input, train_label)
        test_ce = tensor_cross_entropy_loss(model, test_input, test_label)
        test_ce_value = float(test_ce.cpu())
        if scheduler_step_mode == "metric":
            scheduler.step(test_ce_value)
        else:
            scheduler.step()

        train_ce_value = float(train_ce.cpu())
        train_loss = train_ce_value
        test_loss = test_ce_value
        current_lr = float(optimizer.param_groups[0]["lr"])

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_cross_entropy"].append(train_ce_value)
        results["test_cross_entropy"].append(test_ce_value)
        results["train_objective_loss"].append(float(objective_loss.detach().cpu()))
        results["learning_rate"].append(current_lr)

        if test_ce_value < best_test_ce - training.early_stopping_min_delta:
            best_test_ce = test_ce_value
            best_state = clone_state_dict(model)
            best_step = step
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        pbar.set_description(
            "| train_loss: %.3f | test_loss: %.3f | lr: %.2e | best_step: %d | "
            % (train_loss, test_loss, current_lr, best_step)
        )

        if steps_without_improvement >= training.early_stopping_patience:
            stopped_early = True
            results["stop_reason"] = "validation_loss_plateau"
            break

    model.load_state_dict(best_state)
    results["best_step"] = best_step
    results["best_test_cross_entropy"] = best_test_ce
    results["stopped_early"] = stopped_early
    results["trained_steps"] = len(results["train_loss"])
    results["final_learning_rate"] = (
        results["learning_rate"][-1] if results["learning_rate"] else training.learning_rate
    )
    return model, results


def clear_output_dir(output_dir: Path, enabled: bool) -> None:
    """Remove plots from a previous baseline run if requested."""
    if output_dir.exists() and enabled:
        clear_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_loss_curve(results: dict[str, list], output_file: Path) -> None:
    """Plot train and test cross-entropy loss from the manual PyTorch loop."""
    train_loss = np.asarray(results["train_loss"], dtype=float)
    test_loss = np.asarray(results["test_loss"], dtype=float)
    steps = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.plot(steps, train_loss, linewidth=2, label="train")
    ax.plot(steps, test_loss, linewidth=2, linestyle="--", label="test")
    best_step = int(results.get("best_step", 0))
    if best_step > 0:
        ax.axvline(best_step, color="#b85042", linestyle=":", label=f"best step {best_step}")
    ax.set_title("Single-task KAN loss")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("CrossEntropyLoss")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def plot_class_test_metrics(class_test_frame: pd.DataFrame, output_file: Path) -> None:
    """Plot per-class test loss and accuracy for the baseline model."""
    if class_test_frame.empty:
        return

    x = np.arange(len(class_test_frame))
    class_names = class_test_frame["test_class"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].bar(x, class_test_frame["test_loss"], color="#5b7c99")
    axes[0].set_title("Test loss by class")
    axes[0].set_ylabel("CrossEntropyLoss")

    axes[1].bar(x, class_test_frame["test_accuracy"], color="#2a9d8f")
    axes[1].set_title("Test accuracy by class")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)

    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def model_quality_verdict(test_accuracy: float, test_loss: float) -> tuple[str, str]:
    """Return a short qualitative verdict based on accuracy and loss."""
    if test_accuracy >= 0.97 and test_loss <= 0.25:
        return "good", "The model is strong: high test accuracy and low confidence penalty."
    if test_accuracy >= 0.93 and test_loss <= 0.5625:
        return "acceptable", "The model is usable, but confidence or a few classes may still need attention."
    return "weak", "The model needs improvement: inspect the confusion matrix and high-loss examples."


def build_run_config(
    config: BaselineConfig,
    input_names: list[str],
    shape_names: list[str],
    device: torch.device,
    standardization_mean: np.ndarray,
    standardization_std: np.ndarray,
    verdict: str,
) -> dict[str, object]:
    """Collect the parameters needed to reproduce this baseline run."""
    return {
        "experiment": "single_task_conic_baseline",
        "config_path": str(config.config_path),
        "data_path": str(config.data.csv_path),
        "device": str(device),
        "input_names": input_names,
        "shape_names": shape_names,
        "class_to_id": {shape_name: class_id for class_id, shape_name in enumerate(shape_names)},
        "model": {
            "width": [len(input_names), config.model.hidden, len(shape_names)],
            "hidden": config.model.hidden,
            "grid": config.model.grid,
            "spline_order": config.model.spline_order,
            "seed": config.split.seed,
        },
        "training": vars(config.training),
        "standardization": {
            "enabled": config.standardize,
            "mean": standardization_mean.reshape(-1).astype(float).tolist(),
            "std": standardization_std.reshape(-1).astype(float).tolist(),
        },
        "outputs": {
            "plot_dir": None if config.output.plot_dir is None else str(config.output.plot_dir),
            "metrics_csv": None if config.output.metrics_path is None else str(config.output.metrics_path),
            "class_tests_csv": None if config.output.class_tests_path is None else str(config.output.class_tests_path),
            "high_loss_csv": None if config.output.high_loss_path is None else str(config.output.high_loss_path),
            "run_config_json": None if config.output.run_config_path is None else str(config.output.run_config_path),
            "final_model": None if config.output.final_model_path is None else str(config.output.final_model_path),
        },
        "quality_verdict": verdict,
    }


def save_final_model(model: KAN, output_path: Path, run_config: dict[str, object]) -> None:
    """Save model weights and configuration together."""
    torch.save({"state_dict": model.state_dict(), "run_config": run_config}, output_path)


def run_baseline(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    """Train, evaluate, plot, and save the all-classes KAN baseline."""
    config = load_baseline_config(config_path)
    torch.manual_seed(config.split.seed)
    device = detect_device(config.model.device)
    output_dir = config.output.plot_dir
    if output_dir is None:
        raise ValueError("output.plot_dir is required for the baseline experiment.")
    clear_output_dir(output_dir, config.plots.clear_old_plots)

    input_names = load_feature_names(config.data.csv_path, config.data.target_column)
    raw_features, labels, shape_names = load_conic_csv(config.data.csv_path, config.data.target_column)
    train_indices, test_indices, _, test_by_class = stratified_split(
        labels=labels,
        test_ratio=config.split.test_ratio,
        seed=config.split.seed,
    )

    if config.standardize:
        standardization_mean, standardization_std = standardization_stats(raw_features, train_indices)
        features = ((raw_features - standardization_mean) / standardization_std).astype(np.float32)
    else:
        standardization_mean = np.zeros((1, raw_features.shape[1]), dtype=np.float32)
        standardization_std = np.ones((1, raw_features.shape[1]), dtype=np.float32)
        features = raw_features

    dataset = make_kan_dataset(features, labels, train_indices, test_indices, device)
    model = build_model(
        input_dim=features.shape[1],
        output_dim=len(shape_names),
        config=config.model,
        seed=config.split.seed,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Classes: {dict(enumerate(shape_names))}")
    print(f"Training examples: {len(train_indices)}, test examples: {len(test_indices)}")
    print(
        f"Model: width={[features.shape[1], config.model.hidden, len(shape_names)]}, "
        f"grid={config.model.grid}, k={config.model.spline_order}"
    )
    print(
        f"Optimizer: {config.training.optimizer}, lr={config.training.learning_rate}, "
        f"scheduler={config.training.lr_scheduler}, weight_decay={config.training.weight_decay}, "
        f"label_smoothing={config.training.label_smoothing}"
    )

    model, results = train_with_early_stopping(model, dataset, config)

    train_accuracy = accuracy(model, features, labels, train_indices, device)
    test_accuracy = accuracy(model, features, labels, test_indices, device)
    train_cross_entropy = cross_entropy_loss(model, features, labels, train_indices, device)
    test_cross_entropy = cross_entropy_loss(model, features, labels, test_indices, device)
    final_train_loss = cross_entropy_loss(model, features, labels, train_indices, device)
    final_test_loss = cross_entropy_loss(model, features, labels, test_indices, device)
    verdict, verdict_reason = model_quality_verdict(test_accuracy, final_test_loss)

    metrics = pd.DataFrame(
        [
            {
                "model": "single_task_kan",
                "train_examples": int(len(train_indices)),
                "test_examples": int(len(test_indices)),
                "hidden": config.model.hidden,
                "grid": config.model.grid,
                "spline_order": config.model.spline_order,
                "steps": config.training.steps,
                "trained_steps": int(results["trained_steps"]),
                "best_step": int(results["best_step"]),
                "stopped_early": bool(results["stopped_early"]),
                "stop_reason": str(results["stop_reason"]),
                "optimizer": config.training.optimizer,
                "lr_scheduler": config.training.lr_scheduler,
                "learning_rate": config.training.learning_rate,
                "final_learning_rate": float(results["final_learning_rate"]),
                "weight_decay": config.training.weight_decay,
                "label_smoothing": config.training.label_smoothing,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_cross_entropy": train_cross_entropy,
                "test_cross_entropy": test_cross_entropy,
                "final_train_loss": final_train_loss,
                "final_test_loss": final_test_loss,
                "best_train_loss": float(np.min(np.asarray(results["train_loss"], dtype=float))),
                "best_test_loss": float(np.min(np.asarray(results["test_loss"], dtype=float))),
                "best_test_cross_entropy": float(results["best_test_cross_entropy"]),
                "quality_verdict": verdict,
                "quality_reason": verdict_reason,
            }
        ]
    )

    if config.output.metrics_path is not None:
        metrics.to_csv(config.output.metrics_path, index=False)

    class_rows = []
    for class_id, class_name in enumerate(shape_names):
        class_indices = test_by_class[class_id]
        class_rows.append(
            {
                "test_class": class_name,
                "test_examples": int(len(class_indices)),
                "test_loss": cross_entropy_loss(model, features, labels, class_indices, device),
                "test_accuracy": accuracy(model, features, labels, class_indices, device),
            }
        )

    class_tests = pd.DataFrame(class_rows)
    if config.output.class_tests_path is not None:
        class_tests.to_csv(config.output.class_tests_path, index=False)

    loss_history = [
        {
            "stage": 1,
            "task": "single_task_all_classes",
            "train_loss": np.asarray(results["train_loss"], dtype=float),
            "test_loss": np.asarray(results["test_loss"], dtype=float),
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
        }
    ]
    save_final_loss_summary(loss_history, output_dir)
    plot_loss_curve(results, output_dir / "01_loss_curve.png")
    plot_class_test_metrics(class_tests, output_dir / "02_class_test_metrics.png")

    confusion = compute_confusion_matrix(
        model=model,
        features=features,
        labels=labels,
        indices=test_indices,
        device=device,
        class_ids=list(range(len(shape_names))),
    )
    confusion_csv, confusion_image = save_confusion_matrix(
        matrix=confusion,
        class_names=shape_names,
        output_dir=output_dir,
        stage=1,
        task_name="single_task_all_classes",
    )

    plot_prediction_examples(
        model=model,
        model_features=features,
        plot_features=raw_features,
        labels=labels,
        test_by_class=test_by_class,
        shape_names=shape_names,
        device=device,
        output_file=output_dir / "03_prediction_examples.png",
        examples_per_class=config.plots.prediction_examples,
        seed=config.split.seed,
        title="Single-task KAN predictions",
    )

    high_loss_frame = high_loss_examples(
        model=model,
        features=features,
        labels=labels,
        indices=test_indices,
        shape_names=shape_names,
        device=device,
        n_examples=config.plots.high_loss_examples,
    )
    if config.output.high_loss_path is not None:
        high_loss_frame.to_csv(config.output.high_loss_path, index=False)
    plot_high_loss_examples(high_loss_frame, raw_features, output_dir / "04_high_loss_examples.png")

    edge_image_path, edge_table_path = plot_kan_edge_functions(
        model=model,
        sample_input=dataset["train_input"],
        output_dir=output_dir,
        image_name="05_final_edge_functions.png",
        table_name="05_final_edge_functions.csv",
        input_names=input_names,
        output_names=shape_names,
        title="Single-task KAN edge functions",
    )

    run_config = build_run_config(
        config=config,
        input_names=input_names,
        shape_names=shape_names,
        device=device,
        standardization_mean=standardization_mean,
        standardization_std=standardization_std,
        verdict=verdict,
    )
    run_config["final_results"] = metrics.iloc[0].to_dict()
    if config.output.final_model_path is not None:
        save_final_model(model, config.output.final_model_path, run_config)
    if config.output.run_config_path is not None:
        write_json(config.output.run_config_path, run_config)

    print()
    print("Single-task baseline results:")
    print(f"Train accuracy: {train_accuracy:.3f}, test accuracy: {test_accuracy:.3f}")
    print(f"Train loss: {final_train_loss:.3f}, test loss: {final_test_loss:.3f}")
    print(
        f"Best step: {int(results['best_step'])}, "
        f"trained steps: {int(results['trained_steps'])}, "
        f"stopped early: {bool(results['stopped_early'])}"
    )
    print(f"Verdict: {verdict.upper()} - {verdict_reason}")
    if config.output.metrics_path is not None:
        print(f"Metrics saved to: {config.output.metrics_path}")
    if config.output.class_tests_path is not None:
        print(f"Class tests saved to: {config.output.class_tests_path}")
    print(f"Confusion matrix saved to: {confusion_image}")
    print(f"Confusion matrix CSV saved to: {confusion_csv}")
    if config.output.high_loss_path is not None:
        print(f"High-loss examples saved to: {config.output.high_loss_path}")
    print(f"Final edge-function plot saved to: {edge_image_path}")
    print(f"Final edge-function table saved to: {edge_table_path}")
    if config.output.final_model_path is not None:
        print(f"Final model checkpoint saved to: {config.output.final_model_path}")
    if config.output.run_config_path is not None:
        print(f"Run configuration saved to: {config.output.run_config_path}")
    print(f"Plots saved in: {output_dir}")

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the single-task conic KAN baseline.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the TOML config file. Default: {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args(argv)
    run_baseline(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
