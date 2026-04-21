"""Single-task KAN baseline for the conic-section dataset.

This script trains one KAN on all classes at the same time. It is useful as a
baseline for the continual-learning and pruning experiments, because it tells us
how well the same model family performs when it does not have to learn tasks in
sequence.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kan import KAN
from tqdm import tqdm

from continual_conic import (
    DATA_PATH,
    ROOT_DIR,
    accuracy,
    compute_confusion_matrix,
    load_feature_names,
    load_conic_csv,
    make_kan_dataset,
    plot_kan_edge_functions,
    save_confusion_matrix,
    save_final_loss_summary,
    sqrt_cross_entropy_loss,
    standardization_stats,
    stratified_split,
)
from pruning_conic import (
    high_loss_examples,
    plot_high_loss_examples,
    plot_prediction_examples,
)


# CONFIG
# This is the plain all-classes baseline, not a continual-learning run.
HIDDEN = 6
GRID = 3
SPLINE_ORDER = 3
SEED = 1
TEST_RATIO = 0.2
OPTIMIZER = "AdamW"
LEARNING_RATE = 0.01
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
STEPS = 100
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
GRID_UPDATE_EVERY = 10
STOP_GRID_UPDATE_STEP = 30
STANDARDIZE = True

PLOT_DIR = "conic_plots"
PLOT_EXAMPLES = 2
HIGH_LOSS_EXAMPLES = 12
CLEAR_OLD_PLOTS = True

METRICS_OUT = "conic_metrics.csv"
CLASS_TESTS_OUT = "conic_class_tests.csv"
HIGH_LOSS_OUT = "conic_high_loss_examples.csv"
RUN_CONFIG_OUT = "conic_run_config.json"
FINAL_MODEL_OUT = "conic_model_final.pt"


@torch.no_grad()
def cross_entropy_loss(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    """Compute plain CrossEntropyLoss on a fixed split."""
    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    targets = torch.tensor(labels[indices], device=device)
    logits = model(inputs)
    return float(torch.nn.functional.cross_entropy(logits, targets).cpu().item())


@torch.no_grad()
def tensor_cross_entropy_loss(
    model: KAN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute plain CrossEntropyLoss on already-built tensors."""
    model.eval()
    logits = model(inputs)
    return torch.nn.functional.cross_entropy(logits, targets)


def clone_state_dict(model: KAN) -> dict[str, torch.Tensor]:
    """Clone the trainable state so early stopping can restore the best model."""
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def train_with_early_stopping(
    model: KAN,
    dataset: dict[str, torch.Tensor],
) -> tuple[KAN, dict[str, list | float | int | bool | str]]:
    """Train with AdamW, label smoothing, ReduceLROnPlateau, and strict stopping."""
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.get_params() if hasattr(model, "get_params") else model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

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

    pbar = tqdm(range(1, STEPS + 1), desc="single-task training", ncols=100)
    for step in pbar:
        model.train()

        # KAN grids can adapt to the data early on. After a few updates we stop
        # moving the grid and let AdamW refine the spline coefficients safely.
        if (
            GRID_UPDATE_EVERY > 0
            and step <= STOP_GRID_UPDATE_STEP
            and (step == 1 or (step - 1) % GRID_UPDATE_EVERY == 0)
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
        scheduler.step(float(test_ce.cpu()))

        train_ce_value = float(train_ce.cpu())
        test_ce_value = float(test_ce.cpu())
        train_loss = float(torch.sqrt(train_ce).cpu())
        test_loss = float(torch.sqrt(test_ce).cpu())
        current_lr = float(optimizer.param_groups[0]["lr"])

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_cross_entropy"].append(train_ce_value)
        results["test_cross_entropy"].append(test_ce_value)
        results["train_objective_loss"].append(float(objective_loss.detach().cpu()))
        results["learning_rate"].append(current_lr)

        if test_ce_value < best_test_ce - EARLY_STOPPING_MIN_DELTA:
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

        if steps_without_improvement >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            results["stop_reason"] = "validation_loss_plateau"
            break

    model.load_state_dict(best_state)
    results["best_step"] = best_step
    results["best_test_cross_entropy"] = best_test_ce
    results["stopped_early"] = stopped_early
    results["trained_steps"] = len(results["train_loss"])
    results["final_learning_rate"] = (
        results["learning_rate"][-1] if results["learning_rate"] else LEARNING_RATE
    )

    return model, results


def clear_output_dir(output_dir: Path) -> None:
    """Remove old plots from a previous baseline run."""
    if output_dir.exists() and CLEAR_OLD_PLOTS:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def plot_loss_curve(
    results: dict[str, list],
    output_file: Path,
) -> None:
    """Plot train and test loss from the manual PyTorch loop."""
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
    ax.set_ylabel("sqrt(CrossEntropyLoss)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def plot_class_test_metrics(
    class_test_frame: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot per-class test loss and accuracy for the baseline model."""
    if class_test_frame.empty:
        return

    x = np.arange(len(class_test_frame))
    class_names = class_test_frame["test_class"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].bar(x, class_test_frame["test_loss"], color="#5b7c99")
    axes[0].set_title("Test loss by class")
    axes[0].set_ylabel("sqrt(CrossEntropyLoss)")

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
    if test_accuracy >= 0.97 and test_loss <= 0.50:
        return (
            "good",
            "The model is strong: high test accuracy and low confidence penalty.",
        )

    if test_accuracy >= 0.93 and test_loss <= 0.75:
        return (
            "acceptable",
            "The model is usable, but confidence or a few classes may still need attention.",
        )

    return (
        "weak",
        "The model needs improvement: inspect the confusion matrix and high-loss examples.",
    )


def build_run_config(
    input_names: list[str],
    shape_names: list[str],
    device: torch.device,
    model_output_path: Path,
    standardization_mean: np.ndarray,
    standardization_std: np.ndarray,
    verdict: str,
) -> dict[str, object]:
    """Save the parameters needed to reproduce or reload this baseline."""
    return {
        "experiment": "single_task_conic_baseline",
        "data_path": str(DATA_PATH),
        "device": str(device),
        "input_names": input_names,
        "shape_names": shape_names,
        "class_to_id": {
            shape_name: class_id
            for class_id, shape_name in enumerate(shape_names)
        },
        "model_output": str(model_output_path),
        "model": {
            "width": [len(input_names), HIDDEN, len(shape_names)],
            "hidden": HIDDEN,
            "grid": GRID,
            "spline_order": SPLINE_ORDER,
            "seed": SEED,
        },
        "training": {
            "optimizer": OPTIMIZER,
            "learning_rate": LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "steps": STEPS,
            "label_smoothing": LABEL_SMOOTHING,
            "loss_function": "CrossEntropyLoss",
            "early_stopping_monitor": "test_cross_entropy",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "restore_best_model": True,
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_scheduler_factor": LR_SCHEDULER_FACTOR,
            "lr_scheduler_patience": LR_SCHEDULER_PATIENCE,
            "grid_update_every": GRID_UPDATE_EVERY,
            "stop_grid_update_step": STOP_GRID_UPDATE_STEP,
            "test_ratio": TEST_RATIO,
            "standardize": STANDARDIZE,
        },
        "standardization": {
            "mean": standardization_mean.reshape(-1).astype(float).tolist(),
            "std": standardization_std.reshape(-1).astype(float).tolist(),
        },
        "outputs": {
            "plot_dir": PLOT_DIR,
            "metrics_csv": METRICS_OUT,
            "class_tests_csv": CLASS_TESTS_OUT,
            "high_loss_csv": HIGH_LOSS_OUT,
            "run_config_json": RUN_CONFIG_OUT,
            "final_model": FINAL_MODEL_OUT,
        },
        "quality_verdict": verdict,
    }


def save_run_config(config: dict[str, object], output_path: Path) -> None:
    """Save the baseline configuration as JSON."""
    with output_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")


def save_final_model(
    model: KAN,
    output_path: Path,
    run_config: dict[str, object],
) -> None:
    """Save model weights and the baseline configuration together."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "run_config": run_config,
        },
        output_path,
    )


def run_conic_baseline() -> pd.DataFrame:
    """Train, evaluate, plot, and save the all-classes KAN baseline."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ROOT_DIR / PLOT_DIR
    clear_output_dir(output_dir)

    input_names = load_feature_names(DATA_PATH)
    raw_features, labels, shape_names = load_conic_csv(DATA_PATH)
    train_indices, test_indices, train_by_class, test_by_class = stratified_split(
        labels=labels,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    if STANDARDIZE:
        standardization_mean, standardization_std = standardization_stats(
            raw_features,
            train_indices,
        )
        features = ((raw_features - standardization_mean) / standardization_std).astype(
            np.float32
        )
    else:
        standardization_mean = np.zeros((1, raw_features.shape[1]), dtype=np.float32)
        standardization_std = np.ones((1, raw_features.shape[1]), dtype=np.float32)
        features = raw_features

    dataset = make_kan_dataset(
        features=features,
        labels=labels,
        train_indices=train_indices,
        test_indices=test_indices,
        device=device,
    )

    model = KAN(
        width=[features.shape[1], HIDDEN, len(shape_names)],
        grid=GRID,
        k=SPLINE_ORDER,
        seed=SEED,
        auto_save=False,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Classes: {dict(enumerate(shape_names))}")
    print(f"Training examples: {len(train_indices)}, test examples: {len(test_indices)}")
    print(
        f"Model: width={[features.shape[1], HIDDEN, len(shape_names)]}, "
        f"grid={GRID}, k={SPLINE_ORDER}"
    )
    print(
        f"Optimizer: {OPTIMIZER}, lr={LEARNING_RATE}, "
        f"weight_decay={WEIGHT_DECAY}, label_smoothing={LABEL_SMOOTHING}"
    )
    print(
        f"Early stopping: patience={EARLY_STOPPING_PATIENCE}, "
        f"min_delta={EARLY_STOPPING_MIN_DELTA}"
    )

    model, results = train_with_early_stopping(model, dataset)

    train_accuracy = accuracy(model, features, labels, train_indices, device)
    test_accuracy = accuracy(model, features, labels, test_indices, device)
    train_cross_entropy = cross_entropy_loss(model, features, labels, train_indices, device)
    test_cross_entropy = cross_entropy_loss(model, features, labels, test_indices, device)
    final_train_loss = sqrt_cross_entropy_loss(
        model,
        features,
        labels,
        train_indices,
        device,
    )
    final_test_loss = sqrt_cross_entropy_loss(
        model,
        features,
        labels,
        test_indices,
        device,
    )
    verdict, verdict_reason = model_quality_verdict(test_accuracy, final_test_loss)

    metrics = pd.DataFrame(
        [
            {
                "model": "single_task_kan",
                "train_examples": int(len(train_indices)),
                "test_examples": int(len(test_indices)),
                "hidden": HIDDEN,
                "grid": GRID,
                "spline_order": SPLINE_ORDER,
                "steps": STEPS,
                "trained_steps": int(results["trained_steps"]),
                "best_step": int(results["best_step"]),
                "stopped_early": bool(results["stopped_early"]),
                "stop_reason": str(results["stop_reason"]),
                "optimizer": OPTIMIZER,
                "learning_rate": LEARNING_RATE,
                "final_learning_rate": float(results["final_learning_rate"]),
                "weight_decay": WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
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
    metrics_path = ROOT_DIR / METRICS_OUT
    metrics.to_csv(metrics_path, index=False)

    class_rows = []
    for class_id, class_name in enumerate(shape_names):
        class_indices = test_by_class[class_id]
        class_rows.append(
            {
                "test_class": class_name,
                "test_examples": int(len(class_indices)),
                "test_loss": sqrt_cross_entropy_loss(
                    model,
                    features,
                    labels,
                    class_indices,
                    device,
                ),
                "test_accuracy": accuracy(
                    model,
                    features,
                    labels,
                    class_indices,
                    device,
                ),
            }
        )

    class_tests = pd.DataFrame(class_rows)
    class_tests_path = ROOT_DIR / CLASS_TESTS_OUT
    class_tests.to_csv(class_tests_path, index=False)

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
        examples_per_class=PLOT_EXAMPLES,
        seed=SEED,
        title="Single-task KAN predictions",
    )

    high_loss_frame = high_loss_examples(
        model=model,
        features=features,
        labels=labels,
        indices=test_indices,
        shape_names=shape_names,
        device=device,
        n_examples=HIGH_LOSS_EXAMPLES,
    )
    high_loss_path = ROOT_DIR / HIGH_LOSS_OUT
    high_loss_frame.to_csv(high_loss_path, index=False)
    plot_high_loss_examples(
        high_loss_frame,
        raw_features,
        output_dir / "04_high_loss_examples.png",
    )

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

    run_config_path = ROOT_DIR / RUN_CONFIG_OUT
    final_model_path = ROOT_DIR / FINAL_MODEL_OUT
    run_config = build_run_config(
        input_names=input_names,
        shape_names=shape_names,
        device=device,
        model_output_path=final_model_path,
        standardization_mean=standardization_mean,
        standardization_std=standardization_std,
        verdict=verdict,
    )
    run_config["final_results"] = metrics.iloc[0].to_dict()
    save_final_model(model, final_model_path, run_config)
    save_run_config(run_config, run_config_path)

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
    print(f"Metrics saved to: {metrics_path}")
    print(f"Class tests saved to: {class_tests_path}")
    print(f"Confusion matrix saved to: {confusion_image}")
    print(f"Confusion matrix CSV saved to: {confusion_csv}")
    print(f"High-loss examples saved to: {high_loss_path}")
    print(f"Final edge-function plot saved to: {edge_image_path}")
    print(f"Final edge-function table saved to: {edge_table_path}")
    print(f"Final model checkpoint saved to: {final_model_path}")
    print(f"Run configuration saved to: {run_config_path}")
    print(f"Plots saved in: {output_dir}")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "This script does not accept command-line arguments: edit the CONFIG block."
        )

    run_conic_baseline()
