"""Pruning-first experiment for the conic-section dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kan import KAN

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kan_models.common.runtime import configure_matplotlib

configure_matplotlib()

from kan_models.models.conic.config import PruningConfig, load_pruning_config
from kan_models.models.conic.data import load_conic_csv, load_feature_names, make_kan_dataset, standardize_from_train, stratified_split
from kan_models.models.conic.modeling import accuracy, build_model, compute_confusion_matrix, predict_classes
from kan_models.models.conic.plotting import (
    high_loss_examples,
    plot_high_loss_examples,
    plot_kan_edge_functions,
    plot_prediction_examples,
)
from kan_models.common.runtime import detect_device
from kan_models.common.shared import clear_matching_files, copy_kan_model, hidden_units


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "conic" / "pruning.toml"


@torch.no_grad()
def cross_entropy_loss(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    """Compute CrossEntropyLoss on a fixed group of examples."""
    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    targets = torch.tensor(labels[indices], device=device)
    logits = model(inputs)
    return float(torch.nn.functional.cross_entropy(logits, targets).cpu().item())


def clear_old_plots(output_dir: Path) -> None:
    """Remove old pruning plots before starting a new experiment."""
    clear_matching_files(output_dir, "*.png", "*.txt", "*_assets")


def evaluate_model(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray | None,
    device: torch.device,
    stage: str,
    train_steps: int,
    selected_nodes: list[int] | None = None,
    include_test: bool = False,
) -> dict[str, float | int | str]:
    """Collect accuracy, loss, and size information for one experiment stage."""
    hidden = hidden_units(model)
    row = {
        "stage": stage,
        "hidden_units": hidden,
        "train_steps": train_steps,
        "cost_proxy": hidden * train_steps,
        "selected_nodes": "" if selected_nodes is None else " ".join(map(str, selected_nodes)),
        "train_accuracy": accuracy(model, features, labels, train_indices, device),
        "train_loss": cross_entropy_loss(model, features, labels, train_indices, device),
        "test_accuracy": np.nan,
        "test_loss": np.nan,
    }

    if include_test and test_indices is not None:
        row["test_accuracy"] = accuracy(model, features, labels, test_indices, device)
        row["test_loss"] = cross_entropy_loss(model, features, labels, test_indices, device)
    return row


def top_nodes_from_scores(scores: np.ndarray, n_nodes: int) -> list[int]:
    """Select the most important hidden nodes from the probe scores."""
    n_nodes = min(n_nodes, len(scores))
    selected = np.argsort(scores)[-n_nodes:]
    return np.sort(selected).astype(int).tolist()


def score_probe_nodes(model: KAN, dataset: dict[str, torch.Tensor]) -> np.ndarray:
    """Use KAN attribution to score hidden nodes in the probe model."""
    model.get_act(dataset["train_input"])
    model.attribute(plot=False)
    return model.node_scores[1].detach().cpu().numpy()


def plot_loss_curve(results: dict[str, list], output_dir: Path, filename: str, title: str) -> None:
    """Save the training loss curve for a single pruning stage."""
    train_loss = np.asarray(results["train_loss"], dtype=float)
    steps = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(steps, train_loss, label="train", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Training loss returned by KAN.fit")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(output_dir / filename, dpi=170)
    plt.close(fig)


def plot_all_loss_curves(loss_runs: dict[str, dict[str, list]], output_file: Path) -> None:
    """Save all pruning-stage training losses in one comparison plot."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(loss_runs))))

    for color, (name, results) in zip(colors, loss_runs.items()):
        train_loss = np.asarray(results["train_loss"], dtype=float)
        steps = np.arange(1, len(train_loss) + 1)
        ax.plot(steps, train_loss, marker="o", markersize=3.5, linewidth=2, label=name, color=color)

    ax.set_title("Training losses for all pruning stages")
    ax.set_xlabel("Optimization step inside each stage")
    ax.set_ylabel("Training loss returned by KAN.fit")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def plot_node_scores(scores: np.ndarray, selected_nodes: list[int], output_file: Path) -> None:
    """Plot probe node importance and highlight the first nodes kept."""
    selected = set(selected_nodes)
    colors = ["#2a9d8f" if idx in selected else "#b7b7b7" for idx in range(len(scores))]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(np.arange(len(scores)), scores, color=colors)
    ax.set_title("Probe node importance")
    ax.set_xlabel("Hidden node id in the probe model")
    ax.set_ylabel("Attribution score")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.text(
        0.5,
        0.93,
        f"kept nodes: {selected_nodes}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def plot_growth_progress(metrics_frame: pd.DataFrame, output_file: Path) -> None:
    """Plot how accuracy, loss, model size, and cost change across stages."""
    labels = metrics_frame["stage"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), constrained_layout=True)
    axes[0, 0].plot(x, metrics_frame["train_accuracy"], marker="o", linewidth=2)
    axes[0, 0].set_title("Train accuracy")
    axes[0, 0].set_ylim(0, 1)

    final_test = metrics_frame.dropna(subset=["test_accuracy", "test_loss"])
    if not final_test.empty:
        final_row = final_test.iloc[-1]
        final_index = int(final_test.index[-1])
        axes[0, 0].scatter(final_index, final_row["test_accuracy"], s=90, marker="D", color="#b85042", label="final test", zorder=3)
        axes[0, 0].legend()

    axes[0, 1].plot(x, metrics_frame["train_loss"], marker="o", linewidth=2, color="#2a9d8f")
    axes[0, 1].set_title("Train loss")
    if not final_test.empty:
        final_row = final_test.iloc[-1]
        final_index = int(final_test.index[-1])
        axes[0, 1].scatter(final_index, final_row["test_loss"], s=90, marker="D", color="#b85042", label="final test", zorder=3)
        axes[0, 1].legend()

    axes[1, 0].bar(x, metrics_frame["hidden_units"], color="#5b7c99")
    axes[1, 0].set_title("Hidden units used")

    cumulative_cost = metrics_frame["cost_proxy"].cumsum()
    axes[1, 1].plot(x, cumulative_cost, marker="o", linewidth=2, color="#7a5ab8")
    axes[1, 1].set_title("Cumulative cost proxy")
    axes[1, 1].set_ylabel("hidden units x training steps")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)

    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def plot_architecture_schedule(metrics_frame: pd.DataFrame, output_file: Path) -> None:
    """Show the hidden-unit schedule used during the pruning experiment."""
    labels = metrics_frame["stage"].tolist()
    hidden = metrics_frame["hidden_units"].to_numpy()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.step(x, hidden, where="mid", linewidth=2.5, color="#2f6fbb")
    ax.scatter(x, hidden, s=70, color="#2f6fbb")
    ax.set_title("Architecture schedule")
    ax.set_xlabel("Experiment stage")
    ax.set_ylabel("Hidden units")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    for xpos, value in zip(x, hidden):
        ax.text(xpos, value + 0.25, str(int(value)), ha="center", fontsize=10)

    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, shape_names: list[str], title: str, output_file: Path) -> None:
    """Save a labelled confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, pad=14)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(shape_names)))
    ax.set_yticks(np.arange(len(shape_names)))
    ax.set_xticklabels(shape_names, rotation=20, ha="right")
    ax.set_yticklabels(shape_names)

    max_value = max(int(matrix.max()), 1)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if value > max_value / 2 else "black",
            )

    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def train_phase(
    model: KAN,
    dataset: dict[str, torch.Tensor],
    steps: int,
    loss_fn: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    update_grid: bool = True,
) -> dict[str, list]:
    """Train one KAN phase with the shared pruning configuration."""
    return model.fit(
        dataset,
        opt=optimizer_name,
        steps=steps,
        lr=learning_rate,
        update_grid=update_grid,
        loss_fn=loss_fn,
        display_metrics=["train_loss"],
    )


def run_pruning(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    """Run the full probe-based pruning experiment and save its outputs."""
    config = load_pruning_config(config_path)
    torch.manual_seed(config.split.seed)
    device = detect_device(config.model.device)
    output_dir = config.output.plot_dir
    if output_dir is None:
        raise ValueError("output.plot_dir is required for the pruning experiment.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.plots.clear_old_plots:
        clear_old_plots(output_dir)

    input_names = load_feature_names(config.data.csv_path, config.data.target_column)
    raw_features, labels, shape_names = load_conic_csv(config.data.csv_path, config.data.target_column)
    train_indices, test_indices, _, test_by_class = stratified_split(
        labels=labels,
        test_ratio=config.split.test_ratio,
        seed=config.split.seed,
    )

    if config.training.standardize:
        features, _, _ = standardize_from_train(raw_features, train_indices)
    else:
        features = raw_features

    dataset = make_kan_dataset(features, labels, train_indices, train_indices, device)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Probe model: hidden={config.training.probe_hidden}, steps={config.training.probe_steps}")

    probe_model = build_model(
        input_dim=features.shape[1],
        output_dim=len(shape_names),
        config=config.model,
        seed=config.split.seed,
        device=device,
        hidden=config.training.probe_hidden,
    )

    probe_results = train_phase(
        probe_model,
        dataset,
        config.training.probe_steps,
        loss_fn,
        config.training.optimizer,
        config.training.learning_rate,
    )
    loss_runs = {"probe": probe_results}
    plot_loss_curve(probe_results, output_dir, "01_probe_loss.png", "Short probe loss before selecting nodes")

    node_scores = score_probe_nodes(probe_model, dataset)
    first_selected_nodes = top_nodes_from_scores(node_scores, config.training.start_hidden)
    plot_node_scores(node_scores, first_selected_nodes, output_dir / "02_probe_node_importance.png")

    records = [
        evaluate_model(
            probe_model,
            features,
            labels,
            train_indices,
            None,
            device,
            stage="probe",
            train_steps=config.training.probe_steps,
        )
    ]

    current_model = probe_model
    final_hidden = config.training.keep_hidden_schedule[-1]

    for target_hidden in config.training.keep_hidden_schedule:
        selected_nodes = top_nodes_from_scores(node_scores, target_hidden)
        print(f"Training pruned model with top {target_hidden} probe nodes: {selected_nodes}")

        probe_source = copy_kan_model(probe_model)
        probe_source.get_act(dataset["train_input"])
        current_model = probe_source.prune_node(active_neurons_id=[selected_nodes], log_history=False)
        current_model.auto_save = False
        current_model.get_act(dataset["train_input"])

        stage_name = f"keep_{target_hidden}"
        stage_results = train_phase(
            current_model,
            dataset,
            config.training.train_steps_per_model,
            loss_fn,
            config.training.optimizer,
            config.training.learning_rate,
            update_grid=True,
        )
        loss_runs[stage_name] = stage_results
        plot_loss_curve(stage_results, output_dir, f"loss_keep_{target_hidden:02d}.png", f"Training loss with top {target_hidden} probe nodes")

        records.append(
            evaluate_model(
                current_model,
                features,
                labels,
                train_indices,
                test_indices,
                device,
                stage=stage_name,
                train_steps=config.training.train_steps_per_model,
                selected_nodes=selected_nodes,
                include_test=target_hidden == final_hidden,
            )
        )

    metrics_frame = pd.DataFrame(records)
    if config.output.metrics_path is not None:
        metrics_frame.to_csv(config.output.metrics_path, index=False)

    plot_all_loss_curves(loss_runs, output_dir / "03_all_training_losses.png")
    plot_growth_progress(metrics_frame, output_dir / "04_prune_first_progress.png")
    plot_architecture_schedule(metrics_frame, output_dir / "05_architecture_schedule.png")
    plot_confusion(
        compute_confusion_matrix(current_model, features, labels, test_indices, device, list(range(len(shape_names)))),
        shape_names,
        "Confusion matrix after final growth",
        output_dir / "06_confusion_after_final_growth.png",
    )
    plot_prediction_examples(
        current_model,
        features,
        raw_features,
        labels,
        test_by_class,
        shape_names,
        device,
        output_dir / "07_predictions_after_final_growth.png",
        config.plots.prediction_examples,
        config.split.seed,
        "Predictions after final growth",
    )

    high_loss_frame = high_loss_examples(
        current_model,
        features,
        labels,
        test_indices,
        shape_names,
        device,
        config.plots.high_loss_examples,
    )
    if config.output.high_loss_path is not None:
        high_loss_frame.to_csv(config.output.high_loss_path, index=False)
    plot_high_loss_examples(high_loss_frame, raw_features, output_dir / "08_highest_loss_examples.png")

    edge_image_path = None
    edge_table_path = None
    if config.plots.plot_edge_functions:
        edge_image_path, edge_table_path = plot_kan_edge_functions(
            model=current_model,
            sample_input=dataset["train_input"],
            output_dir=output_dir,
            image_name="09_final_edge_functions.png",
            table_name="09_final_edge_functions.csv",
            input_names=input_names,
            output_names=shape_names,
            title="Final pruned KAN edge functions",
        )

    first = metrics_frame.iloc[0]
    final = metrics_frame.iloc[-1]
    print(f"Probe: train accuracy={first['train_accuracy']:.3f}, hidden={int(first['hidden_units'])}")
    print(
        f"Final pruned-first model: test accuracy={final['test_accuracy']:.3f}, "
        f"test loss={final['test_loss']:.3f}, hidden={int(final['hidden_units'])}"
    )
    print(f"Plots saved in: {output_dir}")
    if config.output.metrics_path is not None:
        print(f"Metrics saved in: {config.output.metrics_path}")
    if config.output.high_loss_path is not None:
        print(f"High-loss examples saved in: {config.output.high_loss_path}")
    if edge_image_path is not None and edge_table_path is not None:
        print(f"Final edge functions saved in: {edge_image_path}")
        print(f"Active edge table saved in: {edge_table_path}")

    return metrics_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pruning-first conic KAN experiment.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the TOML config file. Default: {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args(argv)
    run_pruning(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
