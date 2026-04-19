from __future__ import annotations

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

from continual_conic import (
    DATA_PATH,
    ROOT_DIR,
    accuracy,
    load_feature_names,
    load_conic_csv,
    make_kan_dataset,
    predict_classes,
    plot_kan_edge_functions,
    standardize_from_train,
    stratified_split,
)


# CONFIG
# This experiment does a short probe, keeps the most important nodes,
# and then trains progressively wider pruned models.
PROBE_HIDDEN = 24
PROBE_STEPS = 8
START_HIDDEN = 4
KEEP_HIDDEN_SCHEDULE = [4, 6, 8, 12, 16]

TRAIN_STEPS_PER_MODEL = 15

GRID = 3
SPLINE_ORDER = 3
SEED = 1
TEST_RATIO = 0.2
OPTIMIZER = "LBFGS"
LEARNING_RATE = 0.5
STANDARDIZE = True

PLOT_DIR = "pruning_plots"
PLOT_EXAMPLES = 2
PLOT_EDGE_FUNCTIONS = True
HIGH_LOSS_EXAMPLES = 12
METRICS_OUT = "pruning_metrics.csv"
HIGH_LOSS_OUT = "high_loss_examples.csv"
CLEAR_OLD_PLOTS = True


@torch.no_grad()
def cross_entropy_loss(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    """Compute CrossEntropyLoss on a fixed group of examples."""
    # This gives a simple numerical loss for a fixed split.
    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    targets = torch.tensor(labels[indices], device=device)
    logits = model(inputs)
    return float(torch.nn.functional.cross_entropy(logits, targets).cpu().item())


def hidden_units(model: KAN) -> int:
    """Count how many hidden units are active inside the KAN model."""
    # KAN stores widths as [sum_nodes, mult_nodes], so we count both parts.
    total = 0
    for layer_width in model.width[1:-1]:
        if isinstance(layer_width, list):
            total += int(sum(layer_width))
        else:
            total += int(layer_width)
    return total


def clear_old_plots(output_dir: Path) -> None:
    """Remove old pruning plots before starting a new experiment."""
    # This avoids mixing old pruning pictures with the new experiment.
    if not output_dir.exists():
        return

    for file_path in output_dir.glob("*.png"):
        file_path.unlink()

    for file_path in output_dir.glob("*.txt"):
        file_path.unlink()

    for folder_path in output_dir.glob("*_assets"):
        if folder_path.is_dir():
            shutil.rmtree(folder_path)


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
    # We save one row after every important point of the experiment.
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
    # We keep the strongest nodes, then sort their ids for stable pruning.
    n_nodes = min(n_nodes, len(scores))
    selected = np.argsort(scores)[-n_nodes:]
    return np.sort(selected).astype(int).tolist()


def score_probe_nodes(model: KAN, dataset: dict[str, torch.Tensor]) -> np.ndarray:
    """Use KAN attribution to score the hidden nodes of the probe model."""
    # The probe model gives attribution scores; we keep the highest scoring nodes.
    model.get_act(dataset["train_input"])
    model.attribute(plot=False)
    return model.node_scores[1].detach().cpu().numpy()


def confusion_matrix(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    n_classes: int,
) -> np.ndarray:
    """Build a confusion matrix from true and predicted class ids."""
    predictions = predict_classes(model, features, indices, device)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for true_class, predicted_class in zip(labels[indices], predictions):
        matrix[int(true_class), int(predicted_class)] += 1

    return matrix


def plot_loss_curve(results: dict[str, list], output_dir: Path, filename: str, title: str) -> None:
    """Save the training loss curve for a single pruning stage."""
    # This checks whether each training phase is still learning.
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
    # This keeps all the training losses in one picture for an easier comparison.
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(loss_runs))))

    for color, (name, results) in zip(colors, loss_runs.items()):
        train_loss = np.asarray(results["train_loss"], dtype=float)
        steps = np.arange(1, len(train_loss) + 1)
        ax.plot(
            steps,
            train_loss,
            marker="o",
            markersize=3.5,
            linewidth=2,
            label=name,
            color=color,
        )

    ax.set_title("Training losses for all pruning stages")
    ax.set_xlabel("Optimization step inside each stage")
    ax.set_ylabel("Training loss returned by KAN.fit")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def plot_node_scores(scores: np.ndarray, selected_nodes: list[int], output_file: Path) -> None:
    """Plot probe node importance and highlight the first nodes kept."""
    # This is the key plot: it shows which nodes survive the initial pruning.
    selected = set(selected_nodes)
    colors = ["#2a9d8f" if idx in selected else "#b7b7b7" for idx in range(len(scores))]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(np.arange(len(scores)), scores, color=colors)
    ax.set_title("Probe node importance")
    ax.set_xlabel("Hidden node id in the probe model")
    ax.set_ylabel("Attribution score")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)

    note = f"kept nodes: {selected_nodes}"
    ax.text(
        0.5,
        0.93,
        note,
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
    # This plot shows the whole idea: start from important nodes, then allow more nodes.
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
        axes[0, 0].scatter(
            final_index,
            final_row["test_accuracy"],
            s=90,
            marker="D",
            color="#b85042",
            label="final test",
            zorder=3,
        )
        axes[0, 0].legend()

    axes[0, 1].plot(x, metrics_frame["train_loss"], marker="o", linewidth=2, color="#2a9d8f")
    axes[0, 1].set_title("Train loss")
    if not final_test.empty:
        final_row = final_test.iloc[-1]
        final_index = int(final_test.index[-1])
        axes[0, 1].scatter(
            final_index,
            final_row["test_loss"],
            s=90,
            marker="D",
            color="#b85042",
            label="final test",
            zorder=3,
        )
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
    # This makes the pruning-first schedule visible at a glance.
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


def plot_confusion(
    matrix: np.ndarray,
    shape_names: list[str],
    title: str,
    output_file: Path,
) -> None:
    """Save a labelled confusion matrix plot."""
    # Rows are true classes, columns are predicted classes.
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


def plot_prediction_examples(
    model: KAN,
    model_features: np.ndarray,
    plot_features: np.ndarray,
    labels: np.ndarray,
    test_by_class: dict[int, np.ndarray],
    shape_names: list[str],
    device: torch.device,
    output_file: Path,
    examples_per_class: int,
    seed: int,
    title: str,
) -> None:
    """Plot example predictions from the final pruned model."""
    # A small example grid is easier to read than many crowded subplots.
    rng = np.random.default_rng(seed)
    selected_indices = []

    for class_id in range(len(shape_names)):
        candidates = test_by_class[class_id]
        sample_size = min(examples_per_class, len(candidates))
        selected_indices.extend(
            rng.choice(candidates, size=sample_size, replace=False).tolist()
        )

    selected_indices = np.array(selected_indices, dtype=np.int64)
    predictions = predict_classes(model, model_features, selected_indices, device)

    rows = len(shape_names)
    columns = examples_per_class
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.2 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    colors = {
        "parabola": "#2f6fbb",
        "ellipse": "#2a9d8f",
        "hyperbola": "#d16666",
        "circle": "#7a5ab8",
    }

    position = 0
    for row, class_id in enumerate(range(len(shape_names))):
        class_samples = selected_indices[labels[selected_indices] == class_id]

        for col in range(columns):
            ax = axes[row][col]
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.35, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.45, alpha=0.45)
            ax.axvline(0, color="black", linewidth=0.45, alpha=0.45)
            ax.tick_params(labelsize=7)

            if col >= len(class_samples):
                ax.axis("off")
                continue

            sample_index = class_samples[col]
            predicted_class = int(predictions[position])
            position += 1

            coords = plot_features[sample_index].reshape(-1, 2)
            true_name = shape_names[class_id]
            predicted_name = shape_names[predicted_class]
            correct = predicted_class == class_id

            ax.plot(
                coords[:, 0],
                coords[:, 1],
                marker="o",
                linewidth=1.7,
                color=colors.get(predicted_name, "#444444"),
            )
            ax.set_title(
                f"T: {true_name}\nP: {predicted_name}",
                fontsize=9,
                pad=8,
                color="#1f7a3f" if correct else "#a83232",
            )

    fig.suptitle(title, fontsize=13)
    fig.savefig(output_file, dpi=170)
    plt.close(fig)


@torch.no_grad()
def high_loss_examples(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    shape_names: list[str],
    device: torch.device,
    n_examples: int,
) -> pd.DataFrame:
    """Find the test examples with the largest individual loss values."""
    # This finds the test examples that contribute most to CrossEntropyLoss.
    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    targets = torch.tensor(labels[indices], device=device)
    logits = model(inputs)
    losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    probabilities = torch.softmax(logits, dim=1)
    predicted = probabilities.argmax(dim=1)
    true_probability = probabilities[torch.arange(len(targets), device=device), targets]
    predicted_probability = probabilities[torch.arange(len(targets), device=device), predicted]

    n_examples = min(n_examples, len(indices))
    top_positions = torch.argsort(losses, descending=True)[:n_examples].cpu().numpy()

    rows = []
    for position in top_positions:
        true_id = int(targets[position].cpu().item())
        pred_id = int(predicted[position].cpu().item())
        rows.append(
            {
                "dataset_index": int(indices[position]),
                "true_label": shape_names[true_id],
                "predicted_label": shape_names[pred_id],
                "is_correct": bool(true_id == pred_id),
                "loss": float(losses[position].cpu().item()),
                "true_probability": float(true_probability[position].cpu().item()),
                "predicted_probability": float(predicted_probability[position].cpu().item()),
            }
        )

    return pd.DataFrame(rows)


def plot_high_loss_examples(
    high_loss_frame: pd.DataFrame,
    plot_features: np.ndarray,
    output_file: Path,
) -> None:
    """Plot the examples that contribute most to the final test loss."""
    # This draws the examples with the largest test loss.
    if high_loss_frame.empty:
        return

    n_examples = len(high_loss_frame)
    columns = min(4, n_examples)
    rows = int(np.ceil(n_examples / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.1 * columns, 3.4 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    colors = {
        "parabola": "#2f6fbb",
        "ellipse": "#2a9d8f",
        "hyperbola": "#d16666",
        "circle": "#7a5ab8",
    }

    for ax in axes.ravel():
        ax.axis("off")

    for position, (_, row) in enumerate(high_loss_frame.iterrows()):
        ax = axes[position // columns][position % columns]
        ax.axis("on")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.35, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.45, alpha=0.45)
        ax.axvline(0, color="black", linewidth=0.45, alpha=0.45)
        ax.tick_params(labelsize=7)

        coords = plot_features[int(row["dataset_index"])].reshape(-1, 2)
        predicted_label = row["predicted_label"]
        title_color = "#1f7a3f" if row["is_correct"] else "#a83232"

        ax.plot(
            coords[:, 0],
            coords[:, 1],
            marker="o",
            linewidth=1.7,
            color=colors.get(predicted_label, "#444444"),
        )
        ax.set_title(
            f"T: {row['true_label']}\nP: {predicted_label}\n"
            f"loss={row['loss']:.2f}, p_true={row['true_probability']:.3f}",
            fontsize=8.5,
            pad=8,
            color=title_color,
        )

    fig.suptitle("Highest-loss test examples", fontsize=13)
    fig.savefig(output_file, dpi=170)
    plt.close(fig)


def train_phase(
    model: KAN,
    dataset: dict[str, torch.Tensor],
    steps: int,
    loss_fn: torch.nn.Module,
    update_grid: bool = True,
) -> dict[str, list]:
    """Train one KAN phase with the shared pruning configuration."""
    # A small wrapper keeps the main experiment easier to read.
    return model.fit(
        dataset,
        opt=OPTIMIZER,
        steps=steps,
        lr=LEARNING_RATE,
        update_grid=update_grid,
        loss_fn=loss_fn,
        display_metrics=["train_loss"],
    )


def copy_kan_model(model: KAN) -> KAN:
    """Copy a KAN model while avoiding cached tensors that break deepcopy."""
    # KAN has its own copy method; it is safer than copy.deepcopy for torch tensors.
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


def run_pruning() -> pd.DataFrame:
    """Run the full probe-based pruning experiment and save its outputs."""
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ROOT_DIR / PLOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if CLEAR_OLD_PLOTS:
        clear_old_plots(output_dir)

    input_names = load_feature_names(DATA_PATH)
    raw_features, labels, shape_names = load_conic_csv(DATA_PATH)
    train_indices, test_indices, _, test_by_class = stratified_split(
        labels=labels,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    if STANDARDIZE:
        features = standardize_from_train(raw_features, train_indices)
    else:
        features = raw_features

    # KAN.fit expects a test split in the dictionary.
    # Here we use the training split twice, so the real test set is untouched until the end.
    dataset = make_kan_dataset(features, labels, train_indices, train_indices, device)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Probe model: hidden={PROBE_HIDDEN}, steps={PROBE_STEPS}")

    probe_model = KAN(
        width=[features.shape[1], PROBE_HIDDEN, len(shape_names)],
        grid=GRID,
        k=SPLINE_ORDER,
        seed=SEED,
        auto_save=False,
        device=device,
    )

    probe_results = train_phase(probe_model, dataset, PROBE_STEPS, loss_fn)
    loss_runs = {"probe": probe_results}
    plot_loss_curve(
        probe_results,
        output_dir,
        "01_probe_loss.png",
        "Short probe loss before selecting nodes",
    )

    node_scores = score_probe_nodes(probe_model, dataset)
    first_selected_nodes = top_nodes_from_scores(node_scores, START_HIDDEN)
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
            train_steps=PROBE_STEPS,
        ),
    ]

    current_model = probe_model
    final_hidden = KEEP_HIDDEN_SCHEDULE[-1]

    for target_hidden in KEEP_HIDDEN_SCHEDULE:
        selected_nodes = top_nodes_from_scores(node_scores, target_hidden)
        print(f"Training pruned model with top {target_hidden} probe nodes: {selected_nodes}")

        probe_source = copy_kan_model(probe_model)
        probe_source.get_act(dataset["train_input"])
        current_model = probe_source.prune_node(
            active_neurons_id=[selected_nodes],
            log_history=False,
        )
        current_model.auto_save = False
        current_model.get_act(dataset["train_input"])

        stage_name = f"keep_{target_hidden}"
        stage_results = train_phase(
            current_model,
            dataset,
            TRAIN_STEPS_PER_MODEL,
            loss_fn,
            update_grid=True,
        )
        loss_runs[stage_name] = stage_results
        plot_loss_curve(
            stage_results,
            output_dir,
            f"loss_keep_{target_hidden:02d}.png",
            f"Training loss with top {target_hidden} probe nodes",
        )

        records.append(
            evaluate_model(
                current_model,
                features,
                labels,
                train_indices,
                test_indices,
                device,
                stage=stage_name,
                train_steps=TRAIN_STEPS_PER_MODEL,
                selected_nodes=selected_nodes,
                include_test=target_hidden == final_hidden,
            )
        )

    metrics_frame = pd.DataFrame(records)
    metrics_path = ROOT_DIR / METRICS_OUT
    metrics_frame.to_csv(metrics_path, index=False)

    plot_all_loss_curves(loss_runs, output_dir / "03_all_training_losses.png")
    plot_growth_progress(metrics_frame, output_dir / "04_prune_first_progress.png")
    plot_architecture_schedule(metrics_frame, output_dir / "05_architecture_schedule.png")
    plot_confusion(
        confusion_matrix(current_model, features, labels, test_indices, device, len(shape_names)),
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
        PLOT_EXAMPLES,
        SEED,
        "Predictions after final growth",
    )

    high_loss_frame = high_loss_examples(
        current_model,
        features,
        labels,
        test_indices,
        shape_names,
        device,
        HIGH_LOSS_EXAMPLES,
    )
    high_loss_path = ROOT_DIR / HIGH_LOSS_OUT
    high_loss_frame.to_csv(high_loss_path, index=False)
    plot_high_loss_examples(
        high_loss_frame,
        raw_features,
        output_dir / "08_highest_loss_examples.png",
    )

    if PLOT_EDGE_FUNCTIONS:
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
    print(
        f"Probe: train accuracy={first['train_accuracy']:.3f}, "
        f"hidden={int(first['hidden_units'])}"
    )
    print(
        f"Final pruned-first model: test accuracy={final['test_accuracy']:.3f}, "
        f"test loss={final['test_loss']:.3f}, hidden={int(final['hidden_units'])}"
    )
    print(f"Plot salvati in: {output_dir}")
    print(f"Metriche salvate in: {metrics_path}")
    print(f"Esempi con loss alta salvati in: {high_loss_path}")
    if PLOT_EDGE_FUNCTIONS:
        print(f"Funzioni finali sugli archi salvate in: {edge_image_path}")
        print(f"Tabella archi attivi salvata in: {edge_table_path}")

    return metrics_frame


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "Questo script non accetta parametri da terminale: modifica il blocco CONFIG."
        )

    run_pruning()
