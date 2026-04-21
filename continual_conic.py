"""Continual learning experiment for conic-section classification with a KAN.

The script trains one Kolmogorov-Arnold Network across several learning stages.
Each stage may contain new classes and a smaller amount of older data. To make
this setup stable, the code uses balanced mini-batches, masked class outputs,
label smoothing, and Learning without Forgetting.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "dataset" / "Conic-Section_dataset.csv"

# CONFIG
# Change these values here when you want to try a different experiment.
FREEZE_GRID_AFTER_FIRST = False

# Main model and optimization settings.
EPOCHS_PER_TASK = [60, 80, 100, 100]
HIDDEN = 8
GRID = 3
SPLINE_ORDER = 3
SEED = 1
TEST_RATIO = 0.2
OPTIMIZER = "Adam"
LEARNING_RATE = 0.01
MIN_LEARNING_RATE = 1e-4
BATCH_SIZE = 256

# Continual-learning tricks used during training.
MASK_FUTURE_CLASSES = True
LABEL_SMOOTHING = 0.05
USE_BALANCED_SAMPLER = True
USE_LWF = True
INITIAL_LAMBDA_KD = 0.5
LAMBDA_KD_DECAY = 0.8
DISTILLATION_TEMPERATURE = 2.0

# KAN grid and early-stopping settings.
GRID_UPDATE_EVERY = 10
STOP_GRID_UPDATE_EPOCH = 50
EARLY_STOPPING = True
EARLY_STOPPING_CHECK_EVERY = 5
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-4
ANTI_FORGETTING_EARLY_STOPPING = True
ANTI_FORGETTING_TOLERANCE = 0.02
ANTI_FORGETTING_PATIENCE = 5

# Output files and plot switches.
METRICS_OUT = "continual_metrics.csv"
CLASS_TESTS_OUT = "continual_class_tests.csv"
ACCURACY_CONFIG_OUT = "continual_accuracy_config.json"
STANDARDIZE = True
PLOT_PREDICTIONS = True
PLOT_DIR = "continual_plots"
PLOT_EXAMPLES = 2
PLOT_LOSS = True
PLOT_ERROR = True
PLOT_CLASS_TESTS = True
PLOT_EDGE_FUNCTIONS = True

# Each dictionary is one continual stage. The values say how many training
# samples of each class are available in that stage.
TRAINING_SCHEDULE: list[dict[str, int]] = [
    # Stage 1: learn the base problem using all available data for the first classes.
    {"parabola": 800, "ellipse": 800},
    # Stage 2: add hyperbola while keeping a larger old-class buffer.
    {"parabola": 300, "ellipse": 300, "hyperbola": 800},
    # Stage 3: add circle and tighten the old-class buffer evenly.
    {"parabola": 200, "ellipse": 200, "hyperbola": 200, "circle": 800},
    # Stage 4: final balanced consolidation over all classes.
    {"parabola": 200, "ellipse": 200, "hyperbola": 200, "circle": 200},
]


def load_feature_names(data_path: Path) -> list[str]:
    """Read the numerical input names from the CSV header."""
    columns = pd.read_csv(data_path, nrows=0).columns
    return [column for column in columns if column != "shape"]


def load_conic_csv(data_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load points from the CSV and encode shape names as integer labels."""
    # First we read the CSV and separate the numerical points from the shape names.
    df = pd.read_csv(data_path)

    features = df.drop(columns=["shape"]).to_numpy(dtype=np.float32)
    labels, shape_names = pd.factorize(df["shape"])

    return features, labels.astype(np.int64), list(shape_names)


def stratified_split(
    labels: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Split the dataset class by class, so train and test keep the same balance."""
    # We split each class separately, so every shape has its own train and test part.
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    train_by_class = {}
    test_by_class = {}

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)

        n_test = max(1, int(round(len(class_indices) * test_ratio)))
        class_test = np.sort(class_indices[:n_test])
        class_train = np.sort(class_indices[n_test:])

        train_by_class[int(class_id)] = class_train
        test_by_class[int(class_id)] = class_test
        train_indices.append(class_train)
        test_indices.append(class_test)

    return (
        np.concatenate(train_indices),
        np.concatenate(test_indices),
        train_by_class,
        test_by_class,
    )


def standardize_from_train(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Standardize the input coordinates using only training statistics."""
    # We compute mean and standard deviation only from the training data.
    mean = features[train_indices].mean(axis=0, keepdims=True)
    std = features[train_indices].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((features - mean) / std).astype(np.float32)


def make_kan_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create the dictionary format expected by pykan's built-in KAN.fit."""
    # The continual loop no longer uses KAN.fit, but pruning_conic.py still uses
    # this helper for its probe and pruned-model training experiments.
    return {
        "train_input": torch.tensor(features[train_indices], device=device),
        "train_label": torch.tensor(labels[train_indices], device=device),
        "test_input": torch.tensor(features[test_indices], device=device),
        "test_label": torch.tensor(labels[test_indices], device=device),
    }


class ConicStageDataset(Dataset):
    """Tiny Dataset wrapper for the rows selected in one continual stage."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        """Store only the rows selected for the current stage."""
        self.inputs = torch.tensor(features[indices], dtype=torch.float32)
        self.targets = torch.tensor(labels[indices], dtype=torch.long)

    def __len__(self) -> int:
        """Return the raw number of examples before the sampler repeats any row."""
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one input vector and its class id."""
        return self.inputs[index], self.targets[index]


def count_by_class_id(
    indices: np.ndarray,
    labels: np.ndarray,
    class_ids: list[int],
) -> dict[int, int]:
    """Count how many selected rows belong to each class id."""
    return {
        class_id: int(np.sum(labels[indices] == class_id))
        for class_id in class_ids
    }


def make_balanced_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    class_counts: dict[int, int],
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Create balanced mini-batches with a WeightedRandomSampler."""
    dataset = ConicStageDataset(features, labels, train_indices)

    if not USE_BALANCED_SAMPLER:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

    selected_labels = labels[train_indices]
    present_counts = {
        class_id: count
        for class_id, count in class_counts.items()
        if count > 0
    }

    if not present_counts:
        raise ValueError("Lo stage corrente non contiene esempi di training.")

    # Minority classes receive larger sampling probability. We do not also use
    # CrossEntropy class weights, because that would count the same correction twice.
    sample_weights = np.array(
        [1.0 / present_counts[int(label)] for label in selected_labels],
        dtype=np.float64,
    )

    # Each epoch is as large as a perfectly balanced version of the current stage.
    # Example: 800 new samples and 100 old samples become roughly 800 per class.
    samples_per_epoch = max(present_counts.values()) * len(present_counts)
    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
    )


def sample_indices(
    indices: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Take a reproducible random subset of indices without replacement."""
    # This chooses a smaller group of rows without taking the same row twice.
    if sample_size <= 0 or sample_size >= len(indices):
        return indices

    return np.sort(rng.choice(indices, size=sample_size, replace=False))


def count_by_class(indices: np.ndarray, labels: np.ndarray, shape_names: list[str]) -> dict[str, int]:
    """Count how many selected examples belong to each class."""
    return {
        shape_names[class_id]: int(np.sum(labels[indices] == class_id))
        for class_id in range(len(shape_names))
    }


def active_class_tensor(active_classes: list[int] | None, device: torch.device) -> torch.Tensor | None:
    """Convert active class ids into a tensor used to mask future outputs."""
    if active_classes is None:
        return None

    return torch.tensor(active_classes, dtype=torch.long, device=device)


def masked_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_classes: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only active logits and remap labels into that smaller output space."""
    if active_classes is None:
        return logits, labels.long()

    label_map = torch.full(
        (logits.shape[1],),
        -1,
        dtype=torch.long,
        device=logits.device,
    )
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
    """Compute CrossEntropyLoss with label smoothing, but without class weights."""
    masked_logits, mapped_labels = masked_logits_and_labels(
        logits,
        labels,
        active_classes,
    )
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
    """Apply LwF by matching current and teacher logits on old classes only."""
    if len(previous_classes) < 2:
        return current_logits.new_tensor(0.0)

    old_class_tensor = active_class_tensor(previous_classes, device)
    current_old_logits = current_logits.index_select(dim=1, index=old_class_tensor)
    teacher_old_logits = teacher_logits.index_select(dim=1, index=old_class_tensor)

    # Temperature softens the distributions, so the student can copy relative
    # preferences between old classes instead of only copying the hard winner.
    student_log_probs = torch.nn.functional.log_softmax(
        current_old_logits / temperature,
        dim=1,
    )
    teacher_probs = torch.nn.functional.softmax(
        teacher_old_logits / temperature,
        dim=1,
    )

    return torch.nn.functional.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature ** 2)


@torch.no_grad()
def sqrt_cross_entropy_loss(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    active_classes: list[int] | None = None,
) -> float:
    """Measure sqrt(CrossEntropyLoss) on selected examples for easier comparison."""
    if len(indices) == 0:
        return float("nan")

    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    targets = torch.tensor(labels[indices], device=device)
    logits = model(inputs)
    class_mask = active_class_tensor(active_classes, device)
    masked_logits, mapped_targets = masked_logits_and_labels(logits, targets, class_mask)

    # Evaluation loss is deliberately unweighted. It should describe the model's
    # confidence on the real selected examples, not on the oversampled batches.
    loss = torch.nn.functional.cross_entropy(masked_logits, mapped_targets)
    return float(torch.sqrt(loss).cpu().item())


@torch.no_grad()
def accuracy(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    active_classes: list[int] | None = None,
) -> float:
    """Measure classification accuracy on a selected group of examples."""
    if len(indices) == 0:
        return float("nan")

    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    logits = model(inputs)

    # During early stages we optionally hide future classes, because the model
    # should not be penalized for classes it has not learned yet.
    if active_classes is not None:
        active = active_class_tensor(active_classes, device)
        logits = logits.index_select(dim=1, index=active)
        predicted_positions = logits.argmax(dim=1)
        predictions = active[predicted_positions].cpu().numpy()
    else:
        predictions = logits.argmax(dim=1).cpu().numpy()
    return float(np.mean(predictions == labels[indices]))


@torch.no_grad()
def predict_classes(
    model: KAN,
    features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    active_classes: list[int] | None = None,
) -> np.ndarray:
    """Return the predicted class id for each selected example."""
    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    logits = model(inputs)
    if active_classes is not None:
        active = active_class_tensor(active_classes, device)
        logits = logits.index_select(dim=1, index=active)
        predicted_positions = logits.argmax(dim=1)
        return active[predicted_positions].cpu().numpy()

    return logits.argmax(dim=1).cpu().numpy()


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

    rng = np.random.default_rng(seed + stage)
    selected_indices = []

    for class_id in seen_classes:
        candidates = test_by_class[class_id]
        sample_size = min(examples_per_class, len(candidates))
        selected_indices.extend(
            rng.choice(candidates, size=sample_size, replace=False).tolist()
        )

    selected_indices = np.array(selected_indices, dtype=np.int64)

    # The model predicts on standardized features, while the drawing uses the
    # original coordinates so the conic shapes still look natural.
    predictions = predict_classes(
        model,
        model_features,
        selected_indices,
        device,
        active_classes=active_classes,
    )

    columns = max(1, examples_per_class)
    rows = len(seen_classes)
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

    plot_position = 0
    for row, class_id in enumerate(seen_classes):
        class_samples = selected_indices[labels[selected_indices] == class_id]

        for col in range(columns):
            ax = axes[row][col]
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.4, alpha=0.35)
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.tick_params(labelsize=7)

            if col >= len(class_samples):
                ax.axis("off")
                continue

            sample_index = class_samples[col]
            predicted_class = int(predictions[plot_position])
            plot_position += 1

            coords = plot_features[sample_index].reshape(-1, 2)
            true_name = shape_names[class_id]
            predicted_name = shape_names[predicted_class]
            correct = predicted_class == class_id

            ax.plot(
                coords[:, 0],
                coords[:, 1],
                marker="o",
                linewidth=1.5,
                color=colors.get(predicted_name, "#444444"),
            )
            ax.set_title(
                f"T: {true_name}\nP: {predicted_name}",
                fontsize=9,
                pad=8,
                color="#1f7a3f" if correct else "#a83232",
            )

    fig.suptitle(
        f"Stage {stage}: after training on {trained_task}",
        fontsize=13,
    )

    output_file = output_dir / f"stage_{stage:02d}_{trained_task}_predictions.png"
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def short_task_label(task_name: object) -> str:
    """Create compact task names so plot text does not overlap."""
    short_names = {
        "parabola": "par",
        "ellipse": "ell",
        "hyperbola": "hyp",
        "circle": "cir",
    }
    parts = str(task_name).split("+")
    return "+".join(short_names.get(part, part[:3]) for part in parts)


def stage_tick_label(stage: object, task_name: object) -> str:
    """Create a short two-line tick label for one continual stage."""
    return f"S{stage}\n{short_task_label(task_name)}"


def plot_continual_losses(
    loss_history: list[dict[str, object]],
    output_dir: Path,
) -> None:
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
    axes[0].set_ylabel("sqrt(CrossEntropyLoss)")
    axes[0].grid(True, linewidth=0.4, alpha=0.35)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    stages = [item["stage"] for item in loss_history]
    task_labels = [item["task"] for item in loss_history]
    final_train = [
        item.get("final_train_loss", item["train_loss"][-1])
        for item in loss_history
    ]
    final_test = [
        item.get("final_test_loss", item["test_loss"][-1])
        for item in loss_history
    ]
    x = np.arange(len(stages))
    width = 0.36

    axes[1].bar(x - width / 2, final_train, width, label="train")
    axes[1].bar(x + width / 2, final_test, width, label="test")
    axes[1].set_title("Final loss after each task")
    axes[1].set_xlabel("Continual task")
    axes[1].set_ylabel("Final loss")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [stage_tick_label(stage, task) for stage, task in zip(stages, task_labels)],
        fontsize=9,
    )
    axes[1].grid(True, axis="y", linewidth=0.4, alpha=0.35)
    axes[1].legend(loc="upper right", frameon=False)

    fig.savefig(output_dir / "continual_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_final_loss_summary(
    loss_history: list[dict[str, object]],
    output_dir: Path,
) -> tuple[Path, Path] | None:
    """Save final and best loss values inside the continual_plots folder."""
    if not loss_history:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in loss_history:
        train_loss = np.asarray(item["train_loss"], dtype=float)
        test_loss = np.asarray(item["test_loss"], dtype=float)

        rows.append(
            {
                "stage": int(item["stage"]),
                "task": str(item["task"]),
                "final_train_loss": float(
                    item.get("final_train_loss", train_loss[-1])
                ),
                "final_test_loss": float(
                    item.get("final_test_loss", test_loss[-1])
                ),
                "best_train_loss": float(np.min(train_loss)),
                "best_test_loss": float(np.min(test_loss)),
            }
        )

    csv_path = output_dir / "final_loss_summary.csv"
    json_path = output_dir / "final_loss_summary.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as loss_file:
        json.dump(
            {
                "final_stage_loss": rows[-1],
                "loss_by_stage": rows,
            },
            loss_file,
            indent=2,
        )
        loss_file.write("\n")

    return csv_path, json_path


def plot_continual_error(
    records: list[dict[str, object]],
    output_dir: Path,
) -> None:
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
    ax.set_xticklabels(
        [stage_tick_label(stage, task) for stage, task in zip(stages, task_labels)],
        fontsize=9,
    )
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )
    fig.savefig(output_dir / "continual_error.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_test_metrics(
    class_test_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
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
        axes[0].plot(
            group["stage"],
            group["test_loss"],
            marker="o",
            linewidth=2,
            label=test_class,
        )
        axes[1].plot(
            group["stage"],
            group["test_accuracy"],
            marker="o",
            linewidth=2,
            label=test_class,
        )

    axes[0].set_title("Separate test loss by class")
    axes[0].set_xlabel("Continual stage")
    axes[0].set_ylabel("sqrt(CrossEntropyLoss)")

    axes[1].set_title("Separate test accuracy by class")
    axes[1].set_xlabel("Continual stage")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)

    for ax in axes:
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend()

    fig.savefig(output_dir / "continual_class_tests.png", dpi=160)
    plt.close(fig)


def edge_function_summary(
    model: KAN,
    input_names: list[str],
    output_names: list[str],
) -> pd.DataFrame:
    """Create a table describing the active functions on the KAN edges."""
    rows = []
    n_layers = len(model.act_fun)

    for layer_id in range(n_layers):
        numeric_layer = model.act_fun[layer_id]
        symbolic_layer = model.symbolic_fun[layer_id]
        edge_scores = getattr(model, "edge_scores", [])
        layer_scores = edge_scores[layer_id] if layer_id < len(edge_scores) else None

        for source_id in range(model.width_in[layer_id]):
            for target_id in range(model.width_out[layer_id + 1]):
                numeric_mask = float(numeric_layer.mask[source_id][target_id].detach().cpu())
                symbolic_mask = float(symbolic_layer.mask[target_id][source_id].detach().cpu())

                if numeric_mask <= 0 and symbolic_mask <= 0:
                    continue

                if layer_id == 0 and source_id < len(input_names):
                    source_name = input_names[source_id]
                else:
                    source_name = f"layer_{layer_id}_node_{source_id}"

                if layer_id == n_layers - 1 and target_id < len(output_names):
                    target_name = output_names[target_id]
                else:
                    target_name = f"layer_{layer_id + 1}_node_{target_id}"

                if numeric_mask > 0 and symbolic_mask > 0:
                    function_type = "symbolic + numeric spline"
                elif symbolic_mask > 0:
                    function_type = "symbolic"
                else:
                    function_type = "numeric spline"

                symbolic_name = str(symbolic_layer.funs_name[target_id][source_id])
                edge_score = np.nan
                if layer_scores is not None:
                    edge_score = float(layer_scores[target_id][source_id].detach().cpu())

                rows.append(
                    {
                        "layer": layer_id,
                        "source_node": source_name,
                        "target_node": target_name,
                        "function_type": function_type,
                        "symbolic_function": symbolic_name,
                        "numeric_mask": numeric_mask,
                        "symbolic_mask": symbolic_mask,
                        "spline_scale": float(
                            numeric_layer.scale_sp[source_id][target_id].detach().cpu()
                        ),
                        "base_scale": float(
                            numeric_layer.scale_base[source_id][target_id].detach().cpu()
                        ),
                        "edge_score": edge_score,
                    }
                )

    return pd.DataFrame(rows)


def plot_kan_edge_functions(
    model: KAN,
    sample_input: torch.Tensor,
    output_dir: Path,
    image_name: str,
    table_name: str,
    input_names: list[str],
    output_names: list[str],
    title: str,
) -> tuple[Path, Path]:
    """Save the final KAN graph and a CSV summary of its edge functions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / image_name
    table_path = output_dir / table_name
    assets_dir = output_dir / f"{Path(image_name).stem}_assets"

    if assets_dir.exists():
        shutil.rmtree(assets_dir)

    # KAN.plot needs activations from a real forward pass before it can draw
    # meaningful edge scores and learned spline functions.
    model.get_act(sample_input)
    model.attribute(plot=False)
    summary = edge_function_summary(model, input_names, output_names)
    summary.to_csv(table_path, index=False)

    model.plot(
        folder=str(assets_dir),
        metric="backward",
        scale=0.65,
        tick=False,
        in_vars=input_names,
        out_vars=output_names,
        title=title,
        varscale=0.55,
    )
    plt.savefig(image_path, bbox_inches="tight", dpi=220)
    plt.close("all")

    return image_path, table_path


def copy_kan_model(model: KAN) -> KAN:
    """Copy a KAN model while avoiding cached tensors that can break copying."""
    # pykan stores cached activations after forward passes. Those tensors are not
    # always safe to deepcopy, so we clear them only for the moment of copying.
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


def make_frozen_teacher_model(model: KAN) -> KAN:
    """Build the frozen teacher used by Learning without Forgetting."""
    teacher = copy_kan_model(model)
    teacher.eval()
    teacher.save_act = False

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    return teacher


def make_optimizer(model: KAN) -> torch.optim.Optimizer:
    """Create the optimizer used by the manual mini-batch training loop."""
    params = model.get_params() if hasattr(model, "get_params") else model.parameters()
    optimizer_name = OPTIMIZER.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=LEARNING_RATE)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=LEARNING_RATE)

    raise ValueError(
        "Il loop con WeightedRandomSampler usa mini-batch PyTorch: "
        "imposta OPTIMIZER a 'Adam' oppure 'AdamW'."
    )


def epochs_for_stage(stage: int) -> int:
    """Return how many epochs should be used for one continual stage."""
    if not EPOCHS_PER_TASK:
        raise ValueError("EPOCHS_PER_TASK deve contenere almeno un valore.")

    if stage <= len(EPOCHS_PER_TASK):
        epochs = int(EPOCHS_PER_TASK[stage - 1])
    else:
        epochs = int(EPOCHS_PER_TASK[-1])

    if epochs <= 0:
        raise ValueError(f"Numero di epoche non valido per lo stage {stage}: {epochs}")

    return epochs


def lambda_kd_for_stage(stage: int, has_previous_classes: bool) -> float:
    """Decay the distillation weight as the class-incremental task grows."""
    if not has_previous_classes:
        return 0.0

    incremental_index = max(0, stage - 2)
    return float(INITIAL_LAMBDA_KD * (LAMBDA_KD_DECAY ** incremental_index))


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
) -> tuple[KAN, dict[str, list], dict[str, float | int | bool | list[float]]]:
    """Train one continual stage with balanced batches, smoothing, and LwF."""
    train_history = []
    test_history = []
    reg_history = []
    objective_history = []
    supervised_history = []
    distillation_history = []
    old_accuracy_history = []
    learning_rate_history = []

    # The active tensor masks out future classes. This keeps a stage focused only
    # on the classes the experiment has introduced so far.
    active_tensor = active_class_tensor(active_classes, device)

    # The DataLoader balances classes through sampling. CrossEntropy stays
    # unweighted, avoiding the "double dip" that made minority classes too costly.
    train_loader = make_balanced_dataloader(
        features=features,
        labels=labels,
        train_indices=train_indices,
        class_counts=class_counts,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=MIN_LEARNING_RATE,
    )

    # Early stopping keeps the best model seen on the validation/test split used
    # in this experiment. For a final paper, this could be replaced by a true
    # validation split and a final untouched test set.
    best_loss = sqrt_cross_entropy_loss(
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

        # KAN grids can adapt to the current data. We update them only at the
        # beginning and then stop, so later epochs refine functions instead of
        # constantly moving the grid.
        if (
            update_grid
            and GRID_UPDATE_EVERY > 0
            and epoch <= STOP_GRID_UPDATE_EPOCH
            and (epoch == 1 or (epoch - 1) % GRID_UPDATE_EVERY == 0)
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

            # Main classification loss: future classes are masked, label smoothing
            # is kept, and class weights are intentionally not used.
            supervised_loss = masked_classification_loss(
                logits=logits,
                labels=batch_labels,
                active_classes=active_tensor,
                label_smoothing=LABEL_SMOOTHING,
            )

            # LwF compares the current model with the previous frozen model only
            # on classes that were already learned before this stage.
            if USE_LWF and teacher_model is not None and previous_classes:
                with torch.no_grad():
                    teacher_logits = teacher_model(batch_inputs)
                distillation_loss = knowledge_distillation_loss(
                    current_logits=logits,
                    teacher_logits=teacher_logits,
                    previous_classes=previous_classes,
                    device=device,
                    temperature=DISTILLATION_TEMPERATURE,
                )
            else:
                distillation_loss = logits.new_tensor(0.0)

            # Total objective: supervised learning plus a stage-specific
            # distillation term. The KD weight decays in later stages.
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

        train_loss = sqrt_cross_entropy_loss(
            model=model,
            features=features,
            labels=labels,
            indices=train_indices,
            device=device,
            active_classes=active_classes,
        )
        validation_loss = sqrt_cross_entropy_loss(
            model=model,
            features=features,
            labels=labels,
            indices=validation_indices,
            device=device,
            active_classes=active_classes,
        )
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

        # Anti-forgetting stopping checks the old classes every epoch. If their
        # accuracy falls too far from the best value reached in this stage, we
        # stop before the new task over-specializes the model.
        if np.isfinite(old_class_accuracy):
            if old_class_accuracy >= best_old_class_accuracy:
                best_old_class_accuracy = old_class_accuracy
                best_old_class_accuracy_step = epoch
                best_old_class_model = copy_kan_model(model)
                old_accuracy_bad_epochs = 0
            elif best_old_class_accuracy - old_class_accuracy > ANTI_FORGETTING_TOLERANCE:
                old_accuracy_bad_epochs += 1
            else:
                old_accuracy_bad_epochs = 0

        pbar.set_description(
            "| train_loss: %.3f | test_loss: %.3f | old_acc: %.3f | kd: %.3f | lr: %.2e | "
            % (
                train_loss,
                validation_loss,
                old_class_accuracy,
                distillation_history[-1],
                current_lr,
            )
        )

        # Early stopping is checked every few epochs to avoid copying the KAN
        # model at every single pass.
        should_check = (
            epoch % EARLY_STOPPING_CHECK_EVERY == 0
            or epoch == epochs
        )
        if should_check:
            if validation_loss < best_loss - EARLY_STOPPING_MIN_DELTA:
                best_loss = validation_loss
                best_model = copy_kan_model(model)
                best_step = trained_steps
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if EARLY_STOPPING and steps_without_improvement >= EARLY_STOPPING_PATIENCE:
                stopped_early = True
                stop_reason = "validation_loss"
                break

        scheduler.step()

        if (
            ANTI_FORGETTING_EARLY_STOPPING
            and old_accuracy_bad_epochs >= ANTI_FORGETTING_PATIENCE
        ):
            stopped_early = True
            anti_forgetting_stopped = True
            stop_reason = "anti_forgetting"
            break

    if anti_forgetting_stopped and best_old_class_model is not None:
        model = best_old_class_model
    else:
        model = best_model
    final_train_loss = sqrt_cross_entropy_loss(
        model=model,
        features=features,
        labels=labels,
        indices=train_indices,
        device=device,
        active_classes=active_classes,
    )
    final_validation_loss = sqrt_cross_entropy_loss(
        model=model,
        features=features,
        labels=labels,
        indices=validation_indices,
        device=device,
        active_classes=active_classes,
    )
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
        "best_old_class_accuracy": (
            float("nan")
            if best_old_class_accuracy == float("-inf")
            else best_old_class_accuracy
        ),
        "best_old_class_accuracy_step": best_old_class_accuracy_step,
        "final_old_class_accuracy": final_old_class_accuracy,
        "best_test_loss": best_loss,
        "final_train_loss": final_train_loss,
        "final_test_loss": final_validation_loss,
        "final_learning_rate": learning_rate_history[-1] if learning_rate_history else LEARNING_RATE,
        "lambda_kd": lambda_kd,
        "epochs": epochs,
    }

    return model, results, summary


def parse_training_schedule(
    training_schedule: list[dict[str, int]],
    shape_names: list[str],
) -> list[dict[int, int]]:
    """Convert the human-readable schedule into class ids and sample counts."""
    name_to_id = {name: index for index, name in enumerate(shape_names)}
    flat_names = [name for stage in training_schedule for name in stage]
    unknown = [name for name in flat_names if name not in name_to_id]

    if unknown:
        valid = ", ".join(shape_names)
        raise ValueError(f"Task sconosciuti: {unknown}. Task validi: {valid}")

    parsed_schedule = []
    for stage_index, stage in enumerate(training_schedule, start=1):
        if not stage:
            raise ValueError(f"Lo stage {stage_index} non contiene classi.")

        parsed_stage = {}
        for name, count in stage.items():
            if count <= 0:
                raise ValueError(
                    f"Lo stage {stage_index} ha un conteggio non valido per {name}: {count}"
                )
            parsed_stage[name_to_id[name]] = int(count)

        parsed_schedule.append(parsed_stage)

    return parsed_schedule


def json_number(value: object) -> float | int | bool | None:
    """Convert numpy values into clean JSON numbers."""
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def build_accuracy_config(
    records: list[dict[str, object]],
    training_schedule: list[dict[int, int]],
    task_class_order: list[int],
    shape_names: list[str],
    best_accuracy_by_class: dict[str, float],
    final_accuracy_by_class: dict[str, float],
    forgetting_by_class: dict[str, float],
    mean_forgetting: float,
) -> dict[str, object]:
    """Collect stage accuracies and forgetting in a JSON-friendly dictionary."""
    stage_rows = []

    for record in records:
        per_task_accuracy = {}
        for class_id in task_class_order:
            class_name = shape_names[class_id]
            column = f"acc_{class_name}"

            if column in record:
                per_task_accuracy[class_name] = json_number(record[column])

        stage_rows.append(
            {
                "stage": int(record["stage"]),
                "trained_task": str(record["trained_task"]),
                "mean_seen_accuracy": json_number(record.get("mean_seen_accuracy")),
                "per_task_accuracy": per_task_accuracy,
            }
        )

    schedule_rows = []
    for stage, stage_counts in enumerate(training_schedule, start=1):
        schedule_rows.append(
            {
                "stage": stage,
                "class_counts": {
                    shape_names[class_id]: int(count)
                    for class_id, count in stage_counts.items()
                },
            }
        )

    return {
        "training_schedule": schedule_rows,
        "accuracy_by_stage": stage_rows,
        "final_accuracy": {
            class_name: json_number(value)
            for class_name, value in final_accuracy_by_class.items()
        },
        "best_accuracy": {
            class_name: json_number(value)
            for class_name, value in best_accuracy_by_class.items()
        },
        "forgetting": {
            class_name: json_number(value)
            for class_name, value in forgetting_by_class.items()
        },
        "mean_forgetting": json_number(mean_forgetting),
        "final_mean_seen_accuracy": (
            json_number(records[-1].get("mean_seen_accuracy")) if records else None
        ),
    }


def save_accuracy_config(
    accuracy_config: dict[str, object],
    output_path: Path,
) -> None:
    """Save task accuracies in a config-style JSON file."""
    with output_path.open("w", encoding="utf-8") as config_file:
        json.dump(accuracy_config, config_file, indent=2)
        config_file.write("\n")


def train_continual() -> pd.DataFrame:
    """Run the full continual learning experiment from data loading to reports."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the CSV and create a stratified train/test split.
    input_names = load_feature_names(DATA_PATH)
    raw_features, labels, shape_names = load_conic_csv(DATA_PATH)
    train_indices, _, train_by_class, test_by_class = stratified_split(
        labels=labels,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    # 2. Scaling the coordinates usually makes training easier for the model.
    if STANDARDIZE:
        features = standardize_from_train(raw_features, train_indices)
    else:
        features = raw_features

    # 3. Parse the schedule and remember the order in which classes appear.
    training_schedule = parse_training_schedule(TRAINING_SCHEDULE, shape_names)
    task_class_order = []
    for stage_counts in training_schedule:
        for class_id in stage_counts:
            if class_id not in task_class_order:
                task_class_order.append(class_id)
    plot_dir = ROOT_DIR / PLOT_DIR

    # 4. Build one KAN with an output node for every possible class. During early
    # stages we mask future output nodes instead of changing the architecture.
    model = KAN(
        width=[features.shape[1], HIDDEN, len(shape_names)],
        grid=GRID,
        k=SPLINE_ORDER,
        seed=SEED,
        auto_save=False,
        device=device,
    )

    seen_classes = []
    records = []
    class_test_records = []
    loss_history = []

    print(f"Device: {device}")
    print(
        "Training schedule: "
        f"{[{shape_names[class_id]: count for class_id, count in stage.items()} for stage in training_schedule]}"
    )
    print("Continual learning: explicit per-stage class counts")
    print(f"Epochs per task: {EPOCHS_PER_TASK}")
    print(
        f"Learning rate: {LEARNING_RATE} -> {MIN_LEARNING_RATE}, "
        f"label smoothing: {LABEL_SMOOTHING}"
    )
    print(f"Mask future classes: {MASK_FUTURE_CLASSES}")
    print(f"Balanced sampler: {USE_BALANCED_SAMPLER}, class weights in CE: False")
    print(
        "LwF distillation: "
        f"{USE_LWF}, initial_lambda_kd={INITIAL_LAMBDA_KD}, "
        f"decay={LAMBDA_KD_DECAY}, temperature={DISTILLATION_TEMPERATURE}"
    )
    print(
        "Early stopping: "
        f"{EARLY_STOPPING}, check_every={EARLY_STOPPING_CHECK_EVERY}, "
        f"patience={EARLY_STOPPING_PATIENCE}"
    )
    print(
        "Anti-forgetting stop: "
        f"{ANTI_FORGETTING_EARLY_STOPPING}, "
        f"tolerance={ANTI_FORGETTING_TOLERANCE}, "
        f"patience={ANTI_FORGETTING_PATIENCE}"
    )

    # 5. Train one schedule row at a time. Each row is a continual-learning stage.
    for stage, stage_counts in enumerate(training_schedule, start=1):
        current_classes = list(stage_counts.keys())
        task_name = "+".join(shape_names[class_id] for class_id in current_classes)
        train_parts = []

        # The teacher must be copied before training the new stage. It represents
        # what the model knew at the end of the previous stage.
        previous_classes = seen_classes.copy()
        teacher_model = (
            make_frozen_teacher_model(model)
            if USE_LWF and previous_classes
            else None
        )
        stage_epochs = epochs_for_stage(stage)
        stage_lambda_kd = lambda_kd_for_stage(stage, bool(previous_classes))
        previous_test_indices = (
            np.concatenate([test_by_class[item] for item in previous_classes])
            if previous_classes
            else np.array([], dtype=np.int64)
        )

        # Select exactly the number of examples requested by TRAINING_SCHEDULE.
        for class_id, sample_count in stage_counts.items():
            train_parts.append(
                sample_indices(
                    train_by_class[class_id],
                    sample_count,
                    rng,
                )
            )

        current_train_indices = np.concatenate(train_parts)

        # We test on all classes seen so far, not only on the newest classes.
        for class_id in current_classes:
            if class_id not in seen_classes:
                seen_classes.append(class_id)
        seen_test_indices = np.concatenate([test_by_class[item] for item in seen_classes])
        active_classes_for_stage = seen_classes.copy() if MASK_FUTURE_CLASSES else None

        print()
        print(
            f"Task {stage}/{len(training_schedule)}: {task_name} "
            f"({len(current_train_indices)} esempi di training)"
        )
        print(f"Epochs: {stage_epochs}, lambda_kd: {stage_lambda_kd:.4f}")
        if active_classes_for_stage is not None:
            active_names = [shape_names[class_id] for class_id in active_classes_for_stage]
            print(f"Classi attive nella loss: {active_names}")
        train_counts = count_by_class(current_train_indices, labels, shape_names)
        train_mix = " | ".join(
            f"{shape_names[item]}={train_counts[shape_names[item]]}" for item in seen_classes
        )
        print(f"Training mix: {train_mix}")
        if teacher_model is not None:
            old_names = [shape_names[class_id] for class_id in previous_classes]
            print(f"LwF teacher sulle classi vecchie: {old_names}")

        stage_class_counts = count_by_class_id(
            indices=current_train_indices,
            labels=labels,
            class_ids=seen_classes,
        )

        # This is the actual PyTorch continual-learning step: balanced batches,
        # masked CE with label smoothing, optional LwF, and early stopping.
        model, fit_results, stage_training = train_stage_with_early_stopping(
            model=model,
            features=features,
            labels=labels,
            train_indices=current_train_indices,
            validation_indices=seen_test_indices,
            previous_validation_indices=previous_test_indices,
            device=device,
            active_classes=active_classes_for_stage,
            previous_classes=previous_classes,
            teacher_model=teacher_model,
            class_counts=stage_class_counts,
            update_grid=not (FREEZE_GRID_AFTER_FIRST and stage > 1),
            epochs=stage_epochs,
            lambda_kd=stage_lambda_kd,
        )

        train_loss = np.asarray(fit_results["train_loss"], dtype=float)
        test_loss = np.asarray(fit_results["test_loss"], dtype=float)

        # These histories feed the final loss plot.
        loss_history.append(
            {
                "stage": stage,
                "task": task_name,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "final_train_loss": float(stage_training["final_train_loss"]),
                "final_test_loss": float(stage_training["final_test_loss"]),
            }
        )

        # We store one row per stage. The row includes global metrics and one
        # accuracy column for every class seen so far.
        record = {
            "stage": stage,
            "trained_task": task_name,
            "train_examples": int(len(current_train_indices)),
            "epochs_for_task": int(stage_training["epochs"]),
            "trained_steps": int(stage_training["trained_steps"]),
            "best_step": int(stage_training["best_step"]),
            "stopped_early": bool(stage_training["stopped_early"]),
            "stop_reason": str(stage_training["stop_reason"]),
            "anti_forgetting_stopped": bool(stage_training["anti_forgetting_stopped"]),
            "lambda_kd": float(stage_training["lambda_kd"]),
            "final_learning_rate": float(stage_training["final_learning_rate"]),
            "best_old_class_accuracy": float(stage_training["best_old_class_accuracy"]),
            "best_old_class_accuracy_step": int(stage_training["best_old_class_accuracy_step"]),
            "final_old_class_accuracy": float(stage_training["final_old_class_accuracy"]),
            "final_train_loss": float(stage_training["final_train_loss"]),
            "final_test_loss": float(stage_training["final_test_loss"]),
            "best_train_loss": float(train_loss.min()),
            "best_test_loss": float(stage_training["best_test_loss"]),
            "distillation_loss": float(fit_results["distillation_loss"][-1]),
            "objective_loss": float(fit_results["objective_loss"][-1]),
            "mean_seen_accuracy": accuracy(
                model=model,
                features=features,
                labels=labels,
                indices=seen_test_indices,
                device=device,
                active_classes=active_classes_for_stage,
            ),
        }

        for seen_class_id in seen_classes:
            record[f"train_count_{shape_names[seen_class_id]}"] = train_counts[
                shape_names[seen_class_id]
            ]
            record[f"acc_{shape_names[seen_class_id]}"] = accuracy(
                model=model,
                features=features,
                labels=labels,
                indices=test_by_class[seen_class_id],
                device=device,
                active_classes=active_classes_for_stage,
            )

        # Separate class tests make it clear which class is responsible if the
        # global loss increases even when global accuracy stays high.
        separate_test_summary = []
        for test_class_id, test_class_name in enumerate(shape_names):
            class_seen = test_class_id in seen_classes
            class_indices = test_by_class[test_class_id]
            class_loss = sqrt_cross_entropy_loss(
                model=model,
                features=features,
                labels=labels,
                indices=class_indices,
                device=device,
                active_classes=active_classes_for_stage if class_seen else None,
            )
            class_accuracy = accuracy(
                model=model,
                features=features,
                labels=labels,
                indices=class_indices,
                device=device,
                active_classes=active_classes_for_stage if class_seen else None,
            )

            class_test_records.append(
                {
                    "stage": stage,
                    "trained_task": task_name,
                    "test_class": test_class_name,
                    "seen": class_seen,
                    "test_examples": int(len(class_indices)),
                    "test_loss": class_loss,
                    "test_accuracy": class_accuracy,
                }
            )

            if class_seen:
                record[f"test_loss_{test_class_name}"] = class_loss
                record[f"test_accuracy_{test_class_name}"] = class_accuracy
                separate_test_summary.append(
                    f"{test_class_name}: loss={class_loss:.3f}, acc={class_accuracy:.3f}"
                )

        records.append(record)

        per_task = " | ".join(
            f"{shape_names[item]}={record[f'acc_{shape_names[item]}']:.3f}"
            for item in seen_classes
        )
        print(f"Accuracy viste: mean={record['mean_seen_accuracy']:.3f} | {per_task}")
        print(f"Test separati: {' | '.join(separate_test_summary)}")
        print(
            "Early stopping: "
            f"best_step={record['best_step']}, "
            f"trained_steps={record['trained_steps']}, "
            f"best_loss={record['best_test_loss']:.3f}, "
            f"final_loss={record['final_test_loss']:.3f}, "
            f"stopped={record['stopped_early']}, "
            f"reason={record['stop_reason']}, "
            f"best_old_acc={record['best_old_class_accuracy']:.3f}"
        )

        if PLOT_PREDICTIONS:
            plot_stage_predictions(
                model=model,
                model_features=features,
                plot_features=raw_features,
                labels=labels,
                test_by_class=test_by_class,
                seen_classes=seen_classes,
                shape_names=shape_names,
                stage=stage,
                trained_task=task_name,
                device=device,
                output_dir=plot_dir,
                examples_per_class=PLOT_EXAMPLES,
                seed=SEED,
                active_classes=active_classes_for_stage,
            )
            print(f"Plot predizioni salvato in: {plot_dir}")

    if PLOT_LOSS:
        plot_continual_losses(loss_history, plot_dir)
        print(f"Plot loss continual learning salvato in: {plot_dir / 'continual_loss.png'}")
        final_loss_paths = save_final_loss_summary(loss_history, plot_dir)
        if final_loss_paths is not None:
            final_loss_csv, final_loss_json = final_loss_paths
            print(f"Loss finali salvate in: {final_loss_csv}")
            print(f"Loss finali JSON salvate in: {final_loss_json}")

    # 6. Convert the collected dictionaries into CSV-friendly DataFrames.
    metrics = pd.DataFrame(records)
    class_test_metrics = pd.DataFrame(class_test_records)

    if PLOT_ERROR:
        plot_continual_error(records, plot_dir)
        print(f"Plot error continual learning salvato in: {plot_dir / 'continual_error.png'}")

    if PLOT_CLASS_TESTS:
        plot_class_test_metrics(class_test_metrics, plot_dir)
        print(f"Plot test separati salvato in: {plot_dir / 'continual_class_tests.png'}")

    if PLOT_EDGE_FUNCTIONS:
        sample_input = torch.tensor(features[train_indices], device=device)
        edge_image_path, edge_table_path = plot_kan_edge_functions(
            model=model,
            sample_input=sample_input,
            output_dir=plot_dir,
            image_name="final_edge_functions.png",
            table_name="final_edge_functions.csv",
            input_names=input_names,
            output_names=shape_names,
            title="Final continual KAN edge functions",
        )
        print(f"Funzioni finali sugli archi salvate in: {edge_image_path}")
        print(f"Tabella archi attivi salvata in: {edge_table_path}")

    final_record = records[-1]
    forgetting_values = []
    best_accuracy_by_class = {}
    final_accuracy_by_class = {}
    forgetting_by_class = {}

    # 7. Forgetting compares the best past accuracy with the final accuracy for
    # each class. Low forgetting means the model kept old knowledge.
    print()
    print("Forgetting finale:")
    for class_id in task_class_order:
        column = f"acc_{shape_names[class_id]}"
        values = [record[column] for record in records if column in record]
        if not values:
            continue

        class_name = shape_names[class_id]
        best_accuracy = max(values)
        final_accuracy = final_record.get(column, float("nan"))
        forgetting = best_accuracy - final_accuracy
        forgetting_values.append(forgetting)
        best_accuracy_by_class[class_name] = best_accuracy
        final_accuracy_by_class[class_name] = final_accuracy
        forgetting_by_class[class_name] = forgetting
        print(
            f"{class_name}: best={best_accuracy:.3f}, "
            f"final={final_accuracy:.3f}, forgetting={forgetting:.3f}"
        )

    mean_forgetting = float(np.mean(forgetting_values)) if forgetting_values else float("nan")
    print(f"Forgetting medio: {mean_forgetting:.3f}")
    print(f"Accuracy finale media sui task visti: {final_record['mean_seen_accuracy']:.3f}")

    if ACCURACY_CONFIG_OUT:
        accuracy_config_path = ROOT_DIR / ACCURACY_CONFIG_OUT
        accuracy_config = build_accuracy_config(
            records=records,
            training_schedule=training_schedule,
            task_class_order=task_class_order,
            shape_names=shape_names,
            best_accuracy_by_class=best_accuracy_by_class,
            final_accuracy_by_class=final_accuracy_by_class,
            forgetting_by_class=forgetting_by_class,
            mean_forgetting=mean_forgetting,
        )
        save_accuracy_config(accuracy_config, accuracy_config_path)
        print(f"Config accuracy salvata in: {accuracy_config_path}")

    if METRICS_OUT:
        metrics_path = ROOT_DIR / METRICS_OUT
        metrics.to_csv(metrics_path, index=False)
        print(f"Metriche salvate in: {metrics_path}")

    if CLASS_TESTS_OUT:
        class_tests_path = ROOT_DIR / CLASS_TESTS_OUT
        class_test_metrics.to_csv(class_tests_path, index=False)
        print(f"Test separati salvati in: {class_tests_path}")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "Questo script non accetta parametri da terminale: modifica il blocco CONFIG."
        )

    train_continual()
