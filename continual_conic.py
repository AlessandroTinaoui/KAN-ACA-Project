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


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "dataset" / "Conic-Section_dataset.csv"

# CONFIG
# Change these values here when you want to try a different experiment.
MEMORY_SIZE = 400
BALANCE_REPLAY = True
FREEZE_GRID_AFTER_FIRST = False

STEPS = 30
HIDDEN = 8
GRID = 3
SPLINE_ORDER = 3
SEED = 1
TEST_RATIO = 0.2
OPTIMIZER = "LBFGS"
LEARNING_RATE = 1.0

METRICS_OUT = "continual_metrics.csv"
STANDARDIZE = True
PLOT_PREDICTIONS = True
PLOT_DIR = "continual_plots"
PLOT_EXAMPLES = 2
PLOT_LOSS = True
PLOT_ERROR = True
PLOT_EDGE_FUNCTIONS = True

TASK_ORDER: list[str] | None = None


def load_feature_names(data_path: Path) -> list[str]:
    """Read the feature column names from the dataset header."""
    columns = pd.read_csv(data_path, nrows=0).columns
    return [column for column in columns if column != "shape"]


def load_conic_csv(data_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the conic dataset and turn shape names into class ids."""
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
    """Split the dataset while keeping the same class balance in train and test."""
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
    """Scale all features using only the training set statistics."""
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
    """Create the tensor dictionary expected by KAN.fit."""
    # KAN.fit needs a small dictionary of tensors instead of the original DataFrame.
    return {
        "train_input": torch.tensor(features[train_indices], device=device),
        "train_label": torch.tensor(labels[train_indices], device=device),
        "test_input": torch.tensor(features[test_indices], device=device),
        "test_label": torch.tensor(labels[test_indices], device=device),
    }


def build_replay_indices(
    train_by_class: dict[int, np.ndarray],
    seen_classes: list[int],
    memory_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample old training examples to use as replay memory."""
    # The replay memory keeps a few examples from the classes learned before.
    if memory_size <= 0 or len(seen_classes) == 0:
        return np.array([], dtype=np.int64)

    per_class = max(1, memory_size // len(seen_classes))
    replay_parts = []

    for class_id in seen_classes:
        candidates = train_by_class[class_id]
        sample_size = min(per_class, len(candidates))
        replay_parts.append(rng.choice(candidates, size=sample_size, replace=False))

    return np.concatenate(replay_parts).astype(np.int64)


def sample_indices(
    indices: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Take a random subset of indices without replacement."""
    # This chooses a smaller group of rows without taking the same row twice.
    if sample_size <= 0 or sample_size >= len(indices):
        return indices

    return np.sort(rng.choice(indices, size=sample_size, replace=False))


@torch.no_grad()
def accuracy(
    model: KAN,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    """Measure classification accuracy on a selected group of examples."""
    # Accuracy tells us how many selected examples are classified correctly.
    if len(indices) == 0:
        return float("nan")

    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    logits = model(inputs)
    predictions = logits.argmax(dim=1).cpu().numpy()
    return float(np.mean(predictions == labels[indices]))


@torch.no_grad()
def predict_classes(
    model: KAN,
    features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return the predicted class id for each selected example."""
    # For each example, the predicted class is the output with the highest score.
    model.eval()
    inputs = torch.tensor(features[indices], device=device)
    logits = model(inputs)
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
) -> None:
    """Plot a few predictions after one continual learning stage."""
    # After each task, we plot a few test examples with their true and predicted labels.
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

    # The model uses processed features, while the picture uses the original points.
    predictions = predict_classes(model, model_features, selected_indices, device)

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


def plot_continual_losses(
    loss_history: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Save the loss curves collected during continual learning."""
    # This plot shows how train and test loss behave inside each continual step.
    if not loss_history:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    for item in loss_history:
        stage = item["stage"]
        task = item["task"]
        train_loss = item["train_loss"]
        test_loss = item["test_loss"]
        steps = np.arange(1, len(train_loss) + 1)

        axes[0].plot(steps, train_loss, label=f"{stage}: {task} train")
        axes[0].plot(steps, test_loss, linestyle="--", label=f"{stage}: {task} test")

    axes[0].set_title("Loss curves during continual learning")
    axes[0].set_xlabel("Optimization step")
    axes[0].set_ylabel("Loss returned by KAN.fit")
    axes[0].grid(True, linewidth=0.4, alpha=0.35)
    axes[0].legend(fontsize=8)

    stages = [item["stage"] for item in loss_history]
    task_labels = [item["task"] for item in loss_history]
    final_train = [item["train_loss"][-1] for item in loss_history]
    final_test = [item["test_loss"][-1] for item in loss_history]
    x = np.arange(len(stages))
    width = 0.36

    axes[1].bar(x - width / 2, final_train, width, label="train")
    axes[1].bar(x + width / 2, final_test, width, label="test")
    axes[1].set_title("Final loss after each task")
    axes[1].set_xlabel("Continual task")
    axes[1].set_ylabel("Final loss")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{stage}\n{task}" for stage, task in zip(stages, task_labels)])
    axes[1].grid(True, axis="y", linewidth=0.4, alpha=0.35)
    axes[1].legend()

    fig.savefig(output_dir / "continual_loss.png", dpi=160)
    plt.close(fig)


def plot_continual_error(
    records: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Plot the mean error and accuracy after each learned task."""
    # Error is simply 1 - accuracy, so lower is better.
    if not records:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    stages = [record["stage"] for record in records]
    task_labels = [record["trained_task"] for record in records]
    mean_accuracy = np.array([record["mean_seen_accuracy"] for record in records], dtype=float)
    error_rate = 1.0 - mean_accuracy

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.plot(stages, error_rate, marker="o", label="mean error")
    ax.plot(stages, mean_accuracy, marker="o", label="mean accuracy")
    ax.set_title("Continual learning error and accuracy")
    ax.set_xlabel("Continual task")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(stages)
    ax.set_xticklabels([f"{stage}\n{task}" for stage, task in zip(stages, task_labels)])
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(output_dir / "continual_error.png", dpi=160)
    plt.close(fig)


def edge_function_summary(
    model: KAN,
    input_names: list[str],
    output_names: list[str],
) -> pd.DataFrame:
    """Create a table with the active KAN functions between nodes."""
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
    """Save the final KAN graph with the learned functions drawn on its edges."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / image_name
    table_path = output_dir / table_name
    assets_dir = output_dir / f"{Path(image_name).stem}_assets"

    if assets_dir.exists():
        shutil.rmtree(assets_dir)

    # KAN.plot needs activations from a real forward pass.
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


def parse_task_order(task_order: list[str] | None, shape_names: list[str]) -> list[int]:
    """Convert optional task names from the config into class ids."""
    # This converts the optional task names from CONFIG into internal class ids.
    if not task_order:
        return list(range(len(shape_names)))

    name_to_id = {name: index for index, name in enumerate(shape_names)}
    unknown = [name for name in task_order if name not in name_to_id]

    if unknown:
        valid = ", ".join(shape_names)
        raise ValueError(f"Task sconosciuti: {unknown}. Task validi: {valid}")

    return [name_to_id[name] for name in task_order]


def train_continual() -> pd.DataFrame:
    """Run the full replay-based continual learning experiment."""
    # Main experiment: load the data, train one task at a time, and use replay.
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_names = load_feature_names(DATA_PATH)
    raw_features, labels, shape_names = load_conic_csv(DATA_PATH)
    train_indices, _, train_by_class, test_by_class = stratified_split(
        labels=labels,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    # Scaling the coordinates usually makes training easier for the model.
    if STANDARDIZE:
        features = standardize_from_train(raw_features, train_indices)
    else:
        features = raw_features

    task_order = parse_task_order(TASK_ORDER, shape_names)
    plot_dir = ROOT_DIR / PLOT_DIR

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
    loss_history = []
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Task order: {[shape_names[class_id] for class_id in task_order]}")
    print(f"Continual learning: replay, memory: {MEMORY_SIZE}")
    print(f"Replay bilanciato: {BALANCE_REPLAY}")

    for stage, class_id in enumerate(task_order, start=1):
        previous_classes = seen_classes.copy()
        current_indices = train_by_class[class_id]

        # From the second task on, we train on the new class plus replayed old examples.
        if previous_classes:
            if BALANCE_REPLAY:
                # Balanced replay gives each class about the same number of examples.
                per_class = max(1, MEMORY_SIZE // (len(previous_classes) + 1))
                current_indices = sample_indices(current_indices, per_class, rng)
                replay_indices = build_replay_indices(
                    train_by_class=train_by_class,
                    seen_classes=previous_classes,
                    memory_size=per_class * len(previous_classes),
                    rng=rng,
                )
            else:
                # Unbalanced replay keeps the whole new class and adds old examples.
                replay_indices = build_replay_indices(
                    train_by_class=train_by_class,
                    seen_classes=previous_classes,
                    memory_size=MEMORY_SIZE,
                    rng=rng,
                )

            current_train_indices = np.concatenate([current_indices, replay_indices])
        else:
            current_train_indices = current_indices

        # We test on all classes seen so far, not only on the current class.
        seen_classes.append(class_id)
        seen_test_indices = np.concatenate([test_by_class[item] for item in seen_classes])

        print()
        print(
            f"Task {stage}/{len(task_order)}: {shape_names[class_id]} "
            f"({len(current_train_indices)} esempi di training)"
        )

        dataset = make_kan_dataset(
            features=features,
            labels=labels,
            train_indices=current_train_indices,
            test_indices=seen_test_indices,
            device=device,
        )

        model.train()
        fit_results = model.fit(
            dataset,
            opt=OPTIMIZER,
            steps=STEPS,
            lr=LEARNING_RATE,
            update_grid=not (FREEZE_GRID_AFTER_FIRST and stage > 1),
            loss_fn=loss_fn,
        )

        train_loss = np.asarray(fit_results["train_loss"], dtype=float)
        test_loss = np.asarray(fit_results["test_loss"], dtype=float)
        loss_history.append(
            {
                "stage": stage,
                "task": shape_names[class_id],
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
        )

        # We store the average accuracy and the accuracy for each learned class.
        record = {
            "stage": stage,
            "trained_task": shape_names[class_id],
            "final_train_loss": float(train_loss[-1]),
            "final_test_loss": float(test_loss[-1]),
            "best_train_loss": float(train_loss.min()),
            "best_test_loss": float(test_loss.min()),
            "mean_seen_accuracy": accuracy(
                model=model,
                features=features,
                labels=labels,
                indices=seen_test_indices,
                device=device,
            ),
        }

        for seen_class_id in seen_classes:
            record[f"acc_{shape_names[seen_class_id]}"] = accuracy(
                model=model,
                features=features,
                labels=labels,
                indices=test_by_class[seen_class_id],
                device=device,
            )

        records.append(record)

        per_task = " | ".join(
            f"{shape_names[item]}={record[f'acc_{shape_names[item]}']:.3f}"
            for item in seen_classes
        )
        print(f"Accuracy viste: mean={record['mean_seen_accuracy']:.3f} | {per_task}")

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
                trained_task=shape_names[class_id],
                device=device,
                output_dir=plot_dir,
                examples_per_class=PLOT_EXAMPLES,
                seed=SEED,
            )
            print(f"Plot predizioni salvato in: {plot_dir}")

    if PLOT_LOSS:
        plot_continual_losses(loss_history, plot_dir)
        print(f"Plot loss continual learning salvato in: {plot_dir / 'continual_loss.png'}")

    metrics = pd.DataFrame(records)

    if PLOT_ERROR:
        plot_continual_error(records, plot_dir)
        print(f"Plot error continual learning salvato in: {plot_dir / 'continual_error.png'}")

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

    # Forgetting compares the best past accuracy with the final accuracy for each task.
    print()
    print("Forgetting finale:")
    for class_id in task_order:
        column = f"acc_{shape_names[class_id]}"
        values = [record[column] for record in records if column in record]
        best_accuracy = max(values)
        final_accuracy = final_record.get(column, float("nan"))
        forgetting = best_accuracy - final_accuracy
        forgetting_values.append(forgetting)
        print(
            f"{shape_names[class_id]}: best={best_accuracy:.3f}, "
            f"final={final_accuracy:.3f}, forgetting={forgetting:.3f}"
        )

    print(f"Forgetting medio: {float(np.mean(forgetting_values)):.3f}")
    print(f"Accuracy finale media sui task visti: {final_record['mean_seen_accuracy']:.3f}")

    if METRICS_OUT:
        metrics_path = ROOT_DIR / METRICS_OUT
        metrics.to_csv(metrics_path, index=False)
        print(f"Metriche salvate in: {metrics_path}")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "Questo script non accetta parametri da terminale: modifica il blocco CONFIG."
        )

    train_continual()
