"""Run the continual-learning conic experiment from a TOML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kan_models.models.conic.config import ContinualConfig
from kan_models.models.conic.continual.config import CONTINUAL_CONFIG_VARIANTS, DEFAULT_CONFIG_PATH, load_continual_config
from kan_models.models.conic.continual.plotting import plot_class_test_metrics, plot_continual_error, plot_continual_losses, plot_stage_predictions
from kan_models.models.conic.continual.training import (
    epochs_for_stage,
    lambda_kd_for_stage,
    make_frozen_teacher_model,
    parse_training_schedule,
    train_stage_with_early_stopping,
)
from kan_models.models.conic.data import count_by_class, count_by_class_id, load_conic_csv, load_feature_names, sample_indices, standardization_stats, stratified_split
from kan_models.models.conic.modeling import accuracy, build_model, compute_confusion_matrix, cross_entropy_loss
from kan_models.models.conic.plotting import plot_kan_edge_functions, save_confusion_matrix, save_final_loss_summary
from kan_models.common.runtime import detect_device
from kan_models.common.shared import json_number, write_json


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
                "class_counts": {shape_names[class_id]: int(count) for class_id, count in stage_counts.items()},
            }
        )

    return {
        "training_schedule": schedule_rows,
        "accuracy_by_stage": stage_rows,
        "final_accuracy": {class_name: json_number(value) for class_name, value in final_accuracy_by_class.items()},
        "best_accuracy": {class_name: json_number(value) for class_name, value in best_accuracy_by_class.items()},
        "forgetting": {class_name: json_number(value) for class_name, value in forgetting_by_class.items()},
        "mean_forgetting": json_number(mean_forgetting),
        "final_mean_seen_accuracy": json_number(records[-1].get("mean_seen_accuracy")) if records else None,
    }


def named_training_schedule(training_schedule: list[dict[int, int]], shape_names: list[str]) -> list[dict[str, object]]:
    """Turn the parsed schedule back into a readable JSON structure."""
    return [
        {
            "stage": stage,
            "class_counts": {shape_names[class_id]: int(count) for class_id, count in stage_counts.items()},
        }
        for stage, stage_counts in enumerate(training_schedule, start=1)
    ]


def build_run_config(
    config: ContinualConfig,
    input_names: list[str],
    shape_names: list[str],
    training_schedule: list[dict[int, int]],
    device: torch.device,
    standardization_mean: np.ndarray,
    standardization_std: np.ndarray,
) -> dict[str, object]:
    """Collect the setup required to reproduce the continual run."""
    return {
        "config_path": str(config.config_path),
        "data_path": str(config.data.csv_path),
        "device": str(device),
        "input_names": input_names,
        "shape_names": shape_names,
        "class_to_id": {shape_name: class_id for class_id, shape_name in enumerate(shape_names)},
        "model": {
            "hidden": config.model.hidden,
            "grid": config.model.grid,
            "spline_order": config.model.spline_order,
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "min_learning_rate": config.training.min_learning_rate,
            "batch_size": config.training.batch_size,
        },
        "continual_learning": {
            "epochs_per_task": config.training.epochs_per_task,
            "mask_future_classes": config.training.mask_future_classes,
            "label_smoothing": config.training.label_smoothing,
            "use_balanced_sampler": config.training.use_balanced_sampler,
            "use_lwf": config.training.use_lwf,
            "initial_lambda_kd": config.training.initial_lambda_kd,
            "lambda_kd_decay": config.training.lambda_kd_decay,
            "distillation_temperature": config.training.distillation_temperature,
            "training_schedule": named_training_schedule(training_schedule, shape_names),
        },
        "early_stopping": {
            "validation_loss_enabled": config.early_stopping.enabled,
            "check_every": config.early_stopping.check_every,
            "patience": config.early_stopping.patience,
            "min_delta": config.early_stopping.min_delta,
            "anti_forgetting_enabled": config.early_stopping.anti_forgetting_enabled,
            "anti_forgetting_tolerance": config.early_stopping.anti_forgetting_tolerance,
            "anti_forgetting_patience": config.early_stopping.anti_forgetting_patience,
        },
        "standardization": {
            "enabled": config.standardize,
            "mean": standardization_mean.reshape(-1).astype(float).tolist(),
            "std": standardization_std.reshape(-1).astype(float).tolist(),
        },
        "outputs": {
            "metrics_csv": None if config.output.metrics_path is None else str(config.output.metrics_path),
            "class_tests_csv": None if config.output.class_tests_path is None else str(config.output.class_tests_path),
            "accuracy_config_json": None if config.output.accuracy_config_path is None else str(config.output.accuracy_config_path),
            "run_config_json": None if config.output.run_config_path is None else str(config.output.run_config_path),
            "plot_dir": None if config.output.plot_dir is None else str(config.output.plot_dir),
        },
    }


def save_final_model(model: torch.nn.Module, output_path: Path, run_config: dict[str, object]) -> None:
    """Save the final KAN weights together with the run configuration."""
    torch.save({"state_dict": model.state_dict(), "run_config": run_config}, output_path)


def run_continual(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    """Run the full continual learning experiment from data loading to reports."""
    config = load_continual_config(config_path)
    torch.manual_seed(config.split.seed)
    rng = np.random.default_rng(config.split.seed)
    device = detect_device(config.model.device)

    input_names = load_feature_names(config.data.csv_path, config.data.target_column)
    raw_features, labels, shape_names = load_conic_csv(config.data.csv_path, config.data.target_column)
    train_indices, _, train_by_class, test_by_class = stratified_split(
        labels=labels,
        test_ratio=config.split.test_ratio,
        seed=config.split.seed,
    )

    if config.standardize:
        standardization_mean, standardization_std = standardization_stats(raw_features, train_indices)
        features = ((raw_features - standardization_mean) / standardization_std).astype(np.float32)
    else:
        features = raw_features
        standardization_mean = np.zeros((1, raw_features.shape[1]), dtype=np.float32)
        standardization_std = np.ones((1, raw_features.shape[1]), dtype=np.float32)

    training_schedule = parse_training_schedule(config.schedule, shape_names)
    task_class_order: list[int] = []
    for stage_counts in training_schedule:
        for class_id in stage_counts:
            if class_id not in task_class_order:
                task_class_order.append(class_id)

    plot_dir = config.output.plot_dir
    if plot_dir is None:
        raise ValueError("output.plot_dir is required for the continual experiment.")

    model = build_model(
        input_dim=features.shape[1],
        output_dim=len(shape_names),
        config=config.model,
        seed=config.split.seed,
        device=device,
    )

    seen_classes: list[int] = []
    records: list[dict[str, object]] = []
    class_test_records: list[dict[str, object]] = []
    loss_history: list[dict[str, object]] = []

    print(f"Device: {device}")
    print("Training schedule: " f"{[{shape_names[class_id]: count for class_id, count in stage.items()} for stage in training_schedule]}")
    print(f"Epochs per task: {config.training.epochs_per_task}")

    for stage, stage_counts in enumerate(training_schedule, start=1):
        current_classes = list(stage_counts.keys())
        task_name = "+".join(shape_names[class_id] for class_id in current_classes)
        train_parts = []

        previous_classes = seen_classes.copy()
        teacher_model = make_frozen_teacher_model(model) if config.training.use_lwf and previous_classes else None
        stage_epochs = epochs_for_stage(config.training.epochs_per_task, stage)
        stage_lambda_kd = lambda_kd_for_stage(
            config.training.initial_lambda_kd,
            config.training.lambda_kd_decay,
            stage,
            bool(previous_classes),
        )
        previous_test_indices = (
            np.concatenate([test_by_class[item] for item in previous_classes]) if previous_classes else np.array([], dtype=np.int64)
        )

        for class_id, sample_count in stage_counts.items():
            train_parts.append(sample_indices(train_by_class[class_id], sample_count, rng))

        current_train_indices = np.concatenate(train_parts)

        for class_id in current_classes:
            if class_id not in seen_classes:
                seen_classes.append(class_id)
        seen_test_indices = np.concatenate([test_by_class[item] for item in seen_classes])
        active_classes_for_stage = seen_classes.copy() if config.training.mask_future_classes else None

        print()
        print(f"Task {stage}/{len(training_schedule)}: {task_name} ({len(current_train_indices)} training examples)")
        print(f"Epochs: {stage_epochs}, lambda_kd: {stage_lambda_kd:.4f}")

        train_counts = count_by_class(current_train_indices, labels, shape_names)
        print("Training mix: " + " | ".join(f"{shape_names[item]}={train_counts[shape_names[item]]}" for item in seen_classes))

        stage_class_counts = count_by_class_id(indices=current_train_indices, labels=labels, class_ids=seen_classes)
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
            update_grid=not (config.training.freeze_grid_after_first and stage > 1),
            epochs=stage_epochs,
            lambda_kd=stage_lambda_kd,
            config=config,
        )

        train_loss = np.asarray(fit_results["train_loss"], dtype=float)
        test_loss = np.asarray(fit_results["test_loss"], dtype=float)
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
            record[f"train_count_{shape_names[seen_class_id]}"] = train_counts[shape_names[seen_class_id]]
            record[f"acc_{shape_names[seen_class_id]}"] = accuracy(
                model=model,
                features=features,
                labels=labels,
                indices=test_by_class[seen_class_id],
                device=device,
                active_classes=active_classes_for_stage,
            )

        separate_test_summary = []
        for test_class_id, test_class_name in enumerate(shape_names):
            class_seen = test_class_id in seen_classes
            class_indices = test_by_class[test_class_id]
            class_loss = cross_entropy_loss(
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
                separate_test_summary.append(f"{test_class_name}: loss={class_loss:.3f}, acc={class_accuracy:.3f}")

        records.append(record)
        per_task = " | ".join(f"{shape_names[item]}={record[f'acc_{shape_names[item]}']:.3f}" for item in seen_classes)
        print(f"Seen accuracy: mean={record['mean_seen_accuracy']:.3f} | {per_task}")
        print(f"Separate tests: {' | '.join(separate_test_summary)}")

        if config.plots.enable_confusion_matrices:
            confusion_matrix = compute_confusion_matrix(
                model=model,
                features=features,
                labels=labels,
                indices=seen_test_indices,
                device=device,
                class_ids=seen_classes,
                active_classes=active_classes_for_stage,
            )
            confusion_csv, confusion_image = save_confusion_matrix(
                matrix=confusion_matrix,
                class_names=[shape_names[class_id] for class_id in seen_classes],
                output_dir=plot_dir,
                stage=stage,
                task_name=task_name,
            )
            print(f"Confusion matrix saved to: {confusion_image}")
            print(f"Confusion matrix CSV saved to: {confusion_csv}")

        if config.plots.enable_predictions:
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
                examples_per_class=config.plots.prediction_examples,
                seed=config.split.seed,
                active_classes=active_classes_for_stage,
            )
            print(f"Prediction plots saved in: {plot_dir}")

    metrics = pd.DataFrame(records)
    class_test_metrics = pd.DataFrame(class_test_records)

    if config.plots.enable_loss:
        plot_continual_losses(loss_history, plot_dir)
        final_loss_paths = save_final_loss_summary(loss_history, plot_dir)
        print(f"Continual loss plot saved to: {plot_dir / 'continual_loss.png'}")
        if final_loss_paths is not None:
            print(f"Final loss summary saved to: {final_loss_paths[0]}")
            print(f"Final loss JSON saved to: {final_loss_paths[1]}")

    if config.plots.enable_error:
        plot_continual_error(records, plot_dir)
        print(f"Continual error plot saved to: {plot_dir / 'continual_error.png'}")

    if config.plots.enable_class_tests:
        plot_class_test_metrics(class_test_metrics, plot_dir)
        print(f"Separate class test plot saved to: {plot_dir / 'continual_class_tests.png'}")

    if config.plots.enable_edge_functions:
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
        print(f"Final edge-function plot saved to: {edge_image_path}")
        print(f"Active-edge table saved to: {edge_table_path}")

    final_record = records[-1]
    forgetting_values = []
    best_accuracy_by_class: dict[str, float] = {}
    final_accuracy_by_class: dict[str, float] = {}
    forgetting_by_class: dict[str, float] = {}

    print()
    print("Final forgetting:")
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
        print(f"{class_name}: best={best_accuracy:.3f}, final={final_accuracy:.3f}, forgetting={forgetting:.3f}")

    mean_forgetting = float(np.mean(forgetting_values)) if forgetting_values else float("nan")
    print(f"Mean forgetting: {mean_forgetting:.3f}")
    print(f"Final mean accuracy on seen tasks: {final_record['mean_seen_accuracy']:.3f}")

    run_config = build_run_config(
        config=config,
        input_names=input_names,
        shape_names=shape_names,
        training_schedule=training_schedule,
        device=device,
        standardization_mean=standardization_mean,
        standardization_std=standardization_std,
    )
    run_config["final_results"] = {
        "mean_forgetting": json_number(mean_forgetting),
        "final_mean_seen_accuracy": json_number(final_record["mean_seen_accuracy"]),
        "best_accuracy_by_class": {class_name: json_number(value) for class_name, value in best_accuracy_by_class.items()},
        "final_accuracy_by_class": {class_name: json_number(value) for class_name, value in final_accuracy_by_class.items()},
        "forgetting_by_class": {class_name: json_number(value) for class_name, value in forgetting_by_class.items()},
    }

    if config.output.final_model_path is not None:
        save_final_model(model, config.output.final_model_path, run_config)
        print(f"Final model checkpoint saved to: {config.output.final_model_path}")
    if config.output.run_config_path is not None:
        write_json(config.output.run_config_path, run_config)
        print(f"Run configuration saved to: {config.output.run_config_path}")
    if config.output.accuracy_config_path is not None:
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
        write_json(config.output.accuracy_config_path, accuracy_config)
        print(f"Accuracy config saved to: {config.output.accuracy_config_path}")
    if config.output.metrics_path is not None:
        metrics.to_csv(config.output.metrics_path, index=False)
        print(f"Metrics saved to: {config.output.metrics_path}")
    if config.output.class_tests_path is not None:
        class_test_metrics.to_csv(config.output.class_tests_path, index=False)
        print(f"Separate class tests saved to: {config.output.class_tests_path}")

    return metrics


def resolve_cli_config(config_path: str | None, variant: str | None, default_config: str | Path) -> Path:
    """Choose the TOML config from either --config, --variant, or the default path."""
    if config_path:
        return Path(config_path)

    if variant:
        return CONTINUAL_CONFIG_VARIANTS[variant]

    return Path(default_config)


def main(argv: list[str] | None = None, default_config: str | Path = DEFAULT_CONFIG_PATH) -> int:
    parser = argparse.ArgumentParser(description="Run the continual conic KAN experiment.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a custom TOML config file. If provided, it overrides --variant.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(CONTINUAL_CONFIG_VARIANTS),
        default=None,
        help=(
            "Built-in continual config to run. "
            "Use 'standard' for continual.toml or 'reversed' for continual_reversed.toml."
        ),
    )
    args = parser.parse_args(argv)
    selected_config = resolve_cli_config(args.config, args.variant, default_config)
    print(f"Continual config: {selected_config}")
    run_continual(selected_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
