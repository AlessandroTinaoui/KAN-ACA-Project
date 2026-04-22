"""Entry point for tabular KAN experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kan_models.common.tabular.config import ExperimentConfig, load_config
from kan_models.common.tabular.data import build_preprocessor, encode_target, load_dataframe, resolve_feature_groups, split_dataframe, transform_features
from kan_models.common.tabular.training import save_artifacts, train_standard, train_with_pruning


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "credit_default" / "default.toml"


def run_experiment(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Load config, train the model, and save final artifacts."""
    experiment = load_config(config_path)
    frame = load_dataframe(experiment.data)

    if experiment.data.target_column not in frame.columns:
        raise ValueError(f"Colonna target '{experiment.data.target_column}' non trovata nel dataset.")

    X_frame, numeric_columns, categorical_columns = resolve_feature_groups(frame, experiment.data)
    y_encoded, task = encode_target(frame[experiment.data.target_column])
    split = split_dataframe(X_frame, y_encoded, experiment.split)
    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        config=experiment.preprocessing,
    )
    X_train, X_val, X_test = transform_features(preprocessor, split)

    if experiment.pruning.enabled:
        model, metrics = train_with_pruning(
            experiment=experiment,
            task=task,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=split.y_train,
            y_val=split.y_val,
            y_test=split.y_test,
        )
    else:
        model, metrics = train_standard(
            experiment=experiment,
            task=task,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=split.y_train,
            y_val=split.y_val,
            y_test=split.y_test,
        )

    feature_names_out = preprocessor.get_feature_names_out().tolist()
    save_artifacts(
        experiment=experiment,
        model=model,
        preprocessor=preprocessor,
        task=task,
        metrics=metrics,
        feature_names_out=feature_names_out,
    )

    print("\nRisultati finali")
    print(f"Train metrics: {json.dumps(metrics['train'], ensure_ascii=True)}")
    print(f"Validation metrics: {json.dumps(metrics['val'], ensure_ascii=True)}")
    print(f"Test metrics: {json.dumps(metrics['test'], ensure_ascii=True)}")
    print(f"Artefatti salvati in: {experiment.output.output_dir}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Allena una KAN tabellare con pykan usando una configurazione TOML.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Percorso al file TOML di configurazione. Default: {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args(argv)
    run_experiment(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
