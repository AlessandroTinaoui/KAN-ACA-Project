"""Train and export a small MLP for the 1D sine regression task."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kan_models.common.runtime import configure_matplotlib, detect_device, ensure_directory
from kan_models.common.shared import clone_state_dict, load_toml, resolve_path, write_json

configure_matplotlib()

import matplotlib.pyplot as plt


DEFAULT_PARAMS_PATH = Path(__file__).resolve().with_name("params.toml")


@dataclass(frozen=True)
class SineTerm:
    """amplitude * sin(2*pi*frequency*x)"""

    amplitude: float
    frequency: float


@dataclass(frozen=True)
class FunctionConfig:
    """Target function"""

    name: str
    terms: list[SineTerm]


@dataclass(frozen=True)
class ModelConfig:
    """MLP architecture"""

    input_dim: int
    output_dim: int
    hidden_layers: list[int]
    activation: str
    x_min: float
    x_max: float

    @property
    def architecture(self) -> list[int]:
        """List of layer sizes"""
        return [self.input_dim, *self.hidden_layers, self.output_dim]


@dataclass(frozen=True)
class DataConfig:
    """Train, validation and test samples."""

    seed: int
    train_samples: int
    validation_samples: int
    test_samples: int


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization parameters."""

    steps: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    early_stopping_min_delta: float


@dataclass(frozen=True)
class EvaluationConfig:
    """Metrics and plot parameters."""

    accuracy_tolerance: float
    plot_points: int


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime device choice."""

    device: str


@dataclass(frozen=True)
class OutputConfig:
    """Output paths."""

    output_dir: Path
    export_json: Path
    metrics_json: Path
    checkpoint_pt: Path
    fit_plot: Path
    loss_plot: Path


@dataclass(frozen=True)
class MlpSine1DConfig:
    """Full local configuration."""

    config_path: Path
    function: FunctionConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig
    output: OutputConfig


class SineMLP(nn.Module):
    def __init__(self, layer_sizes: list[int], activation: str) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("The MLP needs at least an input and an output layer.")

        activation_module = make_activation(activation)
        layers: list[nn.Module] = []
        for layer_index, (in_features, out_features) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            layers.append(nn.Linear(in_features, out_features))
            if layer_index < len(layer_sizes) - 2:
                layers.append(activation_module())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_activation(name: str) -> type[nn.Module]:
    """Return the activation class used between hidden layers."""
    normalized = name.strip().lower()
    if normalized == "tanh":
        return nn.Tanh
    if normalized == "relu":
        return nn.ReLU
    if normalized == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation '{name}'. Use tanh, relu, or silu.")


def _required_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing [{name}] section in the sine_1d_mlp parameter file.")
    return section


def _load_function_config(section: dict[str, Any]) -> FunctionConfig:
    terms = [
        SineTerm(amplitude=float(item["amplitude"]), frequency=float(item["frequency"]))
        for item in section.get("terms", [])
    ]
    if not terms:
        raise ValueError("At least one [[function.terms]] entry is required.")
    return FunctionConfig(name=str(section.get("name", "custom_sine_function")), terms=terms)


def _load_output_config(config_dir: Path, section: dict[str, Any]) -> OutputConfig:
    output_dir = resolve_path(config_dir, section["output_dir"])
    return OutputConfig(
        output_dir=output_dir,
        export_json=output_dir / str(section["export_json"]),
        metrics_json=output_dir / str(section["metrics_json"]),
        checkpoint_pt=output_dir / str(section["checkpoint_pt"]),
        fit_plot=output_dir / str(section["fit_plot"]),
        loss_plot=output_dir / str(section["loss_plot"]),
    )


def validate_config(config: MlpSine1DConfig) -> None:
    model = config.model
    if model.input_dim != 1 or model.output_dim != 1:
        raise ValueError("The RISC-V MLP export expects input_dim=1 and output_dim=1.")
    if not model.hidden_layers:
        raise ValueError("Use at least one hidden layer for this baseline.")
    if any(width <= 0 for width in model.hidden_layers):
        raise ValueError("All hidden layer sizes must be positive.")
    if model.x_min >= model.x_max:
        raise ValueError("x_min must be smaller than x_max.")


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> MlpSine1DConfig:
    """Load every parameter from the TOML file."""
    config_path, raw = load_toml(path)
    config_dir = config_path.parent

    function = _load_function_config(_required_section(raw, "function"))
    model = ModelConfig(**_required_section(raw, "model"))
    data = DataConfig(**_required_section(raw, "data"))
    training = TrainingConfig(**_required_section(raw, "training"))
    evaluation = EvaluationConfig(**_required_section(raw, "evaluation"))
    runtime = RuntimeConfig(**_required_section(raw, "runtime"))
    output = _load_output_config(config_dir, _required_section(raw, "output"))

    config = MlpSine1DConfig(
        config_path=config_path,
        function=function,
        model=model,
        data=data,
        training=training,
        evaluation=evaluation,
        runtime=runtime,
        output=output,
    )
    validate_config(config)
    return config


def target_function(x: torch.Tensor, config: FunctionConfig) -> torch.Tensor:
    """Scalar function to approximate."""
    y = torch.zeros_like(x)
    for term in config.terms:
        y = y + term.amplitude * torch.sin(2.0 * math.pi * term.frequency * x)
    return y


def make_dataset(config: MlpSine1DConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Build simple train/validation/test in [x_min, x_max]."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.data.seed)

    x_min = config.model.x_min
    x_max = config.model.x_max
    train_x = torch.rand((config.data.train_samples, 1), generator=generator, dtype=torch.float32)
    validation_x = torch.rand((config.data.validation_samples, 1), generator=generator, dtype=torch.float32)
    test_x = torch.rand((config.data.test_samples, 1), generator=generator, dtype=torch.float32)
    train_x = x_min + (x_max - x_min) * train_x
    validation_x = x_min + (x_max - x_min) * validation_x
    test_x = x_min + (x_max - x_min) * test_x

    return {
        "train_input": train_x.to(device),
        "train_label": target_function(train_x, config.function).to(device),
        "validation_input": validation_x.to(device),
        "validation_label": target_function(validation_x, config.function).to(device),
        "test_input": test_x.to(device),
        "test_label": target_function(test_x, config.function).to(device),
    }


def build_model(config: MlpSine1DConfig, device: torch.device) -> SineMLP:
    """Build and initialize the MLP."""
    torch.manual_seed(config.data.seed)
    model = SineMLP(config.model.architecture, config.model.activation).to(device)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return model


@torch.no_grad()
def mse_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Return mean squared error."""
    model.eval()
    prediction = model(x)
    return float(torch.nn.functional.mse_loss(prediction, y).cpu())


@torch.no_grad()
def mae_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Return mean absolute error."""
    model.eval()
    prediction = model(x)
    return float(torch.nn.functional.l1_loss(prediction, y).cpu())


@torch.no_grad()
def tolerance_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, tolerance: float) -> float:
    model.eval()
    prediction = model(x)
    return float((prediction.sub(y).abs() <= tolerance).float().mean().cpu())


@torch.no_grad()
def r2_score(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute R^2."""
    model.eval()
    prediction = model(x)
    residual = torch.sum((y - prediction) ** 2)
    total = torch.sum((y - torch.mean(y)) ** 2)
    return float((1.0 - residual / total).cpu())


def train_model(
    model: nn.Module,
    dataset: dict[str, torch.Tensor],
    config: MlpSine1DConfig,
) -> tuple[nn.Module, dict[str, list[float] | float | int | bool | str]]:
    """Train the MLP."""
    training = config.training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training.steps,
        eta_min=training.min_learning_rate,
    )

    train_x = dataset["train_input"]
    train_y = dataset["train_label"]
    validation_x = dataset["validation_input"]
    validation_y = dataset["validation_label"]

    best_validation_mse = float("inf")
    best_state = clone_state_dict(model)
    best_step = 0
    steps_without_improvement = 0
    stopped_early = False

    history: dict[str, list[float] | float | int | bool | str] = {
        "train_mse": [],
        "validation_mse": [],
        "learning_rate": [],
        "best_step": 0,
        "best_validation_mse": best_validation_mse,
        "stopped_early": False,
        "stop_reason": "max_steps",
    }

    for step in range(1, training.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_x)
        loss = torch.nn.functional.mse_loss(prediction, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_mse_value = mse_loss(model, train_x, train_y)
        validation_mse_value = mse_loss(model, validation_x, validation_y)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_mse"].append(train_mse_value)
        history["validation_mse"].append(validation_mse_value)
        history["learning_rate"].append(current_lr)

        if validation_mse_value < best_validation_mse - training.early_stopping_min_delta:
            best_validation_mse = validation_mse_value
            best_state = clone_state_dict(model)
            best_step = step
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if steps_without_improvement >= training.early_stopping_patience:
            stopped_early = True
            history["stop_reason"] = "validation_mse_plateau"
            break

    model.load_state_dict(best_state)
    history["best_step"] = best_step
    history["best_validation_mse"] = best_validation_mse
    history["stopped_early"] = stopped_early
    history["trained_steps"] = len(history["train_mse"])
    history["final_learning_rate"] = (
        history["learning_rate"][-1] if history["learning_rate"] else training.learning_rate
    )
    return model, history


def extract_linear_layers(model: SineMLP) -> list[dict[str, object]]:
    """Export every layer for inference."""
    exported_layers: list[dict[str, object]] = []
    linear_index = 0
    for module in model.network:
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight.detach().cpu()
        bias = module.bias.detach().cpu()
        exported_layers.append(
            {
                "name": f"linear_{linear_index}",
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "weight_layout": "out_features x in_features",
                "weights": [[float(value) for value in row] for row in weight.tolist()],
                "bias": [float(value) for value in bias.tolist()],
            }
        )
        linear_index += 1
    return exported_layers


def extract_export_payload(
    model: SineMLP,
    config: MlpSine1DConfig,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Extract the exact parameters needed for inference"""
    model_config = config.model
    return {
        "model_type": "sine_1d_mlp",
        "config_path": str(config.config_path),
        "function": config.function.name,
        "function_terms": [vars(term) for term in config.function.terms],
        "input_dim": model_config.input_dim,
        "output_dim": model_config.output_dim,
        "architecture": model_config.architecture,
        "hidden_layers": model_config.hidden_layers,
        "activation": model_config.activation,
        "activation_applied_after": "all hidden linear layers",
        "output_activation": "linear",
        "x_min": model_config.x_min,
        "x_max": model_config.x_max,
        "data_config": vars(config.data),
        "training_config": vars(config.training),
        "layers": extract_linear_layers(model),
        "metrics": metrics,
    }


@torch.no_grad()
def save_fit_plot(model: nn.Module, output_file: Path, device: torch.device, config: MlpSine1DConfig) -> None:
    """Save a comparison plot between the target function and the fitted MLP."""
    model_config = config.model
    x_plot = torch.linspace(
        model_config.x_min,
        model_config.x_max,
        config.evaluation.plot_points,
        dtype=torch.float32,
    ).unsqueeze(1).to(device)
    y_true = target_function(x_plot, config.function).cpu().numpy().reshape(-1)
    y_pred = model(x_plot).cpu().numpy().reshape(-1)

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(x_plot.cpu().numpy().reshape(-1), y_true, label="target", linewidth=2)
    ax.plot(x_plot.cpu().numpy().reshape(-1), y_pred, label="MLP", linewidth=2, linestyle="--")
    ax.set_title("MLP fit on the 1D sinusoidal target")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def save_loss_plot(history: dict[str, list[float] | float | int | bool | str], output_file: Path) -> None:
    """Save the train/validation MSE plots."""
    train_mse = history["train_mse"]
    validation_mse = history["validation_mse"]
    steps = list(range(1, len(train_mse) + 1))

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(steps, train_mse, label="train MSE", linewidth=2)
    ax.plot(steps, validation_mse, label="validation MSE", linewidth=2)
    ax.set_title("MLP training history")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def run(params_path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, object]:
    """Run the MLP NN."""
    config = load_params(params_path)
    ensure_directory(config.output.output_dir)
    device = detect_device(config.runtime.device)
    dataset = make_dataset(config, device)
    model = build_model(config, device)
    model, history = train_model(model, dataset, config)

    train_x = dataset["train_input"]
    train_y = dataset["train_label"]
    validation_x = dataset["validation_input"]
    validation_y = dataset["validation_label"]
    test_x = dataset["test_input"]
    test_y = dataset["test_label"]
    tolerance = config.evaluation.accuracy_tolerance

    metrics: dict[str, object] = {
        "config_path": str(config.config_path),
        "device": str(device),
        "train_mse": mse_loss(model, train_x, train_y),
        "validation_mse": mse_loss(model, validation_x, validation_y),
        "test_mse": mse_loss(model, test_x, test_y),
        "train_mae": mae_loss(model, train_x, train_y),
        "validation_mae": mae_loss(model, validation_x, validation_y),
        "test_mae": mae_loss(model, test_x, test_y),
        "accuracy_tolerance": tolerance,
        "train_accuracy_tol": tolerance_accuracy(model, train_x, train_y, tolerance),
        "validation_accuracy_tol": tolerance_accuracy(model, validation_x, validation_y, tolerance),
        "test_accuracy_tol": tolerance_accuracy(model, test_x, test_y, tolerance),
        "train_r2": r2_score(model, train_x, train_y),
        "validation_r2": r2_score(model, validation_x, validation_y),
        "test_r2": r2_score(model, test_x, test_y),
        "best_step": int(history["best_step"]),
        "trained_steps": int(history["trained_steps"]),
        "best_validation_mse": float(history["best_validation_mse"]),
        "stopped_early": bool(history["stopped_early"]),
        "stop_reason": str(history["stop_reason"]),
    }

    export_payload = extract_export_payload(model, config, metrics)
    write_json(config.output.export_json, export_payload)
    write_json(config.output.metrics_json, metrics)
    torch.save({"state_dict": model.state_dict(), "export": export_payload}, config.output.checkpoint_pt)
    save_fit_plot(model, config.output.fit_plot, device, config)
    save_loss_plot(history, config.output.loss_plot)

    print(f"Config: {config.config_path}")
    print(f"Device: {device}")
    print("MLP trained for 1D sinusoidal regression")
    print(f"Architecture: {config.model.architecture}, activation={config.model.activation}")
    print(f"Train MSE: {metrics['train_mse']:.8f}")
    print(f"Validation MSE: {metrics['validation_mse']:.8f}")
    print(f"Test MSE: {metrics['test_mse']:.8f}")
    print(f"Train accuracy @ |error| <= {tolerance:.4f}: {metrics['train_accuracy_tol']:.4f}")
    print(f"Validation accuracy @ |error| <= {tolerance:.4f}: {metrics['validation_accuracy_tol']:.4f}")
    print(f"Test accuracy @ |error| <= {tolerance:.4f}: {metrics['test_accuracy_tol']:.4f}")
    print(f"Train R^2: {metrics['train_r2']:.6f}")
    print(f"Validation R^2: {metrics['validation_r2']:.6f}")
    print(f"Test R^2: {metrics['test_r2']:.6f}")
    print(f"Best step: {metrics['best_step']}, trained steps: {metrics['trained_steps']}")
    print(f"Export JSON saved to: {config.output.export_json}")
    print(f"Metrics JSON saved to: {config.output.metrics_json}")
    print(f"Checkpoint saved to: {config.output.checkpoint_pt}")
    print(f"Fit plot saved to: {config.output.fit_plot}")
    print(f"Loss plot saved to: {config.output.loss_plot}")

    return export_payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train and export the sine_1d MLP baseline.")
    parser.add_argument(
        "--params",
        default=str(DEFAULT_PARAMS_PATH),
        help=f"Path to the local TOML parameter file. Default: {DEFAULT_PARAMS_PATH}",
    )
    args = parser.parse_args(argv)
    run(args.params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
