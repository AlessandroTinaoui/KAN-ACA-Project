"""Train and export a minimal 1D KAN from a local TOML parameter file."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from kan import KAN

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from kan_models.common.runtime import configure_matplotlib, detect_device, ensure_directory
from kan_models.common.shared import clone_state_dict, load_toml, resolve_path, write_json

configure_matplotlib()

import matplotlib.pyplot as plt


DEFAULT_PARAMS_PATH = Path(__file__).resolve().with_name("params.toml")


@dataclass(frozen=True)
class SineTerm:
    """amplitude * sin(2*pi*frequency*x)."""

    amplitude: float
    frequency: float


@dataclass(frozen=True)
class FunctionConfig:
    """Target function."""

    name: str
    terms: list[SineTerm]


@dataclass(frozen=True)
class ModelConfig:
    """Config for exporting parameters."""

    input_dim: int
    output_dim: int
    degree: int
    num_control_points: int
    num_knots: int
    x_min: float
    x_max: float
    noise_scale: float
    disable_base_branch: bool
    scale_sp: float

    @property
    def num_intervals(self) -> int:
        """pykan grid intervals implied by control points and spline degree."""
        return self.num_control_points - self.degree


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
    """Resolved output paths."""

    output_dir: Path
    export_json: Path
    metrics_json: Path
    checkpoint_pt: Path
    fit_plot: Path


@dataclass(frozen=True)
class Sine1DConfig:
    """Full local configuration"""

    config_path: Path
    function: FunctionConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig
    output: OutputConfig


def _required_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Parsing TOML"""
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing [{name}] section in the sine_1d parameter file.")
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
    )


def validate_config(config: Sine1DConfig) -> None:
    """Check that the exported spline dimensions match the requested layout."""
    model = config.model
    if model.input_dim != 1 or model.output_dim != 1:
        raise ValueError("The RISC-V mini KAN export expects input_dim=1 and output_dim=1.")
    if model.num_intervals <= 0:
        raise ValueError("num_control_points must be larger than degree.")
    expected_knots = model.num_intervals + 1 + 2 * model.degree
    if expected_knots != model.num_knots:
        raise ValueError(
            "Invalid spline layout: "
            f"num_knots should be {expected_knots}, got {model.num_knots}."
        )
    if model.x_min >= model.x_max:
        raise ValueError("x_min must be smaller than x_max.")


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> Sine1DConfig:
    """Load every modifiable parameter from the local TOML file."""
    config_path, raw = load_toml(path)
    config_dir = config_path.parent

    function = _load_function_config(_required_section(raw, "function"))
    model = ModelConfig(**_required_section(raw, "model"))
    data = DataConfig(**_required_section(raw, "data"))
    training = TrainingConfig(**_required_section(raw, "training"))
    evaluation = EvaluationConfig(**_required_section(raw, "evaluation"))
    runtime = RuntimeConfig(**_required_section(raw, "runtime"))
    output = _load_output_config(config_dir, _required_section(raw, "output"))

    config = Sine1DConfig(
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


def make_dataset(config: Sine1DConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Build simple train/validation/test splits in [x_min, x_max]."""
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
    train_y = target_function(train_x, config.function)
    validation_y = target_function(validation_x, config.function)
    test_y = target_function(test_x, config.function)

    return {
        "train_input": train_x.to(device),
        "train_label": train_y.to(device),
        "validation_input": validation_x.to(device),
        "validation_label": validation_y.to(device),
        "test_input": test_x.to(device),
        "test_label": test_y.to(device),
    }


def build_model(config: Sine1DConfig, device: torch.device) -> KAN:
    """Build the smallest useful KAN: one input edge and one output edge."""
    model_config = config.model
    model = KAN(
        width=[model_config.input_dim, model_config.output_dim],
        grid=model_config.num_intervals,
        k=model_config.degree,
        grid_range=[model_config.x_min, model_config.x_max],
        noise_scale=model_config.noise_scale,
        seed=config.data.seed,
        auto_save=False,
        device=device,
    )

    layer = model.act_fun[0]
    if model_config.disable_base_branch:
        # This makes the exported model depend only on knots + control points.
        layer.scale_base.data.zero_()
        layer.scale_base.requires_grad_(False)
    layer.scale_sp.data.fill_(model_config.scale_sp)
    layer.scale_sp.requires_grad_(False)
    layer.mask.data.fill_(1.0)

    return model


@torch.no_grad()
def mse_loss(model: KAN, x: torch.Tensor, y: torch.Tensor) -> float:
    """Return mean squared error."""
    model.eval()
    prediction = model(x)
    return float(torch.nn.functional.mse_loss(prediction, y).cpu())


@torch.no_grad()
def mae_loss(model: KAN, x: torch.Tensor, y: torch.Tensor) -> float:
    """Return mean absolute error."""
    model.eval()
    prediction = model(x)
    return float(torch.nn.functional.l1_loss(prediction, y).cpu())


@torch.no_grad()
def tolerance_accuracy(model: KAN, x: torch.Tensor, y: torch.Tensor, tolerance: float) -> float:
    """Regression accuracy defined as fraction of samples under an absolute-error threshold."""
    model.eval()
    prediction = model(x)
    return float((prediction.sub(y).abs() <= tolerance).float().mean().cpu())


@torch.no_grad()
def r2_score(model: KAN, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute R^2 for regression sanity-checking."""
    model.eval()
    prediction = model(x)
    residual = torch.sum((y - prediction) ** 2)
    total = torch.sum((y - torch.mean(y)) ** 2)
    return float((1.0 - residual / total).cpu())


def train_model(
    model: KAN,
    dataset: dict[str, torch.Tensor],
    config: Sine1DConfig,
) -> tuple[KAN, dict[str, list[float] | float | int | bool | str]]:
    """Train the mini KAN with a simple full-batch loop."""
    training = config.training
    optimizer = torch.optim.Adam(model.get_params(), lr=training.learning_rate)
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


def extract_export_payload(model: KAN, config: Sine1DConfig) -> dict[str, object]:
    """Extract the exact parameters needed for a lightweight spline-only inference path."""
    model_config = config.model
    layer = model.act_fun[0]
    knots = layer.grid.detach().cpu().reshape(-1).tolist()
    control_points = layer.coef.detach().cpu().reshape(-1).tolist()

    if len(knots) != model_config.num_knots:
        raise ValueError(f"Expected {model_config.num_knots} knots, found {len(knots)}.")
    if len(control_points) != model_config.num_control_points:
        raise ValueError(
            f"Expected {model_config.num_control_points} control points, found {len(control_points)}."
        )

    return {
        "model_type": "mini_kan_spline_only",
        "config_path": str(config.config_path),
        "function": config.function.name,
        "function_terms": [vars(term) for term in config.function.terms],
        "input_dim": model_config.input_dim,
        "output_dim": model_config.output_dim,
        "degree": model_config.degree,
        "num_control_points": model_config.num_control_points,
        "num_knots": model_config.num_knots,
        "num_intervals": model_config.num_intervals,
        "x_min": model_config.x_min,
        "x_max": model_config.x_max,
        "knots": [float(value) for value in knots],
        "control_points": [float(value) for value in control_points],
        "base_branch_disabled": model_config.disable_base_branch,
        "scale_base": 0.0 if model_config.disable_base_branch else float(layer.scale_base.detach().cpu().reshape(-1)[0]),
        "scale_sp": float(layer.scale_sp.detach().cpu().reshape(-1)[0]),
    }


@torch.no_grad()
def save_fit_plot(model: KAN, output_file: Path, device: torch.device, config: Sine1DConfig) -> None:
    """Save a simple comparison plot between the target function and the fitted KAN."""
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
    ax.plot(x_plot.cpu().numpy().reshape(-1), y_pred, label="mini KAN", linewidth=2, linestyle="--")
    ax.set_title("Mini KAN fit on the 1D sinusoidal target")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def run(params_path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, object]:
    """Train, validate, export, and save the mini KAN."""
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

    metrics = {
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

    export_payload = extract_export_payload(model, config)
    export_payload["metrics"] = metrics
    write_json(config.output.export_json, export_payload)
    write_json(config.output.metrics_json, metrics)
    torch.save({"state_dict": model.state_dict(), "export": export_payload}, config.output.checkpoint_pt)
    save_fit_plot(model, config.output.fit_plot, device, config)

    print(f"Config: {config.config_path}")
    print(f"Device: {device}")
    print("Mini KAN [1,1] trained for sinusoidal regression")
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

    return export_payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train and export the sine_1d mini KAN.")
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
