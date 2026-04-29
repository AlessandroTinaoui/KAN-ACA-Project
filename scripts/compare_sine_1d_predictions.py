"""Print deterministic 1D regression predictions for KAN and MLP exports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from kan import KAN


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kan_models.common.functions import FunctionTerm, RegressionFunction, expression_from_terms


DEFAULT_KAN_EXPORT = ROOT / "artifacts" / "sine_1d" / "kan_1_4_1" / "mini_kan_riscv_export.json"
LEGACY_KAN_EXPORT = ROOT / "artifacts" / "sine_1d" / "mini_kan_riscv_export.json"
DEFAULT_MLP_EXPORT = ROOT / "artifacts" / "sine_1d_mlp" / "mlp_riscv_export.json"


def function_from_export(export: dict[str, Any]) -> RegressionFunction:
    """Rebuild the target function metadata saved in a model export."""
    terms = [
        FunctionTerm(amplitude=float(term["amplitude"]), frequency=float(term["frequency"]))
        for term in export.get("function_terms", [])
    ]
    expression = str(export.get("function_expression", "")).strip()
    if not expression:
        expression = expression_from_terms(terms)
    return RegressionFunction(
        name=str(export.get("function", expression)),
        expression=expression,
        terms=terms,
    )


def activation(name: str, value: float) -> float:
    normalized = name.strip().lower()
    if normalized == "tanh":
        return math.tanh(value)
    if normalized == "relu":
        return max(0.0, value)
    if normalized == "silu":
        return value / (1.0 + math.exp(-value))
    raise ValueError(f"Unsupported MLP activation: {name}")


def mlp_forward(export: dict[str, Any], x_value: float) -> float:
    """Evaluate the exported MLP in plain Python."""
    values = [x_value]
    activation_name = str(export["activation"])
    layers = export["layers"]
    for layer_index, layer in enumerate(layers):
        next_values = []
        for weights, bias in zip(layer["weights"], layer["bias"]):
            next_values.append(sum(float(w) * v for w, v in zip(weights, values)) + float(bias))
        if layer_index < len(layers) - 1:
            next_values = [activation(activation_name, value) for value in next_values]
        values = next_values
    if len(values) != 1:
        raise ValueError(f"Expected scalar MLP output, got {len(values)} values.")
    return values[0]


def _legacy_kan_layers(export: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt the old one-spline export format to the new layered format."""
    return [
        {
            "layer_index": 0,
            "input_dim": int(export["input_dim"]),
            "output_dim": int(export["output_dim"]),
            "knots": [export["knots"]],
            "control_points": [[export["control_points"]]],
            "scale_base": [[export.get("scale_base", 0.0)]],
            "scale_sp": [[export.get("scale_sp", 1.0)]],
            "mask": [[1.0]],
        }
    ]


def build_kan_from_export(export: dict[str, Any]) -> KAN:
    """Rebuild a pykan model from either the legacy or layered JSON export."""
    layers = export.get("layers")
    if not layers:
        layers = _legacy_kan_layers(export)
        width = [int(export["input_dim"]), int(export["output_dim"])]
    else:
        width = list(export["width"])

    model = KAN(
        width=width,
        grid=int(export["num_intervals"]),
        k=int(export["degree"]),
        grid_range=[float(export["x_min"]), float(export["x_max"])],
        symbolic_enabled=False,
        auto_save=False,
        device="cpu",
    )
    model.eval()

    with torch.no_grad():
        for layer_payload, layer in zip(layers, model.act_fun):
            layer.grid.copy_(torch.tensor(layer_payload["knots"], dtype=torch.float32))
            layer.coef.copy_(torch.tensor(layer_payload["control_points"], dtype=torch.float32))
            layer.scale_base.copy_(torch.tensor(layer_payload["scale_base"], dtype=torch.float32))
            layer.scale_sp.copy_(torch.tensor(layer_payload["scale_sp"], dtype=torch.float32))
            if "mask" in layer_payload:
                layer.mask.copy_(torch.tensor(layer_payload["mask"], dtype=torch.float32))
    return model


@torch.no_grad()
def kan_forward(model: KAN, x_value: float) -> float:
    x_tensor = torch.tensor([[x_value]], dtype=torch.float32)
    return float(model(x_tensor).reshape(-1)[0].cpu())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_kan_export() -> Path:
    """Prefer the architecture-subfolder export, falling back to the legacy path."""
    if DEFAULT_KAN_EXPORT.exists():
        return DEFAULT_KAN_EXPORT
    return LEGACY_KAN_EXPORT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare KAN and MLP regression predictions.")
    parser.add_argument("--kan-export", type=Path, default=default_kan_export())
    parser.add_argument("--mlp-export", type=Path, default=DEFAULT_MLP_EXPORT)
    args = parser.parse_args(argv)

    kan_export = load_json(args.kan_export)
    mlp_export = load_json(args.mlp_export)
    target = function_from_export(kan_export)
    x_min = float(kan_export["x_min"])
    x_max = float(kan_export["x_max"])
    x_values = torch.linspace(x_min, x_max, 10, dtype=torch.float64).tolist()
    kan_model = build_kan_from_export(kan_export)

    print("x,y_target,y_kan_pykan,y_mlp_python")
    for x_value in x_values:
        target_y = target.value(torch.tensor([[x_value]], dtype=torch.float64))
        print(
            f"{x_value:.10f},"
            f"{float(target_y.reshape(-1)[0]):.10f},"
            f"{kan_forward(kan_model, x_value):.10f},"
            f"{mlp_forward(mlp_export, x_value):.10f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
