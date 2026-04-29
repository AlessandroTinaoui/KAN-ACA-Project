"""Target functions for regression experiments."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any

import torch


_FUNCTIONS = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
    "square": torch.square,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


class _ExpressionValidator(ast.NodeVisitor):
    """Validate a small math expression language over the variable x."""

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported syntax in function expression: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id != "x" and node.id not in _FUNCTIONS and node.id not in _CONSTANTS:
            raise ValueError(f"Unsupported name in function expression: {node.id}")

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("Only simple calls to supported math functions are allowed.")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in function expressions.")
        for arg in node.args:
            self.visit(arg)


@dataclass(frozen=True)
class FunctionTerm:
    """Legacy sinusoidal term: amplitude * sin(2*pi*frequency*x)."""

    amplitude: float
    frequency: float


@dataclass(frozen=True)
class RegressionFunction:
    """A named scalar target function defined by a string expression."""

    name: str
    expression: str
    terms: list[FunctionTerm]

    def __post_init__(self) -> None:
        tree = ast.parse(self.expression, mode="eval")
        _ExpressionValidator().visit(tree)
        object.__setattr__(self, "_code", compile(tree, "<regression_function>", "eval"))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the target function at tensor x."""
        namespace: dict[str, Any] = {"x": x, **_FUNCTIONS, **_CONSTANTS}
        result = eval(self._code, {"__builtins__": {}}, namespace)
        if not isinstance(result, torch.Tensor):
            result = torch.as_tensor(result, dtype=x.dtype, device=x.device)
        return result.to(dtype=x.dtype, device=x.device)

    def to_json(self) -> dict[str, Any]:
        """Return JSON-friendly metadata."""
        return {
            "name": self.name,
            "expression": self.expression,
            "terms": [vars(term) for term in self.terms],
        }


def expression_from_terms(terms: list[FunctionTerm]) -> str:
    """Build a generic expression string from legacy sine terms."""
    parts: list[str] = []
    for term in terms:
        parts.append(f"({term.amplitude})*sin(2*pi*({term.frequency})*x)")
    return " + ".join(parts)


def load_regression_function(section: dict[str, Any]) -> RegressionFunction:
    """Load a regression function from a TOML [function] section.

    Preferred format:

        [function]
        name = "my function"
        expression = "sin(2*pi*x) + 0.35*cos(6*pi*x)"

    The previous [[function.terms]] sine-only format remains supported.
    """
    terms = [
        FunctionTerm(amplitude=float(item["amplitude"]), frequency=float(item["frequency"]))
        for item in section.get("terms", [])
    ]
    expression = str(section.get("expression", "")).strip()
    if not expression:
        if not terms:
            raise ValueError("Set function.expression or provide at least one [[function.terms]] entry.")
        expression = expression_from_terms(terms)
    name = str(section.get("name", expression))
    return RegressionFunction(name=name, expression=expression, terms=terms)


def make_regression_dataset(
    function: RegressionFunction,
    *,
    x_min: float,
    x_max: float,
    seed: int,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Build deterministic random train/validation/test splits for a 1D function."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    train_x = torch.rand((train_samples, 1), generator=generator, dtype=dtype)
    validation_x = torch.rand((validation_samples, 1), generator=generator, dtype=dtype)
    test_x = torch.rand((test_samples, 1), generator=generator, dtype=dtype)
    train_x = x_min + (x_max - x_min) * train_x
    validation_x = x_min + (x_max - x_min) * validation_x
    test_x = x_min + (x_max - x_min) * test_x

    return {
        "train_input": train_x.to(device),
        "train_label": function.value(train_x).to(device),
        "validation_input": validation_x.to(device),
        "validation_label": function.value(validation_x).to(device),
        "test_input": test_x.to(device),
        "test_label": function.value(test_x).to(device),
    }
