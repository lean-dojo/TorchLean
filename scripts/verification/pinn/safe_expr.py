#!/usr/bin/env python3
"""
Restricted expression evaluator for PINN training helpers.

The PINN workflows only need a small arithmetic DSL over tensors/scalars plus a short whitelist of
math helper functions. The evaluator avoids Python `eval` so CLI-supplied expressions stay in the
realm of "math on named variables" rather than "arbitrary code execution".
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Mapping

import numpy as np
import torch


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda x: x,
    ast.USub: operator.neg,
}

_ALLOWED_NAMES = {
    "abs": abs,
}

_ALLOWED_ATTRS = {
    "math": {
        "pi": math.pi,
        "e": math.e,
        "sin": math.sin,
        "cos": math.cos,
        "tanh": math.tanh,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
    },
    "np": {
        "pi": np.pi,
        "e": np.e,
        "sin": np.sin,
        "cos": np.cos,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
    },
    "torch": {
        "pi": torch.pi,
        "sin": torch.sin,
        "cos": torch.cos,
        "tanh": torch.tanh,
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
        "sigmoid": torch.sigmoid,
        "abs": torch.abs,
        "zeros_like": torch.zeros_like,
        "ones_like": torch.ones_like,
    },
}


class _SafeExprEvaluator(ast.NodeVisitor):
    """AST visitor for the restricted PINN expression language."""

    def __init__(self, env: Mapping[str, Any]):
        self.env = dict(env)

    def visit_Expression(self, node: ast.Expression) -> Any:
        """Evaluate the root expression node."""
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        """Resolve environment variables and allowlisted bare helper names."""
        if node.id in self.env:
            return self.env[node.id]
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Unknown name '{node.id}'")

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Accept numeric literals and reject other constants."""
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant {node.value!r}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Evaluate an allowlisted binary operator."""
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary operator {type(node.op).__name__}")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Evaluate an allowlisted unary operator."""
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator {type(node.op).__name__}")
        return op(self.visit(node.operand))

    def visit_Call(self, node: ast.Call) -> Any:
        """Evaluate calls to functions reached through allowlisted names/attributes."""
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Resolve allowlisted module attributes such as `torch.sin`."""
        if not isinstance(node.value, ast.Name):
            raise ValueError("Only simple module attributes are allowed")
        base = node.value.id
        allowed = _ALLOWED_ATTRS.get(base)
        if allowed is None or node.attr not in allowed:
            raise ValueError(f"Unsupported attribute '{base}.{node.attr}'")
        return allowed[node.attr]

    def generic_visit(self, node: ast.AST) -> Any:  # pragma: no cover - exercised via failures
        """Reject every syntax form outside the small expression language."""
        raise ValueError(f"Unsupported syntax {type(node).__name__}")


def eval_expr(expr: str, env: Mapping[str, Any]) -> Any:
    """Evaluate one restricted expression against an explicit environment."""
    tree = ast.parse(expr, mode="eval")
    return _SafeExprEvaluator(env).visit(tree)
