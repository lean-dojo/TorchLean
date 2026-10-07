#!/usr/bin/env python3
"""Train a sequential PINN and export weights for TorchLean's checker.

Choose `evolution` for (x,t) in [-1,1] × [0,1], with initial and boundary
conditions, or `stationary` for (x,y) in [-1,1]², with boundary conditions
and optional interior observations. Both use two network inputs; these are
problem choices, not an arbitrary-dimensional PDE interface.

    python3 scripts/verification/pinn/train_pinn.py evolution --steps 25
    python3 scripts/verification/pinn/train_pinn.py stationary --steps 25

Use --pde-expr and --const to set the residual. Evolution also accepts --ic-expr;
both accept --bc-expr and --dataset-json. Dataset rows use x/t or x/y, with u
for observed values. Stationary problems can use --data-expr instead of a
dataset. --hidden-widths and --activation select the sequential network.

Training produces candidate weights, not a PDE certificate. Lean reconstructs
the model from exported metadata; verification is a separate command.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - fail fast when torch missing
    raise SystemExit("PyTorch is required: pip install torch") from exc

try:
    import numpy as np
except Exception:  # pragma: no cover - optional expression aliases
    np = None


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

if np is not None:
    _ALLOWED_ATTRS["np"] = {
        "pi": np.pi,
        "e": np.e,
        "sin": np.sin,
        "cos": np.cos,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
    }


class _SafeExprEvaluator(ast.NodeVisitor):
    def __init__(self, env: Mapping[str, Any]):
        self.env = dict(env)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.env:
            return self.env[node.id]
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Unknown name '{node.id}'")

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant {node.value!r}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary operator {type(node.op).__name__}")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator {type(node.op).__name__}")
        return op(self.visit(node.operand))

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if not isinstance(node.value, ast.Name):
            raise ValueError("Only simple module attributes are allowed")
        base = node.value.id
        allowed = _ALLOWED_ATTRS.get(base)
        if allowed is None or node.attr not in allowed:
            raise ValueError(f"Unsupported attribute '{base}.{node.attr}'")
        return allowed[node.attr]

    def generic_visit(self, node: ast.AST) -> Any:  # pragma: no cover - exercised via failures
        raise ValueError(f"Unsupported syntax {type(node).__name__}")


def eval_expr(expr: str, env: Mapping[str, Any]) -> Any:
    tree = ast.parse(expr, mode="eval")
    return _SafeExprEvaluator(env).visit(tree)


class PinnDataset:
    """Keeps each JSON section as a tensor on the chosen torch device."""

    def __init__(self, device: torch.device):
        self.device = device
        self.sections: Dict[str, Optional[torch.Tensor]] = {}

    @staticmethod
    def _read_entries(entries, keys: Sequence[str], device: torch.device) -> Optional[torch.Tensor]:
        if entries is None:
            return None
        if not isinstance(entries, list):
            raise ValueError("Dataset sections must be lists of objects.")
        rows: list[list[float]] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Dataset entry {idx} is not an object.")
            try:
                row = [float(entry[k]) for k in keys]
            except KeyError as exc:
                raise ValueError(f"Dataset entry {idx} missing key '{exc.args[0]}'") from exc
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"Dataset entry {idx} must contain finite numbers.")
            rows.append(row)
        if not rows:
            return None
        result = torch.tensor(rows, dtype=torch.float32, device=device)
        if not torch.isfinite(result).all():
            raise ValueError("Dataset values must be representable as finite float32 numbers.")
        return result

    @classmethod
    def load(
        cls,
        path: str,
        schema: Mapping[str, Sequence[str]],
        device: torch.device,
    ) -> "PinnDataset":
        payload = json.loads(Path(path).read_text())
        data = cls(device)
        for section, keys in schema.items():
            data.sections[section] = cls._read_entries(payload.get(section), keys, device)
        return data

    def sample(self, section: str, count: int) -> Optional[torch.Tensor]:
        if count <= 0:
            raise ValueError("Sample count must be positive.")
        mat = self.sections.get(section)
        if mat is None:
            return None
        if mat.shape[0] == 0:
            raise ValueError(f"Dataset section '{section}' is empty; cannot sample.")
        idx = torch.randint(0, mat.shape[0], (count,), device=self.device, dtype=torch.long)
        return mat.index_select(0, idx)

    def sample_columns(self, section: str, count: int, columns: int) -> Optional[tuple[torch.Tensor, ...]]:
        samples = self.sample(section, count)
        if samples is None:
            return None
        if not 0 < columns <= samples.shape[1]:
            raise ValueError("Requested columns exceed the dataset schema.")
        return tuple(samples[:, i : i + 1] for i in range(columns))


def _activation_factory(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation '{name}'")


def parse_hidden_widths(raw: str) -> List[int]:
    tokens = [tok.strip() for tok in raw.split(",")]
    widths: List[int] = []
    for tok in tokens:
        if not tok:
            continue
        try:
            width = int(tok)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid hidden width '{tok}'") from exc
        if width <= 0:
            raise argparse.ArgumentTypeError(f"Hidden width must be positive, got {width}")
        widths.append(width)
    if not widths:
        raise argparse.ArgumentTypeError("Provide at least one hidden layer width (e.g., '16,16').")
    return widths


def build_model(in_dim: int, hidden_widths: Iterable[int], activation: str) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    act = activation.lower()
    for width in hidden_widths:
        lin = nn.Linear(prev, width)
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        layers.append(_activation_factory(act))
        prev = width
    out = nn.Linear(prev, 1)
    nn.init.xavier_uniform_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


def to_json_dict(model: nn.Sequential, *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Export sequential-layer weights, including training metadata when supplied."""
    exported: Dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".weight") or name.endswith(".bias"):
            exported[f"layers.{name}"] = tensor.detach().cpu().numpy().tolist()
    if meta is not None:
        exported["meta"] = meta
    return exported


def gradients(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
        create_graph=True,
    )[0]


def eval_pinn_expr(expr: str, **tensors):
    try:
        value = eval_expr(expr, tensors)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise ValueError(f"Failed to evaluate expression '{expr}': {exc}") from exc
    return value


def ensure_tensor(val, like: torch.Tensor) -> torch.Tensor:
    arr = torch.as_tensor(val, dtype=like.dtype, device=like.device)
    if arr.numel() == 1:
        return arr.reshape(()).expand_as(like)
    return arr.reshape_as(like)


def parse_const_flags(items) -> Dict[str, float]:
    constants: Dict[str, float] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"--const expects name=value, got '{raw}'")
        name, value = raw.split("=", 1)
        name = name.strip()
        reserved = {
            "x", "y", "t", "u", "ux", "uy", "ut", "uxx", "uyy", "utt",
            "uxy", "uyx", "uxt", "utx", "u_x", "u_y", "u_t", "u_xx", "u_yy",
            "u_tt", "u_xy", "u_yx", "u_xt", "u_tx", "math", "torch", "np", "abs",
        }
        if not name.isidentifier() or name in reserved:
            raise ValueError(f"Invalid constant name in '{raw}'")
        try:
            constants[name] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid constant value in '{raw}'") from exc
        if not math.isfinite(constants[name]):
            raise ValueError(f"Constant must be finite in '{raw}'")
    return constants


def validate_training_args(args) -> None:
    """Reject empty loss batches and invalid weights before allocating a model."""
    for name in ("collocation_points", "boundary_points", "initial_points"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("steps", "data_points"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    for name in ("weight_ic", "weight_bc", "weight_data"):
        if hasattr(args, name):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if hasattr(args, "nu") and not math.isfinite(args.nu):
        raise ValueError("--nu must be finite")


def export_model(model: nn.Sequential, *, out_ckpt: str, out_json: str, hidden_widths, activation: str) -> None:
    ckpt_path = Path(out_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt_path))
    print(f"Saved checkpoint: {ckpt_path}")

    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "input_dim": 2,
        "output_dim": 1,
        "hidden_layers": list(hidden_widths),
        "activation": activation,
    }
    json_path.write_text(json.dumps(to_json_dict(model, meta=meta), allow_nan=False))
    print(f"Exported weights JSON: {json_path}")



def residual_loss(model, x, second, expression, constants, *, time: bool):
    """Evaluate the residual using first and second derivatives in two coordinates."""
    inputs = torch.cat([x, second], dim=1).requires_grad_(True)
    u = model(inputs)
    du = gradients(u, inputs)
    ux, uy = du[:, :1], du[:, 1:2]
    hx, hy = gradients(ux, inputs), gradients(uy, inputs)
    uxx, uxy = hx[:, :1], hx[:, 1:2]
    uyx, uyy = hy[:, :1], hy[:, 1:2]
    env = {
        "u": u, "ux": ux, "u_x": ux, "uy": uy, "u_y": uy,
        "uxx": uxx, "u_xx": uxx, "uyy": uyy, "u_yy": uyy,
        "uxy": uxy, "u_xy": uxy, "uyx": uyx, "u_yx": uyx,
        "x": x, "y": second,
    }
    if time:
        env.update({
            "t": second, "ut": uy, "u_t": uy,
            "utt": uyy, "u_tt": uyy, "uxt": uxy, "u_xt": uxy,
            "utx": uyx, "u_tx": uyx,
        })
    env.update(constants)
    residual = torch.as_tensor(eval_pinn_expr(expression, **env),
                               dtype=u.dtype, device=u.device)
    if residual.ndim == 1:
        residual = residual.unsqueeze(-1)
    return (residual ** 2).mean()


def train(args):
    """Train an evolution or stationary problem without changing its sampling schedule."""
    if args.problem == "evolution":
        _train_evolution(args)
    elif args.problem == "stationary":
        _train_stationary(args)
    else:
        raise ValueError(f"Unknown PINN problem: {args.problem!r}")


def _train_evolution(args):
    validate_training_args(args)
    constants = {"nu": args.nu}
    constants.update(parse_const_flags(args.const or []))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_lo, x_hi = -1.0, 1.0
    t_lo, t_hi = 0.0, 1.0

    model = build_model(in_dim=2, hidden_widths=args.hidden_widths, activation=args.activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_c = args.collocation_points
    N_i = args.initial_points
    N_b = args.boundary_points
    N_d = args.data_points

    dataset: Optional[PinnDataset] = None
    if args.dataset_json:
        dataset = PinnDataset.load(
            args.dataset_json,
            {
                "collocation": ["x", "t"],
                "initial": ["x", "t", "u"],
                "boundary": ["x", "t", "u"],
                "data": ["x", "t", "u"],
            },
            device,
        )
        print(f"Loaded dataset from {args.dataset_json}")

    def sample_collocation():
        if dataset:
            sampled = dataset.sample_columns("collocation", N_c, 2)
            if sampled is not None:
                return sampled
        x = torch.empty(N_c, 1, device=device).uniform_(x_lo, x_hi)
        t = torch.empty(N_c, 1, device=device).uniform_(t_lo, t_hi)
        return x, t

    def sample_initial():
        if dataset:
            sampled = dataset.sample_columns("initial", N_i, 3)
            if sampled is not None:
                return sampled
        x = torch.empty(N_i, 1, device=device).uniform_(x_lo, x_hi)
        t = torch.zeros(N_i, 1, device=device)
        u0 = ensure_tensor(eval_pinn_expr(args.ic_expr, x=x, t=t, **constants), x)
        return x, t, u0

    def sample_boundary():
        if dataset:
            sampled = dataset.sample_columns("boundary", N_b, 3)
            if sampled is not None:
                return sampled
        left_t = torch.empty(N_b // 2, 1, device=device).uniform_(t_lo, t_hi)
        left_x = torch.full_like(left_t, x_lo)
        right_t = torch.empty(N_b - N_b // 2, 1, device=device).uniform_(t_lo, t_hi)
        right_x = torch.full_like(right_t, x_hi)
        x = torch.cat([left_x, right_x], dim=0)
        t = torch.cat([left_t, right_t], dim=0)
        u_b = ensure_tensor(eval_pinn_expr(args.bc_expr, x=x, t=t, **constants), x)
        return x, t, u_b

    def sample_data():
        if N_d <= 0:
            return None
        if not dataset:
            raise ValueError("--data-points > 0 requires --dataset-json with a 'data' section.")
        sampled = dataset.sample_columns("data", N_d, 3)
        if sampled is None:
            raise ValueError("Dataset has no 'data' entries to sample.")
        return sampled

    model.train()
    for step in range(args.steps):
        opt.zero_grad()

        x_c, t_c = sample_collocation()
        x_c = x_c.to(device)
        t_c = t_c.to(device)
        loss_c = residual_loss(model, x_c, t_c,
                               args.pde_expr, constants, time=True)

        x_i, t_i, u0 = sample_initial()
        xi = torch.cat([x_i.to(device), t_i.to(device)], dim=1)
        u_i = model(xi)
        loss_i = ((u_i - u0.to(device)) ** 2).mean()

        x_b, t_b, u_b = sample_boundary()
        xb = torch.cat([x_b.to(device), t_b.to(device)], dim=1)
        u_b_pred = model(xb)
        loss_b = ((u_b_pred - u_b.to(device)) ** 2).mean()

        loss_d = torch.tensor(0.0, device=device)
        sampled_d = sample_data()
        if sampled_d is not None:
            x_d, t_d, u_d = sampled_d
            xd = torch.cat([x_d.to(device), t_d.to(device)], dim=1)
            u_d_pred = model(xd)
            loss_d = ((u_d_pred - u_d.to(device)) ** 2).mean()

        loss = loss_c + args.weight_ic * loss_i + args.weight_bc * loss_b + args.weight_data * loss_d
        loss.backward()
        opt.step()

        if (step + 1) % max(1, args.steps // 10) == 0:
            print(
                f"step {step + 1}/{args.steps}: loss={loss.item():.5e} "
                f"(c={loss_c.item():.3e}, i={loss_i.item():.3e}, b={loss_b.item():.3e}, d={loss_d.item():.3e})"
            )

    export_model(
        model,
        out_ckpt=args.out_ckpt,
        out_json=args.out_json,
        hidden_widths=args.hidden_widths,
        activation=args.activation,
    )



def _train_stationary(args):
    validate_training_args(args)
    constants = parse_const_flags(args.const or [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_lo, x_hi = -1.0, 1.0
    y_lo, y_hi = -1.0, 1.0

    model = build_model(in_dim=2, hidden_widths=args.hidden_widths, activation=args.activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_c = args.collocation_points
    N_b = args.boundary_points
    N_d = args.data_points

    dataset: Optional[PinnDataset] = None
    if args.dataset_json:
        dataset = PinnDataset.load(
            args.dataset_json,
            {
                "collocation": ["x", "y"],
                "boundary": ["x", "y", "u"],
                "data": ["x", "y", "u"],
            },
            device,
        )
        print(f"Loaded dataset from {args.dataset_json}")

    def sample_collocation():
        if dataset:
            sampled = dataset.sample_columns("collocation", N_c, 2)
            if sampled is not None:
                return sampled
        x = torch.empty(N_c, 1, device=device).uniform_(x_lo, x_hi)
        y = torch.empty(N_c, 1, device=device).uniform_(y_lo, y_hi)
        return x, y

    def sample_boundary():
        if dataset:
            sampled = dataset.sample_columns("boundary", N_b, 3)
            if sampled is not None:
                return sampled
        m, leftover = divmod(N_b, 4)
        counts = [m + int(i < leftover) for i in range(4)]
        x_left = torch.full((counts[0], 1), x_lo, device=device)
        y_left = torch.empty_like(x_left).uniform_(y_lo, y_hi)
        x_right = torch.full((counts[1], 1), x_hi, device=device)
        y_right = torch.empty_like(x_right).uniform_(y_lo, y_hi)
        y_bottom = torch.full((counts[2], 1), y_lo, device=device)
        x_bottom = torch.empty_like(y_bottom).uniform_(x_lo, x_hi)
        y_top = torch.full((counts[3], 1), y_hi, device=device)
        x_top = torch.empty_like(y_top).uniform_(x_lo, x_hi)

        x = torch.cat([x_left, x_right, x_bottom, x_top], dim=0)
        y = torch.cat([y_left, y_right, y_bottom, y_top], dim=0)
        u_b = ensure_tensor(eval_pinn_expr(args.bc_expr, x=x, y=y, **constants), x)
        return x, y, u_b

    def sample_data():
        if N_d <= 0:
            return None
        if dataset:
            sampled = dataset.sample_columns("data", N_d, 3)
            if sampled is not None:
                return sampled
        if args.data_expr is None:
            raise ValueError("--data-points > 0 requires --data-expr or dataset 'data' entries.")
        x = torch.empty(N_d, 1, device=device).uniform_(x_lo, x_hi)
        y = torch.empty(N_d, 1, device=device).uniform_(y_lo, y_hi)
        u_d = ensure_tensor(eval_pinn_expr(args.data_expr, x=x, y=y, **constants), x)
        return x, y, u_d

    model.train()
    for step in range(args.steps):
        opt.zero_grad()

        x_c, y_c = sample_collocation()
        x_c = x_c.to(device)
        y_c = y_c.to(device)
        loss_c = residual_loss(model, x_c, y_c,
                               args.pde_expr, constants, time=False)

        x_b, y_b, u_b = sample_boundary()
        xb = torch.cat([x_b.to(device), y_b.to(device)], dim=1)
        u_b_pred = model(xb)
        loss_b = ((u_b_pred - u_b.to(device)) ** 2).mean()

        data_sample = sample_data()
        if data_sample is not None:
            x_d, y_d, u_d = data_sample
            xd = torch.cat([x_d.to(device), y_d.to(device)], dim=1)
            u_d_pred = model(xd)
            loss_d = ((u_d_pred - u_d.to(device)) ** 2).mean()
        else:
            loss_d = torch.tensor(0.0, device=device)

        loss = loss_c + args.weight_bc * loss_b + args.weight_data * loss_d
        loss.backward()
        opt.step()

        if (step + 1) % max(1, args.steps // 10) == 0:
            print(
                f"step {step + 1}/{args.steps}: loss={loss.item():.5e} "
                f"(c={loss_c.item():.3e}, b={loss_b.item():.3e}, d={loss_d.item():.3e})"
            )

    export_model(
        model,
        out_ckpt=args.out_ckpt,
        out_json=args.out_json,
        hidden_widths=args.hidden_widths,
        activation=args.activation,
    )



def export(args):
    """Export the importer's fixed 16,16 tanh model, without training metadata."""
    # Preserve PyTorch's default initialization, not the trainer's Xavier initialization.
    model = nn.Sequential(
        nn.Linear(args.in_dim, 16), nn.Tanh(),
        nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1),
    )
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        keys = ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias")
        if isinstance(state, dict) and all(key in state for key in keys):
            model.load_state_dict(state)
        elif isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            raise SystemExit("Checkpoint does not contain expected state_dict keys")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_json_dict(model), allow_nan=False))
    print(f"Wrote weights JSON to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    problems = ap.add_subparsers(dest="problem", required=True)
    exporter = problems.add_parser("export", help="Export fresh or checkpoint-loaded weights")
    exporter.add_argument("--in-dim", type=int, choices=[1, 2], required=True)
    exporter.add_argument("--ckpt", help="PyTorch state_dict checkpoint")
    exporter.add_argument("--out", required=True, help="Output weight JSON")
    for problem in ("evolution", "stationary"):
        parser = problems.add_parser(problem)
        parser.add_argument("--steps", type=int, default=500)
        parser.add_argument("--const", action="append", default=[], help="Constant name=value")
        parser.add_argument("--collocation-points", type=int, default=256)
        parser.add_argument(
            "--boundary-points", type=int, default=128 if problem == "evolution" else 256,
        )
        parser.add_argument("--data-points", type=int, default=0)
        parser.add_argument("--weight-bc", type=float, default=1.0)
        parser.add_argument("--weight-data", type=float, default=1.0)
        parser.add_argument(
            "--pde-expr",
            default="u_t + u * u_x - nu * u_xx" if problem == "evolution" else "uxx + uyy",
        )
        parser.add_argument("--bc-expr", default="torch.zeros_like(x)")
        parser.add_argument("--hidden-widths", default="16,16")
        parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
        filename = "pinn1d" if problem == "evolution" else "pinn2d"
        parser.add_argument("--out-ckpt", default=f"_external/pinn/checkpoints/{filename}.pt")
        parser.add_argument("--out-json", default=f"_external/pinn/checkpoints/{filename}.json")
        parser.add_argument("--dataset-json", help="JSON collocation, boundary and supervised samples")
        if problem == "evolution":
            parser.add_argument("--nu", type=float, default=0.01)
            parser.add_argument("--initial-points", type=int, default=128)
            parser.add_argument("--weight-ic", type=float, default=10.0)
            parser.add_argument("--ic-expr", default="-torch.sin(math.pi * x)")
        else:
            parser.add_argument("--data-expr", help="Interior data expression in x and y")
    args = ap.parse_args()
    if args.problem == "export":
        export(args)
        return
    args.hidden_widths = parse_hidden_widths(args.hidden_widths)
    train(args)


if __name__ == "__main__":
    main()
