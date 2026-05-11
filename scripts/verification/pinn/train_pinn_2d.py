#!/usr/bin/env python3
"""
Configurable 2D PINN trainer on (x,y) ∈ [-1,1]×[-1,1].

- Network: defaults to Linear(2→16) → Tanh → Linear(16→16) → Tanh → Linear(16→1);
  override widths/activations with ``--hidden-widths`` and ``--activation``.
- PDE residual: parsed from ``--pde-expr`` using the same DSL tokens as Lean's
  verifier (``u``, ``ux``, ``uy``, ``uxx``, ``uyy``, ``uxy``, etc.).
- Boundary data supplied via ``--bc-expr``; optional interior anchors via
  ``--data-expr`` or ``--dataset-json``.
- Additional constants can be injected with repeated ``--const name=value`` flags.
- Dataset JSON schema mirrors the 1D trainer but uses ``x``/``y`` coordinates, e.g.:

  .. code-block:: json

     {
       "collocation": [ {"x": 0.1, "y": 0.2}, ... ],
       "boundary": [ {"x": -1.0, "y": 0.0, "u": 0.0}, ... ],
       "data": [ {"x": 0.3, "y": -0.2, "u": 0.05}, ... ]
     }

Example (Poisson equation Delta u = 0):
    python3 scripts/verification/pinn/train_pinn_2d.py \
        --steps 500 --pde-expr "uxx + uyy" \
        --out-json _external/pinn/checkpoints/pinn2d.json

The produced JSON weights are compatible with sequential PINN graphs; Lean
rebuilds the architecture from the exported metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from safe_expr import eval_expr

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - fail fast when torch missing
    raise SystemExit("PyTorch is required: pip install torch") from exc


class Dataset2D:
    """Optional sampled dataset for 2D PINN training."""

    def __init__(self, device: torch.device):
        self.device = device
        self.collocation: Optional[torch.Tensor] = None
        self.boundary: Optional[torch.Tensor] = None
        self.data: Optional[torch.Tensor] = None

    @staticmethod
    def _read_entries(entries, keys, device) -> Optional[torch.Tensor]:
        """Read one JSON dataset section into a float tensor with selected keys."""
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
            rows.append(row)
        if not rows:
            return None
        return torch.tensor(rows, dtype=torch.float32, device=device)

    @classmethod
    def load(cls, path: str, device: torch.device) -> "Dataset2D":
        """Load collocation, boundary, and data sections from JSON."""
        import json
        payload = json.loads(Path(path).read_text())
        data = cls(device)
        data.collocation = cls._read_entries(payload.get("collocation"), ["x", "y"], device)
        data.boundary = cls._read_entries(payload.get("boundary"), ["x", "y", "u"], device)
        data.data = cls._read_entries(payload.get("data"), ["x", "y", "u"], device)
        return data

    def _sample_rows(self, mat: torch.Tensor, count: int) -> torch.Tensor:
        """Sample rows with replacement from one dataset tensor."""
        if mat is None or mat.shape[0] == 0:
            raise ValueError("Dataset section is empty; cannot sample.")
        idx = torch.randint(0, mat.shape[0], (count,), device=self.device, dtype=torch.long)
        return mat.index_select(0, idx)

    def sample_collocation(self, count: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Sample `(x, y)` collocation points if available."""
        if self.collocation is None:
            return None
        samples = self._sample_rows(self.collocation, count)
        return samples[:, :1], samples[:, 1:]

    def sample_boundary(self, count: int) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample boundary triples `(x, y, u)` if available."""
        if self.boundary is None:
            return None
        samples = self._sample_rows(self.boundary, count)
        return samples[:, :1], samples[:, 1:2], samples[:, 2:]

    def sample_data(self, count: int) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample supervised data triples `(x, y, u)` if available."""
        if self.data is None:
            return None
        samples = self._sample_rows(self.data, count)
        return samples[:, :1], samples[:, 1:2], samples[:, 2:]


def _activation_factory(name: str) -> nn.Module:
    """Build the requested activation module."""
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation '{name}'")


def parse_hidden_widths(raw: str) -> List[int]:
    """Convert a comma-separated width string into a validated list."""
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
    """Build the sequential MLP used by the 2D PINN trainer."""
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


def to_json_dict(model: nn.Sequential, *, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a sequential PINN model plus metadata into JSON."""
    state = model.state_dict()
    exported: Dict[str, Any] = {}
    for name, tensor in state.items():
        if name.endswith(".weight") or name.endswith(".bias"):
            exported[f"layers.{name}"] = tensor.detach().cpu().numpy().tolist()
    exported["meta"] = meta
    return exported


def gradients(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute `d(output)/d(inputs)` with autograd."""
    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
        create_graph=True,
    )[0]


def _eval_expr(expr: str, **tensors):
    """Evaluate one restricted PDE/data expression and attach context to errors."""
    try:
        value = eval_expr(expr, tensors)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise ValueError(f"Failed to evaluate expression '{expr}': {exc}") from exc
    return value


def _ensure_tensor(val, like: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar expression results to match a reference tensor."""
    if isinstance(val, torch.Tensor):
        return val.to(like)
    arr = torch.as_tensor(val, dtype=like.dtype, device=like.device)
    if arr.numel() == 1:
        return torch.full_like(like, arr.item())
    return arr.reshape_as(like)


def _parse_const_flags(items) -> Dict[str, float]:
    """Parse repeated `--const name=value` flags into a numeric environment."""
    constants: Dict[str, float] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"--const expects name=value, got '{raw}'")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid constant name in '{raw}'")
        try:
            constants[name] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid constant value in '{raw}'") from exc
    return constants


def train(args):
    """Train a 2D PINN and export checkpoint/JSON weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_lo, x_hi = -1.0, 1.0
    y_lo, y_hi = -1.0, 1.0

    model = build_model(in_dim=2, hidden_widths=args.hidden_widths, activation=args.activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_c = args.collocation_points
    N_b = args.boundary_points
    N_d = args.data_points

    constants = _parse_const_flags(args.const or [])

    dataset: Optional[Dataset2D] = None
    if args.dataset_json:
        dataset = Dataset2D.load(args.dataset_json, device)
        print(f"Loaded dataset from {args.dataset_json}")

    def sample_collocation():
        """Sample collocation points from JSON data or the default square."""
        if dataset:
            sampled = dataset.sample_collocation(N_c)
            if sampled is not None:
                return sampled
        x = torch.empty(N_c, 1, device=device).uniform_(x_lo, x_hi)
        y = torch.empty(N_c, 1, device=device).uniform_(y_lo, y_hi)
        return x, y

    def sample_boundary():
        """Sample boundary points from JSON data or `--bc-expr`."""
        if dataset:
            sampled = dataset.sample_boundary(N_b)
            if sampled is not None:
                return sampled
        m = max(1, N_b // 4)
        leftover = N_b - 4 * m
        extras = [0, 0, 0, 0]
        for k in range(leftover):
            extras[k % 4] += 1
        counts = [m + extras[i] for i in range(4)]
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
        u_b = _ensure_tensor(_eval_expr(args.bc_expr, x=x, y=y, **constants), x)
        return x, y, u_b

    def sample_data():
        """Sample optional supervised data points."""
        if dataset:
            sampled = dataset.sample_data(N_d if N_d > 0 else 1)
            if sampled is not None:
                return sampled
        if N_d <= 0 or args.data_expr is None:
            return None
        x = torch.empty(N_d, 1, device=device).uniform_(x_lo, x_hi)
        y = torch.empty(N_d, 1, device=device).uniform_(y_lo, y_hi)
        u_d = _ensure_tensor(_eval_expr(args.data_expr, x=x, y=y, **constants), x)
        return x, y, u_d

    model.train()
    for step in range(args.steps):
        opt.zero_grad()

        x_c, y_c = sample_collocation()
        x_c = x_c.to(device)
        y_c = y_c.to(device)
        inp_c = torch.cat([x_c, y_c], dim=1).requires_grad_(True)
        u_c = model(inp_c)

        du = gradients(u_c, inp_c)
        u_x = du[:, :1]
        u_y = du[:, 1:2]
        h_x = gradients(u_x, inp_c)
        h_y = gradients(u_y, inp_c)
        u_xx = h_x[:, :1]
        u_xy = h_x[:, 1:2]
        u_yx = h_y[:, :1]
        u_yy = h_y[:, 1:2]

        pde_env = {
            "u": u_c,
            "ux": u_x,
            "u_x": u_x,
            "uy": u_y,
            "u_y": u_y,
            "uxx": u_xx,
            "u_xx": u_xx,
            "uyy": u_yy,
            "u_yy": u_yy,
            "uxy": u_xy,
            "u_xy": u_xy,
            "uyx": u_yx,
            "u_yx": u_yx,
            "x": x_c,
            "y": y_c,
        }
        pde_env.update(constants)

        res_raw = _eval_expr(args.pde_expr, **pde_env)
        if isinstance(res_raw, torch.Tensor):
            res = res_raw.to(device=u_c.device, dtype=u_c.dtype)
        else:
            res = torch.as_tensor(res_raw, dtype=u_c.dtype, device=u_c.device)
        if res.ndim == 1:
            res = res.unsqueeze(-1)
        loss_c = (res ** 2).mean()

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

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_ckpt))
    print(f"Saved checkpoint: {out_ckpt}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "input_dim": 2,
        "output_dim": 1,
        "hidden_layers": list(args.hidden_widths),
        "activation": args.activation,
    }
    out_json.write_text(json.dumps(to_json_dict(model, meta=meta)))
    print(f"Exported weights JSON: {out_json}")


def main():
    """CLI entry point for 2D PINN training/export."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--const", action="append", help="Additional constants name=value", default=[])
    ap.add_argument("--collocation-points", type=int, default=256)
    ap.add_argument("--boundary-points", type=int, default=256)
    ap.add_argument("--data-points", type=int, default=0)
    ap.add_argument("--weight-bc", type=float, default=1.0, help="Weight for boundary loss")
    ap.add_argument("--weight-data", type=float, default=1.0, help="Weight for interior data loss")
    ap.add_argument(
        "--pde-expr",
        type=str,
        default="uxx + uyy",
        help="PDE residual expression in TorchLean DSL",
    )
    ap.add_argument(
        "--bc-expr",
        type=str,
        default="torch.zeros_like(x)",
        help="Boundary condition expression in terms of x and y",
    )
    ap.add_argument(
        "--data-expr",
        type=str,
        default=None,
        help="Optional interior data expression in terms of x and y",
    )
    ap.add_argument(
        "--hidden-widths",
        type=str,
        default="16,16",
        help="Comma-separated hidden layer widths (e.g., '32,32,32')",
    )
    ap.add_argument(
        "--activation",
        choices=["tanh", "relu"],
        default="tanh",
        help="Activation used between hidden layers",
    )
    ap.add_argument(
        "--out-ckpt",
        type=str,
        default="_external/pinn/checkpoints/pinn2d.pt",
        help="Path to write the PyTorch checkpoint",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default="_external/pinn/checkpoints/pinn2d.json",
        help="Path to export Lean-compatible weights JSON",
    )
    ap.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="Optional dataset JSON providing collocation/boundary/data samples",
    )
    args = ap.parse_args()
    args.hidden_widths = parse_hidden_widths(args.hidden_widths)
    train(args)


if __name__ == "__main__":
    main()
