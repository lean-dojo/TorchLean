#!/usr/bin/env python3
"""
Configurable 1D PINN trainer on (x,t) ∈ [-1,1]×[0,1].

- Network: defaults to Linear(2→16) → Tanh → Linear(16→16) → Tanh → Linear(16→1);
  override widths/activations with ``--hidden-widths`` and ``--activation``.
- PDE residual: parsed from ``--pde-expr`` using the same DSL tokens as
  Lean's verifier (``u``, ``ux``, ``ut``, ``uxx``, ``utt``, ``uxy``/``uyx``, etc.).
- Boundary / initial data are supplied as expressions via ``--ic-expr`` and
  ``--bc-expr``; their values are evaluated on-the-fly with a restricted AST-based math evaluator.
- Additional constants can be injected with repeated ``--const name=value`` flags.
- Optional ``--dataset-json`` parameter lets you plug in measured points.
  Expected format:

  .. code-block:: json

     {
       "collocation": [ {"x": 0.1, "t": 0.2}, ... ],
       "initial": [ {"x": 0.0, "t": 0.0, "u": 0.0}, ... ],
       "boundary": [ {"x": -1.0, "t": 0.5, "u": 0.0}, ... ],
       "data": [ {"x": 0.3, "t": 0.4, "u": 0.12}, ... ]
     }

Example (viscous Burgers):
    python3 scripts/verification/pinn/train_pinn_1d.py \
        --steps 500 --nu 0.01 \
        --pde-expr "u_t + u * u_x - nu * u_xx" \
        --out-ckpt _external/pinn/checkpoints/pinn1d.pt \
        --out-json _external/pinn/checkpoints/pinn1d.json

Then verify in Lean (mapping t→y in the PDE DSL):
    lake exe verify -- pinn-cli --weights=_external/pinn/checkpoints/pinn1d.json \
        "u_y + u*u_x - 0.01*u_xx" 0.0 0.5 0.01

Notes:
- The defaults stay light for fast local checks; increase steps, sample
  counts, and tune loss weights for serious training.
- The script only produces weights compatible with sequential PINN graphs;
  Lean rebuilds a matching graph from the exported metadata.
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


class Dataset1D:
    """Optional sampled dataset for 1D space-time PINN training."""

    def __init__(self, device: torch.device):
        self.device = device
        self.collocation: Optional[torch.Tensor] = None
        self.initial: Optional[torch.Tensor] = None
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
    def load(cls, path: str, device: torch.device) -> "Dataset1D":
        """Load collocation, initial, boundary, and data sections from JSON."""
        import json
        data = cls(device)
        payload = json.loads(Path(path).read_text())
        data.collocation = cls._read_entries(payload.get("collocation"), ["x", "t"], device)
        data.initial = cls._read_entries(payload.get("initial"), ["x", "t", "u"], device)
        data.boundary = cls._read_entries(payload.get("boundary"), ["x", "t", "u"], device)
        data.data = cls._read_entries(payload.get("data"), ["x", "t", "u"], device)
        return data

    def _sample_rows(self, mat: torch.Tensor, count: int) -> torch.Tensor:
        """Sample rows with replacement from one dataset tensor."""
        if mat is None or mat.shape[0] == 0:
            raise ValueError("Dataset section is empty; cannot sample.")
        idx = torch.randint(0, mat.shape[0], (count,), device=self.device, dtype=torch.long)
        return mat.index_select(0, idx)

    def sample_collocation(self, count: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Sample `(x, t)` collocation points if the dataset provides them."""
        if self.collocation is None:
            return None
        samples = self._sample_rows(self.collocation, count)
        return samples[:, :1], samples[:, 1:]

    def sample_initial(self, count: int) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample initial-condition triples `(x, t, u)`."""
        if self.initial is None:
            return None
        samples = self._sample_rows(self.initial, count)
        return samples[:, :1], samples[:, 1:2], samples[:, 2:]

    def sample_boundary(self, count: int) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample boundary-condition triples `(x, t, u)`."""
        if self.boundary is None:
            return None
        samples = self._sample_rows(self.boundary, count)
        return samples[:, :1], samples[:, 1:2], samples[:, 2:]

    def sample_data(self, count: int) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample supervised data triples `(x, t, u)`."""
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
    """Construct a Sequential network with user-defined widths and nonlinearity."""
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
    """Compute d(output)/d(inputs) with autograd."""
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
    """Train a 1D space-time PINN and export checkpoint/JSON weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_lo, x_hi = -1.0, 1.0
    t_lo, t_hi = 0.0, 1.0

    model = build_model(in_dim=2, hidden_widths=args.hidden_widths, activation=args.activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_c = args.collocation_points
    N_i = args.initial_points
    N_b = args.boundary_points
    N_d = args.data_points

    constants = {"nu": args.nu}
    constants.update(_parse_const_flags(args.const or []))

    dataset: Optional[Dataset1D] = None
    if args.dataset_json:
        dataset = Dataset1D.load(args.dataset_json, device)
        print(f"Loaded dataset from {args.dataset_json}")

    def sample_collocation():
        """Sample collocation points from JSON data or the default box."""
        if dataset:
            sampled = dataset.sample_collocation(N_c)
            if sampled is not None:
                return sampled
        x = torch.empty(N_c, 1, device=device).uniform_(x_lo, x_hi)
        t = torch.empty(N_c, 1, device=device).uniform_(t_lo, t_hi)
        return x, t

    def sample_initial():
        """Sample initial-condition points from JSON data or `--ic-expr`."""
        if dataset:
            sampled = dataset.sample_initial(N_i)
            if sampled is not None:
                return sampled
        x = torch.empty(N_i, 1, device=device).uniform_(x_lo, x_hi)
        t = torch.zeros(N_i, 1, device=device)
        u0 = _ensure_tensor(_eval_expr(args.ic_expr, x=x, t=t, **constants), x)
        return x, t, u0

    def sample_boundary():
        """Sample boundary-condition points from JSON data or `--bc-expr`."""
        if dataset:
            sampled = dataset.sample_boundary(N_b)
            if sampled is not None:
                return sampled
        t1 = torch.empty(N_b // 2, 1, device=device).uniform_(t_lo, t_hi)
        x1 = torch.full_like(t1, x_lo)
        t2 = torch.empty(N_b - N_b // 2, 1, device=device).uniform_(t_lo, t_hi)
        x2 = torch.full_like(t2, x_hi)
        x = torch.cat([x1, x2], dim=0)
        t = torch.cat([t1, t2], dim=0)
        u_b = _ensure_tensor(_eval_expr(args.bc_expr, x=x, t=t, **constants), x)
        return x, t, u_b

    def sample_data():
        """Sample optional supervised data points."""
        if N_d <= 0:
            return None
        if not dataset:
            raise ValueError("--data-points > 0 requires --dataset-json with a 'data' section.")
        sampled = dataset.sample_data(N_d)
        if sampled is None:
            raise ValueError("Dataset has no 'data' entries to sample.")
        return sampled

    model.train()
    for step in range(args.steps):
        opt.zero_grad()

        x_c, t_c = sample_collocation()
        x_c = x_c.to(device)
        t_c = t_c.to(device)
        inp_c = torch.cat([x_c, t_c], dim=1).requires_grad_(True)
        u_c = model(inp_c)

        du = gradients(u_c, inp_c)
        u_x = du[:, :1]
        u_t = du[:, 1:2]
        h_x = gradients(u_x, inp_c)
        h_t = gradients(u_t, inp_c)
        u_xx = h_x[:, :1]
        u_xt = h_x[:, 1:2]
        u_tx = h_t[:, :1]
        u_tt = h_t[:, 1:2]

        pde_env = {
            "u": u_c,
            "ux": u_x,
            "u_x": u_x,
            "ut": u_t,
            "u_t": u_t,
            "uy": u_t,
            "u_y": u_t,
            "uxx": u_xx,
            "u_xx": u_xx,
            "utt": u_tt,
            "u_tt": u_tt,
            "uyy": u_tt,
            "u_yy": u_tt,
            "uxy": u_xt,
            "u_xy": u_xt,
            "uyx": u_tx,
            "u_yx": u_tx,
            "uxt": u_xt,
            "u_tx": u_tx,
            "x": x_c,
            "t": t_c,
            "y": t_c,
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
    """CLI entry point for 1D PINN training/export."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500, help="Training steps")
    ap.add_argument("--nu", type=float, default=0.01, help="Default constant 'nu'")
    ap.add_argument("--const", action="append", help="Additional constants name=value", default=[])
    ap.add_argument("--collocation-points", type=int, default=256)
    ap.add_argument("--initial-points", type=int, default=128)
    ap.add_argument("--boundary-points", type=int, default=128)
    ap.add_argument("--data-points", type=int, default=0, help="Supervised data points per step (requires --dataset-json)")
    ap.add_argument("--weight-ic", type=float, default=10.0, help="Weight for initial condition loss")
    ap.add_argument("--weight-bc", type=float, default=1.0, help="Weight for boundary loss")
    ap.add_argument("--weight-data", type=float, default=1.0, help="Weight for supervised data loss")
    ap.add_argument(
        "--pde-expr",
        type=str,
        default="u_t + u * u_x - nu * u_xx",
        help="PDE residual expression in TorchLean DSL",
    )
    ap.add_argument(
        "--ic-expr",
        type=str,
        default="-torch.sin(math.pi * x)",
        help="Initial condition expression in terms of x and t",
    )
    ap.add_argument(
        "--bc-expr",
        type=str,
        default="torch.zeros_like(x)",
        help="Boundary condition expression in terms of x and t",
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
        default="_external/pinn/checkpoints/pinn1d.pt",
        help="Path to write the PyTorch checkpoint",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default="_external/pinn/checkpoints/pinn1d.json",
        help="Path to export Lean-compatible weights JSON",
    )
    ap.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="Optional dataset JSON providing collocation/initial/boundary/data samples",
    )
    args = ap.parse_args()
    args.hidden_widths = parse_hidden_widths(args.hidden_widths)
    train(args)


if __name__ == "__main__":
    main()
