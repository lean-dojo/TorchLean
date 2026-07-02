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
from typing import Optional

from pinn_common import (
    PinnDataset,
    build_model,
    ensure_tensor,
    eval_pinn_expr,
    export_model,
    gradients,
    parse_const_flags,
    parse_hidden_widths,
    torch,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_lo, x_hi = -1.0, 1.0
    y_lo, y_hi = -1.0, 1.0

    model = build_model(in_dim=2, hidden_widths=args.hidden_widths, activation=args.activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_c = args.collocation_points
    N_b = args.boundary_points
    N_d = args.data_points

    constants = parse_const_flags(args.const or [])

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
        u_b = ensure_tensor(eval_pinn_expr(args.bc_expr, x=x, y=y, **constants), x)
        return x, y, u_b

    def sample_data():
        if dataset:
            sampled = dataset.sample_columns("data", N_d if N_d > 0 else 1, 3)
            if sampled is not None:
                return sampled
        if N_d <= 0 or args.data_expr is None:
            return None
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

        res_raw = eval_pinn_expr(args.pde_expr, **pde_env)
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

    export_model(
        model,
        out_ckpt=args.out_ckpt,
        out_json=args.out_json,
        hidden_widths=args.hidden_widths,
        activation=args.activation,
    )


def main():
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
