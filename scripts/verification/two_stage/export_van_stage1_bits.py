#!/usr/bin/env python3
"""Train/export Stage 1 for the TorchLean two-stage Van-der-Pol workflow.

This script is an *artifact producer*.  It trains a compact PyTorch controller/Lyapunov seed and
writes the learned parameters in the exact order expected by the TorchLean Stage-2 checker.

Why bits?
- Stage 2 can run in TorchLean under `IEEE32Exec`.
- JSON decimal floats can lose the exact binary32 payload that PyTorch trained.
- Exporting uint32 bit patterns lets Lean reconstruct exactly the same Float32 values.

The default output path is `_external/...`, which is ignored by git.  That is intentional: these
weights are regenerated experiment artifacts, not source files.

Output schema (JSON):
{
  "width": 100,
  "dtype": "float32",
  "format": "uint32-bits-as-decimal-strings",
  "wC": ["...", ...],   # len = uDim*xDim = 2
  "bC": ["...", ...],   # len = uDim = 1
  "w1": ["...", ...],   # len = width*xDim
  "b1": ["...", ...],   # len = width
  "w2": ["...", ...],   # len = 1*width
  "b2": ["...", ...]    # len = 1
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


X_DIM = 2
U_DIM = 1
FORMAT = "uint32-bits-as-decimal-strings"


def float32_bits_as_str_list(t: torch.Tensor) -> list[str]:
    """Serialize a tensor as decimal strings containing raw IEEE-754 binary32 bits.

    The `.view(np.uint32)` is the important line: it does not numerically convert the tensor.  It
    reinterprets each 32-bit float as the unsigned integer with the same bits.
    """

    a = t.detach().cpu().contiguous().to(torch.float32).numpy()
    bits = a.view(np.uint32).reshape(-1)
    return [str(int(x)) for x in bits.tolist()]


def build_parameters(width: int) -> list[torch.Tensor]:
    """Create Stage-1 parameters in exactly the order consumed by TorchLean.

    Order:
      Wc, bc: controller `u = tanh(Wc x + bc)`
      W1, b1, W2, b2: scalar Lyapunov network `s = W2 tanh(W1 x + b1) + b2`, `V = s²`
    """

    Wc = torch.empty((U_DIM, X_DIM), dtype=torch.float32)
    bc = torch.zeros((U_DIM,), dtype=torch.float32)
    W1 = torch.empty((width, X_DIM), dtype=torch.float32)
    b1 = torch.zeros((width,), dtype=torch.float32)
    W2 = torch.empty((1, width), dtype=torch.float32)
    b2 = torch.zeros((1,), dtype=torch.float32)

    torch.nn.init.xavier_uniform_(Wc)
    torch.nn.init.xavier_uniform_(W1)
    torch.nn.init.xavier_uniform_(W2)

    return [Wc, bc, W1, b1, W2, b2]


def van_loss(x: torch.Tensor, params: list[torch.Tensor]) -> torch.Tensor:
    """Stage-1 scalar loss shared with the TorchLean Stage-2 workflow.

    The loss has two ReLU penalties:
      positivity: `0.1 * ||x||² <= V(x)`
      decrease:   `Vdot(x) + 0.1 * V(x) <= 0`

    The script computes `grad V` analytically instead of using second-order autograd.  That keeps it
    close to the Lean implementation and makes parameter-order bugs easier to spot.
    """

    Wc, bc, W1, b1, W2, b2 = params
    mu = torch.tensor(1.0, dtype=torch.float32)
    cV = torch.tensor(0.1, dtype=torch.float32)
    cD = torch.tensor(0.1, dtype=torch.float32)
    scaleU = torch.tensor(1.0, dtype=torch.float32)

    u_pre = Wc @ x + bc
    u0 = scaleU * torch.tanh(u_pre[0])

    z1 = W1 @ x + b1
    h1 = torch.tanh(z1)
    s0 = (W2 @ h1 + b2)[0]
    V = s0 * s0

    dh = 1.0 - h1 * h1
    g_hidden = W2[0] * dh
    ds = g_hidden @ W1
    gradV = (2.0 * s0) * ds

    x1, x2 = x[0], x[1]
    dx2 = (-x1) + mu * (1.0 - x1 * x1) * x2 + u0
    f = torch.stack([x2, dx2])

    Vdot = (gradV * f).sum()
    x_sq = (x * x).sum()
    return torch.relu(cV * x_sq - V) + torch.relu(Vdot + cD * V)


def main() -> None:
    """Train/export the stage-1 Van der Pol certificate bits."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rad", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    width = int(args.width)
    params = build_parameters(width)
    for p in params:
        p.requires_grad_(True)

    opt = torch.optim.SGD(params, lr=float(args.lr))

    rad = float(args.rad)
    for step in range(int(args.steps)):
        # Uniformly sample a point from the verification box `[-rad, rad]^2`.
        x = (2.0 * torch.rand((X_DIM,), dtype=torch.float32) - 1.0) * rad
        opt.zero_grad(set_to_none=True)
        loss = van_loss(x, params)
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"[stage1] step={step} loss={loss.item():.6g}")

    out_path = Path(args.out) if args.out else Path(f"_external/van_stage1_w{width}_bits.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "width": width,
        "dtype": "float32",
        "format": FORMAT,
        "wC": float32_bits_as_str_list(params[0]),
        "bC": float32_bits_as_str_list(params[1]),
        "w1": float32_bits_as_str_list(params[2]),
        "b1": float32_bits_as_str_list(params[3]),
        "w2": float32_bits_as_str_list(params[4]),
        "b2": float32_bits_as_str_list(params[5]),
    }
    out_path.write_text(json.dumps(payload))
    print(f"Wrote {out_path} (width={width})")


if __name__ == "__main__":
    main()
