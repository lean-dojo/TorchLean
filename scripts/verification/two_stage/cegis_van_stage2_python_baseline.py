#!/usr/bin/env python3
"""Run the Python Stage-2 baseline for the TorchLean two-stage Van workflow.

The purpose is comparison, not trust.  TorchLean/Lean is the checker side; this script is a
familiar PyTorch reference loop that uses the same scalar loss and the same parameter pack order.

Workflow:
  1. Read exact Stage-1 Float32 values exported by `export_van_stage1_bits.py`.
  2. Search for high-loss points by PGD ascent over `x`.
  3. Take an SGD step on the controller/Lyapunov parameters at the found point.

This is not the full Verified-Intelligence Stage-2 implementation.  It is a compact baseline for
checking that the TorchLean workflow is wired to the same objective.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


X_DIM = 2
U_DIM = 1
FORMAT = "uint32-bits-as-decimal-strings"


def bits_str_list_to_f32_tensor(bits: list[str], shape: tuple[int, ...]) -> torch.Tensor:
    """Decode decimal strings containing raw binary32 bits into a Float32 tensor."""

    expected = int(np.prod(shape))
    if len(bits) != expected:
        raise ValueError(f"expected {expected} bit strings for shape {shape}, got {len(bits)}")
    arr_u32 = np.array([np.uint32(int(x)) for x in bits], dtype=np.uint32)
    arr_f32 = arr_u32.view(np.float32).reshape(shape)
    return torch.from_numpy(arr_f32.copy())


def load_stage1_bits(path: str, width: int) -> tuple[torch.Tensor, ...]:
    """Load the exact Stage-1 parameter pack emitted by `export_van_stage1_bits.py`."""

    j = json.loads(Path(path).read_text())
    if j.get("format") != FORMAT:
        raise ValueError(f"unsupported format {j.get('format')!r}; expected {FORMAT!r}")
    if j.get("dtype") != "float32":
        raise ValueError(f"unsupported dtype {j.get('dtype')!r}; expected 'float32'")
    if int(j["width"]) != width:
        raise ValueError(f"width mismatch: file has {j['width']} but expected {width}")

    Wc = bits_str_list_to_f32_tensor(j["wC"], (U_DIM, X_DIM))
    bc = bits_str_list_to_f32_tensor(j["bC"], (U_DIM,))
    W1 = bits_str_list_to_f32_tensor(j["w1"], (width, X_DIM))
    b1 = bits_str_list_to_f32_tensor(j["b1"], (width,))
    W2 = bits_str_list_to_f32_tensor(j["w2"], (1, width))
    b2 = bits_str_list_to_f32_tensor(j["b2"], (1,))
    return Wc, bc, W1, b1, W2, b2


def clamp_(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clamp without changing dtype/device."""

    return torch.max(torch.min(x, torch.tensor(hi, dtype=x.dtype)), torch.tensor(lo, dtype=x.dtype))


def loss_fn(
    x: torch.Tensor,
    Wc: torch.Tensor,
    bc: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Compute the Stage-2 loss used by both the Python baseline and TorchLean.

    The formulas mirror the Lean workflow:
      controller: `u = tanh(Wc x + bc)`
      Lyapunov candidate: `V = (W2 tanh(W1 x + b1) + b2)^2`
      dynamics: Van-der-Pol-like system with `mu = 1`
      penalties: positivity and decrease violations
    """

    mu = torch.tensor(1.0, dtype=x.dtype)
    cV = torch.tensor(0.1, dtype=x.dtype)
    cD = torch.tensor(0.1, dtype=x.dtype)

    u_pre = Wc @ x + bc  # [1]
    u0 = torch.tanh(u_pre[0])

    z1 = W1 @ x + b1  # [width]
    h1 = torch.tanh(z1)
    s0 = (W2 @ h1 + b2)[0]  # scalar tensor
    V = s0 * s0

    # gradV = 2*s0 * W1^T (W2_row ⊙ (1 - tanh(z1)^2))
    dh = 1.0 - h1 * h1
    g_hidden = W2[0] * dh  # [width]
    ds = g_hidden @ W1  # [2]
    gradV = (2.0 * s0) * ds  # [2]

    x1, x2 = x[0], x[1]
    dx2 = (-x1) + mu * (1.0 - x1 * x1) * x2 + u0
    f = torch.stack([x2, dx2])  # [2]

    Vdot = (gradV * f).sum()
    x_sq = (x * x).sum()
    pos = torch.relu(cV * x_sq - V)
    dec = torch.relu(Vdot + cD * V)
    return pos + dec


def main() -> None:
    """Run the Python CEGIS baseline and emit a compact JSON summary."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str, default="_external/van_stage1_w100_bits.json")
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--stage2_rounds", type=int, default=1)
    ap.add_argument("--candidates", type=int, default=1)
    ap.add_argument("--pgd_steps", type=int, default=1)
    ap.add_argument("--pgd_step", type=float, default=0.05)
    ap.add_argument("--rad", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Wc, bc, W1, b1, W2, b2 = load_stage1_bits(args.weights, args.width)
    params = [Wc, bc, W1, b1, W2, b2]
    for p in params:
        p.requires_grad_(True)

    opt = torch.optim.SGD(params, lr=float(args.lr))

    t0 = time.time()
    pos_count = 0
    total = args.stage2_rounds * args.candidates

    for r in range(args.stage2_rounds):
        for _c in range(args.candidates):
            # Start from a random point in the verification box.
            x = (2.0 * torch.rand((X_DIM,), dtype=torch.float32) - 1.0) * float(args.rad)
            with torch.no_grad():
                l0 = loss_fn(x, Wc, bc, W1, b1, W2, b2).item()

            # PGD ascent over x searches for a counterexample-like point that violates the loss.
            x_adv = x.detach().clone().requires_grad_(True)
            for _k in range(args.pgd_steps):
                l = loss_fn(x_adv, Wc, bc, W1, b1, W2, b2)
                (g,) = torch.autograd.grad(l, x_adv, retain_graph=False, create_graph=False)
                with torch.no_grad():
                    x_adv += float(args.pgd_step) * g
                    x_adv = clamp_(x_adv, -float(args.rad), float(args.rad))
                x_adv.requires_grad_(True)
            x_adv = x_adv.detach()

            # SGD on parameters tries to reduce the violation at the adversarial point.
            opt.zero_grad(set_to_none=True)
            l_adv = loss_fn(x_adv, Wc, bc, W1, b1, W2, b2)
            l_adv.backward()
            opt.step()

            l1 = float(l_adv.detach().item())
            if l1 > 0.0:
                pos_count += 1
            print(f"[stage2] r={r} lossBefore={l0:.6g} lossAfterPGD={l1:.6g}")

    dt = time.time() - t0
    print(f"[stage2] PGD counterexample candidates={total} (positive-loss={pos_count})")
    print(f"[stage2] wall_time_sec={dt:.3f}")


if __name__ == "__main__":
    main()
