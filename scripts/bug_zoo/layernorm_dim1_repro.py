#!/usr/bin/env python3
"""PyTorch repro for the TorchLean LayerNorm one-feature contract.

For normalized_shape=(1,), LayerNorm is mathematically constant in x and weight:

    mean([x]) = x
    var([x]) = 0
    y = ((x - mean) / sqrt(var + eps)) * weight + bias = bias

Therefore dy/dx = 0 and dy/dweight = 0. TorchLean records this contract in
NN/Examples/BugZoo/LayerNormDegenerateAxis.lean.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Observation:
    device: str
    dtype: torch.dtype
    requested_x: float
    stored_x: float
    y: float
    dx: float
    dweight: float
    dbias: float
    native_mean: float
    native_rstd: float


def run_case(device: str, dtype: torch.dtype, requested_x: float) -> Observation:
    x = torch.tensor([requested_x], device=device, dtype=dtype, requires_grad=True)
    weight = torch.tensor([2.0], device=device, dtype=dtype, requires_grad=True)
    bias = torch.tensor([3.0], device=device, dtype=dtype, requires_grad=True)

    y = F.layer_norm(x, (1,), weight, bias, eps=1e-5)
    y.sum().backward()

    with torch.no_grad():
        native_y, native_mean, native_rstd = torch.ops.aten.native_layer_norm.default(
            x.detach(), [1], weight.detach(), bias.detach(), 1e-5
        )

    return Observation(
        device=device,
        dtype=dtype,
        requested_x=requested_x,
        stored_x=float(x.detach().cpu()[0]),
        y=float(y.detach().cpu()[0]),
        dx=float(x.grad.detach().cpu()[0]),
        dweight=float(weight.grad.detach().cpu()[0]),
        dbias=float(bias.grad.detach().cpu()[0]),
        native_mean=float(native_mean.detach().cpu()[0]),
        native_rstd=float(native_rstd.detach().cpu()[0]),
    )


def violates_contract(obs: Observation) -> bool:
    if not math.isfinite(obs.y) or not math.isfinite(obs.dx) or not math.isfinite(obs.dweight):
        return True
    return obs.y != 3.0 or obs.dx != 0.0 or obs.dweight != 0.0 or obs.dbias != 1.0


def print_observation(obs: Observation) -> None:
    status = "VIOLATION" if violates_contract(obs) else "ok"
    print(
        f"{status:9s} device={obs.device:4s} dtype={str(obs.dtype):14s} "
        f"x_request={obs.requested_x:>10.1e} x_stored={obs.stored_x:>14.6g} "
        f"y={obs.y:>14.6g} dx={obs.dx:>14.6g} "
        f"dweight={obs.dweight:>14.6g} dbias={obs.dbias:>8.6g} "
        f"native_mean={obs.native_mean:>14.6g} native_rstd={obs.native_rstd:>10.6g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a nonzero exit code if the current PyTorch violates the contract",
    )
    args = parser.parse_args()

    print(f"torch {torch.__version__}")
    print("contract: y == bias == 3, dx == 0, dweight == 0, dbias == 1")

    observations: list[Observation] = []
    for dtype in (torch.float32, torch.bfloat16):
        for requested_x in (1e2, 1e4, 1e6, 1e8):
            observations.append(run_case("cpu", dtype, requested_x))

    if torch.cuda.is_available():
        print(f"cuda {torch.version.cuda} device={torch.cuda.get_device_name(0)}")
        for requested_x in (1e2, 1e6, 1e8):
            observations.append(run_case("cuda", torch.float32, requested_x))

    for obs in observations:
        print_observation(obs)

    violations = [obs for obs in observations if violates_contract(obs)]
    print(f"\nviolations: {len(violations)} / {len(observations)}")
    return 1 if args.strict and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
