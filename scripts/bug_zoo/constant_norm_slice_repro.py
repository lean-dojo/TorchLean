#!/usr/bin/env python3
"""PyTorch repro for TorchLean's constant normalization-slice contract.

For a constant normalized slice, mean == x and variance == 0. Affine normalization should therefore
return the bias, and the scale-gradient contribution should be zero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class Case:
    name: str
    device: str
    dtype: torch.dtype
    value: float
    shape: tuple[int, ...]
    run: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    stats: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]


@dataclass
class Observation:
    name: str
    device: str
    dtype: torch.dtype
    value: float
    shape: tuple[int, ...]
    y_max: float
    y_diff_from_bias: float
    dweight_max: float
    native_y_max: float | None
    native_mean: float | None
    native_rstd: float | None


def group_stats(groups: int) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    def inner(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, c = x.shape[:2]
        hxw = x.numel() // (n * c)
        return torch.ops.aten.native_group_norm.default(x, weight, bias, n, c, hxw, groups, 1e-5)

    return inner


def batch_stats(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.aten.native_batch_norm.default(x, weight, bias, None, None, True, 0.1, 1e-5)


def run_case(case: Case) -> Observation:
    x = torch.full(case.shape, case.value, device=case.device, dtype=case.dtype, requires_grad=True)
    channels = case.shape[1]
    weight = torch.full((channels,), 2.0, device=case.device, dtype=case.dtype, requires_grad=True)
    bias = torch.full((channels,), 3.0, device=case.device, dtype=case.dtype, requires_grad=True)

    y = case.run(x, weight, bias)
    y.sum().backward()

    bias_view = bias.detach().reshape(1, channels, *([1] * (y.ndim - 2)))
    y_diff = float((y.detach() - bias_view).abs().max().cpu())

    native_y_max: float | None = None
    native_mean: float | None = None
    native_rstd: float | None = None
    try:
        native_y, mean, rstd = case.stats(x.detach(), weight.detach(), bias.detach())
        native_y_max = float(native_y.detach().abs().max().cpu())
        native_mean = float(mean.detach().flatten()[0].cpu()) if mean is not None else None
        native_rstd = float(rstd.detach().flatten()[0].cpu()) if rstd is not None else None
    except RuntimeError:
        pass

    return Observation(
        name=case.name,
        device=case.device,
        dtype=case.dtype,
        value=case.value,
        shape=case.shape,
        y_max=float(y.detach().abs().max().cpu()),
        y_diff_from_bias=y_diff,
        dweight_max=float(weight.grad.detach().abs().max().cpu()),
        native_y_max=native_y_max,
        native_mean=native_mean,
        native_rstd=native_rstd,
    )


def violates_contract(obs: Observation) -> bool:
    return obs.y_diff_from_bias != 0.0 or obs.dweight_max != 0.0


def print_observation(obs: Observation) -> None:
    status = "VIOLATION" if violates_contract(obs) else "ok"
    print(
        f"{status:9s} {obs.name:13s} device={obs.device:4s} dtype={str(obs.dtype):14s} "
        f"value={obs.value:>9.1e} shape={obs.shape!s:14s} "
        f"y_max={obs.y_max:>12.6g} |y-bias|={obs.y_diff_from_bias:>12.6g} "
        f"|dweight|={obs.dweight_max:>10.6g} "
        f"native_y_max={obs.native_y_max!s:>12s} mean={obs.native_mean!s:>12s} "
        f"rstd={obs.native_rstd!s:>12s}"
    )


def build_cases() -> list[Case]:
    cases: list[Case] = [
        Case(
            name="group_norm",
            device="cpu",
            dtype=torch.float32,
            value=1e6,
            shape=(1, 2, 1, 1),
            run=lambda x, w, b: F.group_norm(x, 1, w, b, eps=1e-5),
            stats=group_stats(1),
        ),
        Case(
            name="instance_norm",
            device="cpu",
            dtype=torch.float32,
            value=1e6,
            shape=(1, 2, 1, 2),
            run=lambda x, w, b: F.instance_norm(
                x, running_mean=None, running_var=None, weight=w, bias=b, use_input_stats=True, eps=1e-5
            ),
            stats=group_stats(2),
        ),
        Case(
            name="batch_norm",
            device="cpu",
            dtype=torch.float32,
            value=1e6,
            shape=(2, 2, 1, 1),
            run=lambda x, w, b: F.batch_norm(
                x, running_mean=None, running_var=None, weight=w, bias=b, training=True, eps=1e-5
            ),
            stats=batch_stats,
        ),
        Case(
            name="group_norm_bf16",
            device="cpu",
            dtype=torch.bfloat16,
            value=1e8,
            shape=(1, 2, 1, 1),
            run=lambda x, w, b: F.group_norm(x, 1, w, b, eps=1e-5),
            stats=group_stats(1),
        ),
    ]

    if torch.cuda.is_available():
        cases.extend(
            [
                Case(
                    name="group_norm",
                    device="cuda",
                    dtype=torch.float32,
                    value=1e6,
                    shape=(1, 2, 1, 2),
                    run=lambda x, w, b: F.group_norm(x, 2, w, b, eps=1e-5),
                    stats=group_stats(2),
                ),
                Case(
                    name="batch_norm",
                    device="cuda",
                    dtype=torch.float32,
                    value=1e6,
                    shape=(2, 2, 1, 1),
                    run=lambda x, w, b: F.batch_norm(
                        x, running_mean=None, running_var=None, weight=w, bias=b, training=True, eps=1e-5
                    ),
                    stats=batch_stats,
                ),
            ]
        )

    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a nonzero exit code if the current PyTorch violates the contract",
    )
    args = parser.parse_args()

    print(f"torch {torch.__version__}")
    print("contract: constant normalized slice -> output == bias == 3 and dweight == 0")
    if torch.cuda.is_available():
        print(f"cuda {torch.version.cuda} device={torch.cuda.get_device_name(0)}")

    observations = [run_case(case) for case in build_cases()]
    for obs in observations:
        print_observation(obs)

    violations = [obs for obs in observations if violates_contract(obs)]
    print(f"\nviolations: {len(violations)} / {len(observations)}")
    return 1 if args.strict and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
