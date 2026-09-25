"""Shared helpers for the small LiRPA certificate producers.

These functions are intentionally tiny.  The exporter scripts still spell out the model-specific
graph and parameters; this module only keeps the boring interval arithmetic and JSON writing in one
place so the fixture producers do not drift apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def round_down(x: float) -> float:
    """Move one binary64 value toward negative infinity, matching Lean's host-Float bounds."""

    return math.nextafter(x, -math.inf)


def round_up(x: float) -> float:
    """Move one binary64 value toward positive infinity, matching Lean's host-Float bounds."""

    return math.nextafter(x, math.inf)


def add_down(x: float, y: float) -> float:
    """Add two host floats and widen the result downward by one representable value."""

    return round_down(x + y)


def add_up(x: float, y: float) -> float:
    """Add two host floats and widen the result upward by one representable value."""

    return round_up(x + y)


def mul_down(x: float, y: float) -> float:
    """Multiply two host floats and widen the result downward by one representable value."""

    return round_down(x * y)


def mul_up(x: float, y: float) -> float:
    """Multiply two host floats and widen the result upward by one representable value."""

    return round_up(x * y)


def centered_box(center: list[float], eps: float) -> tuple[list[float], list[float]]:
    """Enclose `center ± eps` with the directed endpoints used by Lean's `lInfBall`."""

    return [round_down(x - eps) for x in center], [add_up(x, eps) for x in center]


def affine_interval(
    weights: list[list[float]],
    bias: list[float],
    lo: list[float],
    hi: list[float],
) -> tuple[list[float], list[float]]:
    """Propagate bounds using the same outward-rounded operation order as Lean's IBP linear rule.

    The zero accumulator, every coefficient (including structural zeros), and the final bias are
    processed in exactly the order used by `IBP.linear`.  Skipping a zero coefficient would still
    change a host-Float result because every primitive deliberately widens by one binary64 value.
    """

    out_lo: list[float] = []
    out_hi: list[float] = []
    for row, b in zip(weights, bias):
        lo_i = 0.0
        hi_i = 0.0
        for a, x_lo, x_hi in zip(row, lo, hi):
            lo_i = add_down(lo_i, min(mul_down(a, x_lo), mul_down(a, x_hi)))
            hi_i = add_up(hi_i, max(mul_up(a, x_lo), mul_up(a, x_hi)))
        out_lo.append(add_down(lo_i, b))
        out_hi.append(add_up(hi_i, b))
    return out_lo, out_hi


def matmul_interval(
    weights: list[list[float]],
    lo: list[float],
    hi: list[float],
) -> tuple[list[float], list[float]]:
    """Propagate interval bounds through a bias-free matrix multiply."""

    return affine_interval(weights, [0.0 for _ in weights], lo, hi)


def relu_interval(lo: list[float], hi: list[float]) -> tuple[list[float], list[float]]:
    """Propagate interval bounds through elementwise ReLU."""

    return [max(0.0, x) for x in lo], [max(0.0, x) for x in hi]


def softmax_range_interval(length: int) -> tuple[list[float], list[float]]:
    """Mirror TorchLean's format-independent softmax range enclosure.

    The executable checker does not use host ``exp`` calls as directed interval operations.  A
    singleton softmax is exactly one; every coordinate of a longer vector lies in ``[0, 1]``.
    """

    if length == 1:
        return [1.0], [1.0]
    return [0.0] * length, [1.0] * length


def layernorm_range_interval(length: int) -> tuple[list[float], list[float]]:
    """Mirror TorchLean's uniform last-axis LayerNorm enclosure.

    ``ibpLayerNormRange?`` uses ``subDown 0 radius`` for the lower endpoint. The host-Float
    instance widens this subtraction as well as the square root used to compute the radius.
    """

    if length <= 1:
        return [0.0] * length, [0.0] * length
    radius = round_up(math.sqrt(float(length)))
    return [round_down(0.0 - radius)] * length, [radius] * length


def sigmoid_range_interval(length: int) -> tuple[list[float], list[float]]:
    """Mirror TorchLean's format-independent sigmoid enclosure."""

    return [0.0] * length, [1.0] * length


def tanh_range_interval(length: int) -> tuple[list[float], list[float]]:
    """Mirror TorchLean's format-independent hyperbolic-tangent enclosure."""

    return [-1.0] * length, [1.0] * length


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write one certificate JSON payload with stable indentation."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out
