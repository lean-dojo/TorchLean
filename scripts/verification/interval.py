"""Outward-rounded binary64 interval helpers shared by the IBP and PINN certificate exporters.

Every primitive widens its host-Float result by one representable value, in the operation order of
TorchLean's interval checkers, so regenerated certificates are bit-for-bit reproducible.
"""

import math


def round_down(x: float) -> float:
    """Step `x` one binary64 value toward negative infinity, as Lean's host-Float bounds do."""

    return math.nextafter(x, -math.inf)


def round_up(x: float) -> float:
    """Step `x` one binary64 value toward positive infinity, as Lean's host-Float bounds do."""

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


def affine_interval(
    weights: list[list[float]],
    bias: list[float],
    lo: list[float],
    hi: list[float],
) -> tuple[list[float], list[float]]:
    """Bound `W x + b` with outward rounding, in the operation order of Lean's `IBP.linear`.

    Each row starts from a zero accumulator, adds every term (structural zeros included) and then
    the bias. Skipping a zero term would change a host-Float result, because every primitive widens
    by one binary64 value.
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


def mul_interval(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Multiply two scalar intervals with outward-rounded endpoint products.

    The products are taken in the order lo * lo, lo * hi, hi * lo, hi * hi, and `min` and `max`
    return the first of equal values, so this order also fixes the sign of a zero endpoint.
    """

    (al, ah), (bl, bh) = a, b
    return (
        min(mul_down(al, bl), mul_down(al, bh), mul_down(ah, bl), mul_down(ah, bh)),
        max(mul_up(al, bl), mul_up(al, bh), mul_up(ah, bl), mul_up(ah, bh)),
    )


def add_interval(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Add two scalar intervals with outward rounding."""

    return add_down(a[0], b[0]), add_up(a[1], b[1])


def tanh_range_interval(length: int) -> tuple[list[float], list[float]]:
    """Return `[-1, 1]` per coordinate, TorchLean's format-independent tanh enclosure."""

    return [-1.0] * length, [1.0] * length
