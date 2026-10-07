#!/usr/bin/env python3
"""Export a compact deterministic PINN residual certificate."""
import json
import math
from typing import List, Dict, Any, Tuple

# Compact PINN-style workflow: 1D network u(x) approximates a solution on x in [0,1].
# Build a small tanh MLP (1 -> 16 -> 16 -> 1) with deterministic weights.
# Compute Interval Bound Propagation (IBP) bounds for u at x-h, x, x+h and produce
# a conservative bound on the finite-difference residual r ≈ (u(x+h) - 2u(x) + u(x-h)) / h^2.
# Use f(x) = 0 (homogeneous Poisson) so the exported certificate stays small and deterministic.

HIDDEN_WIDTH, MIDDLE_WIDTH = 16, 16


def round_down(x: float) -> float:
    """Move one binary64 value toward negative infinity."""

    return math.nextafter(x, -math.inf)


def round_up(x: float) -> float:
    """Move one binary64 value toward positive infinity."""

    return math.nextafter(x, math.inf)


def seed_weights() -> Tuple[
    List[List[float]], List[float], List[List[float]], List[float],
    List[List[float]], List[float],
]:
    """Return deterministic tanh-MLP weights used by the bundled PINN fixture."""
    first_weight = [[(i + 1) * 0.1] for i in range(HIDDEN_WIDTH)]
    first_bias = [0.05 * (i - 8) for i in range(HIDDEN_WIDTH)]
    output_weight = [[0.1 + 0.01 * j for j in range(MIDDLE_WIDTH)]]
    output_bias = [0.0]
    # Middle layer 16x16 with mild coupling
    middle_weight = [[(1.0 if i == j else 0.05) for j in range(MIDDLE_WIDTH)]
                     for i in range(HIDDEN_WIDTH)]
    middle_bias = [0.0 for _ in range(HIDDEN_WIDTH)]
    return first_weight, first_bias, middle_weight, middle_bias, output_weight, output_bias


def ibp_linear(W: List[List[float]], b: List[float], lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate affine bounds in the operation order used by TorchLean's checker."""
    m, n = len(W), len(W[0])
    out_lo, out_hi = [], []
    for i in range(m):
        li, ui = 0.0, 0.0
        for j in range(n):
            a = W[i][j]
            li = round_down(li + min(round_down(a * lo[j]), round_down(a * hi[j])))
            ui = round_up(ui + max(round_up(a * lo[j]), round_up(a * hi[j])))
        out_lo.append(round_down(li + b[i]))
        out_hi.append(round_up(ui + b[i]))
    return out_lo, out_hi


def ibp_tanh(lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Use TorchLean's format-independent global enclosure for ``tanh``."""

    assert len(lo) == len(hi)
    return [-1.0] * len(lo), [1.0] * len(lo)


def fd_residual_bounds(
    u_minus: Tuple[float, float],
    u_center: Tuple[float, float],
    u_plus: Tuple[float, float],
    h: float,
) -> Tuple[float, float]:
    """Produce finite-difference endpoints using ordinary binary64 arithmetic.

    These are producer assertions for the Lean checker, not directed-rounding
    evidence. Keep this schedule distinct from the derivative interval helpers.
    """
    # u_minus, u_center, u_plus are (lo, hi)
    l_minus, h_minus = u_minus
    center_lo, center_hi = u_center
    l_plus, h_plus = u_plus
    # r_num = u+ - 2u0 + u-
    num_lo = l_plus - 2.0 * center_hi + l_minus
    num_hi = h_plus - 2.0 * center_lo + h_minus
    s = 1.0 / (h * h)
    return num_lo * s, num_hi * s


# Interval helpers
def mul_interval(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Multiply two scalar intervals with outward-rounded endpoint products."""
    (al, ah), (bl, bh) = a, b
    products_lo = [round_down(al * bl), round_down(al * bh),
                   round_down(ah * bl), round_down(ah * bh)]
    products_hi = [round_up(al * bl), round_up(al * bh),
                   round_up(ah * bl), round_up(ah * bh)]
    return min(products_lo), max(products_hi)


def add_interval(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Add two scalar intervals with outward rounding."""
    return round_down(a[0] + b[0]), round_up(a[1] + b[1])


def square_interval(a: Tuple[float, float]) -> Tuple[float, float]:
    """Square one scalar interval with outward rounding."""
    l, u = a
    lower_sq = round_down(l * l)
    upper_sq = round_up(u * u)
    if l <= 0.0 <= u:
        return 0.0, max(round_up(l * l), upper_sq)
    return min(lower_sq, round_down(u * u)), max(round_up(l * l), upper_sq)


def lin_deriv(W: List[List[float]], d_lo: List[float], d_hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate derivative interval bounds through a zero-bias linear layer."""
    return ibp_linear(W, [0.0] * len(W), d_lo, d_hi)


def tanh_prime_bounds(y_lo: float, y_hi: float) -> Tuple[float, float]:
    """Use TorchLean's global enclosure for the first derivative of ``tanh``."""
    assert y_lo <= y_hi
    return 0.0, 1.0


def tanh_second_bounds(y_lo: float, y_hi: float) -> Tuple[float, float]:
    """Use TorchLean's global enclosure for the second derivative of ``tanh``."""
    assert y_lo <= y_hi
    return -2.0, 2.0


def deriv_and_second_for_mlp(
    first_weight: List[List[float]], first_bias: List[float],
    middle_weight: List[List[float]], middle_bias: List[float],
    output_weight: List[List[float]],
    x_lo: List[float], x_hi: List[float]
) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]]]:
    """Propagate first- and second-derivative intervals through the tanh MLP."""
    # Value bounds through the network
    first_linear_lo, first_linear_hi = ibp_linear(first_weight, first_bias, x_lo, x_hi)
    first_activation_lo, first_activation_hi = ibp_tanh(first_linear_lo, first_linear_hi)
    middle_linear_lo, middle_linear_hi = ibp_linear(
        middle_weight,
        middle_bias,
        first_activation_lo,
        first_activation_hi,
    )
    middle_activation_lo, middle_activation_hi = ibp_tanh(middle_linear_lo, middle_linear_hi)
    # First derivative intervals
    first_deriv_lo, first_deriv_hi = [1.0], [1.0]
    second_deriv_lo, second_deriv_hi = [0.0], [0.0]
    # Both hidden layers use the same chain rule and endpoint operation order.
    for weight, activation_lo, activation_hi in (
        (first_weight, first_activation_lo, first_activation_hi),
        (middle_weight, middle_activation_lo, middle_activation_hi),
    ):
        first_deriv_lo, first_deriv_hi = lin_deriv(weight, first_deriv_lo, first_deriv_hi)
        second_deriv_lo, second_deriv_hi = lin_deriv(weight, second_deriv_lo, second_deriv_hi)
        next_deriv_lo: List[float] = []
        next_deriv_hi: List[float] = []
        next_second_lo: List[float] = []
        next_second_hi: List[float] = []
        for value_lo, value_hi, deriv_lo, deriv_hi, second_lo, second_hi in zip(
            activation_lo, activation_hi, first_deriv_lo, first_deriv_hi,
            second_deriv_lo, second_deriv_hi,
        ):
            prime = tanh_prime_bounds(value_lo, value_hi)
            second = tanh_second_bounds(value_lo, value_hi)
            deriv_out_lo, deriv_out_hi = mul_interval(prime, (deriv_lo, deriv_hi))
            next_deriv_lo.append(deriv_out_lo)
            next_deriv_hi.append(deriv_out_hi)
            # Second derivative: p2*(z')^2 + p1*z''.
            deriv_sq = square_interval((deriv_lo, deriv_hi))
            curvature_term = mul_interval(second, deriv_sq)
            chain_term = mul_interval(prime, (second_lo, second_hi))
            second_out_lo, second_out_hi = add_interval(curvature_term, chain_term)
            next_second_lo.append(second_out_lo)
            next_second_hi.append(second_out_hi)
        first_deriv_lo, first_deriv_hi = next_deriv_lo, next_deriv_hi
        second_deriv_lo, second_deriv_hi = next_second_lo, next_second_hi
    # Final linear to scalar output
    first_deriv_lo, first_deriv_hi = lin_deriv(output_weight, first_deriv_lo, first_deriv_hi)
    second_deriv_lo, second_deriv_hi = lin_deriv(output_weight, second_deriv_lo, second_deriv_hi)
    return (first_deriv_lo, first_deriv_hi), (second_deriv_lo, second_deriv_hi)


def run_ibp() -> Dict[str, Any]:
    """Compute the bundled PINN certificate payload."""
    first_weight, first_bias, middle_weight, middle_bias, output_weight, output_bias = seed_weights()
    h = 1e-2
    eps = 0.01
    points = [0.25, 0.5, 0.75]
    resid_lo: List[float] = []
    resid_hi: List[float] = []
    resid_lo_deriv: List[float] = []
    resid_hi_deriv: List[float] = []
    u_bounds: List[Dict[str, List[float]]] = []

    for center in points:
        def box(x: float) -> Tuple[List[float], List[float]]:
            """Return the local interval around one scalar point."""
            return [x - eps], [x + eps]

        # Evaluate IBP through MLP at x-h, x, x+h
        res_u: List[Tuple[float, float]] = []
        for x in (center - h, center, center + h):
            x_lo, x_hi = box(x)
            hidden_lo, hidden_hi = ibp_linear(first_weight, first_bias, x_lo, x_hi)
            hidden_lo, hidden_hi = ibp_tanh(hidden_lo, hidden_hi)
            middle_lo, middle_hi = ibp_linear(middle_weight, middle_bias, hidden_lo, hidden_hi)
            middle_lo, middle_hi = ibp_tanh(middle_lo, middle_hi)
            y_lo, y_hi = ibp_linear(output_weight, output_bias, middle_lo, middle_hi)
            # output is scalar
            res_u.append((y_lo[0], y_hi[0]))

        # Finite-difference residual at the center point.
        lo_fd, hi_fd = fd_residual_bounds(res_u[0], res_u[1], res_u[2], h)
        resid_lo.append(lo_fd)
        resid_hi.append(hi_fd)

        # Derivative-based residual: compute u'' bounds at the center point.
        x_lo, x_hi = box(center)
        (_, _), (d2_lo, d2_hi) = deriv_and_second_for_mlp(
            first_weight,
            first_bias,
            middle_weight,
            middle_bias,
            output_weight,
            x_lo,
            x_hi,
        )
        resid_lo_deriv.append(d2_lo[0])
        resid_hi_deriv.append(d2_hi[0])

        u_bounds.append({
            "x": center,
            "u_minus": {"lo": res_u[0][0], "hi": res_u[0][1]},
            "u": {"lo": res_u[1][0], "hi": res_u[1][1]},
            "u_plus": {"lo": res_u[2][0], "hi": res_u[2][1]},
        })

    cert = {
        "model": {
            "arch": [1, 16, 16, 1],
            "activations": ["tanh", "tanh"],
            "weights": {
                "W1": first_weight, "b1": first_bias,
                "Wm": middle_weight, "bm": middle_bias,
                "W2": output_weight, "b2": output_bias,
            }
        },
        "pinn": {
            "pde": "uxx",
            "domain": [0.0, 1.0],
            "h": h,
            "eps": eps,
            "points": points,
        },
        "residual_bounds": {"lo": resid_lo, "hi": resid_hi},
        "residual_bounds_deriv": {"lo": resid_lo_deriv, "hi": resid_hi_deriv},
        "u_bounds": u_bounds,
    }
    return cert


def main():
    """Write the bundled PINN certificate JSON."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/PINN/pinn_cert.json"
    with open(out_path, "w") as f:
        json.dump(cert, f, indent=2, allow_nan=False)
    print(f"Wrote certificate to {out_path}")


if __name__ == "__main__":
    main()
