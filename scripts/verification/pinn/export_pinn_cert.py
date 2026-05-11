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

W1_H, W2_W = 16, 16


def seed_weights() -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    """Return deterministic tanh-MLP weights used by the bundled PINN fixture."""
    # W1: 16x1 with pattern; b1: 16; W2: 1x16; b2: 1
    W1 = [[(i + 1) * 0.1] for i in range(W1_H)]
    b1 = [0.05 * (i - 8) for i in range(W1_H)]
    W2 = [[0.1 + 0.01 * j for j in range(W2_W)]]
    b2 = [0.0]
    # Middle layer 16x16 with mild coupling
    Wm = [[(1.0 if i == j else 0.05) for j in range(W2_W)] for i in range(W1_H)]
    bm = [0.0 for _ in range(W1_H)]
    return W1, b1, Wm, bm, W2, b2


def ibp_linear(W: List[List[float]], b: List[float], lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate interval bounds through an affine layer."""
    m, n = len(W), len(W[0])
    out_lo, out_hi = [], []
    for i in range(m):
        li, ui = b[i], b[i]
        for j in range(n):
            a = W[i][j]
            p, q = a * lo[j], a * hi[j]
            li += min(p, q)
            ui += max(p, q)
        out_lo.append(li)
        out_hi.append(ui)
    return out_lo, out_hi


def ibp_tanh(lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate interval bounds through elementwise tanh."""
    out_lo, out_hi = [], []
    for l, u in zip(lo, hi):
        tl, tu = math.tanh(l), math.tanh(u)
        out_lo.append(min(tl, tu))
        out_hi.append(max(tl, tu))
    return out_lo, out_hi


def fd_residual_bounds(u_minus: Tuple[float, float], u0: Tuple[float, float], u_plus: Tuple[float, float], h: float) -> Tuple[float, float]:
    """Bound the centered finite-difference second derivative residual."""
    # u_minus, u0, u_plus are (lo, hi)
    l_minus, h_minus = u_minus
    l0, h0 = u0
    l_plus, h_plus = u_plus
    # r_num = u+ - 2u0 + u-
    num_lo = l_plus - 2.0 * h0 + l_minus
    num_hi = h_plus - 2.0 * l0 + h_minus
    s = 1.0 / (h * h)
    return num_lo * s, num_hi * s


# Interval helpers
def mul_interval(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Multiply two scalar intervals."""
    (al, ah), (bl, bh) = a, b
    p1, p2, p3, p4 = al * bl, al * bh, ah * bl, ah * bh
    return min(p1, p2, p3, p4), max(p1, p2, p3, p4)


def add_interval(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Add two scalar intervals."""
    return a[0] + b[0], a[1] + b[1]


def square_interval(a: Tuple[float, float]) -> Tuple[float, float]:
    """Square one scalar interval."""
    l, u = a
    l2, u2 = l * l, u * u
    if l <= 0.0 <= u:
        return 0.0, max(l2, u2)
    return (min(l2, u2), max(l2, u2))


def lin_deriv(W: List[List[float]], d_lo: List[float], d_hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate derivative interval bounds through a zero-bias linear layer."""
    # Like ibp_linear but with zero bias
    m, n = len(W), len(W[0])
    out_lo, out_hi = [], []
    for i in range(m):
        li, ui = 0.0, 0.0
        for j in range(n):
            a = W[i][j]
            p, q = a * d_lo[j], a * d_hi[j]
            li += min(p, q)
            ui += max(p, q)
        out_lo.append(li)
        out_hi.append(ui)
    return out_lo, out_hi


def tanh_prime_bounds(y_lo: float, y_hi: float) -> Tuple[float, float]:
    """Bound `d/dz tanh(z) = 1 - tanh(z)^2` from tanh-value bounds."""
    yl2, yh2 = y_lo * y_lo, y_hi * y_hi
    s_max = max(yl2, yh2)
    if y_lo <= 0.0 <= y_hi:
        s_min = 0.0
    else:
        s_min = min(yl2, yh2)
    return 1.0 - s_max, 1.0 - s_min


def tanh_second_bounds(y_lo: float, y_hi: float) -> Tuple[float, float]:
    """Bound the second tanh derivative expressed as a function of `y = tanh(z)`."""
    # f2(y) = -2y(1 - y^2) = -2y + 2y^3, extrema at y = ± 1/sqrt(3)
    def f2(y: float) -> float:
        """Evaluate the second-derivative polynomial in tanh-value space."""
        return -2.0 * y + 2.0 * (y * y * y)
    candidates = [f2(y_lo), f2(y_hi)]
    r = 1.0 / math.sqrt(3.0)
    for c in (-r, r):
        if y_lo < c < y_hi:
            candidates.append(f2(c))
    return min(candidates), max(candidates)


def deriv_and_second_for_mlp(
    W1: List[List[float]], b1: List[float],
    Wm: List[List[float]], bm: List[float],
    W2: List[List[float]], b2: List[float],
    x_lo: List[float], x_hi: List[float]
) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]]]:
    """Propagate first- and second-derivative intervals through the tanh MLP."""
    # Value bounds through the network
    h1_lo, h1_hi = ibp_linear(W1, b1, x_lo, x_hi)
    y1_lo, y1_hi = ibp_tanh(h1_lo, h1_hi)
    h2_lo, h2_hi = ibp_linear(Wm, bm, y1_lo, y1_hi)
    y2_lo, y2_hi = ibp_tanh(h2_lo, h2_hi)
    out_lo, out_hi = ibp_linear(W2, b2, y2_lo, y2_hi)
    # First derivative intervals
    dz1_lo, dz1_hi = [1.0], [1.0]
    dz2_lo, dz2_hi = [0.0], [0.0]
    # Through first linear
    dz1_lo, dz1_hi = lin_deriv(W1, dz1_lo, dz1_hi)
    dz2_lo, dz2_hi = lin_deriv(W1, dz2_lo, dz2_hi)
    # Through tanh1
    d1_t1_lo: List[float] = []
    d1_t1_hi: List[float] = []
    d2_t1_lo: List[float] = []
    d2_t1_hi: List[float] = []
    for yl, yh, z1l, z1h, z2l, z2h in zip(y1_lo, y1_hi, dz1_lo, dz1_hi, dz2_lo, dz2_hi):
        p1_lo, p1_hi = tanh_prime_bounds(yl, yh)
        s2_lo, s2_hi = tanh_second_bounds(yl, yh)
        # first derivative
        d1_lo, d1_hi = mul_interval((p1_lo, p1_hi), (z1l, z1h))
        d1_t1_lo.append(d1_lo); d1_t1_hi.append(d1_hi)
        # second derivative: p2*(z')^2 + p1*z''
        z1_sq = square_interval((z1l, z1h))
        tA = mul_interval((s2_lo, s2_hi), z1_sq)
        tB = mul_interval((p1_lo, p1_hi), (z2l, z2h))
        d2_lo, d2_hi = add_interval(tA, tB)
        d2_t1_lo.append(d2_lo); d2_t1_hi.append(d2_hi)
    dz1_lo, dz1_hi = d1_t1_lo, d1_t1_hi
    dz2_lo, dz2_hi = d2_t1_lo, d2_t1_hi
    # Through middle linear
    dz1_lo, dz1_hi = lin_deriv(Wm, dz1_lo, dz1_hi)
    dz2_lo, dz2_hi = lin_deriv(Wm, dz2_lo, dz2_hi)
    # Through tanh2
    d1_t2_lo: List[float] = []
    d1_t2_hi: List[float] = []
    d2_t2_lo: List[float] = []
    d2_t2_hi: List[float] = []
    for yl, yh, z1l, z1h, z2l, z2h in zip(y2_lo, y2_hi, dz1_lo, dz1_hi, dz2_lo, dz2_hi):
        p1_lo, p1_hi = tanh_prime_bounds(yl, yh)
        s2_lo, s2_hi = tanh_second_bounds(yl, yh)
        d1_lo, d1_hi = mul_interval((p1_lo, p1_hi), (z1l, z1h))
        d1_t2_lo.append(d1_lo); d1_t2_hi.append(d1_hi)
        z1_sq = square_interval((z1l, z1h))
        tA = mul_interval((s2_lo, s2_hi), z1_sq)
        tB = mul_interval((p1_lo, p1_hi), (z2l, z2h))
        d2_lo, d2_hi = add_interval(tA, tB)
        d2_t2_lo.append(d2_lo); d2_t2_hi.append(d2_hi)
    dz1_lo, dz1_hi = d1_t2_lo, d1_t2_hi
    dz2_lo, dz2_hi = d2_t2_lo, d2_t2_hi
    # Final linear to scalar output
    dz1_lo, dz1_hi = lin_deriv(W2, dz1_lo, dz1_hi)
    dz2_lo, dz2_hi = lin_deriv(W2, dz2_lo, dz2_hi)
    return (dz1_lo, dz1_hi), (dz2_lo, dz2_hi)


def run_ibp() -> Dict[str, Any]:
    """Compute the bundled PINN certificate payload."""
    W1, b1, Wm, bm, W2, b2 = seed_weights()
    h = 1e-2
    eps = 0.01
    points = [0.25, 0.5, 0.75]
    resid_lo: List[float] = []
    resid_hi: List[float] = []
    resid_lo_deriv: List[float] = []
    resid_hi_deriv: List[float] = []
    u_bounds: List[Dict[str, List[float]]] = []

    for x0 in points:
        def box(x: float) -> Tuple[List[float], List[float]]:
            """Return the local interval around one scalar point."""
            return [x - eps], [x + eps]

        # Evaluate IBP through MLP at x-h, x, x+h
        res_u: List[Tuple[float, float]] = []
        for x in (x0 - h, x0, x0 + h):
            x_lo, x_hi = box(x)
            h1_lo, h1_hi = ibp_linear(W1, b1, x_lo, x_hi)
            h1_lo, h1_hi = ibp_tanh(h1_lo, h1_hi)
            h2_lo, h2_hi = ibp_linear(Wm, bm, h1_lo, h1_hi)
            h2_lo, h2_hi = ibp_tanh(h2_lo, h2_hi)
            y_lo, y_hi = ibp_linear(W2, b2, h2_lo, h2_hi)
            # output is scalar
            res_u.append((y_lo[0], y_hi[0]))

        # FD residual at x0
        lo_fd, hi_fd = fd_residual_bounds(res_u[0], res_u[1], res_u[2], h)
        resid_lo.append(lo_fd)
        resid_hi.append(hi_fd)

        # Derivative-based residual: compute u''(x0) bounds via derivative passes at x0
        x_lo, x_hi = box(x0)
        (_, _), (d2_lo, d2_hi) = deriv_and_second_for_mlp(W1, b1, Wm, bm, W2, b2, x_lo, x_hi)
        resid_lo_deriv.append(d2_lo[0])
        resid_hi_deriv.append(d2_hi[0])

        u_bounds.append({
            "x": x0,
            "u_minus": {"lo": res_u[0][0], "hi": res_u[0][1]},
            "u": {"lo": res_u[1][0], "hi": res_u[1][1]},
            "u_plus": {"lo": res_u[2][0], "hi": res_u[2][1]},
        })

    cert = {
        "model": {
            "arch": [1, 16, 16, 1],
            "activations": ["tanh", "tanh"],
            "weights": {
                "W1": W1, "b1": b1,
                "Wm": Wm, "bm": bm,
                "W2": W2, "b2": b2,
            }
        },
        "pinn": {
            "pde": "u''(x) = 0",
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
        json.dump(cert, f, indent=2)
    print(f"Wrote certificate to {out_path}")


if __name__ == "__main__":
    main()
