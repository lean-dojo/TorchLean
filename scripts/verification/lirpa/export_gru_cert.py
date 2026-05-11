#!/usr/bin/env python3
"""Export a deterministic GRU-gate interval certificate."""
import json
import math
from typing import List, Dict, Any, Tuple

# GRU gate graph: 0=input(3) -> 1=linear -> 2=sigmoid; 0 -> 3=linear -> 4=tanh; 5=mul_elem

n = 3


def seed_params():
    """Return deterministic shared gate weights and biases."""
    W = [[float(1 + (i + j)) for j in range(n)] for i in range(n)]
    b = [float(i) for i in range(n)]
    return W, b


def seed_input_box(eps: float = 0.5):
    """Return the input interval box centered at `[1, 2, 3]`."""
    x0 = [float(i + 1) for i in range(n)]
    lo = [xi - eps for xi in x0]
    hi = [xi + eps for xi in x0]
    return lo, hi


def ibp_linear(W: List[List[float]], b: List[float], lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate interval bounds through an affine layer."""
    m, n = len(W), len(W[0])
    out_lo = []
    out_hi = []
    for i in range(m):
        lo_i = b[i]
        hi_i = b[i]
        for j in range(n):
            a = W[i][j]
            p = a * lo[j]
            q = a * hi[j]
            lo_i += min(p, q)
            hi_i += max(p, q)
        out_lo.append(lo_i)
        out_hi.append(hi_i)
    return out_lo, out_hi


def ibp_sigmoid(lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate interval bounds through elementwise sigmoid."""
    def s(x: float) -> float:
        """Evaluate the logistic sigmoid at one scalar."""
        return 1.0 / (1.0 + math.exp(-x))
    out_lo = []
    out_hi = []
    for l, u in zip(lo, hi):
        sl, su = s(l), s(u)
        out_lo.append(min(sl, su))
        out_hi.append(max(sl, su))
    return out_lo, out_hi


def ibp_tanh(lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate interval bounds through elementwise tanh."""
    def t(x: float) -> float:
        """Evaluate tanh at one scalar."""
        return math.tanh(x)
    out_lo = []
    out_hi = []
    for l, u in zip(lo, hi):
        tl, tu = t(l), t(u)
        out_lo.append(min(tl, tu))
        out_hi.append(max(tl, tu))
    return out_lo, out_hi


def ibp_mul_elem(x_lo: List[float], x_hi: List[float], y_lo: List[float], y_hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate interval bounds through elementwise multiplication."""
    lo = []
    hi = []
    for lx, ux, ly, uy in zip(x_lo, x_hi, y_lo, y_hi):
        p1 = lx * ly
        p2 = lx * uy
        p3 = ux * ly
        p4 = ux * uy
        lo.append(min(p1, p2, p3, p4))
        hi.append(max(p1, p2, p3, p4))
    return lo, hi


def run_ibp() -> Dict[str, Any]:
    """Compute the GRU-gate certificate payload consumed by Lean."""
    W, b = seed_params()
    x_lo, x_hi = seed_input_box(0.5)
    a_lo, a_hi = ibp_linear(W, b, x_lo, x_hi)  # node 1
    s_lo, s_hi = ibp_sigmoid(a_lo, a_hi)       # node 2
    b_lo, b_hi = ibp_linear(W, b, x_lo, x_hi)  # node 3
    t_lo, t_hi = ibp_tanh(b_lo, b_hi)          # node 4
    y_lo, y_hi = ibp_mul_elem(s_lo, s_hi, t_lo, t_hi)  # node 5
    return {
        "graph": "gru_gate_workflow_v1",
        "input_box": {"id": 0, "dim": n, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 5, "dim": n, "lo": y_lo, "hi": y_hi},
    }


def main():
    """Write the GRU-gate certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/gru_gate_cert.json"
    with open(out_path, "w") as f:
        json.dump(cert, f, indent=2)
    print(f"Wrote certificate to {out_path}")


if __name__ == "__main__":
    main()
