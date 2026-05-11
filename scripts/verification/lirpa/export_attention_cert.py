#!/usr/bin/env python3
"""Export a deterministic attention-softmax interval certificate."""
import json
import math
from typing import List, Dict, Any

# Attention-softmax graph: 0=input(4) -> 1=matmul Wq (5) -> 2=softmax -> 3=matmul Wv (3)

nIn = 4
nScores = 5
nOut = 3


def seed_params():
    """Return deterministic query/value matrices for the attention fixture."""
    Wq = [[float(1 + (i + 2 * j)) for j in range(nIn)] for i in range(nScores)]
    Wv = [[float(2 + (i + j)) for j in range(nScores)] for i in range(nOut)]
    return Wq, Wv


def seed_input_box(eps: float = 0.5):
    """Return the input interval box centered at `[1, 2, 3, 4]`."""
    x0 = [float(i + 1) for i in range(nIn)]
    lo = [xi - eps for xi in x0]
    hi = [xi + eps for xi in x0]
    return lo, hi


def ibp_matmul(W: List[List[float]], lo: List[float], hi: List[float]):
    """Propagate interval bounds through a matrix multiplication."""
    m, n = len(W), len(W[0])
    out_lo = []
    out_hi = []
    for i in range(m):
        lo_i = 0.0
        hi_i = 0.0
        for j in range(n):
            a = W[i][j]
            p = a * lo[j]
            q = a * hi[j]
            lo_i += min(p, q)
            hi_i += max(p, q)
        out_lo.append(lo_i)
        out_hi.append(hi_i)
    return out_lo, out_hi


def ibp_softmax(lo: List[float], hi: List[float]):
    """Compute conservative elementwise softmax interval bounds."""
    elo = [math.exp(x) for x in lo]
    ehi = [math.exp(x) for x in hi]
    total_lo = sum(elo)
    total_hi = sum(ehi)
    out_lo = []
    out_hi = []
    for i in range(len(lo)):
        lo_i = elo[i] / (elo[i] + (total_hi - ehi[i]))
        hi_i = ehi[i] / (ehi[i] + (total_lo - elo[i]))
        out_lo.append(lo_i)
        out_hi.append(hi_i)
    return out_lo, out_hi


def run_ibp() -> Dict[str, Any]:
    """Compute the attention certificate payload consumed by Lean."""
    Wq, Wv = seed_params()
    x_lo, x_hi = seed_input_box(0.5)
    s_lo, s_hi = ibp_matmul(Wq, x_lo, x_hi)   # node 1
    p_lo, p_hi = ibp_softmax(s_lo, s_hi)      # node 2
    y_lo, y_hi = ibp_matmul(Wv, p_lo, p_hi)   # node 3
    return {
        "graph": "attention_softmax_workflow_v1",
        "input_box": {"id": 0, "dim": nIn, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 3, "dim": nOut, "lo": y_lo, "hi": y_hi},
    }


def main():
    """Write the attention-softmax certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/attention_softmax_cert.json"
    with open(out_path, "w") as f:
        json.dump(cert, f, indent=2)
    print(f"Wrote certificate to {out_path}")


if __name__ == "__main__":
    main()
