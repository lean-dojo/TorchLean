#!/usr/bin/env python3
"""Export a tiny transformer-encoder-style interval certificate."""
import json
import math
from typing import List, Dict, Any

# Tiny transformer-encoder-like graph IBP, mirroring
# `NN.Examples.Verification.LiRPA.TransformerEncoderVerify`.
# Node ids: 0=input(4), 1=linear Wq (5), 2=softmax, 3=matmul Wv (4), 4=add residual,
# 5=layernorm, 6=linear W1 (6), 7=relu, 8=linear W2 (4), 9=add, 10=layernorm

nModel = 4
scoresDim = 5
nHidden = 6

def seed_params():
    """Return deterministic weights for the transformer-like fixture graph."""
    # Wq[i,j] = 1 + (i + 2*j); bq[i] = 0.1 * i
    Wq = [[float(1 + (i + 2*j)) for j in range(nModel)] for i in range(scoresDim)]
    bq = [0.1 * float(i) for i in range(scoresDim)]
    # Wv[i,j] = 2 + (i + j)
    Wv = [[float(2 + (i + j)) for j in range(scoresDim)] for i in range(nModel)]
    # W1[i,j] = 1 + ((i + j) % 3); b1[i] = 0.05 * i
    W1 = [[float(1 + ((i + j) % 3)) for j in range(nModel)] for i in range(nHidden)]
    b1 = [0.05 * float(i) for i in range(nHidden)]
    # W2[i,j] = 2 + ((i + j) % 4); b2[i] = 0.02 * i
    W2 = [[float(2 + ((i + j) % 4)) for j in range(nHidden)] for i in range(nModel)]
    b2 = [0.02 * float(i) for i in range(nModel)]
    return Wq, bq, Wv, W1, b1, W2, b2


def seed_input_box(eps: float = 0.5):
    """Return the input interval box centered at `[1, 2, 3, 4]`."""
    x0 = [float(i + 1) for i in range(nModel)]
    lo = [xi - eps for xi in x0]
    hi = [xi + eps for xi in x0]
    return lo, hi


def ibp_linear(W: List[List[float]], b: List[float], lo: List[float], hi: List[float]):
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


def ibp_matmul(W: List[List[float]], lo: List[float], hi: List[float]):
    """Propagate interval bounds through a bias-free matrix multiply."""
    b = [0.0 for _ in range(len(W))]
    return ibp_linear(W, b, lo, hi)


def ibp_relu(lo: List[float], hi: List[float]):
    """Propagate interval bounds through elementwise ReLU."""
    return [max(0.0, x) for x in lo], [max(0.0, x) for x in hi]


def ibp_add(lo1: List[float], hi1: List[float], lo2: List[float], hi2: List[float]):
    """Add two interval vectors elementwise."""
    return [a + c for a, c in zip(lo1, lo2)], [b + d for b, d in zip(hi1, hi2)]


def ibp_softmax(lo: List[float], hi: List[float]):
    """Compute conservative elementwise softmax interval bounds."""
    # Tight elementwise bounds as in Lean: exp(lo)/[exp(lo)+sum_{j!=i}exp(hi_j)] and vice versa
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


def ibp_layernorm(lo: List[float], hi: List[float], eps: float = 1e-6):
    """Propagate coarse interval bounds through a layernorm-like normalization."""
    n = len(lo)
    sum_lo = sum(lo)
    sum_hi = sum(hi)
    mu_lo = sum_lo / n
    mu_hi = sum_hi / n
    # Upper bound on variance via worst-case deviations
    sum_abs_sq = 0.0
    for i in range(n):
        dl = abs(lo[i] - mu_hi)
        du = abs(hi[i] - mu_lo)
        a = max(dl, du)
        sum_abs_sq += a * a
    var_hi = sum_abs_sq / n
    den_lo = math.sqrt(eps)
    den_hi = math.sqrt(var_hi + eps)
    out_lo = []
    out_hi = []
    for i in range(n):
        dl = lo[i] - mu_hi
        du = hi[i] - mu_lo
        cands = [dl / den_lo, dl / den_hi, du / den_lo, du / den_hi]
        out_lo.append(min(cands))
        out_hi.append(max(cands))
    return out_lo, out_hi


def run_ibp() -> Dict[str, Any]:
    """Compute the transformer-like certificate payload consumed by Lean."""
    Wq, bq, Wv, W1, b1, W2, b2 = seed_params()
    x_lo, x_hi = seed_input_box(0.5)
    s_lo, s_hi = ibp_linear(Wq, bq, x_lo, x_hi)          # node 1
    p_lo, p_hi = ibp_softmax(s_lo, s_hi)                 # node 2
    a_lo, a_hi = ibp_matmul(Wv, p_lo, p_hi)              # node 3
    r1_lo, r1_hi = ibp_add(x_lo, x_hi, a_lo, a_hi)       # node 4
    n1_lo, n1_hi = ibp_layernorm(r1_lo, r1_hi)           # node 5
    h_lo, h_hi = ibp_linear(W1, b1, n1_lo, n1_hi)        # node 6
    h_lo, h_hi = ibp_relu(h_lo, h_hi)                    # node 7
    o_lo, o_hi = ibp_linear(W2, b2, h_lo, h_hi)          # node 8
    r2_lo, r2_hi = ibp_add(n1_lo, n1_hi, o_lo, o_hi)     # node 9
    n2_lo, n2_hi = ibp_layernorm(r2_lo, r2_hi)           # node 10

    return {
        "graph": "transformer_encoder_workflow_v1",
        "input_box": {"id": 0, "dim": nModel, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 10, "dim": nModel, "lo": n2_lo, "hi": n2_hi}
    }


def main():
    """Write the transformer-like certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/transformer_encoder_cert.json"
    with open(out_path, "w") as f:
        json.dump(cert, f, indent=2)
    print(f"Wrote certificate to {out_path}")

if __name__ == "__main__":
    main()
