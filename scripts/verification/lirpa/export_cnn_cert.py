#!/usr/bin/env python3
"""Export a deterministic CNN certificate with a ReLU affine relaxation."""
import json
from typing import List, Tuple, Dict, Any

# CNN graph (mirrors the Lean CNN certificate workflow):
# input (1x4x4) -> conv2d (1x3x3, stride=1,pad=0) + ReLU (as CROWN affine) -> flatten -> linear head (2)

inC, inH, inW = 1, 4, 4
outC, kH, kW, stride, padding = 1, 3, 3, 1, 0
outH = (inH + 2 * padding - kH) // stride + 1
outW = (inW + 2 * padding - kW) // stride + 1
nIn = inC * inH * inW
nConv = outC * outH * outW
nOut = 2


def seed_conv_params() -> Tuple[List[List[List[List[float]]]], List[float]]:
    """Return deterministic convolution weights and biases."""
    # kernel[outC][inC][kH][kW] with entries 1 + (i + j)
    kernel = [
        [
            [[float(1 + (i + j)) for j in range(kW)] for i in range(kH)]
            for _ in range(inC)
        ]
        for _ in range(outC)
    ]
    bias = [0.0 for _ in range(outC)]
    return kernel, bias


def seed_input_box(eps: float = 0.1) -> Tuple[List[List[List[float]]], List[List[List[float]]]]:
    """Return a small input image interval box centered at all ones."""
    # center ones
    lo = [[[1.0 - eps for _ in range(inW)] for _ in range(inH)] for _ in range(inC)]
    hi = [[[1.0 + eps for _ in range(inW)] for _ in range(inH)] for _ in range(inC)]
    return lo, hi


def ibp_conv2d_preact(kernel, bias, lo, hi) -> Tuple[List[List[List[float]]], List[List[List[float]]]]:
    """Compute pre-activation interval bounds for the convolution output."""
    # Compute pre-activation bounds y_lo/hi[outC][outH][outW]
    ylo = [[[0.0 for _ in range(outW)] for _ in range(outH)] for _ in range(outC)]
    yhi = [[[0.0 for _ in range(outW)] for _ in range(outH)] for _ in range(outC)]
    for oc in range(outC):
        for i in range(outH):
            for j in range(outW):
                acc_lo = 0.0
                acc_hi = 0.0
                for ic in range(inC):
                    for di in range(kH):
                        for dj in range(kW):
                            pi = i * stride + di
                            pj = j * stride + dj
                            if (pi >= padding) and (pj >= padding):
                                ii = pi - padding
                                jj = pj - padding
                                if 0 <= ii < inH and 0 <= jj < inW:
                                    xlo = lo[ic][ii][jj]
                                    xhi = hi[ic][ii][jj]
                                    a = kernel[oc][ic][di][dj]
                                    p1 = a * xlo
                                    p2 = a * xhi
                                    acc_lo += min(p1, p2)
                                    acc_hi += max(p1, p2)
                ylo[oc][i][j] = acc_lo + bias[oc]
                yhi[oc][i][j] = acc_hi + bias[oc]
    return ylo, yhi


def relu_relax_scalar(l: float, u: float) -> Tuple[float, float]:
    """Return slope/intercept for an upper affine ReLU relaxation on `[l, u]`."""
    if u > 0:
        if l > 0:
            return 1.0, 0.0
        denom = (u - l)
        a = (u / denom) if denom > 1e-12 else 0.0
        b = -a * l
        return a, b
    return 0.0, 0.0


def flatten3(x3: List[List[List[float]]]) -> List[float]:
    """Flatten `[outC][outH][outW]` data in the same order as the Lean checker."""
    return [x3[oc][i][j] for oc in range(outC) for i in range(outH) for j in range(outW)]


def conv2d_linear_matrix(kernel) -> List[List[float]]:
    """Materialize the convolution as a dense matrix for the affine certificate."""
    # Build Wconv[nConv x nIn] as in Lean conv2d_linear_matrix
    def decode_in(c: int) -> Tuple[int, int, int]:
        """Decode one flattened input index into `(channel, row, column)`."""
        c0 = c // (inH * inW)
        r0 = c % (inH * inW)
        i0 = r0 // inW
        j0 = r0 % inW
        return c0, i0, j0

    def encode_out(oc: int, i: int, j: int) -> int:
        """Encode one convolution output coordinate as a flattened row index."""
        return (oc * outH + i) * outW + j

    W = [[0.0 for _ in range(nIn)] for _ in range(nConv)]
    for oc in range(outC):
        for i in range(outH):
            for j in range(outW):
                r = encode_out(oc, i, j)
                for c in range(nIn):
                    ic, ii, jj = decode_in(c)
                    coeff = 0.0
                    for di in range(kH):
                        for dj in range(kW):
                            pi = i * stride + di
                            pj = j * stride + dj
                            if (pi >= padding) and (pj >= padding):
                                iii = pi - padding
                                jjj = pj - padding
                                if (iii == ii) and (jjj == jj):
                                    coeff += kernel[oc][ic][di][dj]
                    W[r][c] = coeff
    return W


def mat_row_scale(W: List[List[float]], v: List[float]) -> List[List[float]]:
    """Scale row `i` of matrix `W` by `v[i]`."""
    return [[v[i] * aij for aij in row] for i, row in enumerate(W)]


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


def run_ibp() -> Dict[str, Any]:
    """Compute the CNN certificate payload consumed by Lean."""
    kernel, bias = seed_conv_params()
    x_lo3, x_hi3 = seed_input_box(0.1)
    # Pre-activation bounds for conv
    z_lo3, z_hi3 = ibp_conv2d_preact(kernel, bias, x_lo3, x_hi3)
    # ReLU relax slope/bias per output
    slope = []
    rbias = []
    for oc in range(outC):
        for i in range(outH):
            for j in range(outW):
                s, b = relu_relax_scalar(z_lo3[oc][i][j], z_hi3[oc][i][j])
                slope.append(s)
                rbias.append(b)
    # Conv linear operator and affine after ReLU relaxation
    Wconv = conv2d_linear_matrix(kernel)
    A = mat_row_scale(Wconv, slope)  # scale rows by slope
    # Broadcast bias per output position (bias is zero here), combine: c = slope*bconv + rbias
    bconv = [bias[0] for _ in range(nConv)]
    c = [slope[i] * bconv[i] + rbias[i] for i in range(nConv)]
    # Flatten input box and propagate linear affine
    x_lo = [x_lo3[0][i][j] for i in range(inH) for j in range(inW)]
    x_hi = [x_hi3[0][i][j] for i in range(inH) for j in range(inW)]
    h_lo, h_hi = ibp_linear(A, c, x_lo, x_hi)
    # Linear head 2 x nConv with W[i,j] = 2 + (i + j), b[i] = i
    Whead = [[float(2 + (i + j)) for j in range(nConv)] for i in range(nOut)]
    bhead = [float(i) for i in range(nOut)]
    y_lo, y_hi = ibp_linear(Whead, bhead, h_lo, h_hi)
    return {
        "graph": "cnn_graph_workflow_v1",
        "input_box": {"id": 0, "dim": nIn, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 2, "dim": nOut, "lo": y_lo, "hi": y_hi},
    }


def main():
    """Write the CNN certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/cnn_cert.json"
    with open(out_path, "w") as f:
        json.dump(cert, f, indent=2)
    print(f"Wrote certificate to {out_path}")


if __name__ == "__main__":
    main()
