#!/usr/bin/env python3
"""Export a deterministic interval-bound certificate for a small CNN."""
from typing import Any

from common import affine_interval, centered_box, write_json

# CNN graph (mirrors the Lean CNN certificate workflow):
# input (1x4x4) -> conv2d (1x3x3, stride=1,pad=0) -> ReLU -> flatten -> linear head (2)

inC, inH, inW = 1, 4, 4
outC, kH, kW, stride, padding = 1, 3, 3, 1, 0
outH = (inH + 2 * padding - kH) // stride + 1
outW = (inW + 2 * padding - kW) // stride + 1
nIn = inC * inH * inW
nConv = outC * outH * outW
nOut = 2


def seed_conv_params() -> tuple[list[list[list[list[float]]]], list[float]]:
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


def seed_input_box(eps: float = 0.1) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Return a small input image interval box centered at all ones."""
    flat_lo, flat_hi = centered_box([1.0] * nIn, eps)
    lo = [
        [[flat_lo[(c * inH + i) * inW + j] for j in range(inW)] for i in range(inH)]
        for c in range(inC)
    ]
    hi = [
        [[flat_hi[(c * inH + i) * inW + j] for j in range(inW)] for i in range(inH)]
        for c in range(inC)
    ]
    return lo, hi


def ibp_conv2d_preact(kernel, bias, lo, hi) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Compute convolution bounds through the same flattened affine matrix used by Lean.

    Materializing structural zeros is intentional: Lean checks this graph as an ordinary dense
    linear node, and its host-Float bound operations widen even a zero product.  A sparse convolution
    loop would therefore produce a different certificate despite denoting the same exact map.
    """

    matrix = [[0.0 for _ in range(nIn)] for _ in range(nConv)]
    for oc in range(outC):
        for i in range(outH):
            for j in range(outW):
                for ic in range(inC):
                    for di in range(kH):
                        for dj in range(kW):
                            pi = i * stride + di
                            pj = j * stride + dj
                            if (pi >= padding) and (pj >= padding):
                                ii = pi - padding
                                jj = pj - padding
                                if 0 <= ii < inH and 0 <= jj < inW:
                                    row = (oc * outH + i) * outW + j
                                    col = (ic * inH + ii) * inW + jj
                                    matrix[row][col] = kernel[oc][ic][di][dj]

    flat_lo = [lo[ic][i][j] for ic in range(inC) for i in range(inH) for j in range(inW)]
    flat_hi = [hi[ic][i][j] for ic in range(inC) for i in range(inH) for j in range(inW)]
    flat_bias = [bias[oc] for oc in range(outC) for _ in range(outH * outW)]
    flat_ylo, flat_yhi = affine_interval(matrix, flat_bias, flat_lo, flat_hi)
    ylo = [
        [
            [flat_ylo[(oc * outH + i) * outW + j] for j in range(outW)]
            for i in range(outH)
        ]
        for oc in range(outC)
    ]
    yhi = [
        [
            [flat_yhi[(oc * outH + i) * outW + j] for j in range(outW)]
            for i in range(outH)
        ]
        for oc in range(outC)
    ]
    return ylo, yhi


def run_ibp() -> dict[str, Any]:
    """Compute the CNN certificate payload consumed by Lean."""
    kernel, bias = seed_conv_params()
    x_lo3, x_hi3 = seed_input_box(0.1)
    # Pre-activation bounds for conv
    z_lo3, z_hi3 = ibp_conv2d_preact(kernel, bias, x_lo3, x_hi3)
    # ReLU is monotone, so its interval image is obtained from the two endpoints.
    h_lo = [
        max(0.0, z_lo3[oc][i][j])
        for oc in range(outC)
        for i in range(outH)
        for j in range(outW)
    ]
    h_hi = [
        max(0.0, z_hi3[oc][i][j])
        for oc in range(outC)
        for i in range(outH)
        for j in range(outW)
    ]
    x_lo = [x_lo3[0][i][j] for i in range(inH) for j in range(inW)]
    x_hi = [x_hi3[0][i][j] for i in range(inH) for j in range(inW)]
    # Linear head 2 x nConv with W[i,j] = 2 + (i + j), b[i] = i
    Whead = [[float(2 + (i + j)) for j in range(nConv)] for i in range(nOut)]
    bhead = [float(i) for i in range(nOut)]
    y_lo, y_hi = affine_interval(Whead, bhead, h_lo, h_hi)
    return {
        "graph": "cnn_graph_workflow_v1",
        "input_box": {"id": 0, "dim": nIn, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 3, "dim": nOut, "lo": y_lo, "hi": y_hi},
    }


def main():
    """Write the CNN certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/cnn_cert.json"
    out = write_json(out_path, cert)
    print(f"Wrote certificate to {out}")


if __name__ == "__main__":
    main()
