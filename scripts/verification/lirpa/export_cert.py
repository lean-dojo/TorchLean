#!/usr/bin/env python3
"""Regenerate the bundled LiRPA certificates without changing their rounding schedules."""

import argparse
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


def certificate(graph, node, inputs, outputs) -> dict[str, Any]:
    """Package source and result endpoints in the format consumed by Lean."""
    x_lo, x_hi = inputs
    y_lo, y_hi = outputs
    return {
        "graph": graph,
        "input_box": {"id": 0, "dim": len(x_lo), "lo": x_lo, "hi": x_hi},
        "result": {"node_id": node, "dim": len(y_lo), "lo": y_lo, "hi": y_hi},
    }


def mlp() -> dict[str, Any]:
    """Bound the bundled input → linear → ReLU → linear graph."""
    n_in, n_hidden, n_out = 3, 4, 2
    hidden_weight = [[float(1 + (i + j)) for j in range(n_in)] for i in range(n_hidden)]
    hidden_bias = [float(i + 1) for i in range(n_hidden)]
    output_weight = [[float(2 + (i + j)) for j in range(n_hidden)] for i in range(n_out)]
    output_bias = [float(i) for i in range(n_out)]
    inputs = centered_box([float(i + 1) for i in range(n_in)], 1.0)
    hidden = relu_interval(*affine_interval(hidden_weight, hidden_bias, *inputs))
    outputs = affine_interval(output_weight, output_bias, *hidden)
    return certificate("mlp_graph_workflow_v1", 3, inputs, outputs)


def image_box(eps: float = 0.1):
    """Enclose the bundled single-channel 4×4 image centered at all ones."""
    lo, hi = centered_box([1.0] * 16, eps)
    return (
        [[[lo[i * 4 + j] for j in range(4)] for i in range(4)]],
        [[[hi[i * 4 + j] for j in range(4)] for i in range(4)]],
    )


def cnn() -> dict[str, Any]:
    """Bound the bundled convolution → ReLU → linear graph."""
    in_channels, in_height, in_width = 1, 4, 4
    out_channels, kernel_height, kernel_width, stride, padding = 1, 3, 3, 1, 0
    out_height = (in_height + 2 * padding - kernel_height) // stride + 1
    out_width = (in_width + 2 * padding - kernel_width) // stride + 1
    n_in = in_channels * in_height * in_width
    n_conv = out_channels * out_height * out_width
    n_out = 2
    kernel = [
        [
            [[float(1 + (i + j)) for j in range(kernel_width)] for i in range(kernel_height)]
            for _ in range(in_channels)
        ]
        for _ in range(out_channels)
    ]
    bias = [0.0 for _ in range(out_channels)]
    lo, hi = image_box(0.1)

    # Lean checks a dense affine node. Its host-Float operations widen even zero products,
    # so skipping structural zeros would change the certificate's numerical endpoints.
    matrix = [[0.0 for _ in range(n_in)] for _ in range(n_conv)]
    for oc in range(out_channels):
        for i in range(out_height):
            for j in range(out_width):
                for ic in range(in_channels):
                    for di in range(kernel_height):
                        for dj in range(kernel_width):
                            pi = i * stride + di
                            pj = j * stride + dj
                            if pi >= padding and pj >= padding:
                                ii, jj = pi - padding, pj - padding
                                if 0 <= ii < in_height and 0 <= jj < in_width:
                                    row = (oc * out_height + i) * out_width + j
                                    col = (ic * in_height + ii) * in_width + jj
                                    matrix[row][col] = kernel[oc][ic][di][dj]
    flat_lo = [lo[c][i][j] for c in range(in_channels)
               for i in range(in_height) for j in range(in_width)]
    flat_hi = [hi[c][i][j] for c in range(in_channels)
               for i in range(in_height) for j in range(in_width)]
    flat_bias = [bias[c] for c in range(out_channels) for _ in range(out_height * out_width)]
    hidden = relu_interval(*affine_interval(matrix, flat_bias, flat_lo, flat_hi))
    weight = [[float(2 + (i + j)) for j in range(n_conv)] for i in range(n_out)]
    outputs = affine_interval(weight, [float(i) for i in range(n_out)], *hidden)
    return certificate("cnn_graph_workflow_v1", 3, (flat_lo, flat_hi), outputs)


def attention() -> dict[str, Any]:
    """Bound the bundled score projection → softmax → value projection graph."""
    n_in, n_scores, n_out = 4, 5, 3
    weight = [[float(2 + (i + j)) for j in range(n_scores)] for i in range(n_out)]
    inputs = centered_box([float(i + 1) for i in range(n_in)], 0.5)
    # The conservative enclosure depends only on row width, not the score bounds.
    outputs = affine_interval(weight, [0.0] * n_out, *softmax_range_interval(n_scores))
    return certificate("attention_softmax_workflow_v1", 3, inputs, outputs)


def gru() -> dict[str, Any]:
    """Bound the bundled sigmoid and tanh branches followed by elementwise multiplication."""
    n = 3
    inputs = centered_box([float(i + 1) for i in range(n)], 0.5)
    x_lo, x_hi = sigmoid_range_interval(n)
    y_lo, y_hi = tanh_range_interval(n)
    lo, hi = [], []
    for lx, ux, ly, uy in zip(x_lo, x_hi, y_lo, y_hi):
        lo.append(min(mul_down(lx, ly), mul_down(lx, uy),
                      mul_down(ux, ly), mul_down(ux, uy)))
        hi.append(max(mul_up(lx, ly), mul_up(lx, uy),
                      mul_up(ux, ly), mul_up(ux, uy)))
    return certificate("gru_gate_workflow_v1", 5, inputs, (lo, hi))


def transformer() -> dict[str, Any]:
    """Bound the bundled transformer-like graph using its final LayerNorm range rule."""
    n = 4
    inputs = centered_box([float(i + 1) for i in range(n)], 0.5)
    # Earlier affine/residual bounds do not affect this conservative final enclosure.
    return certificate("transformer_encoder_workflow_v1", 10,
                       inputs, layernorm_range_interval(n))


EXPORTS = {
    "mlp": (mlp, "mlp_cert.json"),
    "cnn": (cnn, "cnn_cert.json"),
    "attention": (attention, "attention_softmax_cert.json"),
    "gru": (gru, "gru_gate_cert.json"),
    "transformer": (transformer, "transformer_encoder_cert.json"),
}


def main() -> None:
    """Export one fixture, or explicitly regenerate all five with ``all``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=(*EXPORTS, "all"))
    parser.add_argument("--out-dir", type=Path, default=Path("NN/Examples/Verification/LiRPA"))
    args = parser.parse_args()
    for model in EXPORTS if args.model == "all" else (args.model,):
        produce, filename = EXPORTS[model]
        out = write_json(args.out_dir / filename, produce())
        print(f"Wrote certificate to {out}")


if __name__ == "__main__":
    main()
