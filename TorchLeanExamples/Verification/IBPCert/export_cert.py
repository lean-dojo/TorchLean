#!/usr/bin/env python3
"""Regenerate the bundled IBP certificate, rounding outward in the operation order of Lean's IBP."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Import the shared helpers from scripts/verification without writing bytecode under scripts/.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "verification"))
from interval import add_up, affine_interval, round_down  # noqa: E402


def centered_box(center: list[float], eps: float) -> tuple[list[float], list[float]]:
    """Enclose `center ± eps` with the directed endpoints used by Lean's `lInfBall`."""

    return [round_down(x - eps) for x in center], [add_up(x, eps) for x in center]


def relu_interval(lo: list[float], hi: list[float]) -> tuple[list[float], list[float]]:
    """Propagate interval bounds through elementwise ReLU."""

    return [max(0.0, x) for x in lo], [max(0.0, x) for x in hi]


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


def main() -> None:
    """Write `mlp_cert.json` into `--out-dir`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("TorchLeanExamples/Verification/IBPCert"))
    args = parser.parse_args()
    out = write_json(args.out_dir / "mlp_cert.json", mlp())
    print(f"Wrote certificate to {out}")


if __name__ == "__main__":
    main()
