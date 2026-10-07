#!/usr/bin/env python3
"""Export per-example logit-bound reports for robustness evaluation.

This is a focused certificate exporter in the spirit of certified-accuracy
benchmarks (AutoLiRPA / CROWN-style workflows): compute output logit bounds for an
ℓ∞ input perturbation, then export the bounds so Lean can check the margin
predicate and report certified accuracy.

The host arithmetic below uses ordinary binary64 rounding. The exported bounds are producer
assertions, not a directed-rounding proof that they enclose the real network.

This implementation supports a simple linear classifier:
  logits(x) = W x + b

Inputs (JSON):
  - certificate weights: digits_linear_weights.json (layers.0.weight / layers.0.bias)
  - dataset: digits_test.json (examples[].x / examples[].y)

Output (JSON):
  format: robust_margin_cert_v0_1
  examples[].logits_lo / logits_hi: per-class bounds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file for provenance metadata."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def clamp01(x: float) -> float:
    """Clamp one scalar to the normalized image range `[0, 1]`."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def argmax(xs: list[float]) -> int:
    """Return the index of the first maximum value."""
    best_i = 0
    best_v = xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i = i
            best_v = v
    return best_i


def certified_top1(logits_lo: list[float], logits_hi: list[float], label: int) -> bool:
    """Check the robust top-1 margin predicate for one labeled example."""
    lo_y = logits_lo[label]
    max_other = max(logits_hi[j] for j in range(len(logits_hi)) if j != label)
    return lo_y > max_other


def ibp_linear(W: list[list[float]], b: list[float], lo: list[float], hi: list[float]) -> tuple[list[float], list[float]]:
    """Propagate input intervals through a linear classifier."""
    out_dim = len(W)
    in_dim = len(W[0])
    out_lo: list[float] = []
    out_hi: list[float] = []
    for i in range(out_dim):
        lo_i = float(b[i])
        hi_i = float(b[i])
        row = W[i]
        for j in range(in_dim):
            a = float(row[j])
            p = a * float(lo[j])
            q = a * float(hi[j])
            lo_i += min(p, q)
            hi_i += max(p, q)
        out_lo.append(lo_i)
        out_hi.append(hi_i)
    return out_lo, out_hi


def linear_forward(W: list[list[float]], b: list[float], x: list[float]) -> list[float]:
    """Evaluate the nominal linear classifier at one input."""
    out_dim = len(W)
    in_dim = len(W[0])
    out: list[float] = []
    for i in range(out_dim):
        acc = float(b[i])
        row = W[i]
        for j in range(in_dim):
            acc += float(row[j]) * float(x[j])
        out.append(acc)
    return out


def finite_vector(values: list[Any], context: str) -> list[float]:
    """Validate producer inputs without changing the arithmetic evaluation order."""
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"{context}: expected finite numeric entries") from exc
    if not all(math.isfinite(value) for value in result):
        raise SystemExit(f"{context}: expected finite numeric entries")
    return result


def natural(value: Any, context: str) -> int:
    """Read a natural-number identifier without truncating a fractional JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{context}: expected a natural number")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise SystemExit(f"{context}: expected a natural number")
    if value < 0:
        raise SystemExit(f"{context}: expected a natural number")
    return int(value)


def main() -> None:
    """Export per-example logit bounds and margin outcomes for the digits workflow."""
    parser = argparse.ArgumentParser(
        description="Export per-example logit-bound reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("NN/Examples/Verification/Robustness/digits_linear_weights.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("NN/Examples/Verification/Robustness/digits_test.json"),
    )
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--max", type=int, default=360)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("NN/Examples/Verification/Robustness/digits_linear_margin_cert.json"),
    )
    args = parser.parse_args()
    if not math.isfinite(args.eps) or args.eps < 0:
        parser.error("--eps must be finite and nonnegative")

    weights_obj = json.loads(args.weights.read_text())
    dataset_obj = json.loads(args.dataset.read_text())

    W = [finite_vector(row, f"weight row {i}")
         for i, row in enumerate(weights_obj["layers.0.weight"])]
    b = finite_vector(weights_obj["layers.0.bias"], "bias")
    out_dim = len(W)
    if out_dim < 2 or len(b) != out_dim:
        raise SystemExit("expected at least two weight rows and one bias per class")
    in_dim = len(W[0])
    if any(len(row) != in_dim for row in W):
        raise SystemExit("weight rows must have equal lengths")

    examples = dataset_obj.get("examples", [])
    rows = []
    nominal_ok = 0
    certified_ok = 0

    for index, ex in enumerate(examples[:args.max] if args.max > 0 else []):
        x = finite_vector(ex["x"], f"example {index}")
        y = natural(ex["y"], f"example {index} label")
        ex_id = natural(ex.get("id", index), f"example {index} id")
        if len(x) != in_dim:
            raise SystemExit(f"bad example {ex_id}: expected input_dim={in_dim}, got {len(x)}")
        if not (0 <= y < out_dim):
            raise SystemExit(f"bad example {ex_id}: label out of range: {y}")
        if any(not 0.0 <= value <= 1.0 for value in x):
            raise SystemExit(f"bad example {ex_id}: normalized inputs must lie in [0,1]")

        lo = [clamp01(v - args.eps) for v in x]
        hi = [clamp01(v + args.eps) for v in x]
        logits_lo, logits_hi = ibp_linear(W, b, lo, hi)
        finite_vector(logits_lo + logits_hi, f"example {ex_id} propagated bounds")

        nominal_logits = linear_forward(W, b, x)
        finite_vector(nominal_logits, f"example {ex_id} nominal logits")
        pred = argmax(nominal_logits)
        if pred == y:
            nominal_ok += 1
        cert = certified_top1(logits_lo, logits_hi, y)
        if cert:
            certified_ok += 1

        rows.append(
            {
                "id": ex_id,
                "label": y,
                "pred": int(pred),
                "logits_lo": logits_lo,
                "logits_hi": logits_hi,
                "certified": bool(cert),
            }
        )

    total = len(rows)
    cert_obj: dict[str, Any] = {
        "format": "robust_margin_cert_v0_1",
        "norm": "linf",
        "method": "ibp_linear",
        "eps": float(args.eps),
        "input_dim": int(in_dim),
        "num_classes": int(out_dim),
        "meta": {
            "weights_path": str(args.weights),
            "dataset_path": str(args.dataset),
            "weights_sha256": sha256_file(args.weights),
            "dataset_sha256": sha256_file(args.dataset),
        },
        "summary": {
            "examples": int(total),
            "nominal_ok": int(nominal_ok),
            "certified_ok": int(certified_ok),
        },
        "examples": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cert_obj, indent=2, allow_nan=False) + "\n")
    print(f"Wrote: {args.out}")
    print(f"Nominal accuracy:   {nominal_ok}/{total}")
    print(f"Certified accuracy: {certified_ok}/{total} (eps={args.eps})")


if __name__ == "__main__":
    main()
