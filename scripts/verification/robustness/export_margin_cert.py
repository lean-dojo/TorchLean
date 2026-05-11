#!/usr/bin/env python3
"""Export per-example logit-margin certificates for robustness evaluation.

This is a lightweight "certificate exporter" in the spirit of certified-accuracy
benchmarks (AutoLiRPA / CROWN-style workflows): compute output logit bounds for an
ℓ∞ input perturbation, then export the bounds so Lean can check the margin
predicate and report certified accuracy.

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
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file for provenance metadata."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def take_examples(examples: list[dict[str, Any]], max_n: int) -> Iterable[dict[str, Any]]:
    """Return at most `max_n` examples, treating non-positive caps as empty."""
    if max_n <= 0:
        return []
    return examples[:max_n]


def main() -> None:
    """Export per-example logit-margin certificates for the digits workflow."""
    parser = argparse.ArgumentParser(
        description="Export per-example logit bounds (margin certificates)",
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

    weights_obj = json.loads(args.weights.read_text())
    dataset_obj = json.loads(args.dataset.read_text())

    W = weights_obj["layers.0.weight"]
    b = weights_obj["layers.0.bias"]
    out_dim = len(W)
    in_dim = len(W[0])

    examples = dataset_obj.get("examples", [])
    rows = []
    nominal_ok = 0
    certified_ok = 0
    total = 0

    for ex in take_examples(examples, args.max):
        x = ex["x"]
        y = int(ex["y"])
        ex_id = int(ex.get("id", total))
        if len(x) != in_dim:
            raise SystemExit(f"bad example {ex_id}: expected input_dim={in_dim}, got {len(x)}")
        if not (0 <= y < out_dim):
            raise SystemExit(f"bad example {ex_id}: label out of range: {y}")

        lo = [clamp01(float(v) - args.eps) for v in x]
        hi = [clamp01(float(v) + args.eps) for v in x]
        logits_lo, logits_hi = ibp_linear(W, b, lo, hi)

        nominal_logits = linear_forward(W, b, [float(v) for v in x])
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
        total += 1

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
    args.out.write_text(json.dumps(cert_obj, indent=2) + "\n")
    print(f"Wrote: {args.out}")
    print(f"Nominal accuracy:   {nominal_ok}/{total}")
    print(f"Certified accuracy: {certified_ok}/{total} (eps={args.eps})")


if __name__ == "__main__":
    main()
