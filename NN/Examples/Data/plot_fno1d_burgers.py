#!/usr/bin/env python3
"""Plot native TorchLean FNO1D Burgers predictions exported as CSV."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("data/real/fno/predictions.csv"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/real/fno/predictions.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=np.float64, ndmin=1)
    columns = set(data.dtype.names or ())
    input_column = "u0" if "u0" in columns else "input"
    required = {"x", input_column, "target", "prediction"}
    if not required.issubset(columns):
        raise ValueError(f"Missing CSV columns: {sorted(required - columns)}")
    if data.size == 0 or any(not np.isfinite(data[name]).all() for name in required):
        raise ValueError("Prediction CSV must contain finite, nonempty numeric columns")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(data["x"], data[input_column], label="u0(x)", linewidth=1.5, alpha=0.75)
    plt.plot(data["x"], data["target"], label="target u(x,T)", linewidth=2.0)
    plt.plot(data["x"], data["prediction"], label="TorchLean FNO prediction", linewidth=2.0, linestyle="--")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("1D Burgers: native TorchLean FNO")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
