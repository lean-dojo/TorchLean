#!/usr/bin/env python3
"""Plot native TorchLean FNO1D Burgers predictions exported as CSV."""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("data/real/fno/predictions.csv"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/real/fno/predictions.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xs: list[float] = []
    u0: list[float] = []
    target: list[float] = []
    pred: list[float] = []
    with args.csv.open() as f:
        for row in csv.DictReader(f):
            xs.append(float(row["x"]))
            u0.append(float(row.get("u0", row["input"])))
            target.append(float(row["target"]))
            pred.append(float(row["prediction"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, u0, label="u0(x)", linewidth=1.5, alpha=0.75)
    plt.plot(xs, target, label="target u(x,T)", linewidth=2.0)
    plt.plot(xs, pred, label="TorchLean FNO prediction", linewidth=2.0, linestyle="--")
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
