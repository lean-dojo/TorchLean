#!/usr/bin/env python3
"""Generate the small local datasets used by TorchLean data tutorials.

The generated files are intentionally not committed. They are small and deterministic, but keeping
them generated avoids treating example data artifacts as source files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def make_regression() -> tuple[np.ndarray, np.ndarray]:
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    for x1 in np.linspace(-1.0, 1.0, 5, dtype=np.float32):
        for x2 in np.linspace(-1.0, 1.0, 5, dtype=np.float32):
            y = 0.7 * float(x1) - 0.4 * float(x2) + 0.5 * float(x1) * float(x2)
            xs.append([float(x1), float(x2)])
            ys.append([y])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def write_regression(out_dir: Path) -> None:
    X, y = make_regression()
    np.save(out_dir / "toy_regression_X.npy", X)
    np.save(out_dir / "toy_regression_y.npy", y)
    with (out_dir / "toy_regression.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "y"])
        for row, target in zip(X, y):
            writer.writerow([float(row[0]), float(row[1]), float(target[0])])
    print(f"wrote {out_dir / 'toy_regression.csv'} rows={X.shape[0]}")
    print(f"wrote {out_dir / 'toy_regression_X.npy'} shape={X.shape} dtype={X.dtype}")
    print(f"wrote {out_dir / 'toy_regression_y.npy'} shape={y.shape} dtype={y.dtype}")


def make_cifar10like(n_per_class: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes = 10
    n = n_per_class * n_classes

    X = np.zeros((n, 3, 32, 32), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)

    square = 6
    idx = 0
    for k in range(n_classes):
        grid_r = k // 5
        grid_c = k % 5
        base_r = 5 + grid_r * 16
        base_c = 2 + grid_c * 6

        for _ in range(n_per_class):
            dr = int(rng.integers(-1, 2))
            dc = int(rng.integers(-1, 2))
            r0 = int(np.clip(base_r + dr, 0, 32 - square))
            c0 = int(np.clip(base_c + dc, 0, 32 - square))

            img = np.zeros((3, 32, 32), dtype=np.float32)
            ch = k % 3
            img[ch, r0 : r0 + square, c0 : c0 + square] = 1.0
            img += rng.normal(0.0, 0.05, size=img.shape).astype(np.float32)
            X[idx] = np.clip(img, 0.0, 1.0)
            y[idx] = float(k)
            idx += 1

    perm = rng.permutation(n)
    return X[perm], y[perm]


def write_cifar10like(out_dir: Path, n_per_class: int, seed: int) -> None:
    X, y = make_cifar10like(n_per_class=n_per_class, seed=seed)
    np.save(out_dir / "toy_cifar10like_X.npy", X)
    np.save(out_dir / "toy_cifar10like_y.npy", y)
    print(f"wrote {out_dir / 'toy_cifar10like_X.npy'} shape={X.shape} dtype={X.dtype}")
    print(f"wrote {out_dir / 'toy_cifar10like_y.npy'} shape={y.shape} dtype={y.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regression-only", action="store_true")
    parser.add_argument("--cifar-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.cifar_only:
        write_regression(args.out_dir)
    if not args.regression_only:
        write_cifar10like(args.out_dir, args.n_per_class, args.seed)


if __name__ == "__main__":
    main()
