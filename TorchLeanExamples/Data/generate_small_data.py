#!/usr/bin/env python3
"""Generate the small local datasets used by TorchLean data examples.

The files are deterministic. They are written next to this script unless `--out-dir` says
otherwise, and the data tutorials read them from there by default. Git ignores the files, so this
script is their source.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def make_regression() -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    features = np.stack((first, second), axis=-1).reshape(-1, 2)
    # The target is computed in binary64 from the float32 features and rounded to float32 once.
    x1, x2 = features.astype(np.float64).T
    target = 0.7 * x1 - 0.4 * x2 + 0.5 * x1 * x2
    return features, target.astype(np.float32)[:, None]


def write_regression(out_dir: Path) -> None:
    X, y = make_regression()
    np.save(out_dir / "small_regression_X.npy", X)
    np.save(out_dir / "small_regression_y.npy", y)
    with (out_dir / "small_regression.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "y"])
        for row, target in zip(X, y):
            writer.writerow([float(row[0]), float(row[1]), float(target[0])])
    print(f"wrote {out_dir / 'small_regression.csv'} rows={X.shape[0]}")
    print(f"wrote {out_dir / 'small_regression_X.npy'} shape={X.shape} dtype={X.dtype}")
    print(f"wrote {out_dir / 'small_regression_y.npy'} shape={y.shape} dtype={y.dtype}")


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
    np.save(out_dir / "small_cifar10like_X.npy", X)
    np.save(out_dir / "small_cifar10like_y.npy", y)
    print(f"wrote {out_dir / 'small_cifar10like_X.npy'} shape={X.shape} dtype={X.dtype}")
    print(f"wrote {out_dir / 'small_cifar10like_y.npy'} shape={y.shape} dtype={y.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--regression-only", action="store_true")
    modes.add_argument("--cifar-only", action="store_true")
    args = parser.parse_args()

    if args.n_per_class <= 0:
        parser.error("--n-per-class must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.cifar_only:
        write_regression(args.out_dir)
    if not args.regression_only:
        write_cifar10like(args.out_dir, args.n_per_class, args.seed)


if __name__ == "__main__":
    main()
