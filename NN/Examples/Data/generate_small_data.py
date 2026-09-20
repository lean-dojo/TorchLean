#!/usr/bin/env python3
"""Generate the small local datasets used by TorchLean data examples.

The generated files are deterministic and live next to the examples that consume them. Keeping the
generator in source control makes the artifact provenance clear without treating derived arrays as
hand-written source.
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
    # Preserve the original binary64 arithmetic before the final float32 file conversion.
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


def write_tabular_regression(out_dir: Path) -> None:
    """Write a seven-feature CSV for the MLP and KAN command checks."""
    with (out_dir / "small_tabular_regression.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"x{i}" for i in range(1, 8)] + ["y"])
        rows = np.arange(10)[:, None]
        columns = np.arange(7)[None, :]
        features = ((rows + columns) % 10).astype(np.float32) / np.float32(9)
        targets = features.mean(axis=1, dtype=np.float64).astype(np.float32)
        writer.writerows(np.column_stack((features, targets)).tolist())
    print(f"wrote {out_dir / 'small_tabular_regression.csv'} rows=10")


def write_forecast(out_dir: Path, n_rows: int = 4, seq_len: int = 24) -> None:
    """Write deterministic one-feature time-series windows for recurrent-model checks."""
    base = np.linspace(0.0, 1.0, seq_len + 1, dtype=np.float32)
    offsets = np.arange(n_rows, dtype=np.float32)[:, None] / np.float32(n_rows)
    X = (base[None, :-1] + offsets)[:, :, None]
    y = (base[None, 1:] + offsets)[:, :, None]
    np.save(out_dir / "small_forecast_X.npy", X)
    np.save(out_dir / "small_forecast_y.npy", y)
    print(f"wrote {out_dir / 'small_forecast_X.npy'} shape={X.shape} dtype={X.dtype}")
    print(f"wrote {out_dir / 'small_forecast_y.npy'} shape={y.shape} dtype={y.dtype}")


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


def write_fno1d(out_dir: Path, n_rows: int = 4, grid: int = 32) -> None:
    """Write a deterministic periodic operator-learning fixture for FNO smoke tests."""
    x_grid = np.linspace(0.0, 2.0 * np.pi, grid, endpoint=False, dtype=np.float32)
    phases = np.arange(n_rows, dtype=np.float32)[:, None] / np.float32(n_rows)
    X = np.sin(x_grid[None, :] + phases)
    y = (
        np.float32(0.75) * X
        + np.float32(0.1) * np.sin(np.float32(2.0) * x_grid)
    ).astype(np.float32)
    np.save(out_dir / "small_fno1d_X.npy", X)
    np.save(out_dir / "small_fno1d_y.npy", y)
    print(f"wrote {out_dir / 'small_fno1d_X.npy'} shape={X.shape} dtype={X.dtype}")
    print(f"wrote {out_dir / 'small_fno1d_y.npy'} shape={y.shape} dtype={y.dtype}")


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
        write_tabular_regression(args.out_dir)
        write_forecast(args.out_dir)
    if not args.regression_only:
        write_cifar10like(args.out_dir, args.n_per_class, args.seed)
    if not args.regression_only and not args.cifar_only:
        write_fno1d(args.out_dir)


if __name__ == "__main__":
    main()
