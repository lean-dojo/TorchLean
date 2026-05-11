#!/usr/bin/env python3
"""Prepare the 1D Burgers FNO dataset for native TorchLean training.

This script intentionally does **data preparation only**.  Training happens in
`NN/Examples/Models/Operators/Fno1dBurgers.lean`, so the example stays native TorchLean while still
using the standard `burgers_data_R10.mat` file used by many FNO tutorials.

Dataset convention:
  - field `a`: initial condition u_0(x)
  - field `u`: solution u(x, T)

The original file is large, so the defaults export a quick-check subset on a coarse grid. Increase
`--grid`, `--ntrain`, and `--ntest` when you want a larger training run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request

import numpy as np


DEFAULT_URL = (
    "https://huggingface.co/datasets/kks32/sciml-dataset/resolve/main/fno/burgers_data_R10.mat"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=pathlib.Path, default=None, help="Existing burgers_data_R10.mat")
    parser.add_argument("--download", action="store_true", help="Download the public .mat dataset")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dataset URL used with --download")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("data/real/fno"))
    parser.add_argument("--grid", type=int, default=32, help="Output grid resolution")
    parser.add_argument("--ntrain", type=int, default=128, help="Training samples to export")
    parser.add_argument("--ntest", type=int, default=32, help="Test samples to export")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed")
    return parser.parse_args()


def download_if_needed(url: str, dst: pathlib.Path) -> pathlib.Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    print(f"Downloading {url}")
    print(f"  -> {dst}")
    urllib.request.urlretrieve(url, dst)
    return dst


def load_mat(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise SystemExit(
            "scipy is required to read .mat files. Install with: python3 -m pip install scipy"
        ) from exc

    mat = loadmat(path)
    x_key = next((k for k in ("a", "input", "x", "IC") if k in mat), None)
    y_key = next((k for k in ("u", "output", "y", "solution", "usol") if k in mat), None)
    if x_key is None or y_key is None:
        public_keys = sorted(k for k in mat.keys() if not k.startswith("__"))
        raise SystemExit(f"Could not find Burgers fields `a` and `u`; keys were: {public_keys}")

    x = np.asarray(mat[x_key], dtype=np.float32)
    y = np.asarray(mat[y_key], dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise SystemExit(f"Expected 2D arrays, got {x_key}{x.shape} and {y_key}{y.shape}")
    if x.shape != y.shape:
        raise SystemExit(f"Input/target shapes differ: {x.shape} vs {y.shape}")
    return x, y


def subsample_to_grid(a: np.ndarray, grid: int) -> np.ndarray:
    if grid <= 1:
        raise SystemExit("--grid must be > 1")
    if grid > a.shape[1]:
        raise SystemExit(f"--grid={grid} exceeds source resolution {a.shape[1]}")
    # Match the common FNO tutorial pattern: uniform stride when possible, otherwise nearest picks.
    if a.shape[1] % grid == 0:
        return a[:, :: a.shape[1] // grid][:, :grid]
    idx = np.linspace(0, a.shape[1] - 1, grid).round().astype(np.int64)
    return a[:, idx]


def main() -> None:
    args = parse_args()
    mat_path = args.mat
    if args.download:
        mat_path = download_if_needed(args.url, args.out_dir / "burgers_data_R10.mat")
    if mat_path is None:
        raise SystemExit("Pass --mat PATH or --download")

    x, y = load_mat(mat_path)
    x = subsample_to_grid(x, args.grid)
    y = subsample_to_grid(y, args.grid)

    needed = args.ntrain + args.ntest
    if needed > x.shape[0]:
        raise SystemExit(f"Need {needed} samples but dataset has {x.shape[0]}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(x.shape[0])[:needed]
    train_idx = perm[: args.ntrain]
    test_idx = perm[args.ntrain :]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "burgers_train_X.npy", x[train_idx].astype(np.float32))
    np.save(args.out_dir / "burgers_train_y.npy", y[train_idx].astype(np.float32))
    np.save(args.out_dir / "burgers_test_X.npy", x[test_idx].astype(np.float32))
    np.save(args.out_dir / "burgers_test_y.npy", y[test_idx].astype(np.float32))

    meta = {
        "source": str(mat_path),
        "url": args.url if args.download else None,
        "grid": args.grid,
        "ntrain": args.ntrain,
        "ntest": args.ntest,
        "dtype": "float32",
        "fields": {"x": "a", "y": "u"},
    }
    (args.out_dir / "burgers_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote TorchLean arrays under {args.out_dir}")
    print("  burgers_train_X.npy, burgers_train_y.npy")
    print("  burgers_test_X.npy,  burgers_test_y.npy")


if __name__ == "__main__":
    main()
