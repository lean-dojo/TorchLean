#!/usr/bin/env python3
"""Prepare the 1D Burgers FNO dataset for native TorchLean training.

Training happens in `TorchLeanExamples/Models/Operators/FNOBurgers.lean`, so the example stays
native TorchLean while using the standard `burgers_data_R10.mat` file of many FNO tutorials. The
script reads that file from `--mat PATH` or `--download` and writes
`burgers_{train,test}_{X,y}.npy` and `burgers_meta.json` under `--out-dir`.

Dataset convention:
  - field `a`: initial condition u_0(x)
  - field `u`: solution u(x, T)

The original file is large, so the defaults export a small subset on a coarse grid. Increase
`--grid`, `--ntrain`, and `--ntest` for larger training runs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Import the shared dataset helpers without leaving generated files under scripts/.
sys.dont_write_bytecode = True

from scripts.datasets.download_example_data import download
from scripts.datasets.torchlean_data_convert import load_tensor


DEFAULT_URL = (
    "https://huggingface.co/datasets/kks32/sciml-dataset/resolve/main/fno/burgers_data_R10.mat"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=pathlib.Path, default=None, help="Local burgers_data_R10.mat")
    parser.add_argument("--download", action="store_true", help="Download the public .mat dataset")
    parser.add_argument("--url", default=DEFAULT_URL, help="HTTPS dataset URL used with --download")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("data/real/fno"))
    parser.add_argument("--grid", type=int, default=32, help="Output grid resolution")
    parser.add_argument("--ntrain", type=int, default=128, help="Training samples to export")
    parser.add_argument("--ntest", type=int, default=32, help="Test samples to export")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed")
    args = parser.parse_args()
    if args.ntrain <= 0 or args.ntest <= 0:
        parser.error("--ntrain and --ntest must be positive")
    if args.grid <= 1:
        parser.error("--grid must be greater than one")
    return args


def load_mat(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the Burgers fields `a` and `u` as finite float32 2D arrays of one shape."""
    if path.suffix.lower() != ".mat":
        raise SystemExit(f"Expected a .mat file, got {path}")
    x = np.asarray(load_tensor(path, "a"), dtype=np.float32)
    y = np.asarray(load_tensor(path, "u"), dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise SystemExit(f"Expected 2D arrays, got a{x.shape} and u{y.shape}")
    if x.shape != y.shape:
        raise SystemExit(f"Input/target shapes differ: {x.shape} vs {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise SystemExit("Burgers fields must contain finite numeric values")
    return x, y


def resample(a: np.ndarray, grid: int) -> np.ndarray:
    if grid <= 1:
        raise SystemExit("--grid must be > 1")
    if grid > a.shape[1]:
        raise SystemExit(f"--grid={grid} exceeds source resolution {a.shape[1]}")
    # Keep the output on the uniform periodic grid used by the Fourier layer.
    if a.shape[1] % grid == 0:
        return a[:, :: a.shape[1] // grid][:, :grid]
    positions = np.arange(grid, dtype=np.float64) * (a.shape[1] / grid)
    left = np.floor(positions).astype(np.int64)
    right = (left + 1) % a.shape[1]
    fraction = positions - left
    return (a[:, left] * (1.0 - fraction) + a[:, right] * fraction).astype(np.float32)


def main() -> None:
    args = parse_args()
    mat_path = args.mat
    if args.download:
        mat_path = download(args.url, args.out_dir / "burgers_data_R10.mat")
    if mat_path is None:
        raise SystemExit("Pass --mat PATH or --download")

    x, y = load_mat(mat_path)
    source_grid = x.shape[1]
    x = resample(x, args.grid)
    y = resample(y, args.grid)

    needed = args.ntrain + args.ntest
    if needed > x.shape[0]:
        raise SystemExit(f"Need {needed} samples but dataset has {x.shape[0]}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(x.shape[0])[:needed]
    train_idx = perm[: args.ntrain]
    test_idx = perm[args.ntrain :]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "burgers_train_X.npy", x[train_idx])
    np.save(args.out_dir / "burgers_train_y.npy", y[train_idx])
    np.save(args.out_dir / "burgers_test_X.npy", x[test_idx])
    np.save(args.out_dir / "burgers_test_y.npy", y[test_idx])

    meta = {
        "source": str(mat_path),
        "url": args.url if args.download else None,
        "grid": args.grid,
        "ntrain": args.ntrain,
        "ntest": args.ntest,
        "dtype": "float32",
        "seed": args.seed,
        "fields": {"x": "a", "y": "u"},
        "source_grid": source_grid,
        "grid_convention": "periodic endpoint-excluded [0,1)",
        "resampling": "uniform_stride" if source_grid % args.grid == 0 else "periodic_linear",
    }
    (args.out_dir / "burgers_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote TorchLean arrays under {args.out_dir}")
    print("  burgers_train_X.npy, burgers_train_y.npy")
    print("  burgers_test_X.npy,  burgers_test_y.npy")


if __name__ == "__main__":
    main()
