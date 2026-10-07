#!/usr/bin/env python3
"""
Convert the classic Raissi et al. viscous Burgers dataset (MATLAB .mat) into the
JSON schema consumed by `train_pinn.py evolution --dataset-json`.

Source dataset format (as in https://github.com/AdrianDario10/Burgers_Equation1D):
  - x:    (Nx, 1)
  - t:    (Nt, 1)
  - usol: (Nx, Nt)

Output JSON schema:
  {
    "collocation": [ {"x": ..., "t": ...}, ... ],
    "initial":     [ {"x": ..., "t": 0.0, "u": ...}, ... ],
    "boundary":    [ {"x": -1.0, "t": ..., "u": 0.0}, {"x": 1.0, "t": ..., "u": 0.0}, ... ],
    "data":        [ {"x": ..., "t": ..., "u": ...}, ... ]
  }

Notes:
  - For convenience, the exported entries also include `"y": t` aliases so the same JSON can
    be used with Lean-side tools that use (x,y) notation.
  - This script *samples* points by default to keep JSON small; use `--full-grid`
    to emit every grid point as `"data"`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_mat(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the expected Burgers MATLAB arrays."""
    try:
        import scipy.io
    except Exception as exc:  # pragma: no cover
        raise SystemExit("SciPy is required to read .mat files: pip install scipy") from exc

    mat = scipy.io.loadmat(path)
    for k in ("x", "t", "usol"):
        if k not in mat:
            raise SystemExit(f"MAT file missing key '{k}': {path}")
    x = np.asarray(mat["x"]).reshape(-1)
    t = np.asarray(mat["t"]).reshape(-1)
    u = np.asarray(mat["usol"])
    if u.shape != (x.shape[0], t.shape[0]):
        raise SystemExit(f"Unexpected usol shape {u.shape}, expected ({x.shape[0]}, {t.shape[0]})")
    if x.size == 0 or t.size == 0:
        raise SystemExit("MAT coordinates must be nonempty")
    if any(np.iscomplexobj(a) or not np.isfinite(a).all() for a in (x, t, u)):
        raise SystemExit("MAT coordinates and solution must be finite real numbers")
    return x, t, u


def _as_entry(x: float, t: float, u: float | None = None) -> Dict[str, Any]:
    """Build one JSON point entry with the coordinate names expected by the checker."""
    out: Dict[str, Any] = {"x": float(x), "t": float(t), "y": float(t)}
    if u is not None:
        out["u"] = float(u)
    return out


def _sample_indices(rng: random.Random, n: int, k: int) -> List[int]:
    """Sample up to `k` distinct indices from `range(n)`."""
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    return rng.sample(range(n), k)


def main() -> None:
    """Convert a Burgers `.mat` file into the PINN trainer dataset JSON schema."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, required=True, help="Path to burgers_shock.mat")
    ap.add_argument("--out", type=str, required=True, help="Output dataset JSON path")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    ap.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Optional clip: keep only times t <= t_max",
    )
    ap.add_argument("--full-grid", action="store_true", help="Emit every grid point as 'data'")
    ap.add_argument("--max-collocation", type=int, default=2000)
    ap.add_argument("--max-initial", type=int, default=256)
    ap.add_argument("--max-boundary", type=int, default=200)
    ap.add_argument("--max-data", type=int, default=2000)
    args = ap.parse_args()
    if args.t_max is not None and not math.isfinite(args.t_max):
        ap.error("--t-max must be finite")
    if min(args.max_collocation, args.max_initial, args.max_boundary, args.max_data) < 0:
        ap.error("sample counts must be nonnegative")

    x, t, usol = _load_mat(Path(args.mat))

    if args.t_max is not None:
        mask = t <= float(args.t_max)
        if not mask.any():
            raise SystemExit("--t-max removed all points")
        t = t[mask]
        usol = usol[:, mask]

    rng = random.Random(args.seed)

    nx, nt = x.shape[0], t.shape[0]

    # Initial (t = t[0], typically 0.0)
    t0 = float(t[0])
    init_idx = _sample_indices(rng, nx, args.max_initial)
    initial = [_as_entry(float(x[i]), t0, float(usol[i, 0])) for i in init_idx]

    # Boundary (x = x[0] and x = x[-1], typically -1 and +1)
    t_idx = _sample_indices(rng, nt, args.max_boundary)
    boundary: List[Dict[str, Any]] = []
    for j in t_idx:
        boundary.append(_as_entry(float(x[0]), float(t[j]), float(usol[0, j])))
        boundary.append(_as_entry(float(x[-1]), float(t[j]), float(usol[-1, j])))

    # Interior pool (exclude x-boundary and initial time column by default)
    interior_i = range(1, nx - 1) if nx >= 3 else range(nx)
    interior_j = range(1, nt) if nt >= 2 else range(nt)
    pool_size = len(interior_i) * len(interior_j)

    def entry(k: int, *, observed: bool) -> Dict[str, Any]:
        # Decode the same row-major pool index without storing every grid pair.
        row, column = divmod(k, len(interior_j))
        i, j = interior_i[row], interior_j[column]
        return _as_entry(float(x[i]), float(t[j]), float(usol[i, j]) if observed else None)

    # Data points (supervised u)
    if args.full_grid:
        data = [_as_entry(float(x[i]), float(t[j]), float(usol[i, j]))
                for i in range(nx) for j in range(nt)]
    else:
        data_idx = _sample_indices(rng, pool_size, args.max_data)
        data = [entry(k, observed=True) for k in data_idx]

    # Collocation points: reuse the same interior pool but omit u
    colloc_idx = _sample_indices(rng, pool_size, args.max_collocation)
    collocation = [entry(k, observed=False) for k in colloc_idx]

    payload = {
        "meta": {
            "source": str(Path(args.mat)),
            "equation": "viscous Burgers (Raissi et al.)",
            "nu": 0.01 / math.pi,
            "note": "Includes y=t aliases for Lean-side (x,y) notation.",
        },
        "collocation": collocation,
        "initial": initial,
        "boundary": boundary,
        "data": data,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(f"Wrote dataset JSON to {out}")


if __name__ == "__main__":
    main()
