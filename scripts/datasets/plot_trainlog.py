#!/usr/bin/env python3
"""Plot TorchLean TrainLog JSON artifacts.

Examples:
  python3 scripts/datasets/plot_trainlog.py data/model_zoo/mlp_trainlog.json
  python3 scripts/datasets/plot_trainlog.py data/model_zoo/*.json --out-dir plots/model_zoo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse TrainLog plotting arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="TrainLog JSON files to plot")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNGs. Defaults to each JSON file's directory.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def plot_one(path: Path, out_dir: Path | None, dpi: int) -> Path:
    """Plot one TrainLog JSON file and return the generated PNG path.

    Expected schema: top-level `steps`, top-level `series`, and for each series
    a `name`, `values`, plus optional `color`. The optional top-level `title`
    becomes the plot title.
    """
    import matplotlib.pyplot as plt

    with path.open("r", encoding="utf-8") as f:
        log = json.load(f)

    steps = log.get("steps", [])
    series = log.get("series", [])
    if not steps or not series:
        raise ValueError(f"{path}: expected nonempty `steps` and `series` arrays")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for item in series:
        name = item.get("name", "metric")
        values = item.get("values", [])
        color = item.get("color", None)
        if len(values) != len(steps):
            raise ValueError(
                f"{path}: series `{name}` has {len(values)} values but {len(steps)} steps"
            )
        ax.plot(steps, values, marker="o", linewidth=2.0, label=name, color=color)

    ax.set_title(log.get("title", path.stem))
    ax.set_xlabel("step")
    ax.set_ylabel("metric")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    dest_dir = out_dir or path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{path.stem}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main() -> None:
    """Plot every requested TrainLog file."""
    args = parse_args()
    for log in args.logs:
        out = plot_one(log, args.out_dir, args.dpi)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
