#!/usr/bin/env python3
"""Train a linear classifier on sklearn's digits (8x8) and export JSON weights for Lean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class DigitsLinear(nn.Module):
    """Single-layer classifier with stable state-dict keys for Lean import."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Keep keys stable and easy to parse on the Lean side.
        self.layers = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate logits for a batch of flattened digit images."""
        return self.layers(x)


def set_seeds(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducible exported artifacts."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute classification accuracy on a tensor batch."""
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        return float((pred == y).float().mean().item())


def export_weights_json(model: DigitsLinear, in_dim: int, out_dim: int, normalize_div: float) -> dict[str, Any]:
  """Serialize learned weights in the schema consumed by Lean examples."""
  state = model.state_dict()
  return {
    "format": "digits_linear_weights_v0_1",
    "in_dim": in_dim,
    "out_dim": out_dim,
    "normalize_div": normalize_div,
    "layers.0.weight": state["layers.0.weight"].tolist(),
    "layers.0.bias": state["layers.0.bias"].tolist(),
  }


def export_dataset_json(x: np.ndarray, y: np.ndarray, in_dim: int, out_dim: int, normalize_div: float) -> dict[str, Any]:
    """Serialize a normalized test split for certificate generation."""
    examples = []
    for i in range(x.shape[0]):
        examples.append({"id": int(i), "x": x[i].astype(np.float32).tolist(), "y": int(y[i])})
    return {
        "format": "digits_dataset_v0_1",
        "split": "test",
        "input_dim": in_dim,
        "num_classes": out_dim,
        "normalize_div": normalize_div,
        "examples": examples,
    }


def main() -> None:
    """Train the digits classifier and export weights plus a test split."""
    parser = argparse.ArgumentParser(
        description="Train digits linear classifier + export Lean weights JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-test", type=int, default=360, help="Cap exported test examples")
    parser.add_argument(
        "--out-weights",
        type=Path,
        default=Path("NN/Examples/Verification/Robustness/digits_linear_weights.json"),
    )
    parser.add_argument(
        "--out-dataset",
        type=Path,
        default=Path("NN/Examples/Verification/Robustness/digits_test.json"),
    )
    args = parser.parse_args()

    set_seeds(args.seed)

    digits = load_digits()
    x = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)
    # sklearn digits pixels are integers in `[0, 16]`; divide once so Lean and
    # Python use the same normalized input convention.
    normalize_div = 16.0
    x = x / normalize_div

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    if args.max_test > 0:
        x_test = x_test[: args.max_test]
        y_test = y_test[: args.max_test]

    in_dim = x.shape[1]
    out_dim = 10
    model = DigitsLinear(in_dim, out_dim)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    xtr = torch.tensor(x_train)
    ytr = torch.tensor(y_train)
    for epoch in range(args.epochs):
        perm = torch.randperm(xtr.shape[0])
        for i in range(0, xtr.shape[0], args.batch):
            idx = perm[i : i + args.batch]
            xb = xtr[idx]
            yb = ytr[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            acc = accuracy(model, torch.tensor(x_test), torch.tensor(y_test))
            print(f"epoch {epoch:03d}: test_acc={acc:.4f}")

    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    args.out_dataset.parent.mkdir(parents=True, exist_ok=True)
    weights_json = export_weights_json(model, in_dim=in_dim, out_dim=out_dim, normalize_div=normalize_div)
    dataset_json = export_dataset_json(x_test, y_test, in_dim=in_dim, out_dim=out_dim, normalize_div=normalize_div)
    args.out_weights.write_text(json.dumps(weights_json, indent=2) + "\n")
    args.out_dataset.write_text(json.dumps(dataset_json, indent=2) + "\n")
    print(f"Wrote: {args.out_weights}")
    print(f"Wrote: {args.out_dataset}")


if __name__ == "__main__":
    main()
