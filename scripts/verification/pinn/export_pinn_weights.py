#!/usr/bin/env python3
"""
Export weights for a PINN-style MLP compatible with TorchLean's PINN CLI importer.

Architecture (Sequential):
  layers = [
    nn.Linear(in_dim, 16), nn.Tanh(),
    nn.Linear(16, 16),     nn.Tanh(),
    nn.Linear(16, 1)
  ]

This script can:
  - initialize a fresh model and save its weights as JSON, or
  - load a PyTorch checkpoint (.pt with state_dict) and export the same JSON format.

JSON format keys:
  - "layers.0.weight" : 16 × in_dim
  - "layers.0.bias"   : 16
  - "layers.2.weight" : 16 × 16
  - "layers.2.bias"   : 16
  - "layers.4.weight" : 1 × 16
  - "layers.4.bias"   : 1

Usage examples:
  # Fresh random weights for 1D
  python3 scripts/verification/pinn/export_pinn_weights.py --in-dim 1 --out weights_1d.json

  # Load from checkpoint and export
  python3 scripts/verification/pinn/export_pinn_weights.py --in-dim 2 --ckpt my_pinn.pt --out weights_2d.json

Then verify in Lean (1D example):
  lake exe verify -- pinn-cli --weights=weights_1d.json "u_xx + u" 0.25 0.05
"""

import argparse
import json
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise SystemExit("PyTorch is required: pip install torch")


def build_model(in_dim: int):
    """Build the sequential PINN MLP architecture expected by the Lean importer."""
    layers = [
        nn.Linear(in_dim, 16), nn.Tanh(),
        nn.Linear(16, 16),     nn.Tanh(),
        nn.Linear(16, 1),
    ]
    return nn.Sequential(*layers)


def to_json_dict(model: nn.Sequential):
    """Serialize a sequential PINN model into TorchLean's JSON weight schema."""
    sd = model.state_dict()
    def tens(name):
        """Read one state-dict tensor as a nested Python list."""
        return sd[name].detach().cpu().numpy().tolist()
    return {
        "layers.0.weight": tens("0.weight"),
        "layers.0.bias":   tens("0.bias"),
        "layers.2.weight": tens("2.weight"),
        "layers.2.bias":   tens("2.bias"),
        "layers.4.weight": tens("4.weight"),
        "layers.4.bias":   tens("4.bias"),
    }


def main():
    """CLI entry point for exporting fresh or checkpoint-loaded PINN weights."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dim", type=int, choices=[1,2], required=True, help="Input dimension (1 or 2)")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to a PyTorch .pt checkpoint with state_dict")
    ap.add_argument("--out", type=str, required=True, help="Output JSON path")
    args = ap.parse_args()

    model = build_model(args.in_dim)

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        # allow either plain state dict or dict with 'state_dict'
        if isinstance(sd, dict) and all(k in sd for k in ("0.weight","0.bias","2.weight","2.bias","4.weight","4.bias")):
            model.load_state_dict(sd)
        elif isinstance(sd, dict) and "state_dict" in sd:
            model.load_state_dict(sd["state_dict"]) 
        else:
            raise SystemExit("Checkpoint does not contain expected state_dict keys")

    js = to_json_dict(model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(js))
    print(f"Wrote weights JSON to {out}")


if __name__ == "__main__":
    main()
