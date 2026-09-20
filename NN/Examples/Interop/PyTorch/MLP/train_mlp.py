"""
Train the MLP reference example and export its weights to JSON for TorchLean.

The training run uses a fixed input/target pair and writes a JSON object that matches the importer in:

  `NN/Runtime/PyTorch/Import/MLP.lean`.

Run from the repo root:

  `python3 NN/Examples/Interop/PyTorch/MLP/train_mlp.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent


class TestMLP(nn.Module):
    """Reference MLP: Linear(2→3) → ReLU → Linear(3→1)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def get_model_info(self) -> str:
        return "TestMLP(2→3→1, ReLU)"


def init_deterministic_weights(model: TestMLP) -> None:
    """Deterministic init so repeated runs are comparable."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)

def save_mlp_to_json(model: TestMLP, json_path: str):
    """Save MLP weights to JSON.

    Notes:
    - The TorchLean importer supports both named-module keys (`fc1.weight`/`fc2.bias`)
      and sequential-module keys (`layers.0.weight`/`layers.2.bias`).
    - We emit the sequential-style keys to keep the file format stable across
      different PyTorch module naming conventions.
    """
    state_dict = model.state_dict()

    # Convert PyTorch state dict to the TorchLean import key format (layers.X.weight/bias)
    new_format = {}
    new_format['layers.0.weight'] = state_dict['fc1.weight'].tolist()
    new_format['layers.0.bias'] = state_dict['fc1.bias'].tolist()
    new_format['layers.2.weight'] = state_dict['fc2.weight'].tolist()
    new_format['layers.2.bias'] = state_dict['fc2.bias'].tolist()

    payload = {
        # `Import.PyTorch.loadWeights?` accepts `{...}` or `{ "params": {...} }`.
        "params": new_format,
        "meta": {
            "format": "TorchLean.MLP",
            "pytorch_keys": "layers.*",
            "dtype": "float32",
        },
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

def main():
    model = TestMLP()
    init_deterministic_weights(model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # shape: (batch_size, input_dim=2)
    x_train = torch.tensor([[0.5, 0.8]], dtype=torch.float32)  # shape: (1, 2)
    y_train = torch.tensor([[1.0]], dtype=torch.float32)        # shape: (1, 1)

    print(f"Model info: {model.get_model_info()}")
    print(f"Initial output: {model(x_train)}")

    print("\nStarting training...")
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    print(f"Final output: {model(x_train)}")

    # Write directly into this example folder so `Roundtrip.lean` can import it by a stable path.
    out_path = THIS_DIR / "mlp.json"
    save_mlp_to_json(model, str(out_path))
    print(f"Saved trained weights to {out_path}")

    test_model = TestMLP()
    init_deterministic_weights(test_model)
    original_output = test_model(x_train)
    print(f"Original (fresh init) output: {original_output}")
    print(f"Output change from initialization: {abs(model(x_train).item() - original_output.item()):.6f}")

if __name__ == "__main__":
    main()
