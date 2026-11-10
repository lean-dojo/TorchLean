"""
Train the MLP reference example and write its state dict for TorchLean.

The training run uses a fixed input/target pair. `torchlean.state` then writes the float64 copy of
`TestMLP.state_dict()` to `mlp.json`, the words of `fc1.weight`, `fc1.bias`, `fc2.weight` and
`fc2.bias`. `TorchLeanExamples/Interop/PyTorch/Roundtrip.lean` reads them by these names into the
state of the same MLP.

Run from the repo root:

  `python3 TorchLeanExamples/Interop/PyTorch/MLP/train_mlp.py`
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parents[3] / "python"))

from torchlean import state  # noqa: E402


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
    """Seed 0 and uniform weights in [-0.1, 0.1], so repeated runs start from the same model."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)


def main() -> None:
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

    test_model = TestMLP()
    init_deterministic_weights(test_model)
    original_output = test_model(x_train)
    print(f"Original (fresh init) output: {original_output}")
    change = abs(model(x_train).item() - original_output.item())
    print(f"Output change from initialization: {change:.6f}")

    # Write into this example folder so `Roundtrip.lean` reads it by a stable path.
    # The float64 copy holds the trained float32 values exactly.
    out_path = THIS_DIR / "mlp.json"
    state.write(out_path, model.double())
    print(f"Saved the state dict to {out_path}")


if __name__ == "__main__":
    main()
