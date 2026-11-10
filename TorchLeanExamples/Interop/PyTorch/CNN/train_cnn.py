"""
Train the compact CNN example and write its state dict for TorchLean.

`torchlean.state` writes the float64 copy of `TestCNN.state_dict()` to `cnn.json`, the words of
`conv1.weight`, `conv1.bias`, `conv2.weight`, `conv2.bias`, `fc.weight` and `fc.bias`.
`TorchLeanExamples/Interop/PyTorch/Roundtrip.lean` reads them by these names into the state of the
same CNN.

Run from the repo root:

  `python3 TorchLeanExamples/Interop/PyTorch/CNN/train_cnn.py`
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


class TestCNN(nn.Module):
    """
    Reference CNN with the named layers conv1, conv2 and fc.

    The layer shapes match the CNN of
      `TorchLeanExamples/Interop/PyTorch/Roundtrip.lean`, which reads the trained state dict.

    Architecture (batch-first, channel-first 8x8 images):
      Conv2d(1 → 2, k=3, stride=1, pad=1) → ReLU → MaxPool2d(k=2, stride=2)
      Conv2d(2 → 2, k=3, stride=1, pad=1) → ReLU → MaxPool2d(k=2, stride=2)
      Flatten → Linear(8 → 2)
    """

    def __init__(self):
        super().__init__()
        # Input: (N, 1, 8, 8)
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)  # -> (N,2,8,8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                # -> (N,2,4,4)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)  # -> (N,2,4,4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                # -> (N,2,2,2)
        self.act = nn.ReLU()
        self.fc = nn.Linear(8, 2)  # flatten (2*2*2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    def get_model_info(self) -> str:
        return "TestCNN(conv1→relu→pool1→conv2→relu→pool2→flatten→fc)"


def init_deterministic_weights(model: TestCNN) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.05, 0.05)


def main() -> None:
    model = TestCNN()
    init_deterministic_weights(model)

    criterion = nn.MSELoss()
    # A small stable optimizer keeps the trained weights reproducible and non-exploding.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # shape: (batch_size=1, channels=1, height=8, width=8)
    x_train = torch.arange(1, 65, dtype=torch.float32).reshape(1, 1, 8, 8)
    y_train = torch.tensor([[1.0, 0.5]], dtype=torch.float32)  # shape: (1, 2)

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

    test_model = TestCNN()
    init_deterministic_weights(test_model)
    original_output = test_model(x_train)
    print(f"Original model output: {original_output}")
    change = torch.norm(model(x_train) - original_output).item()
    print(f"Output change from initialization: {change:.6f}")

    # Write into this example folder so `Roundtrip.lean` reads it by a stable path.
    # The float64 copy holds the trained float32 values exactly.
    out_path = THIS_DIR / "cnn.json"
    state.write(out_path, model.double())
    print(f"Saved the state dict to {out_path}")


if __name__ == "__main__":
    main()
