"""
Train the compact CNN example and export its weights to JSON for TorchLean.

The JSON schema matches the importer in:

  `NN/Examples/Interop/PyTorch/CNN/Import.lean` (`Import.CNNPyTorch.loadCnnStateDict`),

using keys:
  `conv1.weight`, `conv1.bias`, `conv2.weight`, `conv2.bias`, `fc.weight`, `fc.bias`.

Run from the repo root:

  `python3 NN/Examples/Interop/PyTorch/CNN/train_cnn.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent


class TestCNN(nn.Module):
    """
    Compact CNN with named layers matching TorchLean import keys:
      conv1, conv2, fc

    This model is intentionally aligned with the CNN shape used by:
      `NN/Examples/Interop/PyTorch/Roundtrip.lean` (the Lean side import example)

    Architecture (batch-first, channel-first images):
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
        return "TestCNN(conv1→relu→conv2→relu→flatten→fc)"


def init_deterministic_weights(model: TestCNN) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.05, 0.05)

def save_cnn_to_json(model: TestCNN, json_path: str):
    """Save CNN weights to JSON in the key format expected by TorchLean import."""
    state_dict = model.state_dict()
    
    # Convert PyTorch state dict to the TorchLean CNN import key format (named layers)
    new_format = {}
    new_format['conv1.weight'] = state_dict['conv1.weight'].tolist()
    new_format['conv1.bias'] = state_dict['conv1.bias'].tolist()
    new_format['conv2.weight'] = state_dict['conv2.weight'].tolist()
    new_format['conv2.bias'] = state_dict['conv2.bias'].tolist()
    new_format['fc.weight'] = state_dict['fc.weight'].tolist()
    new_format['fc.bias'] = state_dict['fc.bias'].tolist()
    
    payload = {
        "params": new_format,
        "meta": {
            "format": "TorchLean.CNN2",
            "dtype": "float32",
        },
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

def main():
    # 1. Instantiate model
    model = TestCNN()
    init_deterministic_weights(model)
    
    # 2. Loss & optimizer
    criterion = nn.MSELoss()
    # A small stable optimizer keeps exported weights reproducible and non-exploding.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Training data (fixed Lean input)
    # shape: (batch_size=1, channels=1, height=8, width=8)
    x_train = torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
            [33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
            [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
            [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
            [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]
        ]
    ], dtype=torch.float32).unsqueeze(0)
    # target output
    y_train = torch.tensor([[1.0, 0.5]], dtype=torch.float32)  # shape: (1, 2)
    
    print(f"Model info: {model.get_model_info()}")
    print(f"Initial output: {model(x_train)}")
    
    # 4. Train loop
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
    
    # 5. Save trained weights to JSON in the TorchLean CNN import key format
    # Write directly into this example folder so `Roundtrip.lean` can import it by a stable path.
    out_path = THIS_DIR / "cnn.json"
    save_cnn_to_json(model, str(out_path))
    print(f"Saved trained weights to {out_path}")
    
    # 6. Print a fresh-init baseline so the exported file has an easy sanity check.
    test_model = TestCNN()
    init_deterministic_weights(test_model)
    original_output = test_model(x_train)
    print(f"Original model output: {original_output}")
    print(f"Training improved output by: {torch.norm(model(x_train) - original_output).item():.6f}")

if __name__ == "__main__":
    main()
