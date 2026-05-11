"""Small PyTorch model used by the TorchLean translator widget quickstart.

This is ordinary Python source on disk, not a Lean string literal. Open
`NN/Examples/Quickstart/Widgets.lean` and place the cursor on:

    #pytorch_translate_file "NN/Examples/Quickstart/pytorch_translator_mlp.py"

The widget reads this file, recognizes the common layer stack, and renders a
TorchLean skeleton plus boundary notes. This is an editor preview, not an
import certificate; for checked graph imports, use the torch.export JSON bridge.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
