"""
Create a Transformer encoder reference in PyTorch and write its state dict for TorchLean.

`torchlean.state` writes the float64 copy of `TinyTransformerEncoder.state_dict()` to
`transformer_encoder.json`, twelve entries named `layers.0.mha.q_weight` through
`layers.0.norm2.bias`. `TorchLeanExamples/Interop/PyTorch/Roundtrip.lean` reads them by these
names into the state of `nn.transformerBlock`.

Run from the repo root:

  `python3 TorchLeanExamples/Interop/PyTorch/Transformer/export_transformer.py`

Design note:
The attention projections are bias-free `(input, output)` matrices applied as `x @ weight`, the
layout and computation of TorchLean's executable attention layer. Each one is the transpose of a
fresh `nn.Linear(embed_dim, embed_dim, bias=False)` weight, so it has `nn.Linear`'s
initialization. The feed-forward layers and norms are ordinary `nn.Linear` and `nn.LayerNorm`
modules, whose weights both sides store in PyTorch's `(output, input)` layout.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parents[3] / "python"))

from torchlean import state  # noqa: E402


def projection(embed_dim: int) -> nn.Parameter:
    """An `(input, output)` projection with `nn.Linear`'s initialization, for `x @ weight`."""
    weight = nn.Linear(embed_dim, embed_dim, bias=False).weight.detach()
    return nn.Parameter(weight.t().contiguous())


class MultiHeadAttention(nn.Module):
    """Small multi-head attention block with separate query, key, value and output projections.

    The projections `q_weight`, `k_weight`, `v_weight` and `out_weight` are bias-free
    `(input, output)` matrices, as in TorchLean's runtime.
    """

    def __init__(self, embed_dim: int = 2, num_heads: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.q_weight = projection(embed_dim)
        self.k_weight = projection(embed_dim)
        self.v_weight = projection(embed_dim)
        self.out_weight = projection(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        B, S, E = x.shape
        q = (x @ self.q_weight).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = (x @ self.k_weight).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = (x @ self.v_weight).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, E)
        return out @ self.out_weight


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = 2, hidden_dim: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class TinyTransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int = 2, num_heads: int = 1, hidden_dim: int = 2,
                 num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mha": MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads),
                        "norm1": nn.LayerNorm(embed_dim),
                        "ffn": FeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim),
                        "norm2": nn.LayerNorm(embed_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            attn_out = layer["mha"](x, mask)
            x = layer["norm1"](x + attn_out)
            ffn_out = layer["ffn"](x)
            x = layer["norm2"](x + ffn_out)
        return x


def main():
    torch.manual_seed(0)
    model = TinyTransformerEncoder(embed_dim=2, num_heads=1, hidden_dim=2, num_layers=1)
    # Fixed input to match Lean (seqLen=1, embedDim=2), all elements 1.5
    x = torch.full((1, 1, 2), 1.5, dtype=torch.float32)
    y = model(x)
    print('Input:', x)
    print('Output shape:', y.shape)
    print('Output:', y)

    # The float64 copy holds the float32 parameters exactly.
    out_path = THIS_DIR / "transformer_encoder.json"
    state.write(out_path, model.double())
    print(f"Saved the Transformer encoder state dict to {out_path}")


if __name__ == '__main__':
    main()
