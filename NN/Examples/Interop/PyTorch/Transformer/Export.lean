/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Spec.Models.Transformer

/-!
# Transformer PyTorch Fixture Export

PyTorch code generator for the small Transformer encoder round-trip fixture.

This file produces a readable Python `nn.Module` implementation that follows the usual PyTorch
structure (MHA + residual + LayerNorm + FFN). In the TorchLean repo we mostly use this as a
round-trip companion: generate a reference implementation, train/tweak in Python if needed, and
optionally export parameters back to Lean via JSON in the importer modules.
-/

@[expose] public section


namespace Export
namespace TransformerPyTorch

open Spec
open Tensor
open Export.PyTorch

/-- Render a small Transformer encoder as a Python `nn.Module` class definition.

This produces readable "reference PyTorch" code (MultiHeadAttention + residual + LayerNorm + FFN),
useful for round-trip demos.
-/
def generateTransformerEncoderPyTorchClass (seqLen embedDim headCount hiddenDim numLayers : Nat)
  (className : String := "TransformerEncoder") : String :=
  joinLines <|
  [generatePyTorchImports, "import math", ""] ++ [
    "class MultiHeadAttention(nn.Module):",
    indent2 s!"def __init__(self, embed_dim={embedDim}, num_heads={headCount}):",
    indent4 "super().__init__()",
    indent4 "self.embed_dim = embed_dim",
    indent4 "self.num_heads = num_heads",
    indent4 "self.head_dim = embed_dim // num_heads",
    indent4 "assert embed_dim % num_heads == 0",
    -- TorchLean's spec uses bias-free attention projections (explicit matrices).
    indent4 "self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indent4 "self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indent4 "self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indent4 "self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indent2 "",
    indent2 "def forward(self, x, mask=None):",
    indent4 "B, S, E = x.shape",
    indent4 "q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)",
    indent4 "k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)",
    indent4 "v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)",
    indent4 "scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)",
    indent4 "if mask is not None:",
    indent6 "scores = scores.masked_fill(mask == 0, float('-inf'))",
    indent4 "attn = torch.softmax(scores, dim=-1)",
    indent4 "out = torch.matmul(attn, v)",
    indent4 "out = out.transpose(1, 2).contiguous().view(B, S, E)",
    indent4 "return self.out_proj(out)",
    "",
    "class FeedForward(nn.Module):",
    indent2 s!"def __init__(self, embed_dim={embedDim}, hidden_dim={hiddenDim}):",
    indent4 "super().__init__()",
    indent4 "self.fc1 = nn.Linear(embed_dim, hidden_dim)",
    indent4 "self.fc2 = nn.Linear(hidden_dim, embed_dim)",
    indent2 "",
    indent2 "def forward(self, x):",
    indent4 "return self.fc2(F.relu(self.fc1(x)))",
    "",
    s!"class {className}(nn.Module):",
    (s!"\"\"\"Transformer Encoder with {numLayers} layers, {headCount} heads, " ++
      s!"embed dim {embedDim}, hidden dim {hiddenDim}\"\"\""),
    indent2 "",
    indent2 s!"def __init__(self):",
    indent4 "super().__init__()",
    indent4 "self.layers = nn.ModuleList([nn.ModuleDict({",
    indent6 s!"'mha': MultiHeadAttention({embedDim}, {headCount}),",
    indent6 s!"'norm1': nn.LayerNorm({embedDim}),",
    indent6 s!"'ffn': FeedForward({embedDim}, {hiddenDim}),",
    indent6 s!"'norm2': nn.LayerNorm({embedDim})",
    indent4 s!"}) for _ in range({numLayers})])",
    indent2 "",
    indent2 "def forward(self, x, mask=None):",
    indent4 "# x: (batch, seq_len, embed_dim)",
    indent4 "for layer in self.layers:",
    indent6 "# Self-attention block",
    indent6 "attn_out = layer['mha'](x, mask)",
    indent6 "x = layer['norm1'](x + attn_out)",
    indent6 "# Feed-forward block",
    indent6 "ffn_out = layer['ffn'](x)",
    indent6 "x = layer['norm2'](x + ffn_out)",
    indent4 "return x",
    indent2 "",
    indent2 "@property",
    indent2 "def input_shape(self):",
    indent4 s!"return ({seqLen}, {embedDim})",
    indent4 "",
    indent2 "@property",
    indent2 "def output_shape(self):",
    indent4 s!"return ({seqLen}, {embedDim})",
    indent4 "",
    indent2 "@property",
    indent2 "def layer_count(self):",
    indent4 s!"return {numLayers}",
    indent4 "",
    indent2 "@property",
    indent2 "def operation_types(self):",
    indent4
      "return ['MultiHeadAttention', 'LayerNorm', 'FeedForward', 'LayerNorm'] * self.layer_count",
    indent4 ""
  ] ++
    generateGetModelInfoMethodLines className

/--
Generate a single-layer Transformer encoder module with an embedded `state_dict` initializer.

This is meant for round-trip demos where parameters are loaded from TorchLean tensors.

Important convention:
`NN/Spec` transformer weights are stored in the mathematical `(in, out)` orientation because they
are applied as `X * W`. PyTorch stores linear weights as `(out, in)` and applies them as
`X @ W.T + b`. So for all matrix-valued weights we print the transpose when populating the
PyTorch `state_dict`.
-/
def generateTransformerEncoderWithWeights (seqLen embedDim headCount hiddenDim : Nat)
  (Wq : Tensor Float (.dim embedDim (.dim embedDim .scalar)))
  (Wk : Tensor Float (.dim embedDim (.dim embedDim .scalar)))
  (Wv : Tensor Float (.dim embedDim (.dim embedDim .scalar)))
  (Wo : Tensor Float (.dim embedDim (.dim embedDim .scalar)))
  (W1 : Tensor Float (.dim embedDim (.dim hiddenDim .scalar)))
  (W2 : Tensor Float (.dim hiddenDim (.dim embedDim .scalar)))
  (b1 : Tensor Float (.dim hiddenDim .scalar))
  (b2 : Tensor Float (.dim embedDim .scalar))
  (norm1_gamma : Tensor Float (.dim embedDim .scalar))
  (norm1_beta : Tensor Float (.dim embedDim .scalar))
  (norm2_gamma : Tensor Float (.dim embedDim .scalar))
  (norm2_beta : Tensor Float (.dim embedDim .scalar))
  (className : String := "TransformerEncoder") : String :=
  joinLines [
    generateTransformerEncoderPyTorchClass seqLen embedDim headCount hiddenDim 1 className,
    "",
    "# Weight initialization helpers",
    "def get_transformer_state_dict():",
    indent2 "state_dict = {}",
    indent2 s!"state_dict['layers.0.mha.q_proj.weight'] = torch.tensor({tensor2DToPyT Wq})",
    indent2 s!"state_dict['layers.0.mha.k_proj.weight'] = torch.tensor({tensor2DToPyT Wk})",
    indent2 s!"state_dict['layers.0.mha.v_proj.weight'] = torch.tensor({tensor2DToPyT Wv})",
    indent2 s!"state_dict['layers.0.mha.out_proj.weight'] = torch.tensor({tensor2DToPyT Wo})",
    indent2 s!"state_dict['layers.0.ffn.fc1.weight'] = torch.tensor({tensor2DToPyT W1})",
    indent2 s!"state_dict['layers.0.ffn.fc1.bias'] = torch.tensor({tensor1DToPy b1})",
    indent2 s!"state_dict['layers.0.ffn.fc2.weight'] = torch.tensor({tensor2DToPyT W2})",
    indent2 s!"state_dict['layers.0.ffn.fc2.bias'] = torch.tensor({tensor1DToPy b2})",
    indent2 s!"state_dict['layers.0.norm1.weight'] = torch.tensor({tensor1DToPy norm1_gamma})",
    indent2 s!"state_dict['layers.0.norm1.bias'] = torch.tensor({tensor1DToPy norm1_beta})",
    indent2 s!"state_dict['layers.0.norm2.weight'] = torch.tensor({tensor1DToPy norm2_gamma})",
    indent2 s!"state_dict['layers.0.norm2.bias'] = torch.tensor({tensor1DToPy norm2_beta})",
    indent2 "return state_dict",
    indent2 "",
    "def load_transformer_weights(model):",
    indent2 "model.load_state_dict(get_transformer_state_dict())",
    indent2 "return model",
    indent2 "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indent2 s!"model = {className}()",
    indent2 "model = load_transformer_weights(model)",
    s!"x = torch.randn(1, {seqLen}, {embedDim})  # batch=1, seq_len={seqLen}, embed_dim={embedDim}",
    indent2 "y = model(x)",
    indent2 "print(f\"Input shape: {x.shape}\")",
    indent2 "print(f\"Output shape: {y.shape}\")",
    indent2 "print(f\"Output: {y}\")",
    indent2 "print(f\"Model info: {model.get_model_info()}\")"
  ]

end TransformerPyTorch
end Export
