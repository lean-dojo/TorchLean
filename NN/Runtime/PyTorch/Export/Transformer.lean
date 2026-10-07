/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Tensor

/-!
# Transformer PyTorch Reference Export

PyTorch code generator for the Transformer encoder round-trip reference model.

This file produces a readable Python `nn.Module` implementation that follows the usual PyTorch
structure (MHA + residual + LayerNorm + FFN). In the TorchLean repo we mostly use this as a
round-trip companion: generate a reference implementation, train/tweak in Python if needed, and
optionally export parameters back to Lean via JSON in the importer modules.
-/

@[expose] public section

open Spec TorchLean
open TorchLean.Tensor
open Export.PyTorch

namespace Export.PyTorch.Transformer

/-- Render a small Transformer encoder as a Python `nn.Module` class definition.

This produces readable "reference PyTorch" code (MultiHeadAttention + residual + LayerNorm + FFN),
useful for round-trip examples.
-/
def source
    (sequenceLength modelWidth headCount feedForwardWidth layerCount : Nat)
    (className : String := "TransformerEncoder") : String :=
  joinLines <|
  #[imports, "import math", ""] ++ #[
    "class MultiHeadAttention(nn.Module):",
    indent 2 s!"def __init__(self, embed_dim={modelWidth}, num_heads={headCount}):",
    indent 4 "super().__init__()",
    indent 4 "self.embed_dim = embed_dim",
    indent 4 "self.num_heads = num_heads",
    indent 4 "self.head_dim = embed_dim // num_heads",
    indent 4 "assert embed_dim % num_heads == 0",
    -- TorchLean's executable attention uses bias-free projection matrices.
    indent 4 "self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indent 4 "self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indent 4 "self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indent 4 "self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)",
    indent 2 "",
    indent 2 "def forward(self, x, mask=None):",
    indent 4 "B, S, E = x.shape",
    indent 4 "q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)",
    indent 4 "k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)",
    indent 4 "v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)",
    indent 4 "scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)",
    indent 4 "if mask is not None:",
    indent 6 "scores = scores.masked_fill(mask == 0, float('-inf'))",
    indent 4 "attn = torch.softmax(scores, dim=-1)",
    indent 4 "out = torch.matmul(attn, v)",
    indent 4 "out = out.transpose(1, 2).contiguous().view(B, S, E)",
    indent 4 "return self.out_proj(out)",
    "",
    "class FeedForward(nn.Module):",
    indent 2
      s!"def __init__(self, embed_dim={modelWidth}, hidden_dim={feedForwardWidth}):",
    indent 4 "super().__init__()",
    indent 4 "self.fc1 = nn.Linear(embed_dim, hidden_dim)",
    indent 4 "self.fc2 = nn.Linear(hidden_dim, embed_dim)",
    indent 2 "",
    indent 2 "def forward(self, x):",
    indent 4 "return self.fc2(F.relu(self.fc1(x)))",
    "",
    s!"class {className}(nn.Module):",
    indent 2 (s!"\"\"\"Transformer Encoder with {layerCount} layers, {headCount} heads, " ++
      s!"embed dim {modelWidth}, hidden dim {feedForwardWidth}\"\"\""),
    indent 2 "",
    indent 2 s!"def __init__(self):",
    indent 4 "super().__init__()",
    indent 4 "self.layers = nn.ModuleList([nn.ModuleDict({",
    indent 6 s!"'mha': MultiHeadAttention({modelWidth}, {headCount}),",
    indent 6 s!"'norm1': nn.LayerNorm({modelWidth}),",
    indent 6 s!"'ffn': FeedForward({modelWidth}, {feedForwardWidth}),",
    indent 6 s!"'norm2': nn.LayerNorm({modelWidth})",
    indent 4 s!"}) for _ in range({layerCount})])",
    indent 2 "",
    indent 2 "def forward(self, x, mask=None):",
    indent 4 "# x: (batch, seq_len, embed_dim)",
    indent 4 "for layer in self.layers:",
    indent 6 "# Self-attention block",
    indent 6 "attn_out = layer['mha'](x, mask)",
    indent 6 "x = layer['norm1'](x + attn_out)",
    indent 6 "# Feed-forward block",
    indent 6 "ffn_out = layer['ffn'](x)",
    indent 6 "x = layer['norm2'](x + ffn_out)",
    indent 4 "return x",
    indent 2 "",
    indent 2 "@property",
    indent 2 "def input_shape(self):",
    indent 4 s!"return ({sequenceLength}, {modelWidth})",
    indent 4 "",
    indent 2 "@property",
    indent 2 "def output_shape(self):",
    indent 4 s!"return ({sequenceLength}, {modelWidth})",
    indent 4 "",
    indent 2 "@property",
    indent 2 "def layer_count(self):",
    indent 4 s!"return {layerCount}",
    indent 4 "",
    indent 2 "@property",
    indent 2 "def operation_types(self):",
    indent 4
      "return ['MultiHeadAttention', 'LayerNorm', 'FeedForward', 'LayerNorm'] * self.layer_count",
    indent 4 ""
  ] ++
    metadata className

/--
Generate a single-layer Transformer encoder module with an embedded `state_dict` initializer.

This is meant for round-trip examples where parameters are loaded from TorchLean tensors.

TorchLean attention projections use mathematical `(input, output)` orientation and are transposed
for PyTorch. Feed-forward layers already use PyTorch's `(output, input)` orientation and are emitted
unchanged.
-/
def weights (sequenceLength modelWidth headCount feedForwardWidth : Nat)
  (queryWeight keyWeight valueWeight outputWeight :
    Tensor Float [modelWidth, modelWidth])
  (feedForwardInputWeight : Tensor Float [feedForwardWidth, modelWidth])
  (feedForwardOutputWeight : Tensor Float [modelWidth, feedForwardWidth])
  (feedForwardInputBias : Tensor Float [feedForwardWidth])
  (feedForwardOutputBias norm1Scale norm1Bias norm2Scale norm2Bias :
    Tensor Float [modelWidth])
  (className : String := "TransformerEncoder") : String :=
  joinLines #[
    source sequenceLength modelWidth headCount feedForwardWidth 1 className,
    "",
    "# Weight initialization helpers",
    "def get_transformer_state_dict():",
    indent 2 "state_dict = {}",
    indent 2 (s!"state_dict['layers.0.mha.q_proj.weight'] = "
      ++ s!"torch.tensor({tensorLiteral (Tensor.swapAdjacentAxes queryWeight 0)})"),
    indent 2 (s!"state_dict['layers.0.mha.k_proj.weight'] = "
      ++ s!"torch.tensor({tensorLiteral (Tensor.swapAdjacentAxes keyWeight 0)})"),
    indent 2 (s!"state_dict['layers.0.mha.v_proj.weight'] = "
      ++ s!"torch.tensor({tensorLiteral (Tensor.swapAdjacentAxes valueWeight 0)})"),
    indent 2 (s!"state_dict['layers.0.mha.out_proj.weight'] = "
      ++ s!"torch.tensor({tensorLiteral (Tensor.swapAdjacentAxes outputWeight 0)})"),
    indent 2 (s!"state_dict['layers.0.ffn.fc1.weight'] = "
      ++ s!"torch.tensor({tensorLiteral feedForwardInputWeight})"),
    indent 2 (s!"state_dict['layers.0.ffn.fc1.bias'] = "
      ++ s!"torch.tensor({tensorLiteral feedForwardInputBias})"),
    indent 2 (s!"state_dict['layers.0.ffn.fc2.weight'] = "
      ++ s!"torch.tensor({tensorLiteral feedForwardOutputWeight})"),
    indent 2 (s!"state_dict['layers.0.ffn.fc2.bias'] = "
      ++ s!"torch.tensor({tensorLiteral feedForwardOutputBias})"),
    indent 2 (s!"state_dict['layers.0.norm1.weight'] = "
      ++ s!"torch.tensor({tensorLiteral norm1Scale})"),
    indent 2 (s!"state_dict['layers.0.norm1.bias'] = "
      ++ s!"torch.tensor({tensorLiteral norm1Bias})"),
    indent 2 (s!"state_dict['layers.0.norm2.weight'] = "
      ++ s!"torch.tensor({tensorLiteral norm2Scale})"),
    indent 2 (s!"state_dict['layers.0.norm2.bias'] = "
      ++ s!"torch.tensor({tensorLiteral norm2Bias})"),
    indent 2 "return state_dict",
    indent 2 "",
    "def load_transformer_weights(model):",
    indent 2 "model.load_state_dict(get_transformer_state_dict())",
    indent 2 "return model",
    indent 2 "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indent 2 s!"model = {className}()",
    indent 2 "model = load_transformer_weights(model)",
    indent 2 (s!"x = torch.randn(1, {sequenceLength}, {modelWidth})  " ++
      s!"# batch=1, seq_len={sequenceLength}, embed_dim={modelWidth}"),
    indent 2 "y = model(x)",
    indent 2 "print(f\"Input shape: {x.shape}\")",
    indent 2 "print(f\"Output shape: {y.shape}\")",
    indent 2 "print(f\"Output: {y}\")",
    indent 2 "print(f\"Model info: {model.get_model_info()}\")"
  ]

end Export.PyTorch.Transformer
