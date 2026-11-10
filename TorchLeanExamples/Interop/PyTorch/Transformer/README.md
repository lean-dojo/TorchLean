# Transformer state-dict example

`export_transformer.py` initializes a seeded single-layer Transformer encoder, prints its output
for the input `[[[1.5, 1.5]]]` and writes the float64 copy of its state dict to
`transformer_encoder.json`. The model has one attention head, width two, feed-forward width two,
post-residual LayerNorm, ReLU and bias-free attention projections.

| PyTorch name, after `layers.0.` | Shape and orientation |
| --- | --- |
| `mha.q_weight`, `mha.k_weight`, `mha.v_weight`, `mha.out_weight` | `[2, 2]`, `(input, output)` |
| `ffn.fc1.weight`, `ffn.fc2.weight` | `[2, 2]`, PyTorch `(output, input)` |
| `ffn.fc1.bias`, `ffn.fc2.bias` | `[2]` |
| `norm1.weight`, `norm1.bias`, `norm2.weight`, `norm2.bias` | `[2]` |

The attention projections are applied as `x @ weight`, the layout of TorchLean's executable
attention layer, so both sides store the same numbers in the same order. All four are square, so
the shape check cannot detect a transposed projection.

From the repository root:

```bash
python3 TorchLeanExamples/Interop/PyTorch/Transformer/export_transformer.py
scripts/lake.sh exe torchlean pytorch_roundtrip --model transformer
```

`TorchLeanExamples/Interop/PyTorch/Roundtrip.lean` reads the twelve tensors by these names into
the state of `nn.transformerBlock` and prints the Lean CPU output. The command does not compare it
with PyTorch. `TorchLeanTest/Interop/PyTorchRoundtrip.lean` does, at the same one-token input,
which does not exercise competition between attention positions.
