# Transformer state-dict example

Despite its filename, `train_transformer.py` does not train. It initializes a seeded single-layer
Transformer encoder, prints the output for input `[[[1.5, 1.5]]]`, and overwrites
`transformer_encoder.json`. The model has one attention head, width two, feed-forward width two,
post-residual LayerNorm, ReLU, and bias-free attention projections.

| Explicit JSON keys | Shape and orientation |
| --- | --- |
| `Wq`, `Wk`, `Wv`, `Wo` | `[2, 2]`, `(input, output)` |
| `W1`, `W2` | `[2, 2]`, PyTorch `(output, input)` |
| `b1`, `b2`, `norm1_gamma`, `norm1_beta`, `norm2_gamma`, `norm2_beta` | `[2]` |

The importer also accepts nested generated-module keys such as `layers.0.mha.q_proj.weight`.
Those attention matrices use PyTorch orientation and are transposed on import. Since the matrices
are square, shape checks alone cannot detect an orientation mistake.

From the repository root:

```bash
python3 NN/Examples/Interop/PyTorch/Transformer/train_transformer.py
lake exe torchlean pytorch_roundtrip --model transformer --action import
lake exe torchlean pytorch_roundtrip --model transformer --action export
```

Import prints the Lean CPU output. Export writes `TestTransformer_Encoder.py` and, when JSON
weights exist, `TestTransformer_Encoder_WithWeights.py`. The weighted exporter and importer handle
one layer; the unweighted class generator can describe more layers. A one-token input does not
exercise competition between attention positions. These commands demonstrate parameter transport,
not a general Transformer equivalence proof; see the [interop guide](../README.md).
