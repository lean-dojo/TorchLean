# MLP state-dict example

`train_mlp.py` trains `Linear(2, 3) → ReLU → Linear(3, 1)` for 200 SGD updates on the
single pair `[0.5, 0.8] → [1.0]`. It overwrites `mlp.json`; no dataset download is needed.

| JSON key | Shape |
| --- | --- |
| `layers.0.weight`, `layers.0.bias` | `[3, 2]`, `[3]` |
| `layers.2.weight`, `layers.2.bias` | `[1, 3]`, `[1]` |

The importer also accepts `fc1.*`/`fc2.*`. Weights have PyTorch's `(output, input)` orientation.
`NN/Runtime/PyTorch/Export/MLP.lean` generates readable model classes and optional weight-loading helpers;
`NN/Runtime/PyTorch/Import/MLP.lean` checks the weights and implements the two-layer forward pass.

From the repository root:

```bash
python3 NN/Examples/Interop/PyTorch/MLP/train_mlp.py
lake exe torchlean pytorch_roundtrip --model mlp --action import
lake exe torchlean pytorch_roundtrip --model mlp --action export
```

Import prints the Lean output for `[0.5, 0.8]`. Export writes `TestMLP_PyTorch.py` and, when JSON
weights are present, `TestMLP_WithWeights.py` beside this README. Neither action performs an
automated Python/Lean output comparison. See the [interop guide](../README.md) for that distinction.
