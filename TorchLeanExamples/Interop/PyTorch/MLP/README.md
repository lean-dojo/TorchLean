# MLP state-dict example

`train_mlp.py` trains `Linear(2, 3) → ReLU → Linear(3, 1)` for 200 SGD updates on the single pair
`[0.5, 0.8] → [1.0]` and writes the float64 copy of its state dict to `mlp.json`. No dataset
download is needed.

| PyTorch name | Shape |
| --- | --- |
| `fc1.weight`, `fc1.bias` | `[3, 2]`, `[3]` |
| `fc2.weight`, `fc2.bias` | `[1, 3]`, `[1]` |

Weights have PyTorch's `(output, input)` orientation, which is also the orientation of the state of
`nn.linear`. `TorchLeanExamples/Interop/PyTorch/Roundtrip.lean` reads the four tensors by these
names into the state of `nn.Sequential![nn.linear 2 3, nn.op .relu, nn.linear 3 1]`.

From the repository root:

```bash
python3 TorchLeanExamples/Interop/PyTorch/MLP/train_mlp.py
scripts/lake.sh exe torchlean pytorch_roundtrip --model mlp
```

The second command prints the Lean output for `[0.5, 0.8]`. It does not compare Python and Lean
outputs. `TorchLeanTest/Interop/PyTorchRoundtrip.lean` compares them for all three
examples.
