# CNN state-dict example

`train_cnn.py` trains two convolution/ReLU/max-pool blocks followed by `Linear(8, 2)`.
Its fixed input has shape `[1, 1, 8, 8]` (batch, channels, height, width), containing values 1–64
in row-major order. The target is `[1.0, 0.5]`; 200 Adam updates overwrite `cnn.json`.

| JSON key | Shape |
| --- | --- |
| `conv1.weight`, `conv1.bias` | `[2, 1, 3, 3]`, `[2]` |
| `conv2.weight`, `conv2.bias` | `[2, 2, 3, 3]`, `[2]` |
| `fc.weight`, `fc.bias` | `[2, 8]`, `[2]` |

Both convolutions use stride one and padding one. Each `2 × 2` max-pool has stride two, giving
spatial sizes `8 → 4 → 2`. The classifier consumes the resulting eight values.

From the repository root:

```bash
python3 NN/Examples/Interop/PyTorch/CNN/train_cnn.py
lake exe torchlean pytorch_roundtrip --model cnn --action import
lake exe torchlean pytorch_roundtrip --model cnn --action export
```

`NN/Runtime/PyTorch/Import/CNN.lean` checks tensor shapes. The driver loads them into an executable CPU `nn` module and
prints its two outputs for the same input. `NN/Runtime/PyTorch/Export/CNN.lean` supports spatial ranks 1–3, but this
fixture and the round-trip driver use rank two. Export writes `TestCNN_PyTorch.py` and the optional
`TestCNN_WithWeights.py`. These actions do not assert PyTorch numerical parity; see the
[interop guide](../README.md).
