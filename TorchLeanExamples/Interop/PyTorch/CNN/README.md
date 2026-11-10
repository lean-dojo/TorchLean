# CNN state-dict example

`train_cnn.py` trains two convolution/ReLU/max-pool blocks followed by `Linear(8, 2)`. Its fixed
input has shape `[1, 1, 8, 8]` (batch, channels, height, width) and holds the values 1 to 64 in
row-major order. The target is `[1.0, 0.5]`. After 200 Adam updates the script writes the float64
copy of the state dict to `cnn.json`.

| PyTorch name | Shape |
| --- | --- |
| `conv1.weight`, `conv1.bias` | `[2, 1, 3, 3]`, `[2]` |
| `conv2.weight`, `conv2.bias` | `[2, 2, 3, 3]`, `[2]` |
| `fc.weight`, `fc.bias` | `[2, 8]`, `[2]` |

Both convolutions use stride one and padding one. Each `2 × 2` max-pool has stride two, giving
spatial sizes `8 → 4 → 2`. The classifier consumes the resulting eight values. A convolution
weight holds output channels, input channels, height and width, in that order, on both sides.

From the repository root:

```bash
python3 TorchLeanExamples/Interop/PyTorch/CNN/train_cnn.py
scripts/lake.sh exe torchlean pytorch_roundtrip --model cnn
```

`TorchLeanExamples/Interop/PyTorch/Roundtrip.lean` reads the six tensors by these names into the
state of the same CNN, built from `nn` layers, and prints its two outputs for the same input. The
command does not compare them with PyTorch. `TorchLeanTest/Interop/PyTorchRoundtrip.lean` does.
