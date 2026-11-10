# VNN-COMP MNIST-FC suite

`vnncomp-mnistfc` runs the MNIST-FC benchmark of VNN-COMP 2022 (`mnist_fc`), fully connected ReLU
networks on the 784 pixels of an MNIST image, from JSON exports of one of its networks and of
VNNLIB properties. For each instance it bounds the network's outputs on the input box and tries to
show that none of them lies in the property's unsafe output region.

- `MNISTFC.lean`: the command `main`, the network `model` (a ReLU after every layer but the last)
  and the bound modes of `Mode`.
- `VNNLib.lean`: `Instance`, its unsafe region `Property`, the suite loader `loadSuite` and the
  output-box test `refutedByOutputBox`.

## Input files

We don't ship either file, or a converter for them. Prepare the two JSON files below from one
ONNX network of the benchmark and its VNNLIB properties with an outside tool, and keep them under
`_external/vnncomp/mnist_fc/`, which git ignores.

- `model_weights.json`: a PyTorch-style state dict with `layers.<i>.weight` and `layers.<i>.bias`.
  Each layer's input size must equal the previous layer's output size.
- `suite.json`: `"format": "vnnlib_suite_v0_1"` and an `instances` array. Each instance has an
  `id`, the input box `input_lo` and `input_hi`, and `spec`, a list of terms `{mat, rhs}`. The
  unsafe region is the union over the terms of the sets where `mat · y ≤ rhs` for the output `y`.

## Command

```bash
scripts/lake.sh exe verify -- vnncomp-mnistfc \
  --weights=_external/vnncomp/mnist_fc/model_weights.json \
  --suite=_external/vnncomp/mnist_fc/suite.json
```

These paths are the defaults, so the flags matter when the files live elsewhere. A missing file
stops the command with `missing weights JSON` or `missing suite JSON` and the usage text. `--max=N`
runs the first `N` instances (30 by default). `--mode=ibp`, the default, tests the rows on the IBP
output box and `--mode=crown` on a CROWN output box. `--mode=crownobj` runs one backward CROWN
objective per row, from intermediate IBP boxes: CROWN narrows those boxes when the input has at
most 64 entries, and MNIST has 784.

## What the counts mean

A run's last line reports `numerically_refuted=N unknown=M`. An instance counts as refuted when
every term has a row whose lower bound over the outputs exceeds its `rhs`, in outward-rounded host
`Float`. That is a floating-point observation, and the command labels it `not a Lean safety
theorem`: no proof connects the count to the real-valued network. `unknown` means the bounds were
too loose to decide, and it doesn't show a counterexample. `verify -- all` doesn't run this command.
