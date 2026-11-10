# JAX and PyTorch interop

These examples bring JAX computations and PyTorch checkpoints into TorchLean. Every command runs
from the repository root.

## JAX captures

`JAX/export.py` captures a small MLP, sums, broadcasts, reshapes, axis permutations, a batched
contraction, a comparison and functions returning several tensors, one capture document each. It
names every computation it cannot capture and writes the others. `torchlean jaxpr`, defined in
`JAX.lean`, loads a document with `TorchLean.JAX.load`, runs its program on the recorded arguments
under the reference runtime and compares the element type, shape and interchange words of every
output with the recorded JAX output. The script needs JAX in the running Python.

```bash
python3 TorchLeanExamples/Interop/JAX/export.py /tmp/torchlean-jax-examples
scripts/lake.sh exe torchlean jaxpr /tmp/torchlean-jax-examples/sum_squares.json
```

A passing document prints `bit-for-bit match for this input` for every output. The comparison
covers the recorded arguments only. `JAX.lean` proves that the real-valued evaluation of the
`sum(x * x)` graph is unchanged by every permutation of a vector of any length, and the runner
prints `matches the graph covered by sumSquaresGraph_permutation` for a document holding that graph.

## A trained MNIST classifier

`JAX/mnist.py` fetches MNIST into `data/real/mnist`, checking each archive's MD5, trains a
two-layer classifier in JAX and replays its forward graph on one held-out test image through
`verify`, once each in `float32`, `float16` and `bfloat16`. It compares each replay with JAX, and
the `float16` and `bfloat16` replays with the `float32` one. When the largest logit difference is
below half the margin of the reference logits, it writes a Lean file proving that margin and that
the difference cannot change the predicted class, and checks the file with `lean`. Otherwise it
reports the decision as inconclusive. Training and test accuracy are measured, not proved.

```bash
scripts/lake.sh build verify TorchLean.Verify.Robustness.LipschitzCertified
uv run TorchLeanExamples/Interop/JAX/mnist.py /tmp/torchlean-mnist
```

`uv run` installs the JAX and NumPy versions the script declares. The output directory must not
exist. It receives the trained weights, the capture documents, the `verify` reports, the Lean files
and `summary.json`.

## PyTorch state dicts

`PyTorch/` reads the PyTorch state dicts of an MLP, a CNN and a Transformer encoder block into the
ordinary `nn` models. Its README lists the models and the commands.
