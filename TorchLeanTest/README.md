# TorchLean tests

`TorchLeanTest` holds the checks of TorchLean against references outside Lean: PyTorch, JAX and
ONNX, LibTorch on CUDA, the C kernels in `csrc/`, CPython, mpmath and numpy, the host's floating
point, and files in the NPY, CSV, PPM and certificate formats. `Suite.lean` is the root of the
`nn_tests_suite` executable, which `scripts/lake.sh test` runs. Two modules check at build time
with `#guard`: `OpTable.lean` and `Tensor/OnlineSoftmax.lean`.

- `Data/`: `NPY.lean` decodes `.npy` payloads bit for bit, signed zeros and infinities included, and
  every NaN as the canonical NaN. `Text.lean` covers the decimal parser behind CSV cells, CLI float
  flags and certificate numbers, the CSV final newline and the PPM writer.
- `Floats/`: the developer-only `native_float32_parity` executable. It compares the host's binary32
  `add`, `mul`, `div` and `sqrt` with `ExecFloat.Binary 8 23` and emits the `fma` cases that
  `scripts/checks/cuda.sh parity` runs. `DirectedConversion.lean` checks the native binary64 to
  binary32 conversion.
- `IR/`: `MoEImport.lean` captures Mixtral's mixture-of-experts block with `torch.export` through
  `python3 -m torchlean.export` and compares Lean's binary32 interpreter with PyTorch's output.
  `Pointwise.lean` runs one-node IR artifacts against the words eager PyTorch 2.13 returns on CPU,
  apart from the two deviations it lists, with JAX 0.11.2 for the comparison cases.
  `Capture.lean` writes Lean's op table to a file for the Python captures and runs them on the
  repository's `python` package.
- `OpTable.lean`: the gradients of `scatter_reduce` with `prod`, `amax` and `amin` against
  torch autograd's, in `Float` and `Float32`.
- `CUDA/`: the LibTorch CUDA kernels against the CPU tape or the specification's bits, and stress
  and memory probes. The checks cover softmax, elementwise operators, attention, the selective scan,
  the real FFT, gather and scatter, the deterministic tree sum, graph evaluation and the trainer.
  Mixture-of-experts routing and gradients (`MoE.lean`), learning-rate schedules (`Scheduler.lean`)
  and tied AdamW parameters (`ParameterAliases.lean`) are checked on LibTorch's CUDA backend,
  against the CPU tape or a reference run. `SessionRefIdentity.lean` checks CUDA session references.
  `KernelGradient.lean` runs the tangent and gradient kernels of four custom operations, in
  binary32 and binary64, against their analytic derivatives on the CPU, and their GPU runs against
  the CPU bits. At the kink of the branched operation the tangent is checked against the derivative
  of the branch that the kink selects.
- `Interop/`: the PyTorch module `PyTorch.emit` writes. It also covers the checked-in MLP, CNN and
  transformer weights through PyTorch, strict JSON integers from an RL process, and ONNX models
  through `python3 -m torchlean.onnx`.
- `Tensor/`: `Elementary.lean` against 400-digit mpmath, CPython's `cmath` and numpy,
  `Elementwise.lean` against IEEE 754-2019, Python's `%` and `jax.lax.sign`, and `EinopsLexer.lean`
  against the Unicode predicates of CPython 3.12.13. `Operations.lean` and `Storage.lean` run the C
  kernels of `csrc/cpu/torchlean_tensor.c` against the host's scalar operations.
  `OnlineSoftmax.lean` checks the online softmax step on native `Float32` and `Float`.
- `Tree/`: `Main.lean` is the root of the `tree_tests` executable. It checks tree arenas, derived
  signatures, folds, folds compiled with `Fold.of`, hylomorphisms, unfolds, scalars held as their
  encodings and unary functions as kernel tables on the host, then on the device when LibTorch and a
  device are available. It compares every result with the Lean function bit for bit.
  `scripts/lake.sh test` runs the host checks too, and in builds without LibTorch it checks that
  uploads and device tables are refused and that hylomorphisms and unfolds compute on the host.
- `Verification/`: `BuilderImport.lean` builds operator graphs from checkpoints in PyTorch's
  state-dictionary layout, `CameraCertificates.lean` and `CertificateParsing.lean` read certificate
  JSON, and `CertificatePreconditions.lean` checks the readers and writers of certificate text.
  `OpDeriv.lean` has 61 theorems that prove with `op_deriv` the derivatives and pullbacks of
  programs outside the operator table, among them a two-layer network, attention and
  cross-entropy, and check hand-written backward passes against them. `OpDerivScatter.lean` does
  the same through the scatters and instantiates the certificates of `Op.scatter`, and
  `OpDerivCat.lean` through a concatenation, instantiating the certificate of `Op.cat`.
  `OpCert.lean` checks that `op_cert` rejects derivative certificates that are not in the canonical
  `DerivCert` form, among them the tautology `backward = backward`, and that the library's
  certificates of `relu`, `exp`, `cat` and `pad` have that form. `Recorder.lean` checks that
  `record_function%` records ordinary programs with their checked equality and rejects instances
  that compute something other than the recorded operator.
  `CertificateVerdict.lean` checks that `CROWNNode.check` decides by `FiniteArtifact.accepts`, on
  an artifact that the binary32 replay accepts and the verdict rejects.
  `RoundoffBaseline.lean` checks that a reference that branches on a rounded value makes an
  analysis conditional, and runs the comparison of the roundoff coverage with
  `scripts/checks/roundoff_baseline.jsonl` on rows that differ in one way each.
  `GradientKernel.lean` proves that the gradient kernels of `x / (x² + 1)` and of a branched
  rational function away from its kink compute the adjoints of their derivatives, and
  `KernelMap.lean` proves it for `x * x + 1` written with `Program.of` over binary32 and over
  `Float32`, and for the two-input quotient `a b / (a² + 1)` of the kernel README over `Float32`.
- `Harness.lean`: the Python dependency policy and the tolerance assertion, depending only on `Std`.

`Suite.lean` also runs the native `Float32` softmax cross-entropy example of
`TorchLeanExamples/DeepDives/Floats/RoundingBugs.lean`.

## Running the tests

```bash
scripts/lake.sh build TorchLeanTest
scripts/lake.sh test
```

`scripts/lake.sh exe nn_tests_suite` runs the suite directly. Without LibTorch it prints
`CUDA kernels: skipped (LibTorch not linked)`. Two variables turn a skip into a failure:
`TORCHLEAN_REQUIRE_INTEROP=1` when the `torch`, `numpy`, or `onnx` Python package is missing, and
`TORCHLEAN_REQUIRE_CUDA=1` when LibTorch is not linked. A suite built with `-Kcuda=true` fails
when no CUDA device is visible.

```bash
TORCHLEAN_REQUIRE_INTEROP=1 scripts/lake.sh -Kcuda=false test
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true test
```

The first command needs the Python packages listed in [CPU checks](../scripts/README.md#cpu-checks)
and the second the SDK described in [LibTorch CUDA build](../scripts/README.md#libtorch-cuda-build).
Setting `TORCHLEAN_LIBTORCH_MEMORY_PROBE` to `accounting`, `attention-buffers`, or `oom-recovery`
runs only that LibTorch memory probe, and the CUDA stress suite starts each probe this way in a
fresh process. The GitHub Actions workflow publishes documentation and runs none of these tests.
