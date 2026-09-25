# NN/Tests

`NN/Tests` is TorchLean's executable regression suite.

The proofs live under `NN/Proofs`; these tests have a different job. They run small deterministic
programs through the same public APIs, FFI paths, and LibTorch CUDA operations used in training
and inference. That makes this directory useful for code Lean treats as a trust boundary: native
SDK operations, external buffers, floating point execution, parser/serialization helpers, and CLI
runtime checks.

The suite is organized as follows:

* `NN/Tests/Suite.lean` is the top-level executable umbrella.
* `NN/Tests/Runtime/Floats` checks float autograd, model runtime checks, IR execution, PINN residuals,
  and other runtime regressions.
* `NN/Tests/Runtime/Rationals` runs a small proof oriented scalar backend to catch semantic
  regressions without floating point noise.
* `NN/Tests/Runtime/Cuda` compares Lean driven CPU/eager behavior against the CUDA FFI layer on
  small deterministic inputs.
* `NN/Tests/API`, `NN/Tests/Tensor`, `NN/Tests/IR`, `NN/Tests/GraphSpec`, `NN/Tests/Backend`, and
  `NN/Tests/MLTheory` hold focused checks for the public API (trainer runs, checkpoints, data,
  models, CLI), the tensor front end (lexer, einsum planning, storage, linear algebra), IR shape
  contracts, GraphSpec generality, backend profiles, and CROWN operator guardrails.

## What Tests Are For

Tests are evidence about executable behavior. They are still important because TorchLean deliberately
crosses runtime boundaries:

- LibTorch CUDA operations and FFI buffers,
- executable `Float` and Float32 paths,
- file parsers and serializers,
- public CLI commands,
- PyTorch/ONNX/export adapters,
- training-loop plumbing, logs, and saved artifacts.

The proof files state mathematical facts under hypotheses. The tests make sure the actual code path
that users run still satisfies closed-form checks, parity checks, parser checks, rejection checks,
and artifact-shape expectations. When a theorem and a runtime test cover the same feature, cite
both: the theorem says what the object means, and the test says the executable path being shipped
still reaches that object on representative cases.

## What The Main Buckets Cover

| Bucket | Examples of coverage |
| --- | --- |
| Float runtime | closed-form tensor/autograd checks, model equivalence checks, IR execution parity, spec-vs-runtime MLP checks |
| Rational runtime | exact small scalar checks that avoid floating-point noise |
| CUDA runtime | matmul, reductions, views, broadcast, gather/scatter, attention, convolution/pooling, deterministic reductions, and stress checks |
| Verification-facing runtime | PINN residual paths, TorchLean-to-IR execution checks, and CROWN operator guardrails |
| CLI support | generic argument parsing, runtime selection defaults, and command device/execution requirements |

The curated suite does not dispatch every registered `torchlean` or `verify` command. Run
`scripts/lake.sh exe verify -- all` separately for the verifier registry's default fixture checks;
`scripts/lake.sh exe verify -- list` only lists the registry. Individual commands still need checks
with their own arguments and artifacts.

When adding a runtime feature, add a small test near the boundary that can fail loudly before a large
model run silently changes meaning.

Hosted CI runs the CPU profile. Build and run the curated suite through the Lake wrapper:

```bash
scripts/lake.sh build nn_tests_suite
scripts/lake.sh test
```

The PyTorch round-trip parity and real ONNX bridge checks normally skip when their Python
dependencies cannot be imported. Hosted CI installs the pinned CPU dependencies and enables required
interop checks so a missing or broken import fails the suite. Use Python 3.12, with the same
`python3` on `PATH` for the preflight and the Lean executables:

```bash
python3 -m pip install -r scripts/checks/requirements-interop.txt
python3 scripts/checks/check_interop_dependencies.py
export TORCHLEAN_REQUIRE_INTEROP=1
scripts/lake.sh build nn_tests_suite pytorch_export_check verify
scripts/lake.sh test
scripts/lake.sh exe pytorch_export_check
scripts/lake.sh exe verify -- all
```

`pytorch_export_check` is a separate executable. It runs generated TorchLean-to-PyTorch code and
imports real `torch.export` and FX graphs for numerical checks; building it alone does not execute
those checks. The real ONNX round trip remains part of `nn_tests_suite`.

For an explicit GPU check, select a CUDA-enabled LibTorch SDK and require a visible device:

```bash
export TORCHLEAN_LIBTORCH_HOME=/path/to/torch
scripts/lake.sh -R -K cuda=true build nn_tests_suite
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -R -K cuda=true test
```

`TORCHLEAN_REQUIRE_CUDA=1` fails on CPU stubs or when no CUDA device is visible. Without required
GPU mode, the CPU profile skips the CUDA suite. These GPU commands require a separate GPU-capable
environment; the hosted CI job provides CPU coverage.

`cuda=true` builds the complete backend in `csrc/libtorch`. TorchLean owns the tape and calls
ATen for forward operations and their VJPs. Keep the selected SDK available during execution;
see [native build instructions](../../scripts/README.md#libtorch-cuda-build) for its requirements.

To test the native boundary, run the curated suite under NVIDIA Compute Sanitizer:

```bash
scripts/checks/cuda_sanitize_tests.sh
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

The sanitizer harness enables required GPU mode and exercises the LibTorch backend through the
same FFI calls used by TorchLean training and inference. It uses the selected SDK; a pass covers
the executed paths. `scripts/checks/check.sh --cuda` also enables required GPU mode.

## When To Add A Test

Add a test when the feature:

- crosses an FFI or external-tool boundary,
- depends on file parsing or serialization,
- changes public trainer, prediction, or loader behavior,
- adds or changes a LibTorch operation or VJP rule,
- adds a runtime approximation path that is easy to regress,
- introduces a checker CLI where a malformed artifact should be rejected.

Do not put long proof obligations here. If the claim is mathematical and reusable, put the theorem
under `NN/Proofs`, `NN/MLTheory`, or `NN/Verification` and keep the test as the executable guard.

## Evidence Matrix

| Change | Useful executable evidence |
| --- | --- |
| Public trainer, prediction, optimizer, or loader behavior | `scripts/lake.sh test` plus a focused `scripts/lake.sh exe torchlean ...` command |
| New verifier command or certificate schema | malformed-artifact rejection, accepted fixture check, and `scripts/lake.sh exe verify -- all` when feasible |
| LibTorch operation or VJP rule | `TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -R -K cuda=true test` plus `scripts/checks/cuda_sanitize_tests.sh` for native memory behavior |
| PyTorch/ONNX/ATen/Julia/Gymnasium bridge | round-trip or fixture test that names the imported artifact and remaining producer boundary |
| Documentation for public commands | `scripts/lake.sh exe torchlean --help`, `scripts/lake.sh exe verify --help`, and the relevant Jekyll/Verso build |
