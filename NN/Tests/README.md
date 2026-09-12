# NN/Tests

`NN/Tests` is TorchLean's executable regression suite.

The proofs live under `NN/Proofs`; these tests have a different job. They run small deterministic
programs through the same public APIs, FFI paths, and CUDA kernels that users exercise in training
and inference. That makes this directory useful for code Lean treats as a trust boundary: native
CUDA kernels, external buffers, floating point execution, parser/serialization helpers, and CLI
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

- native CUDA kernels and FFI buffers,
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
| Verification-facing runtime | PINN residual paths, TorchLean-to-IR execution checks, and small certificate-style command paths |
| Commands | registered `torchlean` and `verify` commands, flags, output shapes, and artifact paths |

When adding a runtime feature, add a small test near the boundary that can fail loudly before a large
model run silently changes meaning.

Normal CI should build and run the curated executable:

```bash
lake build nn_tests_suite
lake exe nn_tests_suite
```

CUDA builds exercise the same Lean driven tests against the native CUDA FFI layer:

```bash
lake -R -K cuda=true build nn_tests_suite
lake -R -K cuda=true exe nn_tests_suite
```

To test the native CUDA boundary, run the curated suite under NVIDIA's CUDA memory checker:

```bash
scripts/checks/cuda_sanitize_tests.sh
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

The sanitizer harness is intentionally Lean driven: it checks the CUDA kernels through the same FFI
calls used by TorchLean training and inference.

## When To Add A Test

Add a test when the feature:

- crosses an FFI or external-tool boundary,
- depends on file parsing or serialization,
- changes public trainer, prediction, or loader behavior,
- adds a CUDA kernel or VJP rule,
- adds a runtime approximation path that is easy to regress,
- introduces a checker CLI where a malformed artifact should be rejected.

Do not put long proof obligations here. If the claim is mathematical and reusable, put the theorem
under `NN/Proofs`, `NN/MLTheory`, or `NN/Verification` and keep the test as the executable guard.

## Evidence Matrix

| Change | Useful executable evidence |
| --- | --- |
| Public trainer, prediction, optimizer, or loader behavior | `lake exe nn_tests_suite` plus a focused `lake exe torchlean ...` command |
| New verifier command or certificate schema | malformed-artifact rejection, accepted fixture check, and `lake exe verify -- all` when feasible |
| CUDA kernel or VJP rule | `lake -R -K cuda=true exe nn_tests_suite` plus `scripts/checks/cuda_sanitize_tests.sh` for native memory behavior |
| PyTorch/ONNX/ATen/Julia/Gymnasium bridge | round-trip or fixture test that names the imported artifact and remaining producer boundary |
| Documentation for public commands | `lake exe torchlean --help`, `lake exe verify --help`, and the relevant Jekyll/Verso build |
