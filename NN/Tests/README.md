# NN/Tests

`NN/Tests` is TorchLean's executable regression suite.

The proofs live under `NN/Proofs`; these tests have a different job. They run small deterministic
programs through the same public APIs, FFI paths, and CUDA kernels that users exercise in training
and inference. That makes this directory useful for code Lean treats as a trust boundary: native
CUDA kernels, external buffers, floating point execution, parser/serialization helpers, and CLI
smoke tests.

The suite is organized as follows:

* `NN/Tests/Suite.lean` is the top-level executable umbrella.
* `NN/Tests/Runtime/Floats` checks float autograd, model smoke tests, IR execution, PINN residuals,
  and other runtime regressions.
* `NN/Tests/Runtime/Rationals` runs a small proof oriented scalar backend to catch semantic
  regressions without floating point noise.
* `NN/Tests/Runtime/Cuda` compares Lean driven CPU/eager behavior against the CUDA FFI layer on
  small deterministic inputs.

Normal CI should build and run the curated executable:

```bash
lake build nn_tests_suite
lake exe nn_tests_suite
```

CUDA builds exercise the same Lean driven tests against the native CUDA FFI layer:

```bash
lake build -R -K cuda=true nn_tests_suite
lake exe -K cuda=true nn_tests_suite
```

To test the native CUDA boundary, run the curated suite under NVIDIA's CUDA memory checker:

```bash
scripts/checks/cuda_sanitize_tests.sh
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

The sanitizer harness is intentionally Lean driven: it checks the CUDA kernels through the same FFI
surface used by TorchLean training and inference.
