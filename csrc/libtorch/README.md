# TorchLean LibTorch backend

This directory holds the CUDA backend behind TorchLean's GPU buffer ABI. It calls the selected
LibTorch SDK's ATen operations. Lean checks shapes and dispatches through the extern symbols;
native memory safety, SDK behavior, and floating-point execution remain outside Lean's kernel.

## Layout

The backend is built as one shared library:

| LibTorch source | Responsibility |
| --- | --- |
| `torchlean.cpp` | All runtime, tensor-operation, and backward adapters to ATen. |
| `operations.h` | Shared list generating common operation exports for both build configurations. |
| `torchlean_libtorch.h` | Buffer representation, Lean object and size helpers. |

`unavailable.c` is linked instead when TorchLean is built without LibTorch. It exports the same
symbols, reports `RuntimeStatus.notLinked`, and fails every buffer operation with a message that
says how to rebuild. The Lean tape owns differentiation; native calls use a no-grad guard. The
CPU evaluation dynamic library is loaded for native CPU `#eval` calls. GPU tests run as
compiled executables.

## Build selection

`scripts/lake.sh build` selects the default `pureLean`/`portableCPU` build, without an SDK or
toolkit. `cuda=true` requires the complete LibTorch CUDA backend.

A full SDK contains `include/`,
`lib/`, and `share/cmake/Torch/TorchConfig.cmake`; a partial header snapshot is insufficient.
Use the CUDA-enabled PyTorch package root or an equivalent LibTorch distribution:

```bash
export TORCHLEAN_LIBTORCH_HOME=/path/to/torch
scripts/lake.sh -Kcuda=true build NN NNCI NNExamples NNTests nn_tests_suite
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true test
scripts/checks/check.sh --libtorch-home "$TORCHLEAN_LIBTORCH_HOME" --ci-all
```

`-Klibtorch_home=PATH` overrides `TORCHLEAN_LIBTORCH_HOME`; otherwise the default is `libtorch/`
under the package root. The build requires Linux, CMake 3.22 or newer, Make, the pinned Lean
headers, and a compatible C++20 compiler. SDK CMake discovers the ABI, any stricter C++ standard,
transitive libraries, and rpath. An executable built in the same project checks compiler/link
compatibility without running. SDK discovery may require a matching CUDA
development toolkit, even though TorchLean itself compiles only C++ sources.

## Tested SDK versions

The current adapter compiled and passed the curated CUDA suite and focused attention check on
A100 with pip torch 2.11.0+cu128. Earlier validation used pip torch 2.13.0+cu130 and a PyTorch 2.12
nightly (revision 0291f960b6). These runs do not guarantee compatibility with every intervening or
newer SDK.

The build records the SDK's `TORCH_VERSION` and checks compiler/link compatibility. Attention is
composed in Lean from numerical primitives, without private ATen attention selectors or paired
attention kernels. SDK changes can still affect compilation and numerical results. Rerun the CUDA
suite and the elementwise C++ harness below after changing SDKs.

Optional SDK discovery controls are explicit:

```bash
TORCH_CUDA_ARCH_LIST=8.0 scripts/lake.sh -Kcuda=true -Kcuda_home=/usr/local/cuda build
```

TorchLean sets no architecture override. The SDK's `TORCH_CUDA_ARCH_LIST` controls configure
probes; it does not rebuild the SDK's packaged GPU kernels.
Keep the same SDK/toolkit configuration on later build, `exe`, and `env` commands.

The helper tracks SDK headers/version/ABI, compiler identity, flags, source contents, and
discovered dependencies; replaced SDK libraries are tracked by file metadata. It records the
manifest in `libtorch/build.json` and SDK settings in `libtorch/cmake/sdk.txt` under the selected
build directory. Lake links `libtorch/libtorchlean_libtorch.so` by its resolved absolute path,
so retain that artifact and the selected SDK for execution.
See [`scripts/README.md`](../../scripts/README.md) for compiler controls, cache selection, and
the direct C++ build command.

## Execution and memory

LibTorch supplies PyTorch's C++ libraries; ATen is the tensor operation layer used by this
backend. A CUDA buffer reaches C++ as a Lean external object owning an `at::Tensor`. The bridge
unwraps that tensor, calls ATen, and returns another owned buffer through the Lean C ABI.
These tensor calls run without a Python interpreter. An installed CUDA-enabled PyTorch package
can supply the SDK at build and execution time.

Configure the selected device and process-wide SDK settings before concurrent runtime work,
as required by `LibTorch.Controls`. The live-wrapper check prevents sequential device changes
while owners remain; it is not a lock against concurrent configuration and allocation.
Explicit release likewise requires exclusive use of that buffer. Atomic telemetry supports
concurrent independent owners, but does not make mutation of one owner thread-safe.

For a linear layer, Lean sends matrix multiplication and bias addition to the buffer API,
then records the result, parents, and backward rule on its runtime tape. During backward,
Lean traverses the tape and calls the corresponding gradient operations. Each native call runs under
`at::NoGradGuard`, so LibTorch does not record another autograd graph. Some operations use
explicit SDK backward kernels. Attention's forward and local VJP are composed in
`NN/Runtime/Autograd/Engine/LibTorch/Ops/Attention.lean`: `Buffer.attentionForward` takes Q/K/V,
an optional mask, dimensions, and scale, returning `Except String (Buffer × Buffer)` for the
output and probabilities. `Buffer.attentionBackward` takes Q/K/V, those probabilities, the output
cotangent, dimensions, and scale. The tape retains and releases these ordinary saved buffers;
there is no native attention context or fused attention selection. The full score and probability
matrices require quadratic memory in sequence length.

Spectral layers compose FFT, frequency mixing, and inverse FFT in Lean using the existing
numerical primitives. Model composition and saved-buffer ownership stay with TorchLean.

The build selects the implementation behind those buffer symbols. The default build links
`unavailable.c` and needs no LibTorch SDK. A `cuda=true` build links
`libtorchlean_libtorch.so`, whose ATen calls use the selected SDK's CUDA implementation.
The eager CUDA tape stores Float32 buffers; the separate DGEMM bridge handles Float64 matrix
multiplication. Selecting CUDA does not move every scalar format onto the GPU.

TorchLean's current CUDA path is eager: each autograd step records a Lean runtime tape and dispatches
individual CUDA buffer ops. This already moves the expensive math to the GPU, but it is not CUDA
Graph capture/replay.

Current memory policy:

- trainable parameters are cached as persistent device mirrors across eager CUDA steps,
- optimizer steps can update those mirrors directly on device,
- forward scratch buffers retained only for backward are listed on tape nodes and explicitly
  released after the step,
- overwritten dense-gradient buffers are explicitly released during accumulation,
- the native allocator exposes a collection hook used after large eager CUDA training steps.

This reduces accidental lifetime extension from Lean external object finalizers. Execution
remains eager. `--execution typed-graph` selects TorchLean's proof/SSA graph backend; CUDA Graph
capture/replay is not implemented.

## Sanitizer Harness

Run the compiled Lean CUDA suite under NVIDIA Compute Sanitizer with the selected SDK and
a visible GPU:

```bash
scripts/checks/cuda_sanitize_tests.sh --libtorch-home "$TORCHLEAN_LIBTORCH_HOME"
scripts/checks/cuda_sanitize_tests.sh --libtorch-home "$TORCHLEAN_LIBTORCH_HOME" --all-tools
scripts/checks/cuda_sanitize_tests.sh --cuda-home /usr/local/cuda --tool memcheck
```

The default tool is `memcheck`. `--all-tools` additionally runs `racecheck`, `initcheck`, and
`synccheck`. Findings fail the command with exit code 99. The wrapper enables
`TORCHLEAN_REQUIRE_CUDA=1`, forwards SDK/toolkit settings to both build and execution, and
defaults to `--target-processes application-only` for the suite's intentional self-reexec probe.
With `--skip-build`, select the same profile and SDK as the existing executable. These checks
exercise the native boundary and SDK on tested paths; a pass is not a proof of memory safety.

For performance work, pair the correctness suite with NVIDIA Nsight Systems for end-to-end runtime
traces and Nsight Compute for individual kernel profiles. Those tools are not pass/fail tests, so
they stay outside the default CI gate. Invoke them directly on the executable being investigated:

```bash
scripts/lake.sh -R -K cuda=true build nn_tests_suite
scripts/lake.sh -R -K cuda=true env nsys profile -t cuda,nvtx,osrt \
  -o /tmp/torchlean-cuda .lake/build/bin/nn_tests_suite
scripts/lake.sh -R -K cuda=true env ncu --section SpeedOfLight \
  --section LaunchStats .lake/build/bin/nn_tests_suite
```

Nsight Compute can be slow on the full suite; use a focused executable for kernel-level work.

## CUDA Test Matrix

The CUDA regression suite lives in `NN/Tests/Runtime/Cuda`. The tests compare the Lean CPU eager
tape against the CUDA eager tape on small examples. They run only with `-Kcuda=true`; the default
build skips them, so CPU hosted CI does not validate GPU execution.

Run the full Lean test executable through Lake:

```bash
scripts/lake.sh -Kcuda=false test
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true test
scripts/checks/check.sh --cuda
```

Use the sanitizer harness when changing native memory, indexing, or synchronization behavior:

```bash
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

Current CUDA coverage:

| Test module | Main coverage |
| --- | --- |
| `NN/Tests/Runtime/Cuda/Softmax.lean` | `softmax` and `log_softmax`, forward and backward. |
| `NN/Tests/Runtime/Cuda/Elementwise.lean` | Scalar elementwise ops, activations, safe logs, products, and `sum`. |
| `NN/Tests/Runtime/Cuda/LayerNorm.lean` | Channel/feature normalization, parameter gradients, and input gradients. |
| `NN/Tests/Runtime/Cuda/BatchNorm.lean` | Channel-first batchnorm forward and backward. |
| `NN/Tests/Runtime/Cuda/Attention.lean` | Attention values and gradients against the CPU reference, including batches. |
| `NN/Tests/Runtime/Cuda/ConvPool.lean` | 2D and 3D convolution, max pool, average pool, smooth max pool, padded max-pool edge cases. |
| `NN/Tests/Runtime/Cuda/ConvTranspose.lean` | 2D and 3D transposed convolution forward and backward. |
| `NN/Tests/Runtime/Cuda/GatherScatter.lean` | Rank-one and row gather/scatter-add behavior, including gradients. |
| `NN/Tests/Runtime/Cuda/DeterministicReductions.lean` | Repeatability under the deterministic reduction control. |
| `NN/Tests/Runtime/Cuda/DeviceInfo.lean` | Per-index device identity readback, current-device agreement, index rejection, and a device-switching probe in a fresh process. |
| `NN/Tests/Runtime/Cuda/SelectiveScan.lean` | Diagonal selective-scan buffer primitives used by the Mamba/SSM runtime path. |
| `NN/Tests/Runtime/Cuda/PositionalEncoding.lean` | Sinusoidal positional encodings and RoPE/rotary embedding kernels. |
| `NN/Tests/Runtime/Cuda/Matmul.lean` | `matmul`, `bmm`, and explicit fp32/fp64 dispatch. |
| `NN/Tests/Runtime/Cuda/Fft.lean` | Packed real FFT, inverse FFT, spectral convolution, and finite-difference gradient checks. |
| `NN/Tests/Runtime/Cuda/ViewsBroadcastReduce.lean` | Reshape, transpose, rank-3 permutations, broadcast, reduce-sum/mean, and empty-axis behavior. |
| `NN/Tests/Runtime/Cuda/LinearMseConcatSliceGather.lean` | Linear layer, MSE loss, vector concat/slice, scalar gather, row gather, and gradients. |
| `NN/Tests/Runtime/Cuda/Stress.lean` | RNG determinism, explicit release, duplicate-parent gradient accumulation, large buffers, reductions, and rectangular matmul. |
| `NN/Tests/Runtime/Cuda/Suite.lean` | The unified entrypoint imported by the repository-level test suite. |

Elementwise arithmetic, activations, and whole-buffer reductions share the operation list in
`operations.h`. It generates both the LibTorch exports and their unavailable-build counterparts.
The shared call adapter converts arguments, checks buffer lengths, disables native autograd, and
boxes the result. Calls resolve at compile time, so this adds no string lookup or operator-ID switch.
Operations with shape metadata or saved backward state keep their explicit adapters.

When adding a CUDA symbol outside that list, add its failing export to `unavailable.c`,
update this matrix, and add at least one test against the Lean CPU tape.
If the symbol participates in autograd, test both the forward value and the relevant VJP/gradient
buffers.  If it uses atomics, also decide whether deterministic mode needs a separate test.

The separate [elementwise C++ harness](tests/elementwise/README.md) links the
production backend and consumes Lean binary32 cases. Its wrapper is
`scripts/checks/cuda_float32_parity.sh --libtorch-home PATH --backend-library PATH --lean-prefix PATH`.
It checks exact finite results and signed zeros, reports NaN encoding differences, and includes
staged Adam and no-autograd regressions. A different SDK needs its own audit and regression
results.

Convolution and pooling checks live in `NN/Tests/Runtime/Cuda/ConvPool.lean` and run
through the production FFI and tape. They compare CPU and CUDA values and gradients,
including smooth-max overflow cases with positive and negative inverse temperatures.

## Review Notes

- SDK upgrades can change numerical behavior. Compiler/link compatibility does not establish
  agreement with TorchLean's floating-point contracts; retain the exact SDK/build manifest with
  test results.
- Deterministic controls request supported deterministic SDK algorithms. Repeatability on the
  tested SDK/device does not imply bitwise agreement across releases, devices, or algorithms.
- Attention uses Lean-composed matrix products and the explicit softmax VJP over tape-owned
  probabilities. The attention tests compare values and gradients with the CPU reference and
  exercise saved-buffer ownership.
  Run the compiled Lean CUDA suite and the separate Lean `libtorch_sdpa_test` target.
- Run the GPU suite after changes to native exports, ownership, or numerics.
