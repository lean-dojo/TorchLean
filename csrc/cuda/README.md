# TorchLean CUDA ABI and CPU stubs

This directory contains the shared Lean C ABI headers and portable CPU stubs. The CUDA
implementation lives in [`../libtorch/`](../libtorch/) and uses the selected LibTorch SDK's ATen
operations. Lean checks shapes and dispatches through the existing symbols; native memory
safety, SDK behavior, and floating-point execution remain outside Lean's kernel.

## Layout

- `common/`: shared buffer representation, Lean object/size helpers, and runtime control parsing.
- `tensor/`, `kernels/`, `conv_pool/`, `blas/`: the CPU implementations of the buffer exports.

The complete CUDA backend is built as one shared library:

| LibTorch source | Responsibility |
| --- | --- |
| `runtime.cpp` | Buffer ownership, allocation, transfers, RNG, and runtime controls. |
| `elementwise.cpp` | Elementwise arithmetic, activations, and optimizer buffer operations. |
| `kernels.cpp` | Views, indexing, reductions, normalization, matrix operations, and FFT. |
| `conv_pool.cpp` | Convolution and pooling forward/backward operations. |
| `attention.cpp` | Attention forward/backward operations and retained SDK contexts. |
| `blas.cpp` | Double-precision matrix multiplication bridge. |

`TORCHLEAN_LIBTORCH` selects the ATen-backed buffer representation in the shared headers.
The Lean tape owns differentiation; native calls use a no-grad guard. The
CPU evaluation dynamic library is loaded for native CPU `#eval` calls. GPU tests run as
compiled executables.

## Build selection

`scripts/lake.sh build` selects the default `pureLean`/`portableCPU` build and CPU stubs,
without an SDK or toolkit. `cuda=true` requires the complete LibTorch CUDA backend.

Run migration compilation and execution only in the cluster. A full SDK contains `include/`,
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
the direct six-unit C++ build command.

## Execution and memory

LibTorch supplies PyTorch's C++ libraries; ATen is the tensor operation layer used by this
backend. A CUDA buffer reaches C++ as a Lean external object owning an `at::Tensor`. The bridge
unwraps that tensor, calls ATen, and returns another owned buffer through the Lean C ABI.
These tensor calls run without a Python interpreter. An installed CUDA-enabled PyTorch package
can supply the SDK at build and execution time.

For a linear layer, Lean sends matrix multiplication and bias addition to the buffer API,
then records the result, parents, and backward rule on its runtime tape. During backward,
Lean traverses the tape and calls the corresponding gradient operations. Each native call runs under
`at::NoGradGuard`, so LibTorch does not record another autograd graph. Some operations use
explicit SDK backward kernels; attention also retains the forward context needed by its
backward call.

The build selects the implementation behind those buffer symbols. The default build uses the
portable CPU stubs and needs no LibTorch SDK. A `cuda=true` build links
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
tape against the CUDA eager tape on small examples. In the default build those same externs
route through CPU stubs, which keeps ordinary CI useful without a GPU. With `-Kcuda=true`
they exercise LibTorch. CPU hosted CI does not validate GPU execution.

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
| `NN/Tests/Runtime/Cuda/Attention.lean` | Multi-head attention and fused attention parity against composed operations. |
| `NN/Tests/Runtime/Cuda/ConvPool.lean` | 2D and N-D convolution, max pool, average pool, smooth max pool, padded max-pool edge cases. |
| `NN/Tests/Runtime/Cuda/ConvTranspose.lean` | 2D and 3D transposed convolution forward and backward. |
| `NN/Tests/Runtime/Cuda/GatherScatter.lean` | Rank-one and row gather/scatter-add behavior, including gradients. |
| `NN/Tests/Runtime/Cuda/DeterministicReductions.lean` | Repeatability under the deterministic reduction control. |
| `NN/Tests/Runtime/Cuda/SelectiveScan.lean` | Diagonal selective-scan buffer primitives used by the Mamba/SSM runtime path. |
| `NN/Tests/Runtime/Cuda/PositionalEncoding.lean` | Sinusoidal positional encodings and RoPE/rotary embedding kernels. |
| `NN/Tests/Runtime/Cuda/MatmulBmm.lean` | `matmul`, `bmm`, and explicit fp32/fp64 dispatch. |
| `NN/Tests/Runtime/Cuda/Fft.lean` | Packed real FFT, inverse FFT, spectral convolution, and finite-difference gradient checks. |
| `NN/Tests/Runtime/Cuda/ViewsBroadcastReduce.lean` | Reshape, transpose, rank-3 permutations, broadcast, reduce-sum/mean, and empty-axis behavior. |
| `NN/Tests/Runtime/Cuda/LinearMseConcatSliceGather.lean` | Linear layer, MSE loss, vector concat/slice, scalar gather, row gather, and gradients. |
| `NN/Tests/Runtime/Cuda/Stress.lean` | RNG determinism, explicit release, duplicate-parent gradient accumulation, large buffers, reductions, and rectangular matmul. |
| `NN/Tests/Runtime/Cuda/Suite.lean` | The unified entrypoint imported by the repository-level test suite. |

When adding a CUDA symbol, update this matrix and add at least one CPU-stub/real-CUDA parity test.
If the symbol participates in autograd, test both the forward value and the relevant VJP/gradient
buffers.  If it uses atomics, also decide whether deterministic mode needs a separate test.

The separate [elementwise C++ harness](../libtorch/tests/elementwise/README.md) links the
production backend and consumes Lean binary32 cases. Its wrapper is
`scripts/checks/cuda_float32_parity.sh --libtorch-home PATH --backend-library PATH --lean-prefix PATH`.
It checks exact finite results and signed zeros, reports NaN encoding differences, and includes
staged Adam and no-autograd regressions. A different SDK needs its own audit and regression
results.

## Review Notes

- SDK upgrades can change numerical behavior. Compiler/link compatibility does not establish
  agreement with TorchLean's floating-point contracts; retain the exact SDK/build manifest with
  test results.
- Deterministic controls request supported deterministic SDK algorithms. Repeatability on the
  tested SDK/device does not imply bitwise agreement across releases, devices, or algorithms.
- Attention uses SDK operations with retained forward context for backward. The attention tests
  compare this path with composed `bmm -> mask -> softmax -> bmm` operations.
  The focused `libtorch_sdpa_test` target links the entire numerical backend.
- Run both CPU stub and real GPU checks after changes to native exports, ownership, or numerics.
