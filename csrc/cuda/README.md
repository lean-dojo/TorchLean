# TorchLean CUDA Runtime Sources

This directory contains TorchLean's trusted native runtime boundary for CUDA and CPU stub builds.
Lean checks shapes and dispatches to these symbols, but memory safety, kernel launch behavior, and
float32 arithmetic are outside Lean's kernel.

The public website includes a generated source browser for this directory. Regenerate it with:

```bash
lake build cudaDocs
```

That target writes `home_page/cuda/`, links native files to Lean FFI modules under
`NN/Runtime/Autograd/Engine/Cuda`, and complements `lake build NN:docs`.

## Layout

- `common/`: shared Lean runtime helpers, CUDA error checking, cuBLAS helpers, RNG helpers, and
  environment parsing for deterministic reductions.
- `tensor/`: `Cuda.Buffer` allocation, finalization, host/device copies, scalar elementwise kernels,
  reductions, and seeded RNG buffers.
- `kernels/`: tensor views, broadcast/reduce, gather/scatter, matmul/bmm, and indexing kernels.
- `conv_pool/`: convolution, transposed convolution, max, average, and smooth max pooling, plus
  their backward passes. The 2D specializations live beside the N dimensional kernels for the common
  fast path.
- `blas/`: double-precision GEMM bridge backed by cuBLAS, plus its portable CPU stub.

Each CUDA implementation has a matching `_stub.c` file with the same exported symbols. The stubs are
used by default so `lake build` works without a CUDA toolkit. Real CUDA builds are selected with:

```bash
lake build -R -K cuda=true -K cuda_home=/usr/local/cuda
```

## CUDA Graph Status

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

This reduces accidental lifetime extension from Lean external object finalizers, but it does not
turn eager execution into a static CUDA graph. A real CUDA Graph backend should be added as a
separate runtime layer with:

- persistent device buffers for parameters, activations, and gradients,
- no `cudaMalloc`/host shape decisions inside the captured region,
- graph specialization keyed by static model and input shapes,
- explicit invalidation when shapes, masks, or parameter layouts change,
- regression tests comparing captured replay against the eager CUDA tape for every supported op.

Until that layer exists, `--backend compiled` should be read as TorchLean's proof/SSA graph backend,
not as CUDA Graph execution.

For a CUDA training smoke test:

```bash
lake exe torchlean gpt_adder --steps 500 --log-every 100
```

## Sanitizer Harness

The CUDA sources are a trusted native boundary: Lean checks shapes and proof contracts, but it
cannot prove that a `.cu` kernel avoids every invalid access or race. To test that boundary, run the
Lean driven CUDA suite under NVIDIA Compute Sanitizer:

```bash
scripts/checks/cuda_sanitize_tests.sh
scripts/checks/cuda_sanitize_tests.sh --all-tools
scripts/checks/cuda_sanitize_tests.sh --cuda-home /usr/local/cuda --tool memcheck
```

The default tool is `memcheck`. `--all-tools` additionally runs `racecheck`, `initcheck`, and
`synccheck`. This is separate from ordinary `lake test` because sanitizer runs are much slower and
require a real CUDA installation.

For performance work, pair the correctness suite with NVIDIA Nsight Systems for end-to-end runtime
traces and Nsight Compute for individual kernel profiles. Those tools are not pass/fail tests, so
they stay outside the default CI gate. The helper below writes reports under `data/profiles/cuda/`,
which is treated as local output:

```bash
scripts/checks/cuda_profile_tests.sh
scripts/checks/cuda_profile_tests.sh --both
scripts/checks/cuda_profile_tests.sh --compute
```

Nsight Compute can be slow on the full suite because it profiles kernels in detail. For focused
kernel work, pass a smaller executable with `--target` or forward test arguments after `--`.

## CUDA Test Matrix

The CUDA regression suite lives in `NN/Tests/Runtime/Cuda`.  The tests compare the Lean CPU eager
tape against the CUDA eager tape on small deterministic examples.  In the default build those same
externs route through CPU stubs, which keeps ordinary CI useful without a GPU.  With `-K cuda=true`
they exercise the real CUDA/cuBLAS/cuFFT implementations.

Run the full Lean test executable through Lake:

```bash
lake build nn_tests_suite && lake exe nn_tests_suite
lake build -R -K cuda=true nn_tests_suite && .lake/build/bin/nn_tests_suite
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
| `NN/Tests/Runtime/Cuda/GatherScatter.lean` | Vector/row gather and scatter-add behavior, including gradients. |
| `NN/Tests/Runtime/Cuda/DeterministicReductions.lean` | Exact repeatability when deterministic reduction mode replaces atomic accumulation paths. |
| `NN/Tests/Runtime/Cuda/SelectiveScan.lean` | Diagonal selective-scan buffer primitives used by the Mamba/SSM runtime path. |
| `NN/Tests/Runtime/Cuda/PositionalEncoding.lean` | Sinusoidal positional encodings and RoPE/rotary embedding kernels. |
| `NN/Tests/Runtime/Cuda/MatmulBmm.lean` | `matmul`, `bmm`, and explicit fp32/fp64 fast-kernel dispatch. |
| `NN/Tests/Runtime/Cuda/Fft.lean` | Packed real FFT, inverse FFT, spectral convolution, and finite-difference gradient checks. |
| `NN/Tests/Runtime/Cuda/ViewsBroadcastReduce.lean` | Reshape, transpose, rank-3 permutations, broadcast, reduce-sum/mean, and empty-axis behavior. |
| `NN/Tests/Runtime/Cuda/LinearMseConcatSliceGather.lean` | Linear layer, MSE loss, vector concat/slice, scalar gather, row gather, and gradients. |
| `NN/Tests/Runtime/Cuda/Stress.lean` | RNG determinism, explicit buffer release, duplicate-parent gradient accumulation, large buffers, reductions, and cuBLAS rectangular matmul. |
| `NN/Tests/Runtime/Cuda/Suite.lean` | The unified entrypoint imported by the repository-level test suite. |

When adding a CUDA symbol, update this matrix and add at least one CPU-stub/real-CUDA parity test.
If the symbol participates in autograd, test both the forward value and the relevant VJP/gradient
buffers.  If it uses atomics, also decide whether deterministic mode needs a separate test.

## Review Notes

- Padded max pooling follows the TorchLean spec and PyTorch convention: cells outside the input are
  ignored (equivalently `-inf`), with row major argmax tie breaking.
- Some accumulation paths use `atomicAdd`; enable deterministic reductions when exact repeatability
  is more important than speed.
- The native FlashAttention symbols in `kernels/` are correctness focused fused forward/VJP kernels
  over split heads. They refine the contract in `NN/Spec/Layers/FlashAttention.lean`; they are not
  the IO tiled Dao AILab kernel.
  `NN/Tests/Runtime/Cuda/Attention.lean` compares the fused path against the composed
  `bmm -> mask -> softmax -> bmm` path.
- Keep broad mechanical file moves separate from semantic kernel changes when possible, then run both
  CPU stub and real CUDA test paths.
