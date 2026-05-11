/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
public import NN.Runtime.Autograd.Engine.Cuda.ConvPool
public import NN.Runtime.Autograd.Engine.Cuda.DGemm
public import NN.Runtime.Autograd.Engine.Cuda.Float32Contract
public import NN.Runtime.Autograd.Engine.Cuda.KernelSpec
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Runtime.Autograd.Engine.Cuda.Shape
public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Engine.Cuda.Trusted

/-!
# CUDA native source map

This module is the DocGen-facing home for TorchLean's native CUDA/C runtime notes.

DocGen documents Lean modules, not C or CUDA translation units. Instead of publishing a separate
Jekyll source browser, TorchLean keeps the native-source map here, beside the Lean FFI modules that
call those symbols. The actual native source remains in `csrc/cuda`; this page tells readers which
files form the trusted backend boundary and which Lean modules expose them.

## Trust boundary

The CUDA backend is a validated implementation of TorchLean's float32 eager runtime. Lean does not
prove the compiled CUDA binary correct. The trusted pieces include:

- the CUDA compiler and runtime;
- GPU hardware, cuBLAS, cuFFT, and libdevice;
- the C/CUDA FFI boundary and Lean external-object finalizers;
- platform behavior such as atomics, floating-point contraction, and library math.

TorchLean's proof-side CUDA contract therefore lives one level up: Lean states pure kernel specs,
float32 agreement assumptions, and graph-level semantics; tests validate that the native backend
agrees with CPU stubs and reference cases on the supported path.

## Native source groups

- `csrc/cuda/common/torchlean_cuda_buffer.h`
  Shared boxed-buffer ABI, size guards, deterministic-reduction toggles, and helper declarations.
  Lean-facing modules: `NN.Runtime.Autograd.Engine.Cuda.Trusted`,
  `NN.Runtime.Autograd.Engine.Cuda.Buffer`.

- `csrc/cuda/common/torchlean_cuda_common.h`
  CUDA error checking helpers. Failures cross the FFI boundary as Lean internal panics.

- `csrc/cuda/common/torchlean_cublas_common.h`
  Thread-local cuBLAS handle management and cuBLAS error checking for matrix kernels.
  Lean-facing modules: `NN.Runtime.Autograd.Engine.Cuda.Kernels`,
  `NN.Runtime.Autograd.Engine.Cuda.DGemm`.

- `csrc/cuda/common/torchlean_cuda_deterministic_reductions_env.h`
  Environment-variable parser for deterministic CUDA reduction mode.

- `csrc/cuda/common/torchlean_cuda_rng_common.h`
  Shared SplitMix64 stream used by CUDA kernels and CPU stubs. The contract intentionally fixes the
  low 32 bits of `splitmix64(key + i)` so seeded CPU-stub and CUDA runs match.

- `csrc/cuda/tensor/torchlean_cuda_tensor.cu`
  Device allocation, host/device copies, scalar elementwise kernels, reductions, seeded RNG, and
  buffer release hooks.
  Lean-facing module: `NN.Runtime.Autograd.Engine.Cuda.Buffer`.

- `csrc/cuda/tensor/torchlean_cuda_tensor_stub.c`
  Portable CPU implementation of the tensor-buffer FFI symbols used when TorchLean is built without
  CUDA.

- `csrc/cuda/kernels/torchlean_cuda_kernels.cu`
  Broadcasting, reductions over axes, gather/scatter, transpose, batched matmul, selective scan,
  attention helpers, FFT, and fused spectral convolution kernels.
  Lean-facing modules: `NN.Runtime.Autograd.Engine.Cuda.Kernels`,
  `NN.Runtime.Autograd.Engine.Cuda.Ops`, `NN.Runtime.Autograd.Engine.Cuda.Tape`.

- `csrc/cuda/kernels/torchlean_cuda_kernels_stub.c`
  Portable CPU mirror of the general tensor-kernel FFI surface.

- `csrc/cuda/conv_pool/torchlean_cuda_conv_pool_common.h`
  Shared convolution/pooling shape arithmetic and rank limits.

- `csrc/cuda/conv_pool/torchlean_cuda_conv_pool.cu`
  2D and N-D convolution, transposed convolution, max/average/smooth-max pooling, and backward
  kernels.
  Lean-facing module: `NN.Runtime.Autograd.Engine.Cuda.ConvPool`.

- `csrc/cuda/conv_pool/torchlean_cuda_conv_pool_stub.c`
  Portable CPU mirror of the convolution and pooling FFI surface.

- `csrc/cuda/blas/torchlean_dgemm_cuda.cu`
  Double-precision Lean `FloatArray` matrix multiplication through cuBLAS.
  Lean-facing module: `NN.Runtime.Autograd.Engine.Cuda.DGemm`.

- `csrc/cuda/blas/torchlean_dgemm_cuda_stub.c`
  Portable CPU mirror of the DGEMM FFI symbol.

## What to read next

- `NN.Runtime.Autograd.Engine.Cuda.Float32Contract` states the float32 agreement assumptions.
- `NN.Runtime.Autograd.Engine.Cuda.KernelSpec` gives pure Lean specs for important CUDA-kernel
  algorithms.
- `NN.Tests.Runtime.Cuda.Stress` and `NN.Tests.Runtime.Cuda.Fft` exercise the native/stub parity and
  FFT-backed spectral convolution path.
-/

@[expose] public section

namespace NN.Runtime.Autograd.Engine.Cuda.NativeSources

/--
DocGen anchor for the CUDA native-source map.

The value is intentionally small; the module documentation above is the useful content. Keeping a
public declaration here ensures DocGen emits a navigable page for this map alongside the Lean CUDA
FFI modules.
-/
def docAnchor : String := "TorchLean CUDA native source map"

end NN.Runtime.Autograd.Engine.Cuda.NativeSources
