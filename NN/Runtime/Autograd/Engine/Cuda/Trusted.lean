/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Trusted boundary for CUDA FFI.

Why this file exists:
- TorchLean’s repo policy forbids axioms in general library code.
- The CUDA runtime types are produced by external C/CUDA code, so we need a small trusted bridge
  to make them usable in compiled Lean code.

Everything in this module should be treated as part of the "FFI trust base".
-/

module

/-!
# Trusted CUDA Runtime Boundary

This module contains the opaque CUDA buffer type used by the native runtime. Buffers are created by
explicit FFI allocation/copy functions. The nonemptiness witness below is only what Lean needs to
declare extern functions returning `Buffer`; it is not a default CUDA allocation and should not be
used as one.

The rest of this docstring is the map from native translation units to the Lean modules that call
them. DocGen documents Lean modules, not C or CUDA files, so the map lives here, on the module that
is the trust boundary, rather than in a separate source browser.

## Trust boundary

The CUDA backend is a validated implementation of TorchLean's float32 eager runtime. Lean does not
prove the compiled CUDA binary correct. The trusted pieces include:

- the CUDA compiler and runtime;
- GPU hardware, cuBLAS, cuFFT, and libdevice;
- the C/CUDA FFI boundary and Lean external-object finalizers;
- platform behavior such as atomics, floating-point contraction, and library math.

TorchLean's proof layer CUDA contract therefore lives one level up: Lean states pure kernel specs,
float32 agreement assumptions, and graph-level semantics; tests validate that the native backend
agrees with CPU stubs and reference cases on the supported path.

## Native source groups

- `csrc/cuda/common/torchlean_cuda_buffer.h`
  Shared boxed-buffer ABI, size guards, deterministic-reduction toggles, and helper declarations.
  Lean side modules: `NN.Runtime.Autograd.Engine.Cuda.Trusted`,
  `NN.Runtime.Autograd.Engine.Cuda.Buffer`.

- `csrc/cuda/common/torchlean_cuda_common.h`
  CUDA error checking helpers. Failures cross the FFI boundary as Lean internal panics.

- `csrc/cuda/common/torchlean_cublas_common.h`
  Thread-local cuBLAS handle management and cuBLAS error checking for matrix kernels.
  Lean side modules: `NN.Runtime.Autograd.Engine.Cuda.Kernels`,
  `NN.Runtime.Autograd.Engine.Cuda.DGemm`.

- `csrc/cuda/common/torchlean_cuda_deterministic_reductions_env.h`
  Environment-variable parser for deterministic CUDA reduction mode.

- `csrc/cuda/common/torchlean_cuda_rng_common.h`
  Shared SplitMix64 stream used by CUDA kernels and CPU stubs. The contract fixes the
  low 32 bits of `splitmix64(key + i)` so seeded CPU-stub and CUDA runs match.

- `csrc/cuda/tensor/torchlean_cuda_tensor.cu`
  Device allocation, host/device copies, scalar elementwise kernels, reductions, seeded RNG, and
  buffer release hooks.
  Lean side module: `NN.Runtime.Autograd.Engine.Cuda.Buffer`.

- `csrc/cuda/tensor/torchlean_cuda_tensor_stub.c`
  Portable CPU implementation of the tensor-buffer FFI symbols used when TorchLean is built without
  CUDA.

- `csrc/cuda/kernels/torchlean_cuda_kernels.cu`
  Broadcasting, reductions over axes, gather/scatter, transpose, batched matmul, selective scan,
  attention helpers, FFT, and fused spectral convolution kernels.
  Lean side modules: `NN.Runtime.Autograd.Engine.Cuda.Kernels`,
  `NN.Runtime.Autograd.Engine.Cuda.Ops`, `NN.Runtime.Autograd.Engine.Cuda.Tape`.

- `csrc/cuda/kernels/torchlean_cuda_kernels_stub.c`
  Portable CPU mirror of the general tensor-kernel FFI surface.

- `csrc/cuda/conv_pool/torchlean_cuda_conv_pool_common.h`
  Shared convolution/pooling shape arithmetic and rank limits.

- `csrc/cuda/conv_pool/torchlean_cuda_conv_pool.cu`
  2D and N-D convolution, transposed convolution, max/average/smooth-max pooling, and backward
  kernels.
  Lean side module: `NN.Runtime.Autograd.Engine.Cuda.ConvPool`.

- `csrc/cuda/conv_pool/torchlean_cuda_conv_pool_stub.c`
  Portable CPU mirror of the convolution and pooling FFI surface.

- `csrc/cuda/blas/torchlean_dgemm_cuda.cu`
  Double-precision Lean `FloatArray` matrix multiplication through cuBLAS.
  Lean side module: `NN.Runtime.Autograd.Engine.Cuda.DGemm`.

- `csrc/cuda/blas/torchlean_dgemm_cuda_stub.c`
  Portable CPU mirror of the DGEMM FFI symbol.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

/--
Opaque handle to a contiguous float32 buffer (CUDA device memory when built with `-K cuda=true`,
otherwise a CPU stub buffer).

Implementation:
- CUDA: `csrc/cuda/tensor/torchlean_cuda_tensor.cu`
- CPU stub (default `lake build`): `csrc/cuda/tensor/torchlean_cuda_tensor_stub.c`
-/
opaque BufferImpl : NonemptyType.{0}

/--
Runtime representation used for native CUDA buffer handles.

The `NonemptyType` wrapper is Lean's standard representation for external resources: it gives
extern declarations a nonempty result type while preserving reference-counting information in
compiled code. The underlying value is still created only by the native buffer constructors.
-/
def Buffer : Type := BufferImpl.val

instance : Nonempty Buffer := BufferImpl.property

end Cuda
end Autograd
end Runtime
