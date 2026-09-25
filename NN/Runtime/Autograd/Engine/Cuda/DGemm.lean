/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

LibTorch FFI: host `FloatArray` DGEMM (FP64 / Lean `Float`).
Implementation: `csrc/libtorch/blas.cpp` (ATen `matmul`).

The FP32 matmul path lives in `Engine.Cuda.Kernels` as `Buffer.bmm`, which uses CUDA buffers and
ATen matrix multiplication.
-/

module

/-!
# CUDA DGEMM FFI

Foreign-function declaration for host `FloatArray` FP64 matrix multiplication. The CUDA build
uploads the arrays to the selected device, calls ATen `matmul`, and downloads the result.
Without LibTorch the call fails with a rebuild hint. The float32 buffer matmul path lives in
`NN.Runtime.Autograd.Engine.Cuda.Kernels`.

This lives in its own small module instead of `Cuda.Kernels`:

- `Cuda.Kernels` is the float32 `Cuda.Buffer` surface used by the CUDA eager tape.
- `DGemm` is a host `FloatArray → FloatArray` bridge for Lean `Float` tensors and the
  `FastKernels` CPU-tape acceleration path.

-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

/-- FP64 matrix multiplication on row-major host arrays: `A` is `m × n`, `B` is `n × p`,
and the result is `m × p`.

Inputs and outputs use binary64, as does Lean `Float`. ATen determines the native accumulation
order; the interface does not promise bitwise equality with a Lean reduction. -/
@[extern "torchlean_dgemm_cuda"]
opaque torchleanDgemmCuda (A : @& FloatArray) (B : @& FloatArray)
                          (m n p : UInt32) : FloatArray

end Cuda
end Autograd
end Runtime
