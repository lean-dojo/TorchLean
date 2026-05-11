/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA/cuBLAS FFI: host `FloatArray` DGEMM (FP64 / Lean `Float`).
Implementation: `csrc/cuda/blas/torchlean_dgemm_cuda.cu` (`cublasDgemm`).

The FP32 matmul path lives in `Engine.Cuda.Kernels` as `Buffer.bmm`, which uses the CUDA buffer
stack and cuBLAS SGEMM.
-/

module

/-!
# CUDA DGEMM FFI

Foreign-function declaration for the host `FloatArray` FP64 matrix multiply path backed by
`cublasDgemm` when CUDA is enabled and by a CPU stub otherwise. The float32 buffer matmul path lives
in `NN.Runtime.Autograd.Engine.Cuda.Kernels`.

This intentionally stays in its own small module instead of `Cuda.Kernels`:

- `Cuda.Kernels` is the float32 `Cuda.Buffer` surface used by the CUDA eager tape.
- `DGemm` is a host `FloatArray → FloatArray` bridge for Lean `Float` tensors and the
  `FastKernels` CPU-tape acceleration path.
- It links through a separate native archive (`torchlean_dgemm_cuda`) because the implementation
  is a cuBLAS-DGEMM wrapper rather than a tensor-buffer kernel.

-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

@[extern "torchlean_dgemm_cuda"]
opaque torchleanDgemmCuda (A : FloatArray) (B : FloatArray)
                          (m n p : UInt32) : FloatArray

end Cuda
end Autograd
end Runtime
