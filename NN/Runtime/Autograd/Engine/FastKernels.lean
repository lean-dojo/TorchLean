/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Runtime.Autograd.Engine.Cuda.Convert
public import NN.Runtime.Autograd.Engine.Cuda.DGemm
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.Tape

/-!
# Matmul Reference and cuBLAS Routines

Low-level matrix-multiplication routines used to compare the CPU reference implementation with
the explicit FP32 and FP64 cuBLAS paths. User-facing execution selects kernels through the runtime
device and backend profile; this module does not define a separate execution mode.
-/

@[expose] public section


namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace FastKernels

/--
Precision selector for GPU-backed fast matmul over Lean `Float` tensors.

- `.fp32` routes through `Cuda.Buffer` and cuBLAS SGEMM, matching the precision used by the eager
  CUDA tensor-buffer path.
- `.fp64` routes through the host `FloatArray` DGEMM bridge and cuBLAS DGEMM, preserving Lean
  `Float` precision for matmul-only research paths.
-/
inductive CublasPrecision where
  | fp32
  | fp64
deriving Repr, DecidableEq

/--
Fast (runtime-only) 2D matmul kernel.

This is a tight triple loop over `Fin` indices, reading both operands directly through `Spec.get2`.
It is the CPU reference the cuBLAS paths are compared against; no runtime bounds assertion is
involved.
-/
def matmulReference {α : Type} [TorchLean.Storage α] [Context α]
    {m n p : Nat}
    (a : Tensor α [m, n])
    (b : Tensor α [n, p]) :
    Tensor α [m, p] :=
  Tensor.matrix fun i k =>
    Fin.foldl n (fun acc j => acc + Spec.get2 a i j * Spec.get2 b j k) (0 : α)

namespace Cuda

namespace Internal

/-- Convert an FFI dimension to `UInt32`, failing before the native call on overflow. -/
def natToU32 (n : Nat) : Result UInt32 :=
  Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked n

/-- Unflatten a kernel result, failing if the native call returned an unexpected element count. -/
def unflatten {m p : Nat} (what : String) (flat : FloatArray) : Result (Tensor Float [m, p]) :=
  match Runtime.Autograd.Cuda.Convert.unflattenFloat? (s := .dim m (.dim p .scalar)) flat with
  | some tensor => pure tensor
  | none =>
      throw s!"autograd: fast matmul: {what} returned {flat.size} elements, expected {m * p}"

end Internal

/-- 2D matmul forward via cuBLAS DGEMM (`torchlean_dgemm_cuda` / `Cuda.torchleanDgemmCuda`). -/
def matmulCublas64 {m n p : Nat}
    (a : Tensor Float [m, n])
    (b : Tensor Float [n, p]) :
    Result (Tensor Float [m, p]) := do
  let flatA := Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim m (.dim n .scalar)) a
  let flatB := Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim n (.dim p .scalar)) b
  let flatC := Runtime.Autograd.Cuda.torchleanDgemmCuda flatA flatB
    (← Internal.natToU32 m) (← Internal.natToU32 n) (← Internal.natToU32 p)
  Internal.unflatten "DGEMM" flatC

/--
2D matmul forward via the float32 CUDA buffer path.

This path uploads Lean `Float` values to `Cuda.Buffer` (rounding to float32), calls the existing
`Buffer.bmm` SGEMM implementation with `batch = 1`, then downloads the float32 result back to Lean
`Float`.
-/
def matmulCublas32 {m n p : Nat}
    (a : Tensor Float [m, n])
    (b : Tensor Float [n, p]) :
    Result (Tensor Float [m, p]) := do
  let aBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim m (.dim n .scalar)) a)
  let bBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim n (.dim p .scalar)) b)
  let cBuf := Runtime.Autograd.Cuda.Buffer.bmm aBuf bBuf
    1 (← Internal.natToU32 m) (← Internal.natToU32 n) (← Internal.natToU32 p)
  Internal.unflatten "SGEMM" (Runtime.Autograd.Cuda.Buffer.toFloatArray cBuf)

/-- Dispatch to the requested GPU matmul precision. -/
def matmulCublas (precision : CublasPrecision) {m n p : Nat}
    (a : Tensor Float [m, n])
    (b : Tensor Float [n, p]) :
    Result (Tensor Float [m, p]) :=
  match precision with
  | .fp32 => matmulCublas32 (m := m) (n := n) (p := p) a b
  | .fp64 => matmulCublas64 (m := m) (n := n) (p := p) a b

end Cuda

end FastKernels

end Autograd
end Runtime
