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
# Matmul Reference and LibTorch Routines

Low-level matrix-multiplication routines used to compare the CPU reference implementation with
the explicit FP32 and FP64 LibTorch paths. User-facing execution selects kernels through the runtime
device and backend profile; this module does not define a separate execution mode.
-/

@[expose] public section


namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace FastKernels

/--
Precision selector for LibTorch matmul over Lean `Float` tensors.

- `.fp32` routes through `Cuda.Buffer` and ATen `bmm`, matching the precision used by the eager
  CUDA tensor-buffer path.
- `.fp64` routes through the host `FloatArray` bridge and ATen `matmul`, preserving the binary64
  element type of Lean `Float`.
-/
inductive MatmulPrecision where
  | fp32
  | fp64
deriving Repr, DecidableEq

/--
Fast (runtime-only) 2D matmul kernel.

This is a tight triple loop over `Fin` indices, reading both operands directly through `Spec.get2`.
It is the CPU reference the LibTorch paths are compared against; no runtime bounds assertion is
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

open Runtime.Autograd.Cuda (AnyBuffer)

namespace Internal

/-- Unflatten a kernel result, failing if the native call returned an unexpected element count. -/
def unflatten {m p : Nat} (what : String) (flat : FloatArray) : Result (Tensor Float [m, p]) :=
  match Runtime.Autograd.Cuda.Convert.unflattenFloat? (s := .dim m (.dim p .scalar)) flat with
  | some tensor => pure tensor
  | none =>
      throw s!"autograd: fast matmul: {what} returned {flat.size} elements, expected {m * p}"

end Internal

/--
Matrix multiplication at the requested LibTorch precision.

The `.fp32` path rounds inputs to float32 buffers, calls ATen `bmm` with one matrix pair,
and downloads the result to Lean `Float`. The `.fp64` path preserves binary64 through the
`FloatArray` bridge and ATen `matmul`. Both reject dimensions outside the FFI's `UInt32` range
and unexpected result sizes.
-/
def matmulLibTorch (precision : MatmulPrecision) {m n p : Nat}
    (a : Tensor Float [m, n])
    (b : Tensor Float [n, p]) :
    Result (Tensor Float [m, p]) :=
  match precision with
  | .fp32 => do
      let aBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
        (Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim m (.dim n .scalar)) a)
      let bBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
        (Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim n (.dim p .scalar)) b)
      let cBuf := Runtime.Autograd.Cuda.Buffer.bmm aBuf bBuf
        1 (← AnyBuffer.natToU32Checked m) (← AnyBuffer.natToU32Checked n)
        (← AnyBuffer.natToU32Checked p)
      Internal.unflatten "ATen float32 bmm" (Runtime.Autograd.Cuda.Buffer.toFloatArray cBuf)
  | .fp64 => do
      let flatA := Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim m (.dim n .scalar)) a
      let flatB := Runtime.Autograd.Cuda.Convert.flattenFloat (s := .dim n (.dim p .scalar)) b
      let flatC := Runtime.Autograd.Cuda.torchleanDgemmCuda flatA flatB
        (← AnyBuffer.natToU32Checked m) (← AnyBuffer.natToU32Checked n)
        (← AnyBuffer.natToU32Checked p)
      Internal.unflatten "ATen float64 matmul" flatC

end Cuda

end FastKernels

end Autograd
end Runtime
