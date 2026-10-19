/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Runtime.Autograd.Engine.LibTorch.Convert
public import NN.Runtime.Autograd.Engine.LibTorch.Kernels
public import NN.Runtime.Autograd.Engine.LibTorch.Tape

/-!
# Native matrix multiplication test support

Adapters for comparing the native matrix product in binary32 and binary64 against
`Tensor.matmul`. They call the buffer FFI directly, so the tests exercise both dtypes.
-/

@[expose] public section


namespace Tests.Cuda.MatmulSupport

open Runtime.Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

/--
Precision selector for LibTorch matmul over Lean `Float` tensors.

- `.fp32` rounds the inputs to binary32 buffers, the precision of the eager CUDA tape.
- `.fp64` keeps the binary64 element type of Lean `Float`.
-/
inductive MatmulPrecision where
  | fp32
  | fp64
deriving Repr, DecidableEq

/-- The buffer dtype of a precision. -/
def MatmulPrecision.dtype : MatmulPrecision → Runtime.Autograd.LibTorch.Dtype
  | .fp32 => .float32
  | .fp64 => .float64

open Runtime.Autograd.LibTorch (SomeBuffer Buffer)

namespace Internal

/-- Unflatten a kernel result, failing if the native call returned an unexpected element count. -/
def unflatten {m p : Nat} (what : String) (flat : FloatArray) : Result (Tensor Float [m, p]) :=
  match Runtime.Autograd.LibTorch.Convert.unflattenFloat? (s := .dim m (.dim p .scalar)) flat with
  | some tensor => pure tensor
  | none =>
      throw s!"autograd: native matmul: {what} returned {flat.size} elements, expected {m * p}"

end Internal

/--
Matrix multiplication at the requested LibTorch precision: upload both operands as buffers of the
precision's dtype, call ATen `bmm` with one matrix pair, and download the result to Lean `Float`.
Rejects dimensions outside the FFI's `UInt32` range and unexpected result sizes.
-/
def matmul (precision : MatmulPrecision) {m n p : Nat}
    (a : Tensor Float [m, n])
    (b : Tensor Float [n, p]) :
    Result (Tensor Float [m, p]) := do
  let upload {r c : Nat} (x : Tensor Float [r, c]) :=
    Buffer.ofFloatArray
      (LibTorch.Convert.flattenFloat (s := .dim r (.dim c .scalar)) x) precision.dtype
  let cBuf := Buffer.bmm (upload a) (upload b)
    1 (← SomeBuffer.natToU32Checked m) (← SomeBuffer.natToU32Checked n)
    (← SomeBuffer.natToU32Checked p)
  Internal.unflatten s!"ATen {repr precision} bmm" (Buffer.toFloatArray cBuf)

end Tests.Cuda.MatmulSupport
