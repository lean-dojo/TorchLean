/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops.Core

/-!
# CUDA Tape Operations: Matrix and Loss Nodes
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-!
## Linear algebra
-/

namespace Internal

/-- Backend matrix kernel for a single pair of matrices. -/
@[inline] def matmul {m n p : Nat} (t : Tape SomeBuffer) (aId bId : Nat) : Result (Tape SomeBuffer
    × Nat) := do
  let m32 ← SomeBuffer.natToU32Checked m
  let n32 ← SomeBuffer.natToU32Checked n
  let p32 ← SomeBuffer.natToU32Checked p
  let one32 : UInt32 := 1
  let leftShape : Shape := [m, n]
  let rightShape : Shape := [n, p]
  let outShape : Shape := [m, p]
  binary (t := t) "matmul" aId bId leftShape rightShape outShape
    (forward := fun a b => Buffer.bmm a b one32 m32 n32 p32)
    (backward := fun a b dLdy =>
      let dA := Buffer.bmmRightTranspose dLdy b one32 m32 p32 n32
      let dB := Buffer.bmmLeftTranspose a dLdy one32 n32 m32 p32
      (dA, dB))

/-- Backend matrix kernel over a flattened leading shape. -/
@[inline]
def matmulFlattened {batch m n p : Nat} (t : Tape SomeBuffer) (aId bId : Nat) : Result
    (Tape SomeBuffer × Nat) := do
  let b32 ← SomeBuffer.natToU32Checked batch
  let m32 ← SomeBuffer.natToU32Checked m
  let n32 ← SomeBuffer.natToU32Checked n
  let p32 ← SomeBuffer.natToU32Checked p
  let σ₁ : Shape := .dim batch (.dim m (.dim n .scalar))
  let σ₂ : Shape := .dim batch (.dim n (.dim p .scalar))
  let τ : Shape := .dim batch (.dim m (.dim p .scalar))
  binary (t := t) "bmm" aId bId σ₁ σ₂ τ
    (forward := fun a b => Buffer.bmm a b b32 m32 n32 p32)
    (backward := fun a b dLdy =>
      let dA := Buffer.bmmRightTranspose dLdy b b32 m32 p32 n32
      let dB := Buffer.bmmLeftTranspose a dLdy b32 n32 m32 p32
      (dA, dB))

end Internal

/-!
## Linear layer / losses
-/

/-- Linear layer: `y = W·x + b` with `W : (outDim,inDim)`, `x : inDim`, `b : outDim`. -/
@[inline] def linear {outDim inDim : Nat} (t : Tape SomeBuffer) (wId bId xId : Nat) : Result
    (Tape SomeBuffer × Nat) := do
  let out32 ← SomeBuffer.natToU32Checked outDim
  let in32 ← SomeBuffer.natToU32Checked inDim
  let one32 : UInt32 := 1
  let wBuf ← requireValue (t := t) wId (.dim outDim (.dim inDim .scalar))
  let bBuf ← requireValue (t := t) bId (.dim outDim .scalar)
  let xBuf ← requireValue (t := t) xId (.dim inDim .scalar)
  let wx := Buffer.bmm wBuf xBuf one32 out32 in32 one32
  let yBuf := Buffer.add wx bBuf
  let node : Node SomeBuffer :=
    { name := some "linear"
      value := { s := .dim outDim .scalar, buf := yBuf }
      requiresGrad := (t.getNode? wId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad) ||
        (t.getNode? xId).any (·.requiresGrad)
      parents := #[wId, bId, xId]
      cleanup := #[wx].map SomeBuffer.ofWorkspace
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim outDim .scalar)
        let g := dLdy.buf
        let dW := Buffer.bmm g xBuf one32 out32 one32 in32
        let db := Buffer.copy g
        let dx := Buffer.bmmLeftTranspose wBuf g one32 in32 out32 one32
        pure #[
            (wId, { s := .dim outDim (.dim inDim .scalar), buf := dW })
          , (bId, { s := .dim outDim .scalar, buf := db })
          , (xId, { s := .dim inDim .scalar, buf := dx }) ] }
  pure (t.addNode node)

/-- Mean-squared-error loss with `"mean"` reduction (single scalar output). -/
@[inline] def mseLoss {s : Shape} (t : Tape SomeBuffer) (yhatId targetId : Nat) : Result
    (Tape SomeBuffer × Nat) := do
  let yhat ← requireValue (t := t) yhatId s
  let target ← requireValue (t := t) targetId s
  let diff := Buffer.sub yhat target
  let squared := Buffer.mul diff diff
  let sum := Buffer.reduceSum squared
  let denom : Float := Float.ofNat (Spec.Shape.size s)
  let mean := Buffer.scale sum (1.0 / denom)
  let node : Node SomeBuffer :=
    { name := some "mse_loss"
      value := { s := Shape.scalar, buf := mean }
      requiresGrad := (t.getNode? yhatId).any (·.requiresGrad) ||
        (t.getNode? targetId).any (·.requiresGrad)
      parents := #[yhatId, targetId]
      cleanup := #[diff, squared, sum].map SomeBuffer.ofWorkspace
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny Shape.scalar
        let gBroad := broadcastScalarToShape dLdy.buf s
        let scaleConst : Float := 2.0 / denom
        let diffGrad := Buffer.mul diff gBroad
        let dYhat := Buffer.scale diffGrad scaleConst
        let dTarget := Buffer.releaseThen gBroad <|
          Buffer.releaseThen diffGrad <|
            Buffer.scale dYhat (-1.0)
        pure #[
          (yhatId, { s := s, buf := dYhat }),
          (targetId, { s := s, buf := dTarget })
        ] }
  pure (t.addNode node)
end Tape

end LibTorch
end Autograd
end Runtime
