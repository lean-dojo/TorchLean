/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Core

/-!
# CUDA Tape Operations: Matrix, FFT, and Loss Nodes
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-!
## Linear algebra
-/

namespace Internal

/-- Backend matrix kernel for a single pair of matrices. -/
def matmul {m n p : Nat} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let m32 ← AnyBuffer.natToU32Checked m
  let n32 ← AnyBuffer.natToU32Checked n
  let p32 ← AnyBuffer.natToU32Checked p
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
def matmulFlattened {batch m n p : Nat} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let b32 ← AnyBuffer.natToU32Checked batch
  let m32 ← AnyBuffer.natToU32Checked m
  let n32 ← AnyBuffer.natToU32Checked n
  let p32 ← AnyBuffer.natToU32Checked p
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

namespace Internal

/--
Real-FFT spectral convolution used by the CUDA FNO1D path.

Shapes:
- `x : (grid, width)`,
- `wRe, wIm : (modes, width, width)`,
- output `y : (grid, width)`.

The low-level buffer primitive owns the numerical contract and VJP:
`rfft(x)` is unnormalized, the inverse is normalized, and the backward kernels include the
half-spectrum adjoint factors for real FFTs. This tape node records those three parent
dependencies and checks the runtime shapes before calling the native kernels.
-/
def spectralConv1dRfft {grid width modes : Nat}
    (t : Tape) (xId wReId wImId : Nat) : Result (Tape × Nat) := do
  if grid = 0 then
    throw "autograd: spectralConv1dRfft: grid must be positive"
  if width = 0 then
    throw "autograd: spectralConv1dRfft: width must be positive"
  if modes > grid / 2 + 1 then
    throw "autograd: spectralConv1dRfft: modes exceeds rfft frequency count"
  let grid32 ← AnyBuffer.natToU32Checked grid
  let width32 ← AnyBuffer.natToU32Checked width
  let modes32 ← AnyBuffer.natToU32Checked modes
  let xShape : Shape := .dim grid (.dim width .scalar)
  let wShape : Shape := .dim modes (.dim width (.dim width .scalar))
  let x ← requireValue (t := t) xId xShape
  let wRe ← requireValue (t := t) wReId wShape
  let wIm ← requireValue (t := t) wImId wShape
  let y := Buffer.spectralConv1dRfftFwd x wRe wIm grid32 width32 modes32
  let node : Node :=
    { name := some "spectralConv1dRfft"
      value := { s := xShape, buf := y }
      requiresGrad := (t.getNode? xId).any (·.requiresGrad) ||
        (t.getNode? wReId).any (·.requiresGrad) ||
        (t.getNode? wImId).any (·.requiresGrad)
      parents := #[xId, wReId, wImId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny xShape
        let (dx, dWRe, dWIm) :=
          Buffer.spectralConv1dRfftBwd x wRe wIm dLdy.buf grid32 width32 modes32
        pure #[
            (xId, { s := xShape, buf := dx })
          , (wReId, { s := wShape, buf := dWRe })
          , (wImId, { s := wShape, buf := dWIm }) ] }
  pure (t.addNode node)

end Internal

/-!
## Linear layer / losses
-/

/-- Linear layer: `y = W·x + b` with `W : (outDim,inDim)`, `x : inDim`, `b : outDim`. -/
def linear {outDim inDim : Nat} (t : Tape) (wId bId xId : Nat) : Result (Tape × Nat) := do
  let out32 ← AnyBuffer.natToU32Checked outDim
  let in32 ← AnyBuffer.natToU32Checked inDim
  let one32 : UInt32 := 1
  let wBuf ← requireValue (t := t) wId (.dim outDim (.dim inDim .scalar))
  let bBuf ← requireValue (t := t) bId (.dim outDim .scalar)
  let xBuf ← requireValue (t := t) xId (.dim inDim .scalar)
  let wx := Buffer.bmm wBuf xBuf one32 out32 in32 one32
  let yBuf := Buffer.add wx bBuf
  let node : Node :=
    { name := some "linear"
      value := { s := .dim outDim .scalar, buf := yBuf }
      requiresGrad := (t.getNode? wId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad) ||
        (t.getNode? xId).any (·.requiresGrad)
      parents := #[wId, bId, xId]
      cleanup := #[wx]
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
def mseLoss {s : Shape} (t : Tape) (yhatId targetId : Nat) : Result (Tape × Nat) := do
  let yhat ← requireValue (t := t) yhatId s
  let target ← requireValue (t := t) targetId s
  let diff := Buffer.sub yhat target
  let squared := Buffer.mul diff diff
  let sum := Buffer.reduceSum squared
  let denom : Float := Float.ofNat (Spec.Shape.size s)
  let mean := Buffer.scale sum (1.0 / denom)
  let node : Node :=
    { name := some "mse_loss"
      value := { s := Shape.scalar, buf := mean }
      requiresGrad := (t.getNode? yhatId).any (·.requiresGrad) ||
        (t.getNode? targetId).any (·.requiresGrad)
      parents := #[yhatId, targetId]
      cleanup := #[diff, squared, sum]
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

end Cuda
end Autograd
end Runtime
