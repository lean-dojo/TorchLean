/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Core
public import NN.Runtime.Autograd.Engine.Cuda.Shape

/-!
# CUDA Tape Operations: Shape and Reduction Nodes
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-!
## Reductions / views
-/

/-- Reduce-sum of all entries, producing a scalar. -/
def sum {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let x ← requireValue (t := t) xId s
  let y := Buffer.reduceSum x
  let node : Node :=
    { name := some "sum"
      value := { s := Shape.scalar, buf := y }
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny Shape.scalar
        let dx := broadcastScalarToShape dLdy.buf s
        pure #[(xId, { s := s, buf := dx })] }
  pure (t.addNode node)

/-- Flatten `s` into a 1D vector of length `Spec.Shape.size s`. -/
def flatten {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) :=
  unary (t := t) "flatten" xId s (.dim (Spec.Shape.size s) .scalar)
    (forward := fun x => x)
    (backward := fun _x dLdy => Buffer.copy dLdy)
    (ownsValue := false)

/--
Reshape a buffer while preserving number of elements.

This is a no-copy view operation: it reuses the same contiguous buffer.
-/
def reshape {s₁ s₂ : Shape} (t : Tape) (xId : Nat) (_h : Spec.Shape.size s₁ = Spec.Shape.size s₂) :
    Result (Tape × Nat) :=
  unary (t := t) "reshape" xId s₁ s₂
    (forward := fun x => x)
    (backward := fun _x dLdy => Buffer.copy dLdy)
    (ownsValue := false)

/--
Swap adjacent axes at a given depth in an N-D buffer.

If `depth` is out of range, this is treated as the identity (matches the spec-layer helper).
-/
def swapAdjacentAtDepth {s : Shape} (t : Tape) (depth : Nat) (xId : Nat) : Result (Tape × Nat) := do
  let depth32 ← AnyBuffer.natToU32Checked depth
  let dimsIn : Array Nat := Shape.toArray s
  let outShape : Shape := s.swapAdjacentAtDepth depth
  let dimsOut : Array Nat := Shape.toArray outShape
  let validDepth := depth + 1 < Spec.Shape.rank s
  unary (t := t) "swapAdjacentAtDepth" xId s outShape
    (forward := fun x =>
      if validDepth then
        Buffer.swapAdjacentAtDepth x dimsIn depth32
      else
        Buffer.copy x)
    (backward := fun _x dLdy =>
      if validDepth then
        Buffer.swapAdjacentAtDepth dLdy dimsOut depth32
      else
        Buffer.copy dLdy)

/--
Broadcast `x : s₁` to `s₂`.

Forward: `broadcastTo`.
Backward: sum-reduce broadcasted axes (`reduceFromBroadcastTo`).
-/
def broadcastTo {s₁ s₂ : Shape} (t : Tape) (cb : Shape.CanBroadcastTo s₁ s₂) (xId : Nat) :
    Result (Tape × Nat) := do
  let inDims := Shape.toArray s₁
  let outDims := Shape.toArray s₂
  let axisMap := Broadcast.axisMap cb
  unary (t := t) "broadcastTo" xId s₁ s₂
    (forward := fun x => Buffer.broadcastTo x inDims outDims axisMap)
    (backward := fun _x dLdy => Buffer.reduceFromBroadcastTo dLdy inDims outDims axisMap)

/-- Reduce-sum along `axis`. -/
def reduceSum {s : Shape} (axis : Nat) [_valid : Shape.HasNonemptyAxis axis s]
    [_wf : Shape.WellFormed s]
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let axis32 ← AnyBuffer.natToU32Checked axis
  let dims : Array Nat := Shape.toArray s
  let outShape : Shape := shapeAfterSum s axis
  unary (t := t) s!"reduce_sum(axis={axis})" xId s outShape
    (forward := fun x => Buffer.reduceSumAxis x dims axis32)
    (backward := fun _x dLdy =>
      let (inDims, outDims, axisMap) := Broadcast.afterSumArgs s axis
      Buffer.broadcastTo dLdy inDims outDims axisMap)

/-- Reduce-mean along `axis`. -/
def reduceMean {s : Shape} (axis : Nat) [valid : Shape.HasNonemptyAxis axis s]
    [_wf : Shape.WellFormed s]
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  let axis32 ← AnyBuffer.natToU32Checked axis
  let dims : Array Nat := Shape.toArray s
  let outShape : Shape := shapeAfterSum s axis
  unary (t := t) s!"reduce_mean(axis={axis})" xId s outShape
    (forward := fun x =>
      let sum := Buffer.reduceSumAxis x dims axis32
      letI : Shape.AxisInBounds axis s := valid.proof.toAxisInBounds
      let denomNat := Shape.axisSize s axis
      Buffer.scale sum (1.0 / (Float.ofNat denomNat)))
    (backward := fun _x dLdy =>
      let (inDims, outDims, axisMap) := Broadcast.afterSumArgs s axis
      let dLdx := Buffer.broadcastTo dLdy inDims outDims axisMap
      letI : Shape.AxisInBounds axis s := valid.proof.toAxisInBounds
      let denomNat := Shape.axisSize s axis
      Buffer.releaseThen dLdx <| Buffer.scale dLdx (1.0 / (Float.ofNat denomNat)))
end Tape

end Cuda
end Autograd
end Runtime
