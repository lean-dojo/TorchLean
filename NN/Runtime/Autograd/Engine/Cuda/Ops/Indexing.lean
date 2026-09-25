/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Spec.Core.Tensor.Core

/-!
# CUDA Tape Operations: Concatenation, Slicing, and Indexing
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-!
## Rank-one tensor slice
-/

/-- Slice `len` entries from a one-dimensional CUDA buffer starting at `start`. -/
@[inline] def sliceBuffer {n start len : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  if start + len ≤ n then
    let n32 ← AnyBuffer.natToU32Checked n
    let start32 ← AnyBuffer.natToU32Checked start
    let len32 ← AnyBuffer.natToU32Checked len
    let outShape : Shape := .dim len .scalar
    let x ← requireValue (t := t) xId (.dim n .scalar)
    let y := Buffer.sliceBuffer x n32 start32 len32
    let node : Node :=
      { name := some s!"slice_vector_buffer[{start}:{start+len}]"
        value := { s := outShape, buf := y }
        requiresGrad := (t.getNode? xId).any (·.requiresGrad)
        parents := #[xId]
        backward := fun dLdyAny => do
          let dLdy ← requireGrad dLdyAny outShape
          let pre := Buffer.zeros start32
          let postLen : Nat := n - start - len
          let post32 ← AnyBuffer.natToU32Checked postLen
          let post := Buffer.zeros post32
          let tmp := Buffer.concatBuffers pre dLdy.buf start32 len32
          let startLen32 ← AnyBuffer.natToU32Checked (start + len)
          let dx := Buffer.releaseThen pre <|
            Buffer.releaseThen post <|
              Buffer.releaseThen tmp <|
                Buffer.concatBuffers tmp post startLen32 post32
          pure #[(xId, { s := .dim n .scalar, buf := dx })] }
    pure (t.addNode node)
  else
    throw "autograd: slice_vector_buffer: start+len out of bounds"

/-!
## Concat / slice along dim 0
-/

/-- Concatenate along dim 0 for tensors with leading dimension (CPU tape name). -/
@[inline]
def concatLeadingAxis {n m : Nat} {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let inner : Nat := Spec.Shape.size s
  let nLen : Nat := n * inner
  let mLen : Nat := m * inner
  let nLen32 ← AnyBuffer.natToU32Checked nLen
  let mLen32 ← AnyBuffer.natToU32Checked mLen
  let nmLen32 ← AnyBuffer.natToU32Checked (nLen + mLen)
  binary (t := t) "concat_leading_axis" aId bId (.dim n s) (.dim m s) (.dim (n + m) s)
    (forward := fun a b => Buffer.concatBuffers a b nLen32 mLen32)
    (backward := fun _a _b dLdy =>
      let dA := Buffer.sliceBuffer dLdy nmLen32 0 nLen32
      let dB := Buffer.sliceBuffer dLdy nmLen32 nLen32 mLen32
      (dA, dB))

/-- Slice along dim 0: `x[start:start+len]` (CPU tape name). -/
@[inline] def sliceLeadingAxisRange {n : Nat} {s : Shape} (t : Tape) (xId : Nat) (start len : Nat)
    (_h : start + len ≤ n) : Result (Tape × Nat) := do
  let inner : Nat := Spec.Shape.size s
  let nTot : Nat := n * inner
  let startOff : Nat := start * inner
  let lenTot : Nat := len * inner
  let rightTot : Nat := nTot - (startOff + lenTot)
  let nTot32 ← AnyBuffer.natToU32Checked nTot
  let start32 ← AnyBuffer.natToU32Checked startOff
  let len32 ← AnyBuffer.natToU32Checked lenTot
  let right32 ← AnyBuffer.natToU32Checked rightTot
  let midLen32 ← AnyBuffer.natToU32Checked (startOff + lenTot)
  unary (t := t) "slice_leading_axis_range" xId (.dim n s) (.dim len s)
    (forward := fun x => Buffer.sliceBuffer x nTot32 start32 len32)
    (backward := fun _x dLdy =>
      let left := Buffer.zeros start32
      let right := Buffer.zeros right32
      let tmp := Buffer.concatBuffers left dLdy start32 len32
      Buffer.releaseThen left <|
        Buffer.releaseThen right <|
          Buffer.releaseThen tmp <|
            Buffer.concatBuffers tmp right midLen32 right32)

/-!
## Arbitrary-axis bounded indexing
-/

namespace Indexing

/-- An adjacent-swap depth paired with its checked CUDA representation. -/
abbrev SwapStep := Nat × UInt32

/-- Adjacent swaps that move `axis` to the outermost position. -/
def moveAxisToFrontDepths (axis : Nat) : List Nat :=
  (List.range axis).reverse

/-- Check each adjacent-swap depth before it crosses the CUDA ABI. -/
def checkedSwapSteps (depths : List Nat) : Result (List SwapStep) :=
  depths.mapM fun depth => do
    let depth32 ← AnyBuffer.natToU32Checked depth
    pure (depth, depth32)

/-- Apply swaps to an owned buffer, releasing each consumed intermediate. -/
def permuteOwned (x : Buffer) (s : Shape) : List SwapStep → Buffer
  | [] => x
  | (depth, depth32) :: steps =>
      let y := Buffer.swapAdjacentAtDepth x s.toArray depth32
      permuteOwned (Buffer.releaseThen x y) (s.swapAdjacentAtDepth depth) steps

/-- Apply swaps to a borrowed buffer, owning only the buffers created by the swaps. -/
def permuteBorrowed (x : Buffer) (s : Shape) : List SwapStep → Buffer
  | [] => x
  | (depth, depth32) :: steps =>
      let y := Buffer.swapAdjacentAtDepth x s.toArray depth32
      permuteOwned y (s.swapAdjacentAtDepth depth) steps

/-- Release a permuted buffer exactly when `permuteBorrowed` allocated it. -/
def releasePermutedThen (steps : List SwapStep) (permuted keep : Buffer) : Buffer :=
  match steps with
  | [] => keep
  | _ :: _ => Buffer.releaseThen permuted keep

/-- Convert bounded tensor indices to the host array expected by CUDA row kernels. -/
def finTensorToIndexArray {n count : Nat}
    (indices : Tensor (Fin n) [count]) : Array Nat :=
  Array.ofFn fun i : Fin count => (Tensor.getScalar indices i).val

end Indexing

/-- Select one bounded coordinate from any tensor axis. -/
@[inline] def select {s : Shape} (t : Tape) (xId : Nat) (axis : Nat)
    [Shape.AxisInBounds axis s] (index : Fin (Shape.axisSize s axis)) :
    Result (Tape × Nat) := do
  let outShape := s.eraseAxis axis
  let x ← requireValue (t := t) xId s
  let rows32 ← AnyBuffer.natToU32Checked (Shape.axisSize s axis)
  let cols32 ← AnyBuffer.numelU32 outShape
  let inputSize32 ← AnyBuffer.numelU32 s
  let steps ← Indexing.checkedSwapSteps (Indexing.moveAxisToFrontDepths axis)
  let reverseSteps := steps.reverse
  let one32 : UInt32 := 1
  let indices : Array Nat := #[index.val]
  let moved := Indexing.permuteBorrowed x s steps
  let gathered := Buffer.gatherRows moved rows32 cols32 indices one32
  let y := Indexing.releasePermutedThen steps moved gathered
  let node : Node :=
    { name := some s!"select(axis={axis}, index={index.val})"
      value := { s := outShape, buf := y }
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let zeros := Buffer.zeros inputSize32
        let scattered := Buffer.scatterAddRows zeros dLdy.buf rows32 cols32 indices one32
        let frontDx := Buffer.releaseThen zeros scattered
        let dx :=
          Indexing.permuteOwned frontDx (.dim (Shape.axisSize s axis) outShape) reverseSteps
        pure #[(xId, { s := s, buf := dx })] }
  pure (t.addNode node)

/-- Select several bounded coordinates from any tensor axis. -/
@[inline] def indexSelect {s : Shape} (t : Tape) (xId : Nat) (axis count : Nat)
    [Shape.AxisInBounds axis s]
    (indices : Tensor (Fin (Shape.axisSize s axis)) [count]) : Result (Tape × Nat) := do
  let outShape := s.replaceAxis axis count
  let tailShape := s.eraseAxis axis
  let frontInputShape := .dim (Shape.axisSize s axis) tailShape
  let frontOutputShape := .dim count tailShape
  let x ← requireValue (t := t) xId s
  let rows32 ← AnyBuffer.natToU32Checked (Shape.axisSize s axis)
  let cols32 ← AnyBuffer.numelU32 tailShape
  let count32 ← AnyBuffer.natToU32Checked count
  let inputSize32 ← AnyBuffer.numelU32 s
  let _ ← AnyBuffer.numelU32 outShape
  let steps ← Indexing.checkedSwapSteps (Indexing.moveAxisToFrontDepths axis)
  let reverseSteps := steps.reverse
  let indexArray := Indexing.finTensorToIndexArray indices
  let moved := Indexing.permuteBorrowed x s steps
  let gathered := Buffer.gatherRows moved rows32 cols32 indexArray count32
  let frontY := Indexing.releasePermutedThen steps moved gathered
  let y := Indexing.permuteOwned frontY frontOutputShape reverseSteps
  let node : Node :=
    { name := some s!"index_select(axis={axis})"
      value := { s := outShape, buf := y }
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let movedGrad := Indexing.permuteBorrowed dLdy.buf outShape steps
        let zeros := Buffer.zeros inputSize32
        let scattered :=
          Buffer.scatterAddRows zeros movedGrad rows32 cols32 indexArray count32
        let frontDx := Buffer.releaseThen zeros scattered
        let frontDx := Indexing.releasePermutedThen steps movedGrad frontDx
        let dx := Indexing.permuteOwned frontDx frontInputShape reverseSteps
        pure #[(xId, { s := s, buf := dx })] }
  pure (t.addNode node)

/-- Add indexed source slices into any tensor axis. -/
@[inline] def scatterAdd {s : Shape} (t : Tape) (baseId sourceId : Nat) (axis count : Nat)
    [Shape.AxisInBounds axis s]
    (indices : Tensor (Fin (Shape.axisSize s axis)) [count]) : Result (Tape × Nat) := do
  let sourceShape := s.replaceAxis axis count
  let tailShape := s.eraseAxis axis
  let frontInputShape := .dim (Shape.axisSize s axis) tailShape
  let frontSourceShape := .dim count tailShape
  let base ← requireValue (t := t) baseId s
  let source ← requireValue (t := t) sourceId sourceShape
  let rows32 ← AnyBuffer.natToU32Checked (Shape.axisSize s axis)
  let cols32 ← AnyBuffer.numelU32 tailShape
  let count32 ← AnyBuffer.natToU32Checked count
  let _ ← AnyBuffer.numelU32 s
  let steps ← Indexing.checkedSwapSteps (Indexing.moveAxisToFrontDepths axis)
  let reverseSteps := steps.reverse
  let indexArray := Indexing.finTensorToIndexArray indices
  let movedBase := Indexing.permuteBorrowed base s steps
  let movedSource := Indexing.permuteBorrowed source sourceShape steps
  let scattered :=
    Buffer.scatterAddRows movedBase movedSource rows32 cols32 indexArray count32
  let frontY := Indexing.releasePermutedThen steps movedSource scattered
  let frontY := Indexing.releasePermutedThen steps movedBase frontY
  let y := Indexing.permuteOwned frontY frontInputShape reverseSteps
  let node : Node :=
    { name := some s!"scatter_add(axis={axis})"
      value := { s := s, buf := y }
      requiresGrad := (t.getNode? baseId).any (·.requiresGrad) ||
        (t.getNode? sourceId).any (·.requiresGrad)
      parents := #[baseId, sourceId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny s
        let dBase := Buffer.copy dLdy.buf
        let movedGrad := Indexing.permuteBorrowed dLdy.buf s steps
        let gathered := Buffer.gatherRows movedGrad rows32 cols32 indexArray count32
        let frontDSource := Indexing.releasePermutedThen steps movedGrad gathered
        let dSource := Indexing.permuteOwned frontDSource frontSourceShape reverseSteps
        pure #[
          (baseId, { s := s, buf := dBase }),
          (sourceId, { s := sourceShape, buf := dSource })] }
  pure (t.addNode node)

end Tape

end Cuda
end Autograd
end Runtime
