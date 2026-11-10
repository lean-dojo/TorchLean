/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Sample
public import TorchLean.Tensor.Operations

/-!
# Arbitrary-Rank Block Masks

Masked prediction is not intrinsically an image operation. A model may hide intervals in a signal,
rectangles in an image, cuboids in a volume, or blocks in a higher-dimensional simulation field.
A mask is described by a rank-indexed policy tensor. The extents come from the input tensor's
type, while `blocks : Tensor (Option Nat) [d]` selects the axes that form a block grid.

`none` means that an axis does not participate in the block index. `some k` groups that axis into
consecutive blocks of width `k`. The selected block-grid coordinates are flattened in row-major
order, and one congruence class modulo `period` is hidden. A zero period, a zero block width, a
rank mismatch, an out-of-bounds coordinate, or a policy with no blocked axis hides nothing.

`SelfSupervised.BlockMask.apply` zeroes the hidden blocks of a tensor, and
`BlockMask.apply_scalar_at`, `hidden_scalar_eq_zero` and `visible_scalar_eq_input` state its effect
at each coordinate. `SelfSupervised.BlockMAE.sample` builds a masked-reconstruction training sample
in the style of a masked autoencoder, and `hiddenMask`, `reconstructionWeights` and `hiddenIndices`
describe its hidden entries. The executable mask and these theorems use the same finite predicate,
so the runtime training sample follows the patch convention stated in Lean.
-/

@[expose] public section

open TorchLean.Shape.Coord (unlinearize)

namespace TorchLean
namespace SelfSupervised

open TorchLean.Tensor

namespace BlockMask

variable {α : Type} [Storage α] [Zero α]

/--
Flat index of the block a coordinate falls in. The result is `none` when the coordinate is out of
range, a block width is zero, the shape, policies and coordinate differ in length, or no axis is
blocked at all.

The walk goes over shape and policy together. An axis with `none` contributes nothing, and an axis
with `some blockSize` contributes its block coordinate as one digit of a mixed-radix number whose
radix is the number of blocks along that axis. `used` tells "block 0" apart from "no axis
blocked", a distinction the caller needs.
-/
def blockIndex (shape : Shape) (policies : List (Option Nat))
    (coordinates : List Nat) (flatIndex : Nat) (used : Bool) : Option Nat :=
  match shape, policies, coordinates with
  | Shape.scalar, List.nil, List.nil => if used then some flatIndex else none
  | Shape.dim extent extents, List.cons policy policies, List.cons coordinate coordinates =>
      if coordinate < extent then
        match policy with
        | none => blockIndex extents policies coordinates flatIndex used
        | some blockSize =>
            if blockSize = 0 then
              none
            else
              let blocksAlongAxis := (extent + blockSize - 1) / blockSize
              let blockCoordinate := coordinate / blockSize
              blockIndex extents policies coordinates
                (flatIndex * blocksAlongAxis + blockCoordinate) true
      else
        none
  | _, _, _ => none

/--
Whether the block containing `coordinate` is hidden, that is, whether its flat index is congruent
to `offset` modulo `period`. A zero `period` hides nothing.

Hiding every `period`-th block instead of a random sample makes a mask reproducible from two
numbers, which keeps the self-supervised examples in the guide comparable across machines and
across runs.
-/
def blockHidden (shape : Shape) (blocks : List (Option Nat))
    (period offset : Nat) (coordinate : List Nat) : Bool :=
  if period = 0 then
    false
  else
    match blockIndex shape blocks coordinate 0 false with
    | some index => decide (index % period = offset % period)
    | none => false

/--
Zero out the hidden blocks of a tensor, recursing over the leading axes and collecting the
coordinate prefix on the way down.

The result is rebuilt with `Tensor.dim`, so a masked tensor is an ordinary value and the mask
cannot depend on evaluation order.
-/
def applyPrefix (shape : Shape) (blocks : List (Option Nat))
    (period offset : Nat) (coordinatePrefix : List Nat) :
    (remainingShape : Shape) → Tensor α remainingShape → Tensor α remainingShape
  | .scalar, tensor =>
      if blockHidden shape blocks period offset coordinatePrefix then
        Tensor.full [] 0
      else
        tensor
  | .dim _ remainingShape, tensor =>
      TorchLean.Tensor.dim fun coordinate =>
        applyPrefix shape blocks period offset (coordinatePrefix ++ [coordinate.val]) remainingShape
          (Tensor.unstack tensor coordinate)

/-- Reading one coordinate out of a masked tensor is the same as reading it out of the original and
zeroing it when its block is hidden.

Proofs about `applyPrefix` go through this lemma, since its recursion unstacks one axis at a time
and unfolds to `Tensor.dim` and coordinate prefixes. With the `Option.map` on the right, an
out-of-range coordinate reads `none` on both sides. -/
theorem applyPrefix_scalar_at
    (shape : Shape) (blocks : List (Option Nat)) (period offset : Nat)
    (coordinatePrefix : List Nat) :
    ∀ (remainingShape : Shape) (x : Tensor α remainingShape) (coordinates : List Nat),
      Tensor.get?
          (applyPrefix shape blocks period offset coordinatePrefix remainingShape x) coordinates =
        (Tensor.get? x coordinates).map (fun value =>
          if blockHidden shape blocks period offset (coordinatePrefix ++ coordinates) then
            0
          else
            value) := by
  intro remainingShape
  induction remainingShape generalizing coordinatePrefix with
  | scalar =>
      intro x coordinates
      rw [← Tensor.scalar_item x]
      cases coordinates with
      | nil =>
          by_cases hHidden :
              blockHidden shape blocks period offset coordinatePrefix
          · simp [applyPrefix, hHidden]
          · simp [applyPrefix, hHidden]
      | cons coordinate coordinates =>
          simp [applyPrefix]
  | dim extent remainingShape ih =>
      intro x coordinates
      rw [← Tensor.dim_unstack x]
      cases coordinates with
      | nil => simp [applyPrefix]
      | cons coordinate coordinates =>
          simp only [applyPrefix, Tensor.get?_dim_cons, Tensor.unstack_dim]
          by_cases h : coordinate < extent
          · simp only [dite_eq_left h]
            simpa [List.append_assoc] using
              ih (coordinatePrefix := coordinatePrefix ++ [coordinate])
                (Tensor.unstack x ⟨coordinate, h⟩) coordinates
          · simp [h]

/--
The row-major index of the block containing `coordinate`, or `none` when the coordinate is out of
bounds, a block width is zero, or no axis is blocked.
-/
def index {shape : Shape} (blocks : Tensor (Option Nat) [shape.rank])
    (coordinate : Tensor Nat [shape.rank]) : Option Nat :=
  blockIndex shape (Tensor.to blocks (List (Option Nat)))
    (Tensor.to coordinate (List Nat)) 0 false

/--
Whether `coordinate` lies in a hidden block, one whose index is congruent to `offset` modulo
`period`.
-/
def hidden {shape : Shape} (blocks : Tensor (Option Nat) [shape.rank])
    (period offset : Nat) (coordinate : Tensor Nat [shape.rank]) : Bool :=
  blockHidden shape (Tensor.to blocks (List (Option Nat))) period offset
    (Tensor.to coordinate (List Nat))

/--
`apply x blocks period offset` sets every scalar in a hidden block of `x` to zero and copies the
other entries unchanged. It works at any rank and needs only storage and a zero value.

For example, policies `[none, some 4, some 4]` repeat a 4-by-4 block mask across the first axis,
`[some 8]` masks intervals in a signal, and `[some 2, some 2, some 2]` masks volume blocks.
-/
def apply {shape : Shape} (x : Tensor α shape)
    (blocks : Tensor (Option Nat) [shape.rank]) (period offset : Nat) :
    Tensor α shape :=
  applyPrefix shape (Tensor.to blocks (List (Option Nat)))
    period offset [] shape x

/--
Reading `coordinate` from `apply x blocks period offset` gives the entry of `x`, or zero when
`hidden` holds. An out-of-bounds coordinate reads `none` on both sides.
-/
theorem apply_scalar_at {shape : Shape}
    (blocks : Tensor (Option Nat) [shape.rank]) (period offset : Nat)
    (x : Tensor α shape) (coordinate : Tensor Nat [shape.rank]) :
    Tensor.get? (apply x blocks period offset) (Tensor.to coordinate (List Nat)) =
      (Tensor.get? x (Tensor.to coordinate (List Nat))).map (fun value =>
        if hidden blocks period offset coordinate then
          0
        else value) := by
  simpa [apply, hidden, blockHidden] using
    applyPrefix_scalar_at shape (Tensor.to blocks (List (Option Nat))) period offset [] shape x
      (Tensor.to coordinate (List Nat))

/-- If `x` has an entry at `coordinate` (`hValue`) whose block is hidden (`hHidden`), the masked
tensor reads `0` there. -/
theorem hidden_scalar_eq_zero {shape : Shape}
    (blocks : Tensor (Option Nat) [shape.rank]) (period offset : Nat)
    (x : Tensor α shape) (coordinate : Tensor Nat [shape.rank])
    (value : α) (hValue : Tensor.get? x (Tensor.to coordinate (List Nat)) = some value)
    (hHidden : hidden blocks period offset coordinate = true) :
    Tensor.get? (apply x blocks period offset) (Tensor.to coordinate (List Nat)) =
      some 0 := by
  rw [apply_scalar_at, hValue, hHidden]
  rfl

/-- A visible in-bounds coordinate is copied unchanged by the mask. -/
theorem visible_scalar_eq_input {shape : Shape}
    (blocks : Tensor (Option Nat) [shape.rank]) (period offset : Nat)
    (x : Tensor α shape) (coordinate : Tensor Nat [shape.rank])
    (value : α) (hValue : Tensor.get? x (Tensor.to coordinate (List Nat)) = some value)
    (hVisible : hidden blocks period offset coordinate = false) :
    Tensor.get? (apply x blocks period offset) (Tensor.to coordinate (List Nat)) =
      some value := by
  rw [apply_scalar_at, hValue, hVisible]
  rfl

end BlockMask

namespace BlockMAE

open Shape (Coord)

/--
Create a masked-reconstruction sample, given a proof that the target width fits in the data.

The input is `x` with the hidden blocks of each data tensor set to zero, in its original shape.
The target is the first `reconstructionWidth` entries of each unmasked data tensor in row-major
order, because TorchLean's compact decoder heads produce matrices.
-/
def sampleOfLE {α : Type} [Storage α] [Zero α] [Inhabited α]
    (batchShape : Shape) {dataShape : Shape} (reconstructionWidth : Nat)
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat)
    (hReconstruction : reconstructionWidth ≤ dataShape.size)
    (x : Tensor α (batchShape ++ dataShape)) :
    TorchLean.Sample.Supervised α
      (batchShape ++ dataShape) (batchShape ++ [reconstructionWidth]) :=
  { input :=
      Tensor.mapLeading batchShape
        (fun data => BlockMask.apply data blocks period offset) x
    target :=
      TorchLean.Tensor.flattenThenTake batchShape reconstructionWidth hReconstruction x }

/-- Flattened reconstruction coordinates hidden by the block mask. -/
def hiddenTargetIndicesOfLE {dataShape : Shape} (reconstructionWidth : Nat)
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat)
    (hReconstruction : reconstructionWidth ≤ dataShape.size) :
    Array (Fin reconstructionWidth) :=
  let policies := Tensor.to blocks (List (Option Nat))
  (Array.finRange reconstructionWidth).filter fun i =>
    let coordinate := Coord.unlinearize (i.castLE hReconstruction)
    BlockMask.blockHidden dataShape policies period offset
      (Shape.Coord.toList dataShape coordinate)

/-- Tensor-valued indicator of hidden reconstruction coordinates in row-major order. -/
def hiddenMask {dataShape : Shape}
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat) :
    Tensor Bool [dataShape.size] :=
  let policies := Tensor.to blocks (List (Option Nat))
  Tensor.ofFn fun i : Fin dataShape.size =>
    BlockMask.blockHidden dataShape policies period offset
      (Shape.Coord.toList dataShape (Coord.unlinearize i))

/-- Uniform reconstruction weights on hidden entries, or all zeros when no entry is hidden. Each
weight is the exact ratio `1 / count` cast once into the scalar. -/
def reconstructionWeights {α : Type} [Storage α] [Zero α] [RatCast α]
    {dataShape : Shape}
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat) :
    Tensor α [dataShape.size] :=
  let mask := hiddenMask blocks period offset
  let count := (mask.map fun hidden => if hidden then (1 : Nat) else 0).sum
  mask.map fun hidden => if hidden then ((1 / count : Rat) : α) else 0

/--
Return every hidden coordinate in the flattened data tensor.

The result covers the whole data shape, so no reconstruction-width validation is needed.
-/
def hiddenIndices {dataShape : Shape}
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat) :
    Array (Fin dataShape.size) :=
  hiddenTargetIndicesOfLE dataShape.size blocks period offset (Nat.le_refl _)

/--
Create a masked-reconstruction sample with arbitrary batch and data shapes, retaining its scalar.

The model input retains its original shape. The target is a row-major prefix of the unmasked source.
An invalid reconstruction width is reported at the ordinary executable boundary rather than
requiring callers to provide a theorem.
-/
def sample {α : Type} [Storage α] [Zero α] [Inhabited α]
    (batchShape : Shape) {dataShape : Shape} (reconstructionWidth : Nat)
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat)
    (x : Tensor α (batchShape ++ dataShape)) :
    Except String (TorchLean.Sample.Supervised α
      (batchShape ++ dataShape) (batchShape ++ [reconstructionWidth])) :=
  if h : reconstructionWidth ≤ dataShape.size then
    .ok (sampleOfLE batchShape reconstructionWidth blocks period offset h x)
  else
    .error (s!"BlockMAE reconstruction width {reconstructionWidth} "
      ++ s!"exceeds input width {dataShape.size}")

/--
The flattened coordinates below `reconstructionWidth` that the block mask hides. Fails with an
error message when `reconstructionWidth` exceeds the data size.
-/
def hiddenTargetIndices {dataShape : Shape} (reconstructionWidth : Nat)
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat) :
    Except String (Array (Fin reconstructionWidth)) :=
  if h : reconstructionWidth ≤ dataShape.size then
    .ok (hiddenTargetIndicesOfLE reconstructionWidth blocks period offset h)
  else
    .error (s!"BlockMAE reconstruction width {reconstructionWidth} "
      ++ s!"exceeds input width {dataShape.size}")

end BlockMAE

end SelfSupervised
end TorchLean
