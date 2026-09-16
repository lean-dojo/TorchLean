/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Sample
public import NN.MLTheory.SelfSupervised.PredictiveView
public import NN.Tensor.Conversion
public import NN.Tensor.Operations
public import NN.Tensor.Reductions
public import Std.Tactic.BVDecide.Normalize.Bool
public import NN.API.Arithmetic -- shake: keep
public import NN.Tensor -- shake: keep

/-!
# Arbitrary-Rank Block Masks

Masked prediction is not intrinsically an image operation. A model may hide intervals in a signal,
rectangles in an image, cuboids in a volume, or blocks in a higher-dimensional simulation field.
This module describes a mask by a rank-indexed policy tensor. The extents come from the input
tensor's type, while `blocks : Tensor (Option Nat) [d]` selects the axes that form a block
grid.

`none` means that an axis does not participate in the block index. `some k` groups that axis into
consecutive blocks of width `k`. The selected block-grid coordinates are flattened in row-major
order, and one congruence class modulo `period` is hidden. A zero period, a zero block width, a rank
mismatch, or an out-of-bounds coordinate hides nothing.

The executable mask and its coordinate theorems use the same finite predicate. Consequently the
runtime training sample cannot silently use a different patch convention from the one stated in
Lean.
-/

@[expose] public section

namespace TorchLean
namespace ssl

open Spec TorchLean TorchLean.Tensor

namespace BlockMask

namespace Internal

/--
Flat index of the block a coordinate falls in, or `none` when the coordinate is out of range or no
axis is blocked at all.

The walk goes over shape and policy together. An axis with `none` contributes nothing; an axis with
`some blockSize` contributes its block coordinate as one digit of a mixed-radix number whose radix
is the number of blocks along that axis. `used` is what distinguishes "block 0" from "nothing was
blocked", which the caller has to tell apart.
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
Is the block containing this coordinate hidden by the mask with the given `period` and `offset`?

Hiding every `period`-th block rather than sampling at random is a deliberate choice: a mask is
then reproducible from two numbers, which is what makes the self-supervised examples in the guide
comparable across machines and across runs.
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

The result is rebuilt with `stackLeading` rather than written in place, so a masked tensor is an
ordinary value and the mask cannot depend on evaluation order.
-/
def apply (shape : Shape) (blocks : List (Option Nat))
    (period offset : Nat) (coordinatePrefix : List Nat) :
    (remainingShape : Shape) → Tensor Float remainingShape → Tensor Float remainingShape
  | .scalar, tensor =>
      if blockHidden shape blocks period offset coordinatePrefix then
        Tensor.full [] 0.0
      else
        tensor
  | .dim _ remainingShape, tensor =>
      TorchLean.Tensor.stackLeading fun coordinate =>
        apply shape blocks period offset (coordinatePrefix ++ [coordinate.val]) remainingShape
          (Tensor.unstack tensor coordinate)

/-- Reading one coordinate out of a masked tensor is the same as reading it out of the original and
zeroing it when its block is hidden.

This is the lemma that makes `apply` usable in proofs. `apply` is defined by recursion on the
remaining shape, unstacking one axis at a time, so its unfolded form talks about `stackLeading` and
prefixes of coordinates rather than about masking. Stated pointwise it says what a reader expects a
mask to mean, and the `Option.map` on the right is just `getSpec` reporting an out-of-range
coordinate the same way on both sides. -/
theorem apply_scalar_at
    (shape : Shape) (blocks : List (Option Nat)) (period offset : Nat)
    (coordinatePrefix : List Nat) :
    ∀ (remainingShape : Shape) (x : Tensor Float remainingShape) (coordinates : List Nat),
      Spec.getSpec
          (apply shape blocks period offset coordinatePrefix remainingShape x) coordinates =
        (Spec.getSpec x coordinates).map (fun value =>
          if blockHidden shape blocks period offset (coordinatePrefix ++ coordinates) then
            0.0
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
          · simp [apply, hHidden, Tensor.full]
          · simp [apply, hHidden]
      | cons coordinate coordinates =>
          simp [apply]
  | dim extent remainingShape ih =>
      intro x coordinates
      rw [← Tensor.dim_unstack x]
      cases coordinates with
      | nil => simp [apply]
      | cons coordinate coordinates =>
          simp only [apply, Spec.get_spec_dim_cons, Tensor.unstack_dim]
          by_cases h : coordinate < extent
          · simp only [dite_eq_left h]
            simpa [List.append_assoc] using
              ih (coordinatePrefix := coordinatePrefix ++ [coordinate])
                (Tensor.unstack x ⟨coordinate, h⟩) coordinates
          · simp [h]

end Internal

/-- Row-major block index, or `none` for an invalid/degenerate block description. -/
def index {shape : Shape} (blocks : Tensor (Option Nat) [shape.rank])
    (coordinate : Tensor Nat [shape.rank]) : Option Nat :=
  Internal.blockIndex shape (Tensor.to blocks (List (Option Nat)))
    (Tensor.to coordinate (List Nat)) 0 false

/-- Whether a coordinate belongs to the selected congruence class of blocks. -/
def hidden {shape : Shape} (blocks : Tensor (Option Nat) [shape.rank])
    (period offset : Nat) (coordinate : Tensor Nat [shape.rank]) : Bool :=
  if period = 0 then
    false
  else
    match index blocks coordinate with
    | some index => decide (index % period = offset % period)
    | none => false

/--
Set every scalar in a selected block to zero, preserving the tensor's arbitrary-rank shape.

For example, policies `[none, some 4, some 4]` repeat a 4-by-4 block mask across the first axis;
`[some 8]` masks intervals in a signal; and `[some 2, some 2, some 2]` masks volume blocks.
-/
def apply {shape : Shape} (x : Tensor Float shape)
    (blocks : Tensor (Option Nat) [shape.rank]) (period offset : Nat) :
    Tensor Float shape :=
  Internal.apply shape (Tensor.to blocks (List (Option Nat)))
    period offset [] shape x

/-- Exact coordinate semantics of `apply`, including out-of-bounds coordinates. -/
theorem apply_scalar_at {shape : Shape}
    (blocks : Tensor (Option Nat) [shape.rank]) (period offset : Nat)
    (x : Tensor Float shape) (coordinate : Tensor Nat [shape.rank]) :
    Spec.getSpec (apply x blocks period offset) (Tensor.to coordinate (List Nat)) =
      (Spec.getSpec x (Tensor.to coordinate (List Nat))).map (fun value =>
        if hidden blocks period offset coordinate then
          0.0
        else value) := by
  simpa [apply, hidden, index, Internal.blockHidden] using
    Internal.apply_scalar_at shape (Tensor.to blocks (List (Option Nat))) period offset [] shape x
      (Tensor.to coordinate (List Nat))

/-- A selected in-bounds coordinate is exactly zero after masking. -/
theorem hidden_scalar_eq_zero {shape : Shape}
    (blocks : Tensor (Option Nat) [shape.rank]) (period offset : Nat)
    (x : Tensor Float shape) (coordinate : Tensor Nat [shape.rank])
    (value : Float) (hValue : Spec.getSpec x (Tensor.to coordinate (List Nat)) = some value)
    (hHidden : hidden blocks period offset coordinate = true) :
    Spec.getSpec (apply x blocks period offset) (Tensor.to coordinate (List Nat)) =
      some 0.0 := by
  rw [apply_scalar_at, hValue, hHidden]
  rfl

/-- A visible in-bounds coordinate is copied unchanged by the mask. -/
theorem visible_scalar_eq_input {shape : Shape}
    (blocks : Tensor (Option Nat) [shape.rank]) (period offset : Nat)
    (x : Tensor Float shape) (coordinate : Tensor Nat [shape.rank])
    (value : Float) (hValue : Spec.getSpec x (Tensor.to coordinate (List Nat)) = some value)
    (hVisible : hidden blocks period offset coordinate = false) :
    Spec.getSpec (apply x blocks period offset) (Tensor.to coordinate (List Nat)) =
      some value := by
  rw [apply_scalar_at, hValue, hVisible]
  rfl

end BlockMask

namespace BlockMAE

namespace Internal

/--
Create a masked-reconstruction sample after validating the requested target width.

The model input retains its original shape. The target is a row-major prefix of the unmasked source
because TorchLean's compact decoder heads produce matrices.
-/
def sample (batchShape : Shape) {dataShape : Shape} (reconstructionWidth : Nat)
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat)
    (hReconstruction : reconstructionWidth ≤ dataShape.size)
    (x : Tensor Float (batchShape.concat dataShape)) :
    TorchLean.Sample.Supervised Float
      (batchShape.concat dataShape) (batchShape.appendDim reconstructionWidth) :=
  { input :=
      Tensor.mapLeading batchShape
        (fun data => BlockMask.apply data blocks period offset) x
    target :=
      TorchLean.Tensor.flattenThenTake batchShape reconstructionWidth hReconstruction x }

/-- Flattened reconstruction coordinates hidden by the block mask. -/
def hiddenReconstructionIndices {dataShape : Shape} (reconstructionWidth : Nat)
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat)
    (hReconstruction : reconstructionWidth ≤ dataShape.size) :
    Array (Fin reconstructionWidth) :=
  let ones : Tensor Float dataShape := Tensor.ones (α := Float) dataShape
  let masked := BlockMask.apply ones blocks period offset
  let flatMask : Tensor Float [reconstructionWidth] :=
    TorchLean.Tensor.flattenThenTake [] reconstructionWidth hReconstruction masked
  (Array.finRange reconstructionWidth).filter fun i =>
    flatMask[i] == 0.0

end Internal

/-- Tensor-valued indicator of hidden reconstruction coordinates in row-major order. -/
def hiddenMask {dataShape : Shape}
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat) :
    Tensor Bool [dataShape.size] :=
  let visible := BlockMask.apply (Tensor.ones (α := Float) dataShape) blocks period offset
  (TorchLean.Tensor.flattenThenTake [] dataShape.size (Nat.le_refl _) visible).map
    (fun value => value == 0.0)

/-- Uniform reconstruction weights on hidden entries, or all zeros when no entry is hidden. -/
def reconstructionWeights {α : Type} [TorchLean.Storage α] [Context α] {dataShape : Shape}
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat) :
    Tensor α [dataShape.size] :=
  let mask := hiddenMask blocks period offset
  let count := (mask.map fun hidden => if hidden then (1 : Nat) else 0).sum
  mask.map fun hidden => if hidden then (1 : α) / (count : α) else 0

/--
Return every hidden coordinate in the flattened data tensor.

The result covers the whole data shape, so no reconstruction-width validation is needed.
-/
def hiddenIndices {dataShape : Shape}
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat) :
    Array (Fin dataShape.size) :=
  Internal.hiddenReconstructionIndices dataShape.size blocks period offset (Nat.le_refl _)

/--
Create a masked-reconstruction sample with arbitrary batch and data shapes.

The model input retains its original shape. The target is a row-major prefix of the unmasked source.
An invalid reconstruction width is reported at the ordinary executable boundary rather than
requiring callers to provide a theorem.
-/
def sample (batchShape : Shape) {dataShape : Shape} (reconstructionWidth : Nat)
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat)
    (x : Tensor Float (batchShape.concat dataShape)) :
    Except String (TorchLean.Sample.Supervised Float
      (batchShape.concat dataShape) (batchShape.appendDim reconstructionWidth)) :=
  if h : reconstructionWidth ≤ dataShape.size then
    .ok (Internal.sample batchShape reconstructionWidth blocks period offset h x)
  else
    .error (s!"BlockMAE reconstruction width {reconstructionWidth} "
      ++ s!"exceeds input width {dataShape.size}")

/--
Return the flattened reconstruction coordinates hidden by the block mask.

An invalid reconstruction width is reported explicitly.
-/
def hiddenReconstructionIndices {dataShape : Shape} (reconstructionWidth : Nat)
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat) :
    Except String (Array (Fin reconstructionWidth)) :=
  if h : reconstructionWidth ≤ dataShape.size then
    .ok (Internal.hiddenReconstructionIndices reconstructionWidth blocks period offset h)
  else
    .error (s!"BlockMAE reconstruction width {reconstructionWidth} "
      ++ s!"exceeds input width {dataShape.size}")

namespace Proof

/-- One batch row of block-MAE training as a finite predictive-view contract. -/
def rowPredictiveContract {dataShape : Shape} (batch reconstructionWidth : Nat)
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat)
    (hReconstruction : reconstructionWidth ≤ dataShape.size)
    (x : Tensor Float (dataShape.prependDim batch))
    (prediction : Tensor Float [batch, reconstructionWidth])
    (row : Fin batch) (loss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.PredictiveViewContract
      reconstructionWidth Unit Float Float Float :=
    NN.MLTheory.SelfSupervised.maeAsPredictiveViewContract
    (Internal.hiddenReconstructionIndices reconstructionWidth blocks period offset hReconstruction)
    (fun j => TorchLean.Tensor.item <|
      Spec.get (Spec.get
        (Internal.sample [batch] reconstructionWidth blocks period offset hReconstruction x).target
          row) j)
    (fun j => prediction[row][j])
    loss

/-- The runnable block-MAE row objective is exactly the finite MAE objective. -/
theorem row_predictive_objective_eq_mae_loss {dataShape : Shape}
    (batch reconstructionWidth : Nat)
    (blocks : Tensor (Option Nat) [dataShape.rank]) (period offset : Nat)
    (hReconstruction : reconstructionWidth ≤ dataShape.size)
    (x : Tensor Float (dataShape.prependDim batch))
    (prediction : Tensor Float [batch, reconstructionWidth])
    (row : Fin batch) (loss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.predictiveViewObjective
        (rowPredictiveContract batch reconstructionWidth blocks period offset hReconstruction
          x prediction row loss) =
      NN.MLTheory.SelfSupervised.maeLoss
        (Internal.hiddenReconstructionIndices reconstructionWidth blocks period offset
          hReconstruction)
        (fun j => TorchLean.Tensor.item <|
          Spec.get (Spec.get
            (Internal.sample [batch] reconstructionWidth blocks period offset hReconstruction
              x).target row) j)
        (fun j => prediction[row][j])
        loss := by
  exact NN.MLTheory.SelfSupervised.mae_is_predictive_view_objective
    (Internal.hiddenReconstructionIndices reconstructionWidth blocks period offset hReconstruction)
    (fun j => TorchLean.Tensor.item <|
      Spec.get (Spec.get
        (Internal.sample [batch] reconstructionWidth blocks period offset hReconstruction x).target
          row) j)
    (fun j => prediction[row][j])
    loss

end Proof
end BlockMAE

end ssl
end TorchLean
