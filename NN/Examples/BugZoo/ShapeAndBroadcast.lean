/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# BugZoo: shape and broadcasting boundaries

This file records the TorchLean response to a very common bug class: tensor shape mistakes that
would normally appear only at runtime, or worse, silently broadcast to the wrong expression.

The empirical motivation is concrete. Wu, Shen, and Chen build SFData, a corpus of 146 crashing
tensor-shape bugs, and use missing batch dimensions as a representative pattern:

- Wu, Shen, and Chen, “Detecting Tensor Shape Faults in Deep Learning Systems”, ISSTA 2022.
  https://doi.org/10.1145/3533767.3534383

Wang et al. also describe numerical bugs where a missing `keep_dims=True` changes a reduction shape;
NumPy/PyTorch-style broadcasting then makes a later expression typecheck while computing the wrong
loss:

- Wang et al., “An Empirical Study on Numerical Bugs in Deep Learning Programs”, ASE NIER 2022.
  https://doi.org/10.1145/3551349.3559561

TorchLean makes this case explicit: ordinary elementwise ops require the same shape, and
broadcasting requires `Shape.CanBroadcastTo` evidence. The examples below show the intended
workflow. A missing batch dimension is not silently accepted; we write the singleton batch
insertion. A reduced vector is not silently expanded back into a matrix; we carry the broadcast
proof.

Bug-shaped PyTorch sketch:

```python
# Crashes or silently changes later code depending on where it appears:
image = torch.randn(100, 100, 3)
model(image)          # model expected [1, 100, 100, 3]

# Easy to miss: reduction drops a dimension, then a later op broadcasts it back.
row_sum = x.sum(dim=0)          # [3], not [1, 3]
loss = ((x - row_sum) ** 2).sum()
```

TorchLean equivalent:

```lean
def batched := addSingletonBatch image
def row := reduceRows x
def explicit := broadcastRowToMatrix row
```

The important part is not the syntax; it is that the shape change and broadcast are named terms
with types and proof evidence.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.BugZoo.ShapeAndBroadcast

open TorchLean.Tensor

/-- A single HWC image: `100 x 100` pixels, three channels. -/
abbrev ImageShape : Spec.Shape :=
  [100, 100, 3]

/-- The same image with a batch axis of one in front. -/
abbrev SingletonBatchImageShape : Spec.Shape :=
  ImageShape.prependDim 1

/--
Insert an explicit singleton batch dimension.

This is the TorchLean version of the fix for the classic “forgot the batch axis” bug: we do not let
`Tensor α [100,100,3]` masquerade as `Tensor α [1,100,100,3]`; the user has to name the reshape.
-/
def addSingletonBatch {α : Type} [Storage α] (x : Tensor α ImageShape) :
    Tensor α SingletonBatchImageShape :=
  Tensor.repeatLeading 1 x

/-- Reading the only batch entry after `addSingletonBatch` gives back the original image. -/
@[simp] theorem addSingletonBatch_zero {α : Type} [Storage α]
    (x : Tensor α ImageShape) :
    (addSingletonBatch x)[0] = x := by
  change Tensor.unstack (addSingletonBatch x) ⟨0, by decide⟩ = x
  simp [addSingletonBatch]

/-- A small `2 x 3` matrix, used to show what a reduction does to the shape. -/
abbrev MatrixShape : Spec.Shape :=
  [2, 3]

/-- The row vector left after reducing `MatrixShape` over its outer axis. -/
abbrev RowShape : Spec.Shape :=
  [3]

/-- Sum over the outer axis of a `2 × 3` tensor, dropping that axis and producing a row vector. -/
def reduceRows {α : Type} [Storage α] [Add α] [Zero α]
    (x : Tensor α MatrixShape) :
    Tensor α RowShape :=
  Tensor.reduceSum 0 x Spec.Shape.NonemptyAxis.zero

/--
Evidence that a row vector can be broadcast back across the outer dimension of a `2 × 3` matrix.

This is exactly the piece TorchLean wants users and proof scripts to make visible: if a reduction
dropped a dimension, any later expansion is an explicit broadcast, not an accidental side effect.
-/
theorem rowBroadcastToMatrix : Spec.Shape.CanBroadcastTo RowShape MatrixShape :=
  Spec.Shape.CanBroadcastTo.expand_dims
    (Spec.Shape.CanBroadcastTo.refl RowShape)

/-- Broadcast a row vector to every row of a `2 × 3` matrix, using the evidence above. -/
def broadcastRowToMatrix {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α RowShape) :
    Tensor α MatrixShape :=
  Tensor.broadcastTo rowBroadcastToMatrix x

/-- The inferred broadcast follows NumPy/PyTorch's right-aligned convention. -/
def inferredRowBroadcastToMatrix {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α RowShape) :
    Tensor α MatrixShape :=
  Tensor.broadcastTo Spec.Shape.BroadcastTo.proof x

/-- The first row of an explicit broadcast is definitionally the original row. -/
@[simp] theorem broadcastRowToMatrix_firstRow {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α RowShape) :
    (broadcastRowToMatrix x)[0] = x := by
  change
    Tensor.unstack (broadcastRowToMatrix x) ⟨0, by decide⟩ = x
  simp [broadcastRowToMatrix]

/-- Inference and the explicit right-aligned witness compute the same matrix. -/
theorem inferredRowBroadcastToMatrix_eq {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α RowShape) :
    inferredRowBroadcastToMatrix x = broadcastRowToMatrix x := by
  rfl

/-!
The dimensions in the preceding example are different, so a reversed alignment convention would
fail rather than compute a different tensor. The square example below is the sharper regression:
both axes have length two, but NumPy/PyTorch broadcasting still requires the vector to align with
the final axis.
-/

abbrev SquareRowShape : Spec.Shape :=
  [2]

/-- A `2 x 2` matrix, the target of the broadcast below. -/
abbrev SquareMatrixShape : Spec.Shape :=
  [2, 2]

/-- Broadcast a length-two vector across the rows of a `2 x 2` matrix. -/
def inferredSquareBroadcast {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α SquareRowShape) : Tensor α SquareMatrixShape :=
  Tensor.broadcastTo Spec.Shape.BroadcastTo.proof x

/-- The inferred square broadcast is right-aligned: each matrix row is the source vector. -/
theorem inferredSquareBroadcast_rows {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α SquareRowShape) :
    (inferredSquareBroadcast x)[0] = x ∧
      (inferredSquareBroadcast x)[1] = x := by
  have hRows : ∀ i : Fin 2, Tensor.unstack (inferredSquareBroadcast x) i = x := by
    intro i
    unfold inferredSquareBroadcast
    rw [Tensor.broadcastTo_dim_self, Tensor.unstack_dim]
  exact ⟨hRows ⟨0, by decide⟩, hRows ⟨1, by decide⟩⟩

end NN.Examples.BugZoo.ShapeAndBroadcast
