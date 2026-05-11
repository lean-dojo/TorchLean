/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

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
  https://conf.researchr.org/details/ase-2022/ase-2022-nier-track/18/An-Empirical-Study-on-Numerical-Bugs-in-Deep-Learning-Programs

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

namespace NN.Examples.BugZoo.ShapeAndBroadcast

open Spec
open Spec.Tensor

abbrev ImageShape : Spec.Shape :=
  .dim 100 (.dim 100 (.dim 3 .scalar))

abbrev SingletonBatchImageShape : Spec.Shape :=
  .dim 1 ImageShape

/--
Insert an explicit singleton batch dimension.

This is the TorchLean version of the fix for the classic “forgot the batch axis” bug: we do not let
`Tensor α [100,100,3]` masquerade as `Tensor α [1,100,100,3]`; the user has to name the reshape.
-/
def addSingletonBatch {α : Type} (x : Spec.Tensor α ImageShape) :
    Spec.Tensor α SingletonBatchImageShape :=
  Spec.Tensor.dim fun _ => x

/-- Reading the only batch entry after `addSingletonBatch` gives back the original image. -/
@[simp] theorem addSingletonBatch_zero {α : Type} (x : Spec.Tensor α ImageShape) :
    match addSingletonBatch x with
    | Spec.Tensor.dim batch => batch ⟨0, by decide⟩ = x := by
  rfl

abbrev MatrixShape : Spec.Shape :=
  .dim 2 (.dim 3 .scalar)

abbrev RowShape : Spec.Shape :=
  .dim 3 .scalar

/-- Sum over the outer axis of a `2 × 3` tensor, dropping that axis and producing a row vector. -/
def reduceRows {α : Type} [Add α] [Zero α] (x : Spec.Tensor α MatrixShape) :
    Spec.Tensor α RowShape :=
  Spec.Tensor.reduceSumAuto 0 x

/--
Evidence that a row vector can be broadcast back across the outer dimension of a `2 × 3` matrix.

This is exactly the piece TorchLean wants users and proof scripts to make visible: if a reduction
dropped a dimension, any later expansion is an explicit broadcast, not an accidental side effect.
-/
def rowBroadcastToMatrix : Spec.Shape.CanBroadcastTo RowShape MatrixShape :=
  Spec.Shape.CanBroadcastTo.expand_dims
    (Spec.Shape.CanBroadcastTo.dim_eq (Spec.Shape.CanBroadcastTo.scalar_to_any .scalar))

/-- Broadcast a row vector to every row of a `2 × 3` matrix, using the evidence above. -/
def broadcastRowToMatrix {α : Type} [Inhabited α] (x : Spec.Tensor α RowShape) :
    Spec.Tensor α MatrixShape :=
  Spec.Tensor.broadcastTo rowBroadcastToMatrix x

/-- The first row of an explicit broadcast is definitionally the original row. -/
@[simp] theorem broadcastRowToMatrix_firstRow {α : Type} [Inhabited α]
    (x : Spec.Tensor α RowShape) :
    match broadcastRowToMatrix x with
    | Spec.Tensor.dim rows => rows ⟨0, by decide⟩ = x := by
  cases x with
  | dim xs =>
      rfl

end NN.Examples.BugZoo.ShapeAndBroadcast
