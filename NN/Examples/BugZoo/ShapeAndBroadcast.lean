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
def batched := Tensor.repeatLeading 1 image
def row := Tensor.reduceSum 0 x Spec.Shape.NonemptyAxis.zero
def explicit : Tensor Float [2, 3] :=
  Tensor.broadcastTo Spec.Shape.BroadcastTo.proof row
```

The important part is not the syntax; it is that the shape change and broadcast are named terms
with types and proof evidence.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.BugZoo.ShapeAndBroadcast

/-- Reading the inserted singleton batch recovers the image. -/
@[simp] theorem repeatLeading_zero {α : Type} [Storage α]
    (x : Tensor α [100, 100, 3]) :
    (Tensor.repeatLeading 1 x)[0] = x := by
  change Tensor.unstack (Tensor.repeatLeading 1 x) ⟨0, by decide⟩ = x
  exact Tensor.unstack_repeatLeading x _

/--
Evidence that a row vector can be broadcast back across the outer dimension of a `2 × 3` matrix.

This is exactly the piece TorchLean wants users and proof scripts to make visible: if a reduction
dropped a dimension, any later expansion is an explicit broadcast, not an accidental side effect.
-/
theorem row_broadcast : Spec.Shape.CanBroadcastTo [3] [2, 3] :=
  Spec.Shape.CanBroadcastTo.expand_dims
    (Spec.Shape.CanBroadcastTo.refl [3])

/-- The first row of an explicit broadcast is definitionally the original row. -/
@[simp] theorem broadcast_first {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α [3]) :
    (Tensor.broadcastTo row_broadcast x)[0] = x := by
  change
    Tensor.unstack (Tensor.broadcastTo row_broadcast x) ⟨0, by decide⟩ = x
  simp

/-- Inference and the explicit right-aligned witness compute the same matrix. -/
theorem broadcast_infer_eq {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α [3]) :
    Tensor.broadcastTo (s₂ := [2, 3]) Spec.Shape.BroadcastTo.proof x =
      Tensor.broadcastTo row_broadcast x := by
  rfl

/-!
The dimensions in the preceding example are different, so a reversed alignment convention would
fail rather than compute a different tensor. The square example below is the sharper regression:
both axes have length two, but NumPy/PyTorch broadcasting still requires the vector to align with
the final axis.
-/

/-- The inferred square broadcast is right-aligned: each matrix row is the source vector. -/
theorem broadcast_square_rows {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α [2]) :
    (Tensor.broadcastTo (s₂ := [2, 2]) Spec.Shape.BroadcastTo.proof x)[0] = x ∧
      (Tensor.broadcastTo (s₂ := [2, 2]) Spec.Shape.BroadcastTo.proof x)[1] = x := by
  have hRows : ∀ i : Fin 2,
      Tensor.unstack (Tensor.broadcastTo (s₂ := [2, 2]) Spec.Shape.BroadcastTo.proof x) i = x := by
    intro i
    rw [Tensor.broadcastTo_dim_self, Tensor.unstack_dim]
  exact ⟨hRows ⟨0, by decide⟩, hRows ⟨1, by decide⟩⟩

end NN.Examples.BugZoo.ShapeAndBroadcast
