/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Operations
public import TorchLean.Tensor.Tactic.Init

/-!
# BugZoo: shape and broadcasting boundaries

Tensor shape mistakes are a very common bug class. Usually they surface only at runtime, and the
worse ones broadcast silently into the wrong expression.

The empirical motivation is concrete. Wu, Shen, and Chen build SFData, a corpus of 146 crashing
tensor-shape bugs, and use missing batch dimensions as a representative pattern:

- Wu, Shen, and Chen, “Detecting Tensor Shape Faults in Deep Learning Systems”, ISSTA 2022.
  https://doi.org/10.1145/3533767.3534383

Wang et al. describe numerical bugs in which a missing `keep_dims=True` changes a reduction shape,
and NumPy/PyTorch-style broadcasting then lets a later expression typecheck while it computes the
wrong loss:

- Wang et al., “An Empirical Study on Numerical Bugs in Deep Learning Programs”, ASE NIER 2022.
  https://doi.org/10.1145/3551349.3559561

In TorchLean, ordinary elementwise ops require equal shapes, and broadcasting requires
`Shape.CanBroadcastTo` evidence. The examples follow the intended workflow. A missing batch
dimension gets a written singleton batch insertion, and a reduced vector that has to become a matrix
again carries its broadcast proof.

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
def row := Tensor.castShape (Tensor.reduce .sum x [⟨0, by decide⟩] false)
  (TorchLean.Shape.reduce_singleton _ _)
def explicit : Tensor Float [2, 3] :=
  Tensor.broadcast row [2, 3] row_broadcast.dims row_broadcast.broadcastDims
```

What matters is that the shape change and the broadcast are named terms with types and proof
evidence.
-/

@[expose] public section

namespace TorchLean.Examples.BugZoo.Broadcasting

/--
Evidence that a length-3 vector broadcasts across the outer dimension of a `2 × 3` matrix.

TorchLean wants users and proof scripts to write this piece down: after a reduction drops a
dimension, any later expansion is a named broadcast with a proof attached.
-/
theorem row_broadcast : Shape.CanBroadcastTo [3] [2, 3] :=
  Shape.CanBroadcastTo.expand_dims
    (Shape.CanBroadcastTo.refl [3])

/-- For every witness `h`, each row of the broadcast of `x : Tensor α [m]` to `[n, m]` is `x`. -/
theorem unstack_broadcast_row {α : Type} [Storage α] {n m : Nat} (x : Tensor α [m])
    (h : Shape.CanBroadcastTo [m] [n, m]) (i : Fin n) :
    Tensor.unstack (Tensor.broadcast x [n, m] h.dims h.broadcastDims) i = x := by
  apply Tensor.Rep.ext
  intro c
  simp only [Tensor.unstack_apply]
  rw [Tensor.broadcast_apply]
  congr 1
  apply Shape.Coord.ext_val
  intro d
  fin_cases d
  simp only [Shape.Coord.get_ofFun]
  split_ifs with h1
  · have := c.1.isLt
    change m = 1 at h1
    show 0 = c.1.val
    omega
  · rfl

/-- Row `0` of the `row_broadcast` expansion is the original vector. -/
@[simp] theorem broadcast_first {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α [3]) :
    (Tensor.broadcast x [2, 3] row_broadcast.dims row_broadcast.broadcastDims)[0] = x := by
  change Tensor.unstack (Tensor.broadcast x [2, 3] row_broadcast.dims row_broadcast.broadcastDims)
    ⟨0, by decide⟩ = x
  exact unstack_broadcast_row x row_broadcast _

/-- The witness found by instance inference for the same shapes. -/
theorem inferred_broadcast : Shape.CanBroadcastTo [3] [2, 3] := Shape.BroadcastTo.proof

/-- The inferred witness and the hand-built `row_broadcast` give the same broadcast matrix. -/
theorem broadcast_infer_eq {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α [3]) :
    Tensor.broadcast x [2, 3] inferred_broadcast.dims inferred_broadcast.broadcastDims =
      Tensor.broadcast x [2, 3] row_broadcast.dims row_broadcast.broadcastDims := by
  rfl

/-!
In the `[3]` to `[2, 3]` example the dimensions differ, so a reversed alignment convention finds
no witness at all. The square case is the sharper regression test: both axes have length two, so
either alignment would fit, and NumPy/PyTorch broadcasting puts the vector along the final axis.
-/

/-- The witness found by instance inference for a vector of length two and a `2 × 2` matrix. -/
theorem square_broadcast : Shape.CanBroadcastTo [2] [2, 2] := Shape.BroadcastTo.proof

/-- The inferred square broadcast is right-aligned: each matrix row is the source vector. -/
theorem broadcast_square_rows {α : Type} [Storage α] [Inhabited α]
    (x : Tensor α [2]) :
    (Tensor.broadcast x [2, 2] square_broadcast.dims square_broadcast.broadcastDims)[0] = x ∧
      (Tensor.broadcast x [2, 2] square_broadcast.dims square_broadcast.broadcastDims)[1] = x :=
  ⟨unstack_broadcast_row x square_broadcast ⟨0, by decide⟩,
    unstack_broadcast_row x square_broadcast ⟨1, by decide⟩⟩

end TorchLean.Examples.BugZoo.Broadcasting
