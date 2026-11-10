/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Ops
public import TorchLean.NN.Batch

/-!
# Tensor Operations over Backend References

`Ops.mapBatch` maps a function over the leading axis by slicing and concatenating references, the
shared traversal for a backend with no fused batch operation.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

variable {m : Type → Type} {α : Type} [Storage α] [Context α] [Monad m]
    [Ops (m := m) (α := α)]

/--
Apply `f` to each leading-axis slice and concatenate the results in order.

An empty leading axis returns an empty reference without calling `f`.
-/
def Ops.mapBatch {σ τ : Shape}
    (f : Ref (m := m) (α := α) σ → m (Ref (m := m) (α := α) τ)) {n : Nat}
    (x : Ref (m := m) (α := α) (Shape.dim n σ)) :
    m (Ref (m := m) (α := α) (Shape.dim n τ)) :=
  TorchLean.mapBatch
    (Ops.const (m := m) (α := α) (s := Shape.dim 0 τ) (Tensor.full (Shape.dim 0 τ) (0 : α)))
    (fun x start len h =>
      Ref.cast (Shape.slice_singleton (.dim _ _) ⟨0, by simp⟩ start len) <$>
        Ops.apply (m := m) (α := α)
          (.slice (s := .dim _ _) [⟨0, by simp⟩] [start] [start + len] [1]
            (Shape.sliceValid_singleton _ ⟨0, by simp⟩ start len h)) (x, ()))
    (fun x h => Ops.apply (m := m) (α := α) (.reshape h) (x, ()))
    (fun a b => Ops.apply (m := m) (α := α) (.cat (s := .dim 0 _) ⟨0, by simp⟩ [_, _]) (a, b, ()))
    f x

end TorchLean
