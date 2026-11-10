/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.Idx

/-!
# Lemmas on typed context indices

The context indices `Idx` and their tensor selection `TensorPack.getIdx` are defined in
`TorchLean.NN.TypedGraph.Idx`. This module proves their extensionality and that
selection commutes with shape-preserving pack maps.

## Main declarations

- `TorchLean.Idx.ext`: two indices at the same position are equal.
- `TorchLean.TensorPack.getIdx_map`: a shape-preserving pack map commutes with typed selection.
-/

@[expose] public section

namespace TorchLean

/-- Two typed indices with the same position are equal. -/
@[ext] theorem Idx.ext {Γ : List Shape} {s : Shape} {a b : Idx Γ s}
    (h : a.i = b.i) : a = b := by
  cases a
  cases b
  cases h
  rfl

/-- A shape-preserving pack map commutes with typed tensor selection. -/
@[simp] theorem TensorPack.getIdx_map {α β : Type} [Storage α] [Storage β] {shapes : List Shape}
    (f : ∀ {s : Shape}, Tensor α s → Tensor β s) (xs : TensorPack α shapes)
    {shape : Shape} (index : Idx shapes shape) :
    getIdx (xs.map f) index = f (getIdx xs index) := by
  obtain ⟨i, rfl⟩ := index
  induction shapes with
  | nil => exact Fin.elim0 i
  | cons s ss ih =>
    cases xs with
    | cons x xs =>
      cases i using Fin.cases with
      | zero => rfl
      | succ i => exact ih xs i

end TorchLean
