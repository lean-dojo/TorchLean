/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Tactic.Attr.Core

/-!
# List fold lemmas for runtime-approximation proofs

These two list-generic lemmas align the flat and nested index traversals used by convolution and
linear-algebra error bounds.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox
namespace NFBackend

/-- Replace the step function of a left fold by a pointwise equal function. -/
theorem foldl_congr {α β : Type} (l : List β) (f g : α → β → α) (init : α)
    (h : ∀ a b, f a b = g a b) :
    l.foldl f init = l.foldl g init := by
  induction l generalizing init with
  | nil => rfl
  | cons b tail ih =>
      simp [List.foldl, h, ih]

/-- A fold over `flatMap` agrees with the corresponding nested fold. -/
theorem foldl_flatMap {α β γ : Type}
    (l : List α) (g : α → List β) (f : γ → β → γ) (init : γ) :
    (l.flatMap g).foldl f init = l.foldl (fun acc a => (g a).foldl f acc) init := by
  induction l generalizing init with
  | nil => simp
  | cons a tail ih =>
      simp [List.flatMap_cons, List.foldl_append, ih]

end NFBackend
end RuntimeApprox
end Proofs
