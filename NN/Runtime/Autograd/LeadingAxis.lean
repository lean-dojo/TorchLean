/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Shape

/-!
# Leading-Axis Reference Maps

A backend-independent recursion for applying an operation pointwise over the outer axis of a
shape-typed reference. Eager and typed graph execution supply their own slicing, reshaping, and
concatenation operations; the traversal itself remains shared.
-/

@[expose] public section

namespace Runtime.Autograd

open Spec

/--
Apply a reference-level operation independently to every entry of a leading axis.

The callbacks isolate the four structural operations needed by the recursion. Device backends may
replace this reference traversal with a fused primitive when the fused operation has the same
per-entry semantics.

The axis is split in half and each half is mapped recursively, so `f` still runs on entries
`0, ..., n - 1` in order while every entry is sliced and concatenated once per level of a balanced
tree. Peeling one entry at a time would copy the remaining tail at every step.
-/
def mapOuterAxisWith {m : Type → Type} [Monad m] {Ref : Shape → Type} {σ τ : Shape}
    (empty : m (Ref (.dim 0 τ)))
    (slice : ∀ {n : Nat}, Ref (.dim n σ) → (start len : Nat) →
      (h : start + len ≤ n) → m (Ref (.dim len σ)))
    (reshape : ∀ {s₁ s₂ : Shape}, Ref s₁ → Shape.size s₁ = Shape.size s₂ → m (Ref s₂))
    (concat : ∀ {n k : Nat}, Ref (.dim n τ) → Ref (.dim k τ) →
      m (Ref (.dim (n + k) τ)))
    (f : Ref σ → m (Ref τ)) {n : Nat} (x : Ref (.dim n σ)) : m (Ref (.dim n τ)) :=
  match n with
  | 0 => empty
  | 1 => do
      let head ← reshape (s₁ := .dim 1 σ) (s₂ := σ) x (by simp [Shape.size])
      let yHead ← f head
      reshape (s₁ := τ) (s₂ := .dim 1 τ) yHead (by simp [Shape.size])
  | k + 2 => do
      let half := (k + 2) / 2
      let left ← slice (n := k + 2) x 0 half (by omega)
      let right ← slice (n := k + 2) x half (k + 2 - half) (by omega)
      let yLeft ← mapOuterAxisWith empty slice reshape concat f (n := half) left
      let yRight ← mapOuterAxisWith empty slice reshape concat f (n := k + 2 - half) right
      let y ← concat (n := half) (k := k + 2 - half) yLeft yRight
      have hSize : half + (k + 2 - half) = k + 2 := by omega
      pure (hSize ▸ y)
termination_by n

end Runtime.Autograd
