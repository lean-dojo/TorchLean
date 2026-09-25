/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/

module

public import NN.Tensor.Internal.Representation.Basic.Pointwise

/-!
# Finite optional traversal

A successful finite traversal retains the value returned at every coordinate, including when the
index type is empty. The temporary buffer is confined to the internal representation proof.
-/

public section

namespace TorchLean.Tensor.Internal

/-- Sequencing a pure finite family retains the original indexed values. -/
@[simp] theorem sequenceFinM_pure {m : Type → Type} [Monad m] [LawfulMonad m]
    {α : Type} {n : Nat} (values : Fin n → α) :
    sequenceFinM (pure ∘ values : Fin n → m α) = pure values := by
  unfold sequenceFinM
  rw [Vector.ofFnM_pure_comp]
  simp only [pure_bind]
  congr 1
  funext index
  simp [Vector.get]
  congr 1

/-- Each coordinate of a successful finite traversal is the value returned at that coordinate. -/
private theorem vector_ofFnM_get_of_eq_some {α : Type} {n : Nat}
    {f : Fin n → Option α} {values : Vector α n}
    (h : Vector.ofFnM f = some values) (i : Fin n) : f i = some (values.get i) := by
  induction n with
  | zero => exact Fin.elim0 i
  | succ n ih =>
      rw [Vector.ofFnM_succ] at h
      obtain ⟨collected, hp, ht⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨last, hl, hv⟩ := Option.bind_eq_some_iff.mp ht
      have heq : collected.push last = values := Option.some.inj hv
      subst values
      refine Fin.lastCases ?_ (fun j => ?_) i
      · change f (Fin.last n) = some ((collected.push last)[n])
        simpa using hl
      · change f j.castSucc = some ((collected.push last)[j.val])
        rw [Vector.getElem_push_lt j.isLt]
        exact ih hp j

/-- Unpack a successful finite sequence without assumptions about positive cardinality. -/
theorem sequenceFinM_get_of_eq_some {α : Type} {n : Nat}
    {f : Fin n → Option α} {values : Fin n → α}
    (h : sequenceFinM f = some values) (i : Fin n) :
    f i = some (values i) := by
  unfold sequenceFinM at h
  obtain ⟨vector, hv, heq⟩ := Option.bind_eq_some_iff.mp h
  have heq : vector.get = values := Option.some.inj heq
  rw [← heq]
  exact vector_ofFnM_get_of_eq_some hv i

end TorchLean.Tensor.Internal
