/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.SelfSupervised.Masking

/-!
# Joint-Embedding Predictive Objective Semantics

JEPA-style objectives predict target-block representations from context-block representations.
This file records the finite-index objective shape without committing to a particular vision
backbone, target encoder, or predictor architecture.

Paper anchor: “Self-Supervised Learning from Images with a Joint-Embedding Predictive
Architecture” (Assran et al., 2023), arXiv:2301.08243.  I-JEPA predicts target-block
representations from context-block representations rather than reconstructing pixels.  The target
branch is treated as a target representation at the objective boundary; this is why
`jepaLoss_target_ext` is useful: the loss depends only on target values at the selected target
indices.
-/

@[expose] public section

namespace NN.MLTheory.SelfSupervised

/--
JEPA loss over target block indices.

`context` abstracts the context encoder output, `target` abstracts target-block representations,
and `predict` abstracts the predictor head.  This keeps the objective theorem independent from any
particular image backbone.
-/
def jepaLoss {n : Nat} {Context Target Pred : Type}
    (targetIdxs : List (Fin n))
    (context : Context)
    (target : Fin n → Target)
    (predict : Context → Fin n → Pred)
    (repLoss : Target → Pred → Nat) : Nat :=
  maskedLoss targetIdxs (fun i => repLoss (target i) (predict context i))

@[simp] theorem jepaLoss_nil {n : Nat} {Context Target Pred : Type}
    (context : Context) (target : Fin n → Target) (predict : Context → Fin n → Pred)
    (repLoss : Target → Pred → Nat) :
    jepaLoss ([] : List (Fin n)) context target predict repLoss = 0 := by
  simp [jepaLoss]

theorem jepaLoss_append {n : Nat} {Context Target Pred : Type}
    (xs ys : List (Fin n)) (context : Context) (target : Fin n → Target)
    (predict : Context → Fin n → Pred) (repLoss : Target → Pred → Nat) :
    jepaLoss (xs ++ ys) context target predict repLoss =
      jepaLoss xs context target predict repLoss +
      jepaLoss ys context target predict repLoss := by
  simp [jepaLoss, maskedLoss_append]

/-- JEPA target-block prediction is invariant under reversing the target-index order. -/
theorem jepaLoss_reverse {n : Nat} {Context Target Pred : Type}
    (idxs : List (Fin n)) (context : Context) (target : Fin n → Target)
    (predict : Context → Fin n → Pred) (repLoss : Target → Pred → Nat) :
    jepaLoss idxs.reverse context target predict repLoss =
      jepaLoss idxs context target predict repLoss := by
  simp [jepaLoss, maskedLoss_reverse]

/--
Stop-gradient is modeled at the objective boundary: the target representation is an ordinary value
passed into the loss, not an output of the online predictor.  This theorem states the corresponding
extensional property: if two target branches agree on the selected indices, the JEPA loss is the
same.
-/
theorem jepaLoss_target_ext {n : Nat} {Context Target Pred : Type}
    (idxs : List (Fin n)) (context : Context)
    (target₁ target₂ : Fin n → Target)
    (predict : Context → Fin n → Pred) (repLoss : Target → Pred → Nat)
    (h : ∀ i ∈ idxs, target₁ i = target₂ i) :
    jepaLoss idxs context target₁ predict repLoss =
      jepaLoss idxs context target₂ predict repLoss := by
  induction idxs with
  | nil => simp
  | cons i rest ih =>
      have hi : target₁ i = target₂ i := h i (by simp)
      have hrest : ∀ j ∈ rest, target₁ j = target₂ j := by
        intro j hj
        exact h j (by simp [hj])
      simp [jepaLoss, maskedLoss_cons, hi]
      exact ih hrest

end NN.MLTheory.SelfSupervised
