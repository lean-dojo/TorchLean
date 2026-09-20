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
and `predict` abstracts the predictor head. The objective theorem is independent from any
particular image backbone.
-/
def jepaLoss {n : Nat} {Context Target Pred : Type}
    (targetIdxs : Array (Fin n))
    (context : Context)
    (target : Fin n → Target)
    (predict : Context → Fin n → Pred)
    (repLoss : Target → Pred → Nat) : Nat :=
  maskedLoss targetIdxs (fun i => repLoss (target i) (predict context i))

/-- Predicting no targets costs nothing. -/
@[simp] theorem jepaLoss_nil {n : Nat} {Context Target Pred : Type}
    (context : Context) (target : Fin n → Target) (predict : Context → Fin n → Pred)
    (repLoss : Target → Pred → Nat) :
    jepaLoss (#[] : Array (Fin n)) context target predict repLoss = 0 := by
  simp [jepaLoss]

/-- The natural-valued JEPA loss is additive under concatenating target-index arrays. This identity
does not assert equality between different floating-point reduction orders. -/
theorem jepaLoss_append {n : Nat} {Context Target Pred : Type}
    (xs ys : Array (Fin n)) (context : Context) (target : Fin n → Target)
    (predict : Context → Fin n → Pred) (repLoss : Target → Pred → Nat) :
    jepaLoss (xs ++ ys) context target predict repLoss =
      jepaLoss xs context target predict repLoss +
      jepaLoss ys context target predict repLoss := by
  simp [jepaLoss, maskedLoss_append]

/-- JEPA target-block prediction is invariant under reversing the target-index order. -/
theorem jepaLoss_reverse {n : Nat} {Context Target Pred : Type}
    (idxs : Array (Fin n)) (context : Context) (target : Fin n → Target)
    (predict : Context → Fin n → Pred) (repLoss : Target → Pred → Nat) :
    jepaLoss idxs.reverse context target predict repLoss =
      jepaLoss idxs context target predict repLoss := by
  simp [jepaLoss, maskedLoss_reverse]

/--
If two target branches agree on the selected indices, the JEPA loss is the same. Targets are
ordinary values here; this extensional identity does not specify stop-gradient or differentiate
either branch.
-/
theorem jepaLoss_target_ext {n : Nat} {Context Target Pred : Type}
    (idxs : Array (Fin n)) (context : Context)
    (target₁ target₂ : Fin n → Target)
    (predict : Context → Fin n → Pred) (repLoss : Target → Pred → Nat)
    (h : ∀ i ∈ idxs, target₁ i = target₂ i) :
    jepaLoss idxs context target₁ predict repLoss =
      jepaLoss idxs context target₂ predict repLoss := by
  apply congrArg Array.sum
  apply Array.ext <;> simp only [Array.size_map]
  intro i hi₁ hi₂
  simp only [Array.getElem_map]
  rw [h idxs[i] (Array.getElem_mem hi₁)]

end NN.MLTheory.SelfSupervised
