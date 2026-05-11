/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Data.List.Basic

/-!
# Masking primitives for self-supervised objectives

This file gives a small finite-index vocabulary for masked prediction objectives.  It is deliberately
independent of any particular image or transformer implementation: a patch/token collection is just
`Fin n → α`, and a mask is a Boolean predicate on `Fin n`.

The goal is to make MAE/JEPA-style objectives precise enough to prove local invariants before we
connect them to larger executable models.
-/

@[expose] public section

namespace NN.MLTheory.SelfSupervised

/-- A finite mask over `n` patches/tokens. `true` means the index is selected. -/
abbrev Mask (n : Nat) := Fin n → Bool

/-- Predicate form of a mask. -/
def selected {n : Nat} (m : Mask n) (i : Fin n) : Prop :=
  m i = true

/-- The all-visible/all-target mask. -/
def allMask (n : Nat) : Mask n :=
  fun _ => true

/-- The empty mask. -/
def emptyMask (n : Nat) : Mask n :=
  fun _ => false

/-- Complement a mask. -/
def complement {n : Nat} (m : Mask n) : Mask n :=
  fun i => !(m i)

@[simp] theorem allMask_selected {n : Nat} (i : Fin n) :
    selected (allMask n) i := by
  simp [selected, allMask]

@[simp] theorem emptyMask_not_selected {n : Nat} (i : Fin n) :
    ¬ selected (emptyMask n) i := by
  simp [selected, emptyMask]

@[simp] theorem complement_selected_iff {n : Nat} (m : Mask n) (i : Fin n) :
    selected (complement m) i ↔ ¬ selected m i := by
  simp [selected, complement]

/--
Generic masked loss over an explicit list of selected indices.

For theory files we keep the scalar loss as `Nat`; concrete runtime losses can instantiate this
with squared-error bins, quantized patch losses, or any other executable per-patch score.
-/
def maskedLoss {n : Nat} (idxs : List (Fin n)) (perPatchLoss : Fin n → Nat) : Nat :=
  (idxs.map perPatchLoss).sum

@[simp] theorem maskedLoss_nil {n : Nat} (perPatchLoss : Fin n → Nat) :
    maskedLoss ([] : List (Fin n)) perPatchLoss = 0 := by
  simp [maskedLoss]

@[simp] theorem maskedLoss_cons {n : Nat} (i : Fin n) (idxs : List (Fin n))
    (perPatchLoss : Fin n → Nat) :
    maskedLoss (i :: idxs) perPatchLoss =
      perPatchLoss i + maskedLoss idxs perPatchLoss := by
  simp [maskedLoss]

theorem maskedLoss_append {n : Nat} (xs ys : List (Fin n)) (perPatchLoss : Fin n → Nat) :
    maskedLoss (xs ++ ys) perPatchLoss =
      maskedLoss xs perPatchLoss + maskedLoss ys perPatchLoss := by
  simp [maskedLoss, List.map_append, List.sum_append]

theorem maskedLoss_reverse {n : Nat} (idxs : List (Fin n)) (perPatchLoss : Fin n → Nat) :
    maskedLoss idxs.reverse perPatchLoss = maskedLoss idxs perPatchLoss := by
  simp [maskedLoss]

theorem maskedLoss_eq_zero_of_all_zero {n : Nat} (idxs : List (Fin n))
    (perPatchLoss : Fin n → Nat) (h : ∀ i ∈ idxs, perPatchLoss i = 0) :
    maskedLoss idxs perPatchLoss = 0 := by
  induction idxs with
  | nil => simp
  | cons i rest ih =>
      have hi : perPatchLoss i = 0 := h i (by simp)
      have hrest : ∀ j ∈ rest, perPatchLoss j = 0 := by
        intro j hj
        exact h j (by simp [hj])
      simp [maskedLoss_cons, hi, ih hrest]

end NN.MLTheory.SelfSupervised
