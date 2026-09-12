/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Masking primitives for self-supervised objectives

This file gives a small finite-index vocabulary for masked prediction objectives. It stays
independent of any particular image or transformer implementation: a patch/token collection is just
`Fin n → α`, and a mask is a Boolean predicate on `Fin n`.

The definitions make MAE/JEPA-style objectives precise enough for local invariants before they are
connected to larger executable models.
-/

@[expose] public section

namespace NN.MLTheory.SelfSupervised

/-- A finite mask over `n` patches/tokens. `true` means the index is selected. -/
abbrev Mask (n : Nat) := Fin n → Bool

/-- Proposition stating that index `i` is selected by the Boolean mask `m`. -/
def selected {n : Nat} (m : Mask n) (i : Fin n) : Prop :=
  m i = true

/-- The all-visible/all-target mask. -/
def allMask (n : Nat) : Mask n :=
  fun _ => true

/-- Mask selecting no positions. -/
def emptyMask (n : Nat) : Mask n :=
  fun _ => false

/-- Pointwise Boolean complement of a mask. -/
def complement {n : Nat} (m : Mask n) : Mask n :=
  fun i => !(m i)

/-- Every index is selected by the full mask. -/
@[simp] theorem allMask_selected {n : Nat} (i : Fin n) :
    selected (allMask n) i := by
  simp [selected, allMask]

/-- No index is selected by the empty mask. -/
@[simp] theorem emptyMask_not_selected {n : Nat} (i : Fin n) :
    ¬ selected (emptyMask n) i := by
  simp [selected, emptyMask]

/-- Complementing a mask negates selection pointwise. This supports a context/target partition when
the caller chooses complementary masks; the objective definitions do not enforce that choice. -/
@[simp] theorem complement_selected_iff {n : Nat} (m : Mask n) (i : Fin n) :
    selected (complement m) i ↔ ¬ selected m i := by
  simp [selected, complement]

/--
Generic masked loss over an explicit array of selected indices.

The scalar loss is `Nat`, so this models natural-valued scores such as quantized patch losses.
Each array occurrence contributes once; duplicate indices are not removed.
-/
def maskedLoss {n : Nat} (idxs : Array (Fin n)) (perPatchLoss : Fin n → Nat) : Nat :=
  (idxs.map perPatchLoss).sum

/-- An empty index array contributes no loss. -/
@[simp] theorem maskedLoss_nil {n : Nat} (perPatchLoss : Fin n → Nat) :
    maskedLoss (#[] : Array (Fin n)) perPatchLoss = 0 := by
  simp [maskedLoss]

/-- Pushing one index adds that patch's loss. -/
@[simp] theorem maskedLoss_push {n : Nat} (idxs : Array (Fin n)) (i : Fin n)
    (perPatchLoss : Fin n → Nat) :
    maskedLoss (idxs.push i) perPatchLoss =
      maskedLoss idxs perPatchLoss + perPatchLoss i := by
  simp [maskedLoss, Array.map_push]

/-- Masked loss is additive in the index array. -/
theorem maskedLoss_append {n : Nat} (xs ys : Array (Fin n)) (perPatchLoss : Fin n → Nat) :
    maskedLoss (xs ++ ys) perPatchLoss =
      maskedLoss xs perPatchLoss + maskedLoss ys perPatchLoss := by
  simp [maskedLoss, Array.map_append, Array.sum_append]

/-- Masked loss does not depend on the order of the indices, so a shuffled mask scores the same. -/
theorem maskedLoss_reverse {n : Nat} (idxs : Array (Fin n)) (perPatchLoss : Fin n → Nat) :
    maskedLoss idxs.reverse perPatchLoss = maskedLoss idxs perPatchLoss := by
  simp [maskedLoss, Array.map_reverse, Array.sum_reverse]

/-- Zero per-patch loss at every selected index gives zero total loss. This theorem states only that
direction; it does not assume the per-patch score characterizes perfect reconstruction. -/
theorem maskedLoss_eq_zero_of_all_zero {n : Nat} (idxs : Array (Fin n))
    (perPatchLoss : Fin n → Nat) (h : ∀ i ∈ idxs, perPatchLoss i = 0) :
    maskedLoss idxs perPatchLoss = 0 := by
  have hmap : idxs.map perPatchLoss = Array.replicate idxs.size 0 := by
    apply Array.ext <;> simp only [Array.size_map, Array.size_replicate]
    intro i hiMap hiReplicate
    rw [Array.getElem_map, Array.getElem_replicate]
    exact h idxs[i] (Array.getElem_mem hiMap)
  rw [maskedLoss, hmap]
  simp

end NN.MLTheory.SelfSupervised
