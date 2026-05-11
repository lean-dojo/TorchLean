/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Hopfield

/-!
# Hopfield: basic lemmas

This file contains small, reusable lemmas about the spec-level Hopfield definitions.

The “paper theorems” (energy monotonicity, convergence, Hebbian stability) should live in
separate files once proved.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.Hopfield

open Spec

open Spec.Hopfield

@[simp] lemma updateAt_apply_eq_update {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    {n : Nat} (p : Params α n) (s : State n) (u : Fin n) :
    updateAt (α := α) p s u =
      Function.update s u (decide (p.θ u ≤ net (α := α) p s u)) := by
  rfl

@[simp] lemma updateAt_apply_self {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    {n : Nat} (p : Params α n) (s : State n) (u : Fin n) :
    updateAt (α := α) p s u u = decide (p.θ u ≤ net (α := α) p s u) := by
  simp [updateAt]

@[simp] lemma updateAt_apply_ne {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    {n : Nat} (p : Params α n) (s : State n) {u v : Fin n} (h : v ≠ u) :
    updateAt (α := α) p s u v = s v := by
  simp [updateAt, h]

lemma pluses_updateAt_eq_succ_of_set_true {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    {n : Nat} (p : Params α n) (s : State n) (u : Fin n)
    (hsu : s u = false)
    (hdec : decide (p.θ u ≤ net (α := α) p s u) = true) :
    pluses (n := n) (updateAt (α := α) p s u) = pluses (n := n) s + 1 := by
  classical
  -- `updateAt` only changes coordinate `u`, and it is set to `true`.
  have hu' : updateAt (α := α) p s u u = true := by
    have hu'' :
        updateAt (α := α) p s u u = decide (p.θ u ≤ net (α := α) p s u) := by
      simp [updateAt]
    exact hu''.trans hdec
  -- Describe the filtered true-set after the update.
  let A : Finset (Fin n) := Finset.univ.filter fun i : Fin n => s i = true
  let A' : Finset (Fin n) := Finset.univ.filter fun i : Fin n => updateAt (α := α) p s u i = true
  have huA : u ∉ A := by
    simp [A, hsu]
  have hA' : A' = insert u A := by
    apply Finset.ext
    intro i
    by_cases hi : i = u
    · subst i
      constructor
      · intro _
        exact Finset.mem_insert_self u A
      · intro _
        -- `u ∈ A'` since `u ∈ univ` and the predicate holds by `hu'`.
        have hupred : updateAt (α := α) p s u u = true := hu'
        have : u ∈ (Finset.univ.filter fun i : Fin n => updateAt (α := α) p s u i = true) :=
          Finset.mem_filter.2 ⟨by simp, hupred⟩
        simpa [A'] using this
    · simp [A', A, hi, Finset.mem_insert]
  -- Convert back to `pluses`.
  have hcard : A'.card = A.card + 1 := by
    simp [hA', Finset.card_insert_of_notMem huA]
  simpa [Spec.Hopfield.pluses, A, A'] using hcard

lemma pluses_updateAt_eq_pred_of_set_false {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    {n : Nat} (p : Params α n) (s : State n) (u : Fin n)
    (hsu : s u = true)
    (hdec : decide (p.θ u ≤ net (α := α) p s u) = false) :
    pluses (n := n) (updateAt (α := α) p s u) + 1 = pluses (n := n) s := by
  classical
  -- `updateAt` only changes coordinate `u`, and it is set to `false`.
  have hu' : updateAt (α := α) p s u u = false := by
    have hu'' :
        updateAt (α := α) p s u u = decide (p.θ u ≤ net (α := α) p s u) := by
      simp [updateAt]
    exact hu''.trans hdec
  let A : Finset (Fin n) := Finset.univ.filter fun i : Fin n => s i = true
  let A' : Finset (Fin n) := Finset.univ.filter fun i : Fin n => updateAt (α := α) p s u i = true
  have huA : u ∈ A := by
    simp [A, hsu]
  have hA' : A' = A.erase u := by
    apply Finset.ext
    intro i
    by_cases hi : i = u
    · subst i
      constructor
      · intro hmem
        -- `u ∈ A'` would imply `updateAt ... u = true`, contradicting `hu' : ... = false`.
        have hpred :
            updateAt (α := α) p s u u = true := by
          -- Unfold the membership in the defining filter.
          have hmem' :
              u ∈ (Finset.univ.filter fun i : Fin n => updateAt (α := α) p s u i = true) := by
            simpa [A'] using hmem
          exact (Finset.mem_filter.1 hmem').2
        exact Bool.noConfusion (Eq.trans (Eq.symm hpred) hu')
      · intro hmem
        -- `u ∈ A.erase u` is impossible.
        exact False.elim ((Finset.notMem_erase u A) hmem)
    · simp [A', A, hi, Finset.mem_erase]
  -- `card (A.erase u) + 1 = card A` since `u ∈ A`.
  have : A'.card + 1 = A.card := by
    simpa [hA'] using (Finset.card_erase_add_one huA)
  simpa [Spec.Hopfield.pluses, A, A'] using this

end NN.MLTheory.Proofs.Hopfield
