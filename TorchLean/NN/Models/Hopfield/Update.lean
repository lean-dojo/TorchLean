/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Hopfield
public import Std.Tactic.BVDecide.Normalize.Prop

/-!
# Hopfield: basic lemmas

An asynchronous update changes one Boolean coordinate. The lemmas here describe that change and
its effect on the active-unit count, used to break energy ties in the convergence argument.
-/

@[expose] public section


namespace TorchLean.Hopfield

/-- After an update at `u`, unit `u` holds the threshold test `p.θ u ≤ net p s u`. -/
@[simp] theorem updateAt_apply_self {α : Type} [Add α] [Mul α] [Zero α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    {n : Nat} (p : Params α n) (s : State n) (u : Fin n) :
    updateAt (α := α) p s u u = decide (p.θ u ≤ net (α := α) p s u) := by
  simp [updateAt]

/-- An update at `u` leaves every other unit untouched, which is what makes it asynchronous. -/
@[simp] theorem updateAt_apply_ne {α : Type} [Add α] [Mul α] [Zero α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    {n : Nat} (p : Params α n) (s : State n) {u v : Fin n} (h : v ≠ u) :
    updateAt (α := α) p s u v = s v := by
  simp [updateAt, h]

/-- If the update at `u` changes the state, then `s u` is the negation of the threshold test:
every other unit is untouched, so the change happens at `u` itself. -/
theorem apply_eq_not_of_updateAt_ne {α : Type} [Add α] [Mul α] [Zero α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    {n : Nat} (p : Params α n) (s : State n) (u : Fin n)
    (h : updateAt (α := α) p s u ≠ s) :
    s u = !decide (p.θ u ≤ net (α := α) p s u) := by
  have hu : updateAt (α := α) p s u u ≠ s u := by
    intro hEq
    apply h
    funext i
    by_cases hi : i = u
    · subst hi; exact hEq
    · exact updateAt_apply_ne p s hi
  rw [updateAt_apply_self] at hu
  exact Bool.eq_not_of_ne hu.symm

/-- If unit `u` is `false` and its threshold test holds, the update raises the active count
`pluses` by one.

The active count breaks energy ties in the convergence proof: a state change that leaves the energy
flat has to raise this counter, and the counter cannot rise forever. -/
theorem pluses_updateAt_eq_succ {α : Type} [Add α] [Mul α] [Zero α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    {n : Nat} (p : Params α n) (s : State n) (u : Fin n)
    (hsu : s u = false)
    (hdec : decide (p.θ u ≤ net (α := α) p s u) = true) :
    pluses (n := n) (updateAt (α := α) p s u) = pluses (n := n) s + 1 := by
  classical
  have hu' : updateAt (α := α) p s u u = true := (updateAt_apply_self p s u).trans hdec
  let A : Finset (Fin n) := Finset.univ.filter fun i : Fin n => s i = true
  let A' : Finset (Fin n) := Finset.univ.filter fun i : Fin n => updateAt (α := α) p s u i = true
  have huA : u ∉ A := by
    simp [A, hsu]
  have hA' : A' = insert u A := by
    apply Finset.ext
    intro i
    by_cases hi : i = u
    · subst i
      simp only [A', Finset.mem_filter, Finset.mem_univ, true_and, hu',
        Finset.mem_insert_self]
    · simp [A', A, hi, Finset.mem_insert]
  have hcard : A'.card = A.card + 1 := by
    simp [hA', Finset.card_insert_of_notMem huA]
  simpa [Hopfield.pluses, A, A'] using hcard

/-- If unit `u` is `true` and its threshold test fails, the update lowers `pluses` by one. -/
theorem pluses_updateAt_eq_pred {α : Type} [Add α] [Mul α] [Zero α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    {n : Nat} (p : Params α n) (s : State n) (u : Fin n)
    (hsu : s u = true)
    (hdec : decide (p.θ u ≤ net (α := α) p s u) = false) :
    pluses (n := n) (updateAt (α := α) p s u) + 1 = pluses (n := n) s := by
  classical
  have hu' : updateAt (α := α) p s u u = false := (updateAt_apply_self p s u).trans hdec
  let A : Finset (Fin n) := Finset.univ.filter fun i : Fin n => s i = true
  let A' : Finset (Fin n) := Finset.univ.filter fun i : Fin n => updateAt (α := α) p s u i = true
  have huA : u ∈ A := by
    simp [A, hsu]
  have hA' : A' = A.erase u := by
    apply Finset.ext
    intro i
    by_cases hi : i = u
    · subst i
      simp only [A', Finset.mem_filter, Finset.mem_univ, true_and, hu', Bool.false_eq_true,
        Finset.notMem_erase]
    · simp [A', A, hi, Finset.mem_erase]
  have : A'.card + 1 = A.card := by
    simpa [hA'] using (Finset.card_erase_add_one huA)
  simpa [Hopfield.pluses, A, A'] using this

end TorchLean.Hopfield
