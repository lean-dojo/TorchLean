/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Fintype.Card
public import Mathlib.Data.Fintype.Pigeonhole
public import Mathlib.Logic.Function.Iterate
public import Mathlib.Order.Monotone.Basic
public import NN.MLTheory.Proofs.Hopfield.Progress

/-!
# Hopfield cyclic-sweep convergence (finite-state argument)

This file uses the tie-handling progress lemma from `progress.lean` to prove the classical
finite-state global dynamics facts for cyclic sweeps:

* **No nontrivial cycles** for the full-sweep update `cycleUpdate` (hence convergence).
* A coarse **convergence bound** of at most `2^n` sweeps (and therefore `n * 2^n` single-coordinate
  updates) from any initial state, by a pigeonhole argument on the finite state space `Bool^n`.

We keep the statement at the “sweep level” (one full pass over coordinates). Connecting this to
`seqStates` with `cyclicUseq` is routine and can be layered on top.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.Hopfield

open scoped BigOperators
open Spec

open Spec.Hopfield

variable {n : Nat}

private lemma pluses_le_dim (s : State n) : pluses (n := n) s ≤ n := by
  classical
  -- `pluses` is the cardinality of a filtered subset of `Finset.univ`.
  unfold Spec.Hopfield.pluses
  simpa using
    (le_trans (Finset.card_filter_le (s := (Finset.univ : Finset (Fin n)))
      (p := fun i : Fin n => s i = true)) (by simp))

section

variable (p : Params ℝ n)

noncomputable def f : State n → State n := cycleUpdate (n := n) p

private lemma energy_iterate_le
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (k : Nat) (s : State n) :
    energy (α := ℝ) p ((f (n := n) p)^[k] s) ≤ energy (α := ℝ) p s := by
  classical
  induction k with
  | zero =>
      simp
  | succ k IH =>
      have hstep :
          energy (α := ℝ) p ((f (n := n) p) ((f (n := n) p)^[k] s))
            ≤
          energy (α := ℝ) p ((f (n := n) p)^[k] s) :=
        energy_cycleUpdate_le (n := n) (p := p) hsym hdiag ((f (n := n) p)^[k] s)
      simpa [Function.iterate_succ_apply'] using le_trans hstep IH

private lemma energy_iterate_antitone
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    {i j : Nat} (hij : i ≤ j) (s : State n) :
    energy (α := ℝ) p ((f (n := n) p)^[j] s) ≤ energy (α := ℝ) p ((f (n := n) p)^[i] s) := by
  classical
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hij
  -- Apply the `d`-step bound from the state `f^[i] s`, then rewrite the iterate as `f^[i+d] s`.
  have h0 :
      energy (α := ℝ) p ((f (n := n) p)^[d] ((f (n := n) p)^[i] s)) ≤
        energy (α := ℝ) p ((f (n := n) p)^[i] s) :=
    energy_iterate_le (n := n) (p := p) hsym hdiag d ((f (n := n) p)^[i] s)
  have e :
      (f (n := n) p)^[d + i] s = (f (n := n) p)^[d] ((f (n := n) p)^[i] s) := by
    simpa using (Function.iterate_add_apply (f := f (n := n) p) d i s)
  have h1 :
      energy (α := ℝ) p ((f (n := n) p)^[d + i] s) ≤
        energy (α := ℝ) p ((f (n := n) p)^[i] s) := by
    -- Rewrite the left-hand side of `h0` via `e` (in the reverse direction).
    have h0' := h0
    -- `h0' : energy p (f^[d] (f^[i] s)) ≤ ...`
    -- Replace `f^[d] (f^[i] s)` with `f^[d+i] s`.
    rw [← e] at h0'
    exact h0'
  -- Commute the addition in the iterate index.
  simpa [Nat.add_comm] using h1

private lemma energy_iterate_eq_of_iterate_eq
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    {k i : Nat} (hi : i ≤ k) {s : State n}
    (hcyc : (f (n := n) p)^[k] s = s) :
    energy (α := ℝ) p ((f (n := n) p)^[i] s) = energy (α := ℝ) p s := by
  have hle : energy (α := ℝ) p ((f (n := n) p)^[i] s) ≤ energy (α := ℝ) p s :=
    energy_iterate_le (n := n) (p := p) hsym hdiag i s
  have hle' : energy (α := ℝ) p s ≤ energy (α := ℝ) p ((f (n := n) p)^[i] s) := by
    have hk_le :
        energy (α := ℝ) p ((f (n := n) p)^[k] s) ≤ energy (α := ℝ) p ((f (n := n) p)^[i] s) :=
      energy_iterate_antitone (n := n) (p := p) hsym hdiag hi s
    simpa [hcyc] using hk_le
  exact le_antisymm hle hle'

private lemma pluses_cycleUpdate_ge_of_energy_eq
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (s : State n)
    (hE : energy (α := ℝ) p ((f (n := n) p) s) = energy (α := ℝ) p s) :
    pluses (n := n) ((f (n := n) p) s) ≥ pluses (n := n) s := by
  classical
  by_cases hfix : (f (n := n) p) s = s
  · exact ge_of_eq (by simpa using congrArg (pluses (n := n)) hfix)
  · have hprog := cycleUpdate_progress (n := n) (p := p) hsym hdiag s hfix
    rcases hprog with hlt | ⟨heq, hpl⟩
    · exact False.elim (hlt.ne hE)
    · exact le_of_lt hpl

private lemma pluses_iterate_step_mono_of_iterate_eq
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    {k i : Nat} (hi : i < k) {s : State n}
    (hcyc : (f (n := n) p)^[k] s = s) :
    pluses (n := n) ((f (n := n) p)^[i] s) ≤ pluses (n := n) ((f (n := n) p)^[i + 1] s) := by
  have hEi :
      energy (α := ℝ) p ((f (n := n) p)^[i] s) = energy (α := ℝ) p s :=
    energy_iterate_eq_of_iterate_eq (n := n) (p := p) hsym hdiag (Nat.le_of_lt hi) hcyc
  have hEi1 :
      energy (α := ℝ) p ((f (n := n) p)^[i + 1] s) = energy (α := ℝ) p s :=
    energy_iterate_eq_of_iterate_eq (n := n) (p := p) hsym hdiag (Nat.succ_le_of_lt hi) hcyc
  have hE_step :
      energy (α := ℝ) p ((f (n := n) p) ((f (n := n) p)^[i] s)) =
        energy (α := ℝ) p ((f (n := n) p)^[i] s) := by
    -- Both sides equal `energy p s` along the cycle.
    calc
      energy (α := ℝ) p ((f (n := n) p) ((f (n := n) p)^[i] s))
          = energy (α := ℝ) p ((f (n := n) p)^[i + 1] s) := by
              simp [Function.iterate_succ_apply']
      _ = energy (α := ℝ) p s := hEi1
      _ = energy (α := ℝ) p ((f (n := n) p)^[i] s) := hEi.symm
  have hpl :=
    pluses_cycleUpdate_ge_of_energy_eq (n := n) (p := p) hsym hdiag ((f (n := n) p)^[i] s) hE_step
  simpa [Function.iterate_succ_apply'] using hpl

theorem cycleUpdate_no_nontrivial_cycles
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    {k : Nat} (hk : 0 < k) (s : State n)
    (hcyc : (f (n := n) p)^[k] s = s) :
    (f (n := n) p) s = s := by
  classical
  by_contra hne
  -- Energy is constant along the cycle.
  have hE1 :
      energy (α := ℝ) p ((f (n := n) p) s) = energy (α := ℝ) p s := by
    simpa using
      energy_iterate_eq_of_iterate_eq (n := n) (p := p) hsym hdiag (Nat.succ_le_of_lt hk) hcyc
  -- So by the progress lemma, pluses strictly increases on the first step.
  have hprog := cycleUpdate_progress (n := n) (p := p) hsym hdiag s hne
  have hpl1 :
      pluses (n := n) ((f (n := n) p) s) > pluses (n := n) s := by
    rcases hprog with hlt | ⟨heq, hpl⟩
    · exact False.elim (hlt.ne hE1)
    · exact hpl
  -- Along a cycle, pluses is stepwise non-decreasing (since energy is constant).
  have hpl_mono : ∀ i, i < k →
      pluses (n := n) ((f (n := n) p)^[i] s) ≤ pluses (n := n) ((f (n := n) p)^[i + 1] s) := by
    intro i hi
    exact pluses_iterate_step_mono_of_iterate_eq (n := n) (p := p) hsym hdiag (k := k) (i := i) hi
      hcyc
  -- Hence `pluses (f s) ≤ pluses (f^[k] s)` by monotonicity on the initial segment.
  have hpl1k : pluses (n := n) ((f (n := n) p) s) ≤ pluses (n := n) ((f (n := n) p)^[k] s) := by
    have hk1 : 1 ≤ k := Nat.succ_le_of_lt hk
    -- Truncate the sequence at `k` so we can use `monotone_nat_of_le_succ`.
    let g : Nat → Nat := fun i => pluses (n := n) ((f (n := n) p)^[Nat.min i k] s)
    have hg_step : ∀ i, g i ≤ g (i + 1) := by
      intro i
      by_cases hi : i < k
      · have hi' : Nat.min i k = i := Nat.min_eq_left (Nat.le_of_lt hi)
        have hi1' : Nat.min (i + 1) k = i + 1 := by
          apply Nat.min_eq_left
          exact Nat.succ_le_of_lt hi
        -- Use the stepwise monotonicity from the cycle hypothesis.
        simpa [g, hi', hi1', Nat.add_assoc] using hpl_mono i hi
      · have hk_le : k ≤ i := Nat.le_of_not_gt hi
        have hi' : Nat.min i k = k := Nat.min_eq_right hk_le
        have hi1' : Nat.min (i + 1) k = k := Nat.min_eq_right (Nat.le_trans hk_le (Nat.le_succ _))
        simp [g, hi', hi1']
    have hg : Monotone g := monotone_nat_of_le_succ hg_step
    have hg1k : g 1 ≤ g k := hg hk1
    -- Untruncate at `1` and `k`.
    have h1min : Nat.min 1 k = 1 := Nat.min_eq_left hk1
    have hkmin : Nat.min k k = k := Nat.min_self k
    simpa [g, h1min, hkmin, Function.iterate_one] using hg1k
  -- But `f^[k] s = s`, so pluses returns to its original value, contradiction.
  have hkPl : pluses (n := n) ((f (n := n) p)^[k] s) = pluses (n := n) s := by
    simp [hcyc]
  have : pluses (n := n) s < pluses (n := n) s := by
    have hlt : pluses (n := n) s < pluses (n := n) ((f (n := n) p)^[k] s) :=
      lt_of_lt_of_le hpl1 hpl1k
    rw [hkPl] at hlt
    exact hlt
  exact (lt_irrefl _ this)

theorem cycleUpdate_exists_fixedpoint_le_card
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (s0 : State n) :
    ∃ m ≤ Fintype.card (State n), (f (n := n) p)^[m + 1] s0 = (f (n := n) p)^[m] s0 := by
  classical
  let N : Nat := Fintype.card (State n)
  let g : Fin (N + 1) → State n := fun t => (f (n := n) p)^[t.1] s0
  have hlt : Fintype.card (State n) < Fintype.card (Fin (N + 1)) := by
    simp [N, Fintype.card_fin]
  rcases Fintype.exists_ne_map_eq_of_card_lt g hlt with ⟨i, j, hij, hijEq⟩
  have hijNat : i.1 ≠ j.1 := by
    intro h
    apply hij
    exact Fin.ext h
  have hijLt : i.1 < j.1 ∨ j.1 < i.1 := lt_or_gt_of_ne hijNat
  -- Reduce to the ordered case by swapping if necessary.
  refine (hijLt.elim (fun hijLt => ?_) (fun hjiLt => ?_))
  · -- Case `i < j`.
    let m : Nat := i.1
    let d : Nat := j.1 - i.1
    have hdpos : 0 < d := Nat.sub_pos_of_lt hijLt
    have hm_le : m ≤ N := Nat.lt_succ_iff.mp i.2
    let t : State n := (f (n := n) p)^[m] s0
    -- Turn the repetition into a cycle at `t`.
    have hcycle : (f (n := n) p)^[d] t = t := by
      have hj : j.1 = m + d := by
        simp [m, d, Nat.add_sub_of_le (Nat.le_of_lt hijLt)]
      have hijEqNat :
          (f (n := n) p)^[m] s0 = (f (n := n) p)^[m + d] s0 := by
        simpa [g, m, hj] using hijEq
      -- `f^[m+d] s0 = f^[d] (f^[m] s0)`.
      have : (f (n := n) p)^[d] ((f (n := n) p)^[m] s0) = (f (n := n) p)^[m] s0 := by
        -- Rewrite the RHS using `iterate_add_apply` and `hijEqNat`.
        calc
          (f (n := n) p)^[d] ((f (n := n) p)^[m] s0)
              = (f (n := n) p)^[d + m] s0 := by
                  simp [Function.iterate_add_apply]
          _ = (f (n := n) p)^[m + d] s0 := by simp [Nat.add_comm]
          _ = (f (n := n) p)^[m] s0 := hijEqNat.symm
      simpa [t] using this
    have hfix : (f (n := n) p) t = t :=
      cycleUpdate_no_nontrivial_cycles (n := n) (p := p) hsym hdiag (k := d) hdpos t hcycle
    refine ⟨m, hm_le, ?_⟩
    -- Translate the fixed-point equation back to iterates from `s0`.
    simpa [t, Function.iterate_succ_apply'] using hfix
  · -- Case `j < i` (swap).
    have hijEq' : g j = g i := by simpa [g] using hijEq.symm
    -- Apply the previous case with swapped indices.
    have : ∃ m ≤ Fintype.card (State n), (f (n := n) p)^[m + 1] s0 = (f (n := n) p)^[m] s0 := by
      -- reuse the same proof by recursion on the left branch
      let m' : Nat := j.1
      let d' : Nat := i.1 - j.1
      have hdpos : 0 < d' := Nat.sub_pos_of_lt hjiLt
      have hm_le : m' ≤ N := Nat.lt_succ_iff.mp j.2
      let t : State n := (f (n := n) p)^[m'] s0
      have hj : i.1 = m' + d' := by
        simp [m', d', Nat.add_sub_of_le (Nat.le_of_lt hjiLt)]
      have hijEqNat :
          (f (n := n) p)^[m'] s0 = (f (n := n) p)^[m' + d'] s0 := by
        simpa [g, m', hj] using hijEq'
      have hcycle : (f (n := n) p)^[d'] t = t := by
        have : (f (n := n) p)^[d'] ((f (n := n) p)^[m'] s0) = (f (n := n) p)^[m'] s0 := by
          calc
            (f (n := n) p)^[d'] ((f (n := n) p)^[m'] s0)
                = (f (n := n) p)^[d' + m'] s0 := by
                    simp [Function.iterate_add_apply]
            _ = (f (n := n) p)^[m' + d'] s0 := by simp [Nat.add_comm]
            _ = (f (n := n) p)^[m'] s0 := hijEqNat.symm
        simpa [t] using this
      have hfix : (f (n := n) p) t = t :=
        cycleUpdate_no_nontrivial_cycles (n := n) (p := p) hsym hdiag (k := d') hdpos t hcycle
      refine ⟨m', hm_le, ?_⟩
      simpa [t, Function.iterate_succ_apply'] using hfix
    exact this

theorem cycleUpdate_exists_fixedpoint_le_pow
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (s0 : State n) :
    ∃ m ≤ (2 : Nat) ^ n, (f (n := n) p)^[m + 1] s0 = (f (n := n) p)^[m] s0 := by
  classical
  have hcard : Fintype.card (State n) = (2 : Nat) ^ n := by
    -- `State n = Fin n → Bool`, so `#State n = 2^n`.
    simp [Spec.Hopfield.State]
  rcases cycleUpdate_exists_fixedpoint_le_card (n := n) (p := p) hsym hdiag s0 with ⟨m, hm, hfix⟩
  refine ⟨m, ?_, hfix⟩
  simpa [hcard] using hm

end

end NN.MLTheory.Proofs.Hopfield
