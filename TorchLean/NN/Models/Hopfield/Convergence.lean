/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Hopfield.Progress

/-!
# Hopfield cyclic-sweep convergence (finite-state argument)

For symmetric weights with zero diagonal, the full sweep `cycleUpdate` has no nontrivial cycles.
By `cycleUpdate_progress`, a sweep that changes the state either lowers the energy or keeps it and
raises the active count `pluses`, so a state the sweep has left never comes back. Since
`State n = Fin n → Bool` has `2^n` elements, pigeonhole then gives a fixed point within `2^n`
sweeps from any initial state, that is, within `n * 2^n` single-unit updates.

The statements are about full sweeps (one pass over the coordinates in order), written with
Mathlib's iterate notation `(cycleUpdate p)^[k]`. No theorem here relates a sweep to `n` steps of
`seqStates` under the cyclic schedule `Fin.ofNat n`.
-/

@[expose] public section


namespace TorchLean.Hopfield

open scoped BigOperators

variable {n : Nat}

section

variable (p : Params ℝ n)

/-- Energy is antitone along the sweep iterates, by composing the one-sweep bound. -/
private theorem energy_iterate_antitone
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0) (s : State n) :
    Antitone fun k => energy (α := ℝ) p ((cycleUpdate (n := n) p)^[k] s) :=
  antitone_nat_of_succ_le fun k => by
    simpa [Function.iterate_succ_apply'] using
      energy_cycleUpdate_le (n := n) (p := p) hsym hdiag ((cycleUpdate (n := n) p)^[k] s)

/-- If `k` sweeps return `s` to itself, every iterate `i ≤ k` has the energy of `s`. -/
private theorem energy_iterate_eq_of_cycle
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    {k i : Nat} (hi : i ≤ k) {s : State n}
    (hcyc : (cycleUpdate (n := n) p)^[k] s = s) :
    energy (α := ℝ) p ((cycleUpdate (n := n) p)^[i] s) = energy (α := ℝ) p s := by
  have hanti := energy_iterate_antitone (n := n) (p := p) hsym hdiag s
  refine le_antisymm (by simpa using hanti (Nat.zero_le i)) ?_
  simpa [hcyc] using hanti hi

/-- A sweep that keeps the energy fixed cannot lower the number of active units. -/
private theorem pluses_cycleUpdate_ge
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s : State n)
    (hE : energy (α := ℝ) p (cycleUpdate (n := n) p s) = energy (α := ℝ) p s) :
    pluses (n := n) (cycleUpdate (n := n) p s) ≥ pluses (n := n) s := by
  by_cases hfix : cycleUpdate (n := n) p s = s
  · exact (congrArg (pluses (n := n)) hfix).ge
  · rcases cycleUpdate_progress (n := n) (p := p) hsym hdiag s hfix with hlt | ⟨_, hpl⟩
    · exact absurd hE hlt.ne
    · exact hpl.le

private theorem pluses_iterate_mono_of_cycle
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    {k i : Nat} (hi : i < k) {s : State n}
    (hcyc : (cycleUpdate (n := n) p)^[k] s = s) :
    pluses (n := n) ((cycleUpdate (n := n) p)^[i] s) ≤
      pluses (n := n) ((cycleUpdate (n := n) p)^[i + 1] s) := by
  -- Along the cycle the energy is constant, so the sweep from `f^[i] s` keeps it fixed.
  have hE_step :
      energy (α := ℝ) p (cycleUpdate (n := n) p ((cycleUpdate (n := n) p)^[i] s)) =
        energy (α := ℝ) p ((cycleUpdate (n := n) p)^[i] s) := by
    calc
      energy (α := ℝ) p (cycleUpdate (n := n) p ((cycleUpdate (n := n) p)^[i] s))
          = energy (α := ℝ) p ((cycleUpdate (n := n) p)^[i + 1] s) := by
              simp only [Function.iterate_succ_apply']
      _ = energy (α := ℝ) p s :=
              energy_iterate_eq_of_cycle (n := n) (p := p) hsym hdiag
                (Nat.succ_le_of_lt hi) hcyc
      _ = energy (α := ℝ) p ((cycleUpdate (n := n) p)^[i] s) :=
              (energy_iterate_eq_of_cycle (n := n) (p := p) hsym hdiag hi.le hcyc).symm
  rw [Function.iterate_succ_apply']
  exact pluses_cycleUpdate_ge (n := n) (p := p) hsym hdiag _ hE_step

/-- With symmetric weights and zero diagonal, a periodic orbit of the sweep is a fixed point.

The proof is the heart of the Hopfield convergence argument: energy never increases along a sweep,
so it is constant on a cycle, and then the active count would rise on the first sweep, never fall
afterwards, and yet return to its starting value. -/
theorem cycleUpdate_eq_self_of_cycle
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    {k : Nat} (hk : 0 < k) (s : State n)
    (hcyc : (cycleUpdate (n := n) p)^[k] s = s) :
    cycleUpdate (n := n) p s = s := by
  classical
  by_contra hne
  -- Energy is constant along the cycle.
  have hE1 :
      energy (α := ℝ) p (cycleUpdate (n := n) p s) = energy (α := ℝ) p s := by
    simpa using
      energy_iterate_eq_of_cycle (n := n) (p := p) hsym hdiag (Nat.succ_le_of_lt hk) hcyc
  -- So by the progress lemma, pluses strictly increases on the first step.
  have hpl1 :
      pluses (n := n) (cycleUpdate (n := n) p s) > pluses (n := n) s := by
    rcases cycleUpdate_progress (n := n) (p := p) hsym hdiag s hne with hlt | ⟨_, hpl⟩
    · exact absurd hE1 hlt.ne
    · exact hpl
  -- Along a cycle, pluses is stepwise non-decreasing (since energy is constant).
  have hpl_mono : ∀ i, i < k →
      pluses (n := n) ((cycleUpdate (n := n) p)^[i] s) ≤
        pluses (n := n) ((cycleUpdate (n := n) p)^[i + 1] s) :=
    fun i hi => pluses_iterate_mono_of_cycle (n := n) (p := p) hsym hdiag hi hcyc
  -- Hence `pluses (f s) ≤ pluses (f^[k] s)` by monotonicity on the initial segment.
  have hpl1k :
      pluses (n := n) (cycleUpdate (n := n) p s) ≤
        pluses (n := n) ((cycleUpdate (n := n) p)^[k] s) := by
    have hk1 : 1 ≤ k := Nat.succ_le_of_lt hk
    have hbound : ∀ i, 1 ≤ i → i ≤ k →
        pluses (n := n) ((cycleUpdate (n := n) p)^[1] s) ≤
          pluses (n := n) ((cycleUpdate (n := n) p)^[i] s) := by
      intro i hi
      induction i, hi using Nat.le_induction with
      | base => intro _; exact le_rfl
      | succ i hi ih =>
          intro hik
          exact (ih (Nat.le_trans (Nat.le_succ i) hik)).trans
            (hpl_mono i (Nat.lt_of_succ_le hik))
    simpa only [Function.iterate_one] using hbound k hk1 le_rfl
  -- But `f^[k] s = s`, so pluses returns to its original value, contradiction.
  have hkPl : pluses (n := n) ((cycleUpdate (n := n) p)^[k] s) = pluses (n := n) s := by
    rw [hcyc]
  have hlt : pluses (n := n) s < pluses (n := n) s := by
    have h := lt_of_lt_of_le hpl1 hpl1k
    rwa [hkPl] at h
  exact lt_irrefl _ hlt

/-- If two iterates of the sweep coincide, the earlier one is already a fixed point.

The repetition closes a cycle of positive length through the earlier iterate, and
`cycleUpdate_eq_self_of_cycle` forbids nontrivial cycles. -/
private theorem iterate_succ_eq_of_iterate_eq
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s0 : State n) {i j : Nat} (hij : i < j)
    (h : (cycleUpdate (n := n) p)^[i] s0 = (cycleUpdate (n := n) p)^[j] s0) :
    (cycleUpdate (n := n) p)^[i + 1] s0 = (cycleUpdate (n := n) p)^[i] s0 := by
  have hcycle :
      (cycleUpdate (n := n) p)^[j - i] ((cycleUpdate (n := n) p)^[i] s0) =
        (cycleUpdate (n := n) p)^[i] s0 := by
    rw [← Function.iterate_add_apply, Nat.sub_add_cancel hij.le, ← h]
  rw [Function.iterate_succ_apply']
  exact cycleUpdate_eq_self_of_cycle (n := n) (p := p) hsym hdiag (Nat.sub_pos_of_lt hij) _
    hcycle

/-- With symmetric weights and zero diagonal, the sweep reaches a fixed point within
`Fintype.card (State n)` sweeps from any `s0`.

Finitely many states and no nontrivial cycles give termination, and the bound is the crude
pigeonhole count. -/
theorem exists_fixedPoint_le_card
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s0 : State n) :
    ∃ m ≤ Fintype.card (State n),
      (cycleUpdate (n := n) p)^[m + 1] s0 = (cycleUpdate (n := n) p)^[m] s0 := by
  classical
  -- Pigeonhole: among the first `card + 1` iterates, two coincide.
  obtain ⟨i, j, hij, hijEq⟩ : ∃ i j : Fin (Fintype.card (State n) + 1), i ≠ j ∧
      (cycleUpdate (n := n) p)^[i.1] s0 = (cycleUpdate (n := n) p)^[j.1] s0 :=
    Fintype.exists_ne_map_eq_of_card_lt
      (fun t : Fin (Fintype.card (State n) + 1) => (cycleUpdate (n := n) p)^[t.1] s0) (by simp)
  have hijNat : i.1 ≠ j.1 := fun h => hij (Fin.ext h)
  -- The earlier of the two repeated indices is the fixed point.
  rcases lt_or_gt_of_ne hijNat with hlt | hlt
  · exact ⟨i.1, Nat.le_of_lt_succ i.2,
      iterate_succ_eq_of_iterate_eq (n := n) (p := p) hsym hdiag s0 hlt hijEq⟩
  · exact ⟨j.1, Nat.le_of_lt_succ j.2,
      iterate_succ_eq_of_iterate_eq (n := n) (p := p) hsym hdiag s0 hlt hijEq.symm⟩

/-- With symmetric weights and zero diagonal, the sweep reaches a fixed point within `2 ^ n` sweeps
from any `s0`, since a state is one bit per unit. -/
theorem exists_fixedPoint_le_pow
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s0 : State n) :
    ∃ m ≤ (2 : Nat) ^ n,
      (cycleUpdate (n := n) p)^[m + 1] s0 = (cycleUpdate (n := n) p)^[m] s0 := by
  classical
  -- `State n = Fin n → Bool`, so there are `2 ^ n` states.
  have hcard : Fintype.card (State n) = (2 : Nat) ^ n := by
    simp [Hopfield.State]
  obtain ⟨m, hm, hfix⟩ := exists_fixedPoint_le_card (n := n) (p := p) hsym hdiag s0
  exact ⟨m, by rw [← hcard]; exact hm, hfix⟩

end

end TorchLean.Hopfield
