/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Hopfield.Energy

/-!
# Hopfield global dynamics: energy along trajectories

This file lifts the single-step energy lemma to whole trajectories:
for any (possibly adversarial) asynchronous update schedule `useq`, the energy sequence is
non-increasing.

Stronger theorems from the paper (convergence for fair schedules; cyclic convergence bounds) can
be built on top of these monotonicity facts.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.Hopfield

open scoped BigOperators
open Spec TorchLean

open Spec.Hopfield

variable {n : Nat}

/-- Energy is nonincreasing along any update sequence, one step at a time. -/
theorem energy_seqStates_succ_le (p : Params ℝ n)
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (useq : Nat → Fin n) (s0 : State n) (k : Nat) :
    energy (α := ℝ) p (seqStates (α := ℝ) p useq s0 (k + 1))
      ≤
    energy (α := ℝ) p (seqStates (α := ℝ) p useq s0 k) := by
  -- Unfold one step and apply the single-update lemma.
  simp [Spec.Hopfield.seqStates]
  exact energy_updateAt_le (n := n) p hsym hdiag _ _

/-- Hence energy never exceeds its starting value, for any schedule of units.

The schedule is an arbitrary `Nat → Fin n`, so this covers random and cyclic orders alike. -/
theorem energy_seqStates_le_start (p : Params ℝ n)
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (useq : Nat → Fin n) (s0 : State n) :
    ∀ k, energy (α := ℝ) p (seqStates (α := ℝ) p useq s0 k) ≤ energy (α := ℝ) p s0
  | 0 => by simp [Spec.Hopfield.seqStates]
  | k + 1 => by
      have hk : energy (α := ℝ) p (seqStates (α := ℝ) p useq s0 (k + 1))
          ≤ energy (α := ℝ) p (seqStates (α := ℝ) p useq s0 k) :=
        energy_seqStates_succ_le (n := n) p hsym hdiag useq s0 k
      exact le_trans hk (energy_seqStates_le_start p hsym hdiag useq s0 k)

end NN.MLTheory.Proofs.Hopfield
