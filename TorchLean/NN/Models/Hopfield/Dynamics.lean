/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Hopfield.Energy

/-!
# Hopfield global dynamics: energy along trajectories

For symmetric weights with zero diagonal, the single-step lemma `energy_updateAt_le` extends to
whole trajectories: under any asynchronous schedule `useq : Nat → Fin n`, adversarial or not, the
energy of `seqStates p useq s0` never increases.

Monotone energy alone does not give convergence, because a schedule might never update an unstable
unit. `TorchLean.NN.Models.Hopfield.Convergence` proves that the cyclic sweep `cycleUpdate`
reaches a fixed point, with a bound on the number of sweeps.
-/

@[expose] public section


namespace TorchLean.Hopfield

open scoped BigOperators

variable {n : Nat}

/-- For symmetric weights with zero diagonal, step `k + 1` of any schedule has energy at most that
of step `k`. -/
theorem energy_seqStates_succ_le (p : Params ℝ n)
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (useq : Nat → Fin n) (s0 : State n) (k : Nat) :
    energy (α := ℝ) p (seqStates (α := ℝ) p useq s0 (k + 1))
      ≤
    energy (α := ℝ) p (seqStates (α := ℝ) p useq s0 k) := by
  simp only [Hopfield.seqStates]
  exact energy_updateAt_le (n := n) p hsym hdiag _ _

/-- For symmetric weights with zero diagonal, the energy along any schedule never exceeds the energy
of the starting state `s0`.

The schedule is an arbitrary `Nat → Fin n`, so random and cyclic orders are both covered. -/
theorem energy_seqStates_le_start (p : Params ℝ n)
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (useq : Nat → Fin n) (s0 : State n) :
    ∀ k, energy (α := ℝ) p (seqStates (α := ℝ) p useq s0 k) ≤ energy (α := ℝ) p s0
  | 0 => by simp [Hopfield.seqStates]
  | k + 1 => by
      have hk : energy (α := ℝ) p (seqStates (α := ℝ) p useq s0 (k + 1))
          ≤ energy (α := ℝ) p (seqStates (α := ℝ) p useq s0 k) :=
        energy_seqStates_succ_le (n := n) p hsym hdiag useq s0 k
      exact le_trans hk (energy_seqStates_le_start p hsym hdiag useq s0 k)

end TorchLean.Hopfield
