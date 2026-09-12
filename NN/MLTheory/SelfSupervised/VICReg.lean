/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# VICReg and Barlow-Twins style collapse guards

This file formalizes the parts of recent redundancy-reduction SSL objectives that are cleanly
checkable without importing probability, asymptotics, or differentiable optimization theory.

Paper anchors:
- VICReg, “Variance-Invariance-Covariance Regularization for Self-Supervised Learning”
  (Bardes, Ponce, LeCun, 2021), arXiv:2105.04906.  VICReg combines an invariance loss between two
  views with variance and covariance regularizers; the variance term explicitly discourages
  collapsed embeddings by penalizing coordinates whose standard deviation falls below a threshold.
- Barlow Twins, “Self-Supervised Learning via Redundancy Reduction” (Zbontar et al., ICML 2021),
  arXiv:2103.03230.  Barlow Twins pushes the empirical cross-correlation matrix between two views
  toward the identity: diagonal entries should be one, off-diagonal entries should be zero.

The Lean statements below intentionally prove finite-objective facts, not full learning guarantees:
- a fully collapsed variance vector pays a positive VICReg variance penalty;
- an identity correlation summary has zero Barlow-style redundancy loss;
- a collapsed diagonal entry pays positive redundancy penalty.

The scalar quantities are natural numbers here. Runtime examples can use floating-point losses; the
theory captures the algebraic shape of the collapse guard without importing numerical analysis.
-/

@[expose] public section

namespace NN.MLTheory.SelfSupervised

/--
Hinge penalty for a variance floor: $\max(0,\gamma-v)$, written over `Nat`.

This is the discrete analogue of the VICReg variance hinge.  In the floating-point objective, `v`
would be a per-coordinate standard deviation; in this finite formalization it is an already-computed
nonnegative summary.
-/
def varianceFloorPenalty (gamma variance : Nat) : Nat :=
  gamma - variance

/-- Sum of per-coordinate variance-floor penalties for one embedding branch. -/
def varianceTerm (gamma : Nat) (variances : Array Nat) : Nat :=
  (variances.map (varianceFloorPenalty gamma)).sum

/--
A compact VICReg-style objective over already-computed nonnegative summary terms.

`invariance`, `variance`, and `covariance` are summaries; this file does not claim to formalize the
statistical estimator used to produce them.
-/
def vicregObjective (lambda mu nu invariance variance covariance : Nat) : Nat :=
  lambda * invariance + mu * variance + nu * covariance

/-- A fully collapsed coordinate pays the whole floor `γ`, the worst case of the hinge. -/
@[simp] theorem varianceFloorPenalty_zero (gamma : Nat) :
    varianceFloorPenalty gamma 0 = gamma := by
  simp [varianceFloorPenalty]

/-- No coordinates, no variance penalty. -/
@[simp] theorem varianceTerm_nil (gamma : Nat) :
    varianceTerm gamma #[] = 0 := by
  simp [varianceTerm]

/-- Adding a coordinate adds its hinge penalty. -/
@[simp] theorem varianceTerm_push (gamma v : Nat) (vs : Array Nat) :
    varianceTerm gamma (vs.push v) =
      varianceTerm gamma vs + varianceFloorPenalty gamma v := by
  simp [varianceTerm, Array.map_push]

/-- The variance term is additive in the coordinates, since it is a sum of independent hinges. -/
theorem varianceTerm_append (gamma : Nat) (xs ys : Array Nat) :
    varianceTerm gamma (xs ++ ys) =
      varianceTerm gamma xs + varianceTerm gamma ys := by
  simp [varianceTerm, Array.map_append, Array.sum_append]

/--
Collapsed coordinates ($\mathrm{variance}=0$) pay exactly $d\gamma$.

The penalty is positive when both the number of coordinates and the floor are positive.
-/
theorem varianceTerm_replicate_zero (gamma d : Nat) :
    varianceTerm gamma (Array.replicate d 0) = d * gamma := by
  induction d with
  | zero => simp [varianceTerm]
  | succ d ih =>
      rw [Array.replicate_succ, varianceTerm_push, varianceFloorPenalty_zero, ih, Nat.succ_mul]

/-- If $\gamma>0$ and there is at least one collapsed coordinate, the variance term is positive. -/
theorem varianceTerm_collapsed_positive {gamma d : Nat} (hγ : 0 < gamma) :
    0 < varianceTerm gamma (Array.replicate (d + 1) 0) := by
  rw [varianceTerm_replicate_zero]
  exact Nat.mul_pos (Nat.succ_pos d) hγ

/-- With all weights and all summaries zero, the objective is zero. -/
@[simp] theorem vicregObjective_zero :
    vicregObjective 0 0 0 0 0 0 = 0 := by
  simp [vicregObjective]

/-- A positive variance summary with a positive weight gives a positive variance-only objective.
This does not exclude a collapsed minimizer: that conclusion also needs an attainable objective
value below this penalty. -/
theorem vicregObjective_variance_positive {μ variance : Nat}
    (hμ : 0 < μ) (hv : 0 < variance) :
    0 < vicregObjective 0 μ 0 0 variance 0 := by
  simp [vicregObjective]
  exact Nat.mul_pos hμ hv

/-! ## Barlow/VICReg-style redundancy penalties -/

/--
Penalty for a diagonal cross-correlation entry that should be one.

The `Nat` version is an absolute-deviation hinge around $1$. The Barlow Twins paper uses squared
floating-point deviations, but both objectives share the key finite property proved below: diagonal
value $1$ is free, while collapsed diagonal value $0$ is not.
-/
def diagonalRedundancyPenalty (c : Nat) : Nat :=
  (c - 1) + (1 - c)

/-- Penalty for an off-diagonal cross-correlation entry that should be zero. -/
def offDiagonalRedundancyPenalty (c : Nat) : Nat :=
  c

/--
Barlow-style redundancy-reduction objective over already-computed diagonal and off-diagonal
correlation summaries.

Over `Nat`, the diagonal penalty is zero at $1$; the off-diagonal penalty is zero at $0$. Runtime
versions can use squared floating-point deviations.
-/
def redundancyReductionObjective (lambda : Nat) (diag offDiag : Array Nat) : Nat :=
  (diag.map diagonalRedundancyPenalty).sum +
    lambda * (offDiag.map offDiagonalRedundancyPenalty).sum

/-- The ideal diagonal entry `1` pays nothing. -/
@[simp] theorem diagonalRedundancyPenalty_one :
    diagonalRedundancyPenalty 1 = 0 := by
  simp [diagonalRedundancyPenalty]

/-- The ideal off-diagonal entry `0` pays nothing. -/
@[simp] theorem offDiagonalRedundancyPenalty_zero :
    offDiagonalRedundancyPenalty 0 = 0 := by
  simp [offDiagonalRedundancyPenalty]

/--
The ideal Barlow-style correlation summary has zero redundancy loss: all diagonal entries are $1$
and all off-diagonal entries are $0$.
-/
@[simp] theorem redundancyReductionObjective_identity (lambda d k : Nat) :
    redundancyReductionObjective lambda (Array.replicate d 1) (Array.replicate k 0) = 0 := by
  simp [redundancyReductionObjective]

/-- A diagonal entry collapsed to `0` pays a positive penalty, which is what Barlow Twins is for. -/
theorem diagonalRedundancyPenalty_zero_positive :
    0 < diagonalRedundancyPenalty 0 := by
  simp [diagonalRedundancyPenalty]

/--
If even one diagonal entry is collapsed to $0$ while all other entries are ideal, the redundancy
objective is positive.
-/
theorem redundancyReductionObjective_collapsed_diag_positive
    {lambda d k : Nat} :
    0 < redundancyReductionObjective lambda
      ((Array.replicate d 1).push 0) (Array.replicate k 0) := by
  simp [redundancyReductionObjective, diagonalRedundancyPenalty, Array.map_push]

end NN.MLTheory.SelfSupervised
