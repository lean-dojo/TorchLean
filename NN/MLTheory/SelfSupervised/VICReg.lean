/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Data.List.Basic

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
Hinge penalty for a variance floor: `max(0, gamma - v)`, written over `Nat`.

This is the discrete analogue of the VICReg variance hinge.  In the floating-point objective, `v`
would be a per-coordinate standard deviation; in this finite formalization it is an already-computed
nonnegative summary.
-/
def varianceFloorPenalty (gamma variance : Nat) : Nat :=
  gamma - variance

/-- Sum of per-coordinate variance-floor penalties for one embedding branch. -/
def varianceTerm (gamma : Nat) (variances : List Nat) : Nat :=
  (variances.map (varianceFloorPenalty gamma)).sum

/--
A compact VICReg-style objective over already-computed nonnegative summary terms.

`invariance`, `variance`, and `covariance` are summaries; this file does not claim to formalize the
statistical estimator used to produce them.
-/
def vicregObjective (lambda mu nu invariance variance covariance : Nat) : Nat :=
  lambda * invariance + mu * variance + nu * covariance

@[simp] theorem varianceFloorPenalty_zero (gamma : Nat) :
    varianceFloorPenalty gamma 0 = gamma := by
  simp [varianceFloorPenalty]

@[simp] theorem varianceTerm_nil (gamma : Nat) :
    varianceTerm gamma [] = 0 := by
  simp [varianceTerm]

@[simp] theorem varianceTerm_cons (gamma v : Nat) (vs : List Nat) :
    varianceTerm gamma (v :: vs) =
      varianceFloorPenalty gamma v + varianceTerm gamma vs := by
  simp [varianceTerm]

theorem varianceTerm_append (gamma : Nat) (xs ys : List Nat) :
    varianceTerm gamma (xs ++ ys) =
      varianceTerm gamma xs + varianceTerm gamma ys := by
  simp [varianceTerm, List.map_append, List.sum_append]

/--
Collapsed coordinates (`variance = 0`) pay exactly `d * gamma`.

This is the direct anti-collapse fact: if every coordinate has zero variance, the variance floor
does not silently accept it.
-/
theorem varianceTerm_replicate_zero (gamma d : Nat) :
    varianceTerm gamma (List.replicate d 0) = d * gamma := by
  induction d with
  | zero => simp [varianceTerm]
  | succ d ih =>
      rw [List.replicate_succ, varianceTerm_cons, varianceFloorPenalty_zero, ih, Nat.succ_mul]
      exact Nat.add_comm gamma (d * gamma)

/-- If `gamma > 0` and there is at least one collapsed coordinate, the variance term is positive. -/
theorem varianceTerm_collapsed_positive {gamma d : Nat} (hγ : 0 < gamma) :
    0 < varianceTerm gamma (List.replicate (d + 1) 0) := by
  rw [varianceTerm_replicate_zero]
  exact Nat.mul_pos (Nat.succ_pos d) hγ

@[simp] theorem vicregObjective_zero :
    vicregObjective 0 0 0 0 0 0 = 0 := by
  simp [vicregObjective]

theorem vicregObjective_variance_positive {μ variance : Nat}
    (hμ : 0 < μ) (hv : 0 < variance) :
    0 < vicregObjective 0 μ 0 0 variance 0 := by
  simp [vicregObjective]
  exact Nat.mul_pos hμ hv

/-! ## Barlow/VICReg-style redundancy penalties -/

/--
Penalty for a diagonal cross-correlation entry that should be one.

The `Nat` version is an absolute-deviation hinge around `1`.  The Barlow Twins paper uses squared
floating-point deviations, but both objectives share the key finite property proved below: diagonal
value `1` is free, while collapsed diagonal value `0` is not.
-/
def diagonalRedundancyPenalty (c : Nat) : Nat :=
  (c - 1) + (1 - c)

/-- Penalty for an off-diagonal cross-correlation entry that should be zero. -/
def offDiagonalRedundancyPenalty (c : Nat) : Nat :=
  c

/--
Barlow-style redundancy-reduction objective over already-computed diagonal and off-diagonal
correlation summaries.

Over `Nat`, the diagonal penalty is zero at `1`; the off-diagonal penalty is zero at `0`. Runtime
versions can use squared floating-point deviations.
-/
def redundancyReductionObjective (lambda : Nat) (diag offDiag : List Nat) : Nat :=
  (diag.map diagonalRedundancyPenalty).sum +
    lambda * (offDiag.map offDiagonalRedundancyPenalty).sum

@[simp] theorem diagonalRedundancyPenalty_one :
    diagonalRedundancyPenalty 1 = 0 := by
  simp [diagonalRedundancyPenalty]

@[simp] theorem offDiagonalRedundancyPenalty_zero :
    offDiagonalRedundancyPenalty 0 = 0 := by
  simp [offDiagonalRedundancyPenalty]

/--
The ideal Barlow-style correlation summary has zero redundancy loss: all diagonal entries are `1`
and all off-diagonal entries are `0`.
-/
@[simp] theorem redundancyReductionObjective_identity (lambda d k : Nat) :
    redundancyReductionObjective lambda (List.replicate d 1) (List.replicate k 0) = 0 := by
  simp [redundancyReductionObjective]

theorem diagonalRedundancyPenalty_zero_positive :
    0 < diagonalRedundancyPenalty 0 := by
  simp [diagonalRedundancyPenalty]

/--
If even one diagonal entry is collapsed to `0` while all other entries are ideal, the redundancy
objective is positive.
-/
theorem redundancyReductionObjective_collapsed_diag_positive
    {lambda d k : Nat} :
    0 < redundancyReductionObjective lambda (0 :: List.replicate d 1) (List.replicate k 0) := by
  simp [redundancyReductionObjective, diagonalRedundancyPenalty]

end NN.MLTheory.SelfSupervised
