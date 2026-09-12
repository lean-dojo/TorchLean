/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.SafeDivSigmoid
public import NN.Proofs.RuntimeApprox.NF.Ops.Sum

/-!
# NF Reduction Operators

NF (rounded) backend: approximation lemmas for matrix reductions used by LayerNorm and attention.

The row and column lemmas use explicit `Shape.NonemptyAxis` evidence derived from positivity of the
matrix dimensions.

## PyTorch correspondence / citations
This file targets reduction patterns used by normalization/attention (sums, means, maxes along an
axis), analogous to operations like `torch.sum`, `torch.mean`, and `torch.max`.
https://pytorch.org/docs/stable/generated/torch.sum.html
https://pytorch.org/docs/stable/generated/torch.mean.html
https://pytorch.org/docs/stable/generated/torch.max.html

Current scope: row and column reductions on matrices. Broader-rank reductions can reuse the same
argument after moving the selected axis into a matrix view.

## Mean bound
The row mean divides the rounded row sum by the rounded row length `n`. Its budget `meanRowBound`
is derived from the conditioned division bound `divPosErrorBound`: the exact denominator is `n`,
the denominator rounding error is `natCastError n = |n̂ - n|`, and the certificate `natCastError n
< n` keeps the rounded denominator positive. When `n` is exactly representable the budget is
`ulp(Σ̂ / n) / 2 + sumBound eps row / n` (`mean_row_bound_of_exact`); if the certificate fails
the budget falls back to an always valid triangle-inequality bound.
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd


-- ---------------------------------------------------------------------------
-- Definitional unfoldings for matrix reductions (axis 0/1)
-- ---------------------------------------------------------------------------

private theorem reduce_sum_by_row_get
    {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {m n : Nat} (x : Tensor α [m, n])
    (hRed : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (i : Fin m) :
    (TorchLean.Tensor.reduceSum (α := α) (s := .dim m (.dim n .scalar)) 1 x hRed).unstack i =
      Tensor.scalar (sumSpec (α := α) (s := [n]) (x.unstack i)) := by
  rw [show x = Tensor.dim (Tensor.unstack x) from (Tensor.dim_unstack x).symm]
  cases hRed
  simp [reduceSum, reduceDim, TorchLean.Tensor.Reduction.Internal.reduceDimCore]

private theorem reduce_mean_by_row_get
    {α : Type} [TorchLean.Storage α] [Context α]
    {m n : Nat} (x : Tensor α [m, n])
    (hRed : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (i : Fin m) :
    (TorchLean.Tensor.reduceMean (α := α) (s := .dim m (.dim n .scalar)) 1 x hRed).unstack i =
      Tensor.scalar (sumSpec (α := α) (s := [n]) (x.unstack i) / (n : α)) := by
  change
    (Tensor.map (fun value => value / (n : α))
      (TorchLean.Tensor.reduceSum (α := α) 1 x hRed)).unstack i =
        Tensor.scalar (sumSpec (α := α) (s := [n]) (x.unstack i) / (n : α))
  rw [Tensor.unstack_map]
  rw [reduce_sum_by_row_get x hRed i]
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  cases coordinate
  simp [Tensor.map, Tensor.scalar]

private theorem reduce_sum_by_column_get
    {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {m n : Nat} (x : Tensor α [m, n])
    (hRed : Shape.NonemptyAxis 0 (.dim m (.dim n .scalar))) (j : Fin n) :
    (TorchLean.Tensor.reduceSum (α := α) (s := .dim m (.dim n .scalar)) 0 x hRed).unstack j =
      Tensor.scalar (sumSpec (α := α) (s := .dim m .scalar)
        (Tensor.dim fun i : Fin m => (x.unstack i).unstack j)) := by
  cases hRed
  simp [reduceSum, reduceDim, TorchLean.Tensor.Reduction.Internal.reduceDimCore,
    TorchLean.Tensor.Reduction.Internal.reduceOuterAxis,
    Spec.get]

-- ---------------------------------------------------------------------------
-- Row-wise matrix sum (axis=1)
-- ---------------------------------------------------------------------------

/-- Axis-1 nonemptiness of a matrix shape is exactly positivity of the row length, so a theorem that
already asks for the reduction's axis evidence does not need a separate `0 < n` hypothesis. -/
private theorem pos_of_row_axis {m n : Nat}
    (hRed : Shape.NonemptyAxis 1 ([m, n] : Shape)) : 0 < n := by
  cases hRed with
  | succ inner => cases inner; exact Nat.succ_pos _

/-- Axis-0 nonemptiness of a matrix shape is positivity of the column length. -/
private theorem pos_of_column_axis {m n : Nat}
    (hRed : Shape.NonemptyAxis 0 ([m, n] : Shape)) : 0 < m := by
  cases hRed; exact Nat.succ_pos _

/-- Row-wise budget vector for an `m × n` runtime matrix: entry `i` is the accumulated rounding
budget of runtime row `i`, so a row of large magnitudes is allowed a larger error than a row of
small ones. This is the sum-side counterpart of `meanRowBoundVec` below, and naming it keeps the
theorem statement readable instead of inlining the whole vector into the tolerance slot. -/
def sumRowBoundVec {m n : Nat} (eps : ℝ) (xR : Tensor R [m, n]) : SpecTensor [m] :=
  Tensor.dim fun i =>
    Tensor.scalar (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps (xR.unstack i))

/-- Row-wise `reduceSum` along axis `1` approximates the exact row sums within
`linfNorm (sumRowBoundVec eps xR)`. The reduction's own nonempty-axis evidence is the hypothesis, so
the statement asks for exactly what `reduceSum` needs and nothing more. -/
theorem approxTensor_reduce_sum_rows
    {m n : Nat}
    {xS : SpecTensor [m, n]}
    {xR : Tensor R [m, n]}
    {eps : ℝ}
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps)
    (hRed : Shape.NonemptyAxis 1 ([m, n] : Shape)) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TorchLean.Tensor.reduceSum (α := ℝ) (s := [m, n]) 1 xS hRed)
      (TorchLean.Tensor.reduceSum (α := R) (s := [m, n]) 1 xR hRed)
      (linfNorm (sumRowBoundVec (β := β) (fexp := fexp) (rnd := rnd) eps xR)) := by
  classical
  have hε : 0 ≤ eps := approxTensor_eps_nonneg (s := [m, n]) hx
  let boundVec : SpecTensor [m] :=
    Tensor.dim fun i =>
      Tensor.scalar
        (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar)
          eps (xR.unstack i))
  have hBoundNonneg : 0 ≤ linfNorm boundVec := linf_norm_nonneg (t := boundVec)
  have hRed' : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar)) := hRed

  refine approxTensor_dim_of_forall
    (n := m) (s := .scalar)
    (xS := TorchLean.Tensor.reduceSum (α := ℝ) (s := .dim m (.dim n .scalar)) 1 xS hRed')
    (xR := TorchLean.Tensor.reduceSum (α := R) (s := .dim m (.dim n .scalar)) 1 xR hRed')
    (eps := linfNorm boundVec) hBoundNonneg ?_
  intro i
  have hxRow :=
    approxTensor_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (xS := xS) (xR := xR) (eps := eps) hx i
  have hSum :=
    approxTensor_sum_spec (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar)
      (xS := xS.unstack i) (xR := xR.unstack i) (eps := eps) hxRow
  have hSumScalar :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).1 hSum
  have hle : sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar)
      eps (xR.unstack i) ≤ linfNorm boundVec := by
    have hcomponent := linf_norm_le_get_dim (t := boundVec) i
    have habs : abs (sumBound (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim n .scalar) eps (xR.unstack i)) ≤ linfNorm boundVec := by
      simpa [boundVec, linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm,
        MathFunctions.abs] using hcomponent
    exact le_trans (le_abs_self _) habs
  have herror : abs
      (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (sumSpec (α := R) (s := .dim n .scalar) (xR.unstack i)) -
        sumSpec (α := ℝ) (s := .dim n .scalar) (xS.unstack i))
      ≤ linfNorm boundVec := le_trans hSumScalar hle
  have hScalarApprox :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.scalar (sumSpec (α := ℝ) (s := .dim n .scalar) (xS.unstack i)))
        (Tensor.scalar (sumSpec (α := R) (s := .dim n .scalar) (xR.unstack i)))
        (linfNorm boundVec) :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2
      (by simpa [abs_sub_comm] using herror)
  rw [reduce_sum_by_row_get xS hRed' i, reduce_sum_by_row_get xR hRed' i]
  exact hScalarApprox

-- ---------------------------------------------------------------------------
-- Row-wise matrix mean (axis=1)
-- ---------------------------------------------------------------------------

/-- Rounding error of the runtime row length. The natural number `n` is embedded into `NF` by
rounding, and `natCastError n` is exactly how far that rounded value sits from `n`. It vanishes
whenever `n` is representable in the format. -/
def natCastError (n : Nat) : ℝ :=
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (n : R) - (n : ℝ))

/-- Per-row forward error budget for `reduceMean` along the last axis of a matrix.

The runtime mean divides the sequentially rounded row sum `Σ̂` by the rounded row length `n̂`.
When the row-length rounding error is below `n`, the budget is the conditioned division bound
`divPosErrorBound` with exact denominator lower bound `n`, numerator error `sumBound eps rowR`, and
denominator error `natCastError n`. For an exactly represented `n` this is
`ulp(Σ̂ / n) / 2 + sumBound eps rowR / n` (see `mean_row_bound_of_exact`). If the certificate fails
the budget falls back to the always valid `|Σ̂ / n̂| + (|Σ̂| + sumBound eps rowR) / n`. -/
def meanRowBound {n : Nat} (eps : ℝ) (rowR : Tensor R [n]) : ℝ :=
  if natCastError (β := β) (fexp := fexp) (rnd := rnd) n < (n : ℝ) then
    divPosErrorBound (β := β) (fexp := fexp) (n : ℝ)
      (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR)
      (natCastError (β := β) (fexp := fexp) (rnd := rnd) n)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := [n]) rowR))
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (n : R))
  else
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
        (sumSpec (α := R) (s := [n]) rowR / (n : R))) +
      (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := [n]) rowR)) +
        sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR) / (n : ℝ)

/-- Row-wise mean budget vector for an `m × n` runtime matrix. -/
def meanRowBoundVec {m n : Nat} (eps : ℝ) (xR : Tensor R [m, n]) : SpecTensor [m] :=
  Tensor.dim fun i =>
    Tensor.scalar (meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i))

/-- Scalar certificate for one row mean: the rounded quotient `Σ̂ / n̂` approximates `Σ / n` within
`meanRowBound`. -/
theorem approx_mean_row_nf {n : Nat} (hn : 0 < n)
    {rowS : SpecTensor [n]} {rowR : Tensor R [n]} {eps : ℝ}
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      rowS rowR eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (sumSpec (α := R) (s := [n]) rowR / (n : R)) -
        sumSpec (α := ℝ) (s := [n]) rowS / (n : ℝ)) ≤
      meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps rowR := by
  have hSum := approxTensor_sum_spec (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) hx
  have hSumScalar :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := [n]) rowR) -
          sumSpec (α := ℝ) (s := [n]) rowS) ≤
        sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).1 hSum
  have hnpos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hN : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (n : R) - (n : ℝ)) ≤
      natCastError (β := β) (fexp := fexp) (rnd := rnd) n := le_rfl
  unfold meanRowBound
  split_ifs with hcert
  · exact approx_div_nf_of_pos_lb (β := β) (fexp := fexp) (rnd := rnd) (η := (n : ℝ))
      le_rfl hcert hSumScalar hN
  · have hsum_abs : abs (sumSpec (α := ℝ) (s := [n]) rowS) ≤
        abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := [n]) rowR)) +
          sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR := by
      have h := abs_sub_abs_le_abs_sub (sumSpec (α := ℝ) (s := [n]) rowS)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := [n]) rowR))
      rw [abs_sub_comm] at h
      linarith
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (sumSpec (α := R) (s := [n]) rowR / (n : R)) -
          sumSpec (α := ℝ) (s := [n]) rowS / (n : ℝ))
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (sumSpec (α := R) (s := [n]) rowR / (n : R))) +
            abs (sumSpec (α := ℝ) (s := [n]) rowS / (n : ℝ)) := abs_sub _ _
      _ = abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (sumSpec (α := R) (s := [n]) rowR / (n : R))) +
            abs (sumSpec (α := ℝ) (s := [n]) rowS) / (n : ℝ) := by
          rw [abs_div, abs_of_pos hnpos]
      _ ≤ _ := by gcongr

omit [NeuralValidRndToNearest rnd] in
/-- Regression: when the row length is exactly representable, the mean budget is precisely one
output half ulp plus the row-sum budget divided by `n`. -/
theorem mean_row_bound_of_exact {n : Nat} (hn : 0 < n) (eps : ℝ) (rowR : Tensor R [n])
    (hexact : toSpec (β := β) (fexp := fexp) (rnd := rnd) (n : R) = (n : ℝ)) :
    meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps rowR =
      neuralUlp β fexp
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := [n]) rowR) /
            (n : ℝ)) / 2 +
        sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR / (n : ℝ) := by
  have hnpos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hzero : natCastError (β := β) (fexp := fexp) (rnd := rnd) n = 0 := by
    unfold natCastError
    rw [hexact, sub_self, abs_zero]
  unfold meanRowBound
  rw [if_pos (by rw [hzero]; exact hnpos), hzero, hexact]
  simp only [divPosErrorBound, sub_zero, zero_div, mul_zero, add_zero]
  ring

omit [NeuralValidRndToNearest rnd] in
/-- Regression: under the half-margin certificate `natCastError n ≤ n / 2`, the mean budget is
linear in the row-sum budget, the row-length rounding error, and one output rounding. The
hypothesis `hsum` records that the row-sum budget is nonnegative, which holds for every budget
produced by `approxTensor_sum_spec`. -/
theorem mean_row_bound_le_of_natCastError_le_half {n : Nat} (hn : 0 < n) (eps : ℝ)
    (rowR : Tensor R [n])
    (hsum : 0 ≤ sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR)
    (hcast : natCastError (β := β) (fexp := fexp) (rnd := rnd) n ≤ (n : ℝ) / 2) :
    meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps rowR ≤
      (2 / (n : ℝ)) * sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR +
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := [n]) rowR)) +
            sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR) *
          (4 * natCastError (β := β) (fexp := fexp) (rnd := rnd) n / ((n : ℝ) * (n : ℝ))) +
        neuralUlp β fexp
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := [n]) rowR) /
            toSpec (β := β) (fexp := fexp) (rnd := rnd) (n : R)) / 2 := by
  have hnpos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hcast0 : 0 ≤ natCastError (β := β) (fexp := fexp) (rnd := rnd) n := abs_nonneg _
  have hlt : natCastError (β := β) (fexp := fexp) (rnd := rnd) n < (n : ℝ) := by linarith
  unfold meanRowBound
  rw [if_pos hlt]
  exact divPosErrorBound_le_of_epsy_le_half (β := β) (fexp := fexp) hnpos hsum hcast0 hcast

/-- Row-wise `reduceMean` along axis `1` of an `m × n` matrix approximates the exact row means
within `linfNorm (meanRowBoundVec eps xR)`. Each entry is the certified division budget
`meanRowBound` for its row. -/
theorem approxTensor_reduce_mean_rows
    {m n : Nat}
    {xS : SpecTensor [m, n]}
    {xR : Tensor R [m, n]}
    {eps : ℝ}
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps)
    (hRed : Shape.NonemptyAxis 1 ([m, n] : Shape)) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TorchLean.Tensor.reduceMean (α := ℝ) (s := [m, n]) 1 xS hRed)
      (TorchLean.Tensor.reduceMean (α := R) (s := [m, n]) 1 xR hRed)
      (linfNorm (meanRowBoundVec (β := β) (fexp := fexp) (rnd := rnd) eps xR)) := by
  classical
  have hn : 0 < n := pos_of_row_axis hRed
  set boundVec : SpecTensor [m] :=
    meanRowBoundVec (β := β) (fexp := fexp) (rnd := rnd) eps xR with hboundVec
  have hBoundNonneg : 0 ≤ linfNorm boundVec := linf_norm_nonneg (t := boundVec)
  have hRed' : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar)) := hRed
  refine approxTensor_dim_of_forall
    (n := m) (s := .scalar)
    (xS := TorchLean.Tensor.reduceMean (α := ℝ) 1 xS hRed')
    (xR := TorchLean.Tensor.reduceMean (α := R) 1 xR hRed')
    (eps := linfNorm boundVec) hBoundNonneg ?_
  intro i
  have hxRow :=
    approxTensor_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (xS := xS) (xR := xR) (eps := eps) hx i
  have hlocal := approx_mean_row_nf (β := β) (fexp := fexp) (rnd := rnd) hn hxRow
  have hunstack : boundVec.unstack i =
      Tensor.scalar (meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i)) := by
    simp [hboundVec, meanRowBoundVec]
  have hle : meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i) ≤
      linfNorm boundVec := by
    have hcomponent := linf_norm_le_get_dim (t := boundVec) i
    rw [hunstack] at hcomponent
    have habs : abs (meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i)) ≤
        linfNorm boundVec := by
      simpa [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, MathFunctions.abs] using hcomponent
    exact le_trans (le_abs_self _) habs
  have hScalarApprox :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.scalar (sumSpec (α := ℝ) (s := [n]) (xS.unstack i) / (n : ℝ)))
        (Tensor.scalar (sumSpec (α := R) (s := [n]) (xR.unstack i) / (n : R)))
        (linfNorm boundVec) :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2
      (by simpa [abs_sub_comm] using le_trans hlocal hle)
  rw [reduce_mean_by_row_get xS hRed' i, reduce_mean_by_row_get xR hRed' i]
  exact hScalarApprox

-- ---------------------------------------------------------------------------
-- Column-wise matrix sum (axis=0)
-- ---------------------------------------------------------------------------

/-- Extract column `j` from a runtime `m×n` tensor. -/
def colR {m n : Nat} (xR : Tensor R [m, n]) (j : Fin n) : Tensor R [m] :=
  Tensor.dim fun i => (xR.unstack i).unstack j

/-- Extract column `j` from a spec `m×n` tensor. -/
def colS {m n : Nat} (xS : SpecTensor [m, n]) (j : Fin n) : SpecTensor [m] :=
  Tensor.dim fun i => (xS.unstack i).unstack j

/-- Column-wise budget vector for an `m × n` runtime matrix, the axis-0 mirror of
`sumRowBoundVec`: entry `j` is the accumulated rounding budget of runtime column `j`. -/
def sumColumnBoundVec {m n : Nat} (eps : ℝ) (xR : Tensor R [m, n]) : SpecTensor [n] :=
  Tensor.dim fun j =>
    Tensor.scalar (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [m]) eps
      (colR (m := m) (n := n) xR j))

/-- Column-wise `reduceSum` along axis `0` approximates the exact column sums within
`linfNorm (sumColumnBoundVec eps xR)`. Note that this is not the row statement with the arguments
renamed: the budget reads columns, so a caller cannot reuse one bound for the other reduction. -/
theorem approxTensor_reduce_sum_columns
    {m n : Nat}
    {xS : SpecTensor [m, n]}
    {xR : Tensor R [m, n]}
    {eps : ℝ}
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps)
    (hRed : Shape.NonemptyAxis 0 ([m, n] : Shape)) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TorchLean.Tensor.reduceSum (α := ℝ) (s := [m, n]) 0 xS hRed)
      (TorchLean.Tensor.reduceSum (α := R) (s := [m, n]) 0 xR hRed)
      (linfNorm (sumColumnBoundVec (β := β) (fexp := fexp) (rnd := rnd) eps xR)) := by
  classical
  have hm : 0 < m := pos_of_column_axis hRed
  have hε : 0 ≤ eps := approxTensor_eps_nonneg (s := [m, n]) hx
  let boundVec : SpecTensor [n] :=
    Tensor.dim (fun j =>
      Tensor.scalar (sumBound (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim m .scalar) eps (colR (m := m) (n := n) xR j)))
  have hBoundNonneg : 0 ≤ linfNorm boundVec := linf_norm_nonneg (t := boundVec)

  refine approxTensor_dim_of_forall
    (n := n) (s := .scalar)
    (xS := TorchLean.Tensor.reduceSum (α := ℝ) (s := [m, n]) 0 xS hRed)
    (xR := TorchLean.Tensor.reduceSum (α := R) (s := [m, n]) 0 xR hRed)
    (eps := linfNorm boundVec) hBoundNonneg ?_
  intro j
  have hcol :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (colS (m := m) (n := n) xS j) (colR (m := m) (n := n) xR j) eps := by
    refine approxTensor_dim_of_forall
      (n := m) (s := .scalar)
      (xS := colS (m := m) (n := n) xS j)
      (xR := colR (m := m) (n := n) xR j)
      (eps := eps) hε ?_
    intro i
    have hrow :=
      approxTensor_dim_get (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i
    have hij :=
      approxTensor_dim_get (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hrow j
    simpa [colS, colR] using hij

  have hSum :=
    approxTensor_sum_spec (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar)
      (xS := colS (m := m) (n := n) xS j)
      (xR := colR (m := m) (n := n) xR j) (eps := eps) hcol
  have hSumScalar :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).1 hSum
  have hle :
      sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar)
          eps (colR (m := m) (n := n) xR j) ≤
        linfNorm boundVec := by
    have hcomponent := linf_norm_le_get_dim (t := boundVec) j
    have habs :
        abs (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar)
          eps (colR (m := m) (n := n) xR j)) ≤ linfNorm boundVec := by
      simpa [boundVec, linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm,
        MathFunctions.abs] using hcomponent
    exact le_trans (le_abs_self _) habs
  have herror : abs
      (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (sumSpec (α := R) (s := .dim m .scalar) (colR (m := m) (n := n) xR j)) -
        sumSpec (α := ℝ) (s := .dim m .scalar) (colS (m := m) (n := n) xS j))
      ≤ linfNorm boundVec := le_trans hSumScalar hle
  have hScalarApprox :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.scalar
          (sumSpec (α := ℝ) (s := .dim m .scalar) (colS (m := m) (n := n) xS j)))
        (Tensor.scalar
          (sumSpec (α := R) (s := .dim m .scalar) (colR (m := m) (n := n) xR j)))
        (linfNorm boundVec) :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2
      (by simpa [abs_sub_comm] using herror)
  have hEqS :=
    reduce_sum_by_column_get (α := SpecScalar) (m := m) (n := n) xS hRed j
  have hEqR :=
    reduce_sum_by_column_get (α := R) (m := m) (n := n) xR hRed j
  rw [hEqS, hEqR]
  simpa [colS, colR] using hScalarApprox

end NFBackend

end
end RuntimeApprox
end Proofs
