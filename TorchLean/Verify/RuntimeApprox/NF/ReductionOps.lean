/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Elementwise.SafeDivSigmoid
public import TorchLean.Verify.RuntimeApprox.NF.Sum
public import TorchLean.Tensor.Reduce.Backward

/-!
# NF Reduction Operators

NF (rounded) backend: approximation lemmas for matrix row sums, column sums and row means.

The sums are `reduce .sum` over one axis, cast to the shape without that axis. The row mean divides
the row sum by the row length and takes the row axis to be nonempty.

Scope: row and column reductions on matrices, with sequential addition along the
selected axis. Extending this proof through a matrix view requires a bridge preserving the
element order and rounding schedule, not only the exact real sum.

## Mean bound
The row mean divides the rounded row sum `Σ̂` by the exact row length `n` with
`TorchLean.Context.divNat`, which for `NF` rounds the exact real quotient `Σ̂ / n` once (the count
is never rounded first). Its budget `meanRowBound` is `ulp(Σ̂ / n) / 2 + sumBound eps row / n` at
every positive `n`.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open Robustness

noncomputable section

section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open TorchLean.NF

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd


-- ---------------------------------------------------------------------------
-- Definitional unfoldings for matrix reductions (axis 0/1)
-- ---------------------------------------------------------------------------

private theorem NF.reduce_sum_by_row_get
    {α : Type} [Storage α] [Add α] [Zero α]
    {m n : Nat} (x : Tensor α [m, n]) (i : Fin m) :
    Tensor.unstack (castShape (reduce .sum x [⟨1, Nat.one_lt_two⟩] false)
        (Shape.reduce_singleton _ _) : Tensor α [m]) i =
      Tensor.scalar (Tensor.sum (α := α) (s := [n]) (x.unstack i)) := by
  apply Tensor.Rep.ext
  intro c
  cases c
  rw [Tensor.unstack_apply, Tensor.scalar_apply]
  erw [castShape_reduce_sum_singleton_mk_apply x 1]
  rw [Tensor.sum_eq_foldl]
  rfl

private theorem NF.reduce_mean_by_row_get
    {α : Type} [Storage α] [Context α]
    {m n : Nat} (x : Tensor α [m, n]) (i : Fin m) :
    (Tensor.map (fun s => Context.divNat s n)
        (castShape (reduce .sum x [⟨1, Nat.one_lt_two⟩] false) (Shape.reduce_singleton _ _)) :
        Tensor α [m]).unstack i =
      Tensor.scalar (Context.divNat (Tensor.sum (α := α) (s := [n]) (x.unstack i)) n) := by
  rw [Tensor.unstack_map]
  rw [reduce_sum_by_row_get x i]
  apply Tensor.Rep.ext
  intro coordinate
  cases coordinate
  simp [Tensor.scalar]

private theorem NF.reduce_sum_by_column_get
    {α : Type} [Storage α] [Add α] [Zero α]
    {m n : Nat} (x : Tensor α [m, n]) (j : Fin n) :
    Tensor.unstack (castShape (reduce .sum x [⟨0, Nat.two_pos⟩] false)
        (Shape.reduce_singleton _ _) : Tensor α [n]) j =
      Tensor.scalar (Tensor.sum (α := α) (s := .dim m .scalar)
        (Tensor.dim fun i : Fin m => (x.unstack i).unstack j)) := by
  apply Tensor.Rep.ext
  intro c
  cases c
  rw [Tensor.unstack_apply, Tensor.scalar_apply]
  erw [castShape_reduce_sum_singleton_mk_apply x 0]
  rw [Tensor.sum_eq_foldl]
  congr 1
  apply Tensor.Rep.ext
  rintro ⟨i, u⟩
  cases u
  simp only [Tensor.ofFn, Tensor.Rep.get_ofFn]
  erw [Tensor.dim_apply, Tensor.unstack_apply,
    Tensor.unstack_apply]
  rfl

-- ---------------------------------------------------------------------------
-- Row-wise matrix sum (axis=1)
-- ---------------------------------------------------------------------------

/-- Axis-1 nonemptiness of a matrix shape is exactly positivity of the row length, so a theorem that
already asks for the reduction's axis evidence does not need a separate `0 < n` hypothesis. -/
private theorem NF.pos_of_row_axis {m n : Nat}
    (hRed : Shape.NonemptyAxis 1 ([m, n] : Shape)) : 0 < n := by
  cases hRed with
  | succ inner => cases inner; exact Nat.succ_pos _

/-- Per-row sequential summation budgets for an `m × n` NF matrix. -/
def NF.sumRowBoundVec {m n : Nat} (eps : ℝ) (xR : Tensor R [m, n]) : Tensor ℝ [m] :=
  Tensor.dim fun i =>
    Tensor.scalar (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps (xR.unstack i))

/-- The sum along axis `1` approximates the exact row sums within
`linfNorm (sumRowBoundVec eps xR)`. -/
theorem ApproxTensor.reduce_sum_rows
    {m n : Nat}
    {xS : Tensor ℝ [m, n]}
    {xR : Tensor R [m, n]}
    {eps : ℝ}
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (castShape (reduce (α := ℝ) .sum xS [⟨1, Nat.one_lt_two⟩] false)
        (Shape.reduce_singleton _ _))
      (castShape (reduce (α := R) .sum xR [⟨1, Nat.one_lt_two⟩] false)
        (Shape.reduce_singleton _ _))
      (linfNorm (sumRowBoundVec (β := β) (fexp := fexp) (rnd := rnd) eps xR)) := by
  classical
  let boundVec := sumRowBoundVec (β := β) (fexp := fexp) (rnd := rnd) eps xR
  have hBoundNonneg : 0 ≤ linfNorm boundVec := linfNorm_nonneg (t := boundVec)

  refine approxTensor_dim_of_forall
    (n := m) (s := .scalar)
    (xS := castShape (reduce (α := ℝ) .sum xS [⟨1, Nat.one_lt_two⟩] false)
      (Shape.reduce_singleton _ _))
    (xR := castShape (reduce (α := R) .sum xR [⟨1, Nat.one_lt_two⟩] false)
      (Shape.reduce_singleton _ _))
    (eps := linfNorm boundVec) hBoundNonneg ?_
  intro i
  have hxRow :=
    unstack (α := R) (toSpec := NF.toReal)
      (xS := xS) (xR := xR) (eps := eps) hx i
  have hSum :=
    sum (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar)
      (xS := xS.unstack i) (xR := xR.unstack i) (eps := eps) hxRow
  have hSumScalar :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := NF.toReal)).1 hSum
  have hle : sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar)
      eps (xR.unstack i) ≤ linfNorm boundVec := by
    have hcomponent := linfNorm_unstack_le (t := boundVec) i
    have habs : abs (sumBound (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim n .scalar) eps (xR.unstack i)) ≤ linfNorm boundVec := by
      simpa [boundVec, sumRowBoundVec, linfNorm,
        Numerics.MathFunctions.abs] using hcomponent
    exact le_trans (le_abs_self _) habs
  have herror : abs
      (NF.toReal
          (Tensor.sum (α := R) (s := .dim n .scalar) (xR.unstack i)) -
        Tensor.sum (α := ℝ) (s := .dim n .scalar) (xS.unstack i))
      ≤ linfNorm boundVec := le_trans hSumScalar hle
  have hScalarApprox :
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (Tensor.scalar (Tensor.sum (α := ℝ) (s := .dim n .scalar) (xS.unstack i)))
        (Tensor.scalar (Tensor.sum (α := R) (s := .dim n .scalar) (xR.unstack i)))
        (linfNorm boundVec) :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := NF.toReal)).2
      (by simpa [abs_sub_comm] using herror)
  rw [reduce_sum_by_row_get xS i, reduce_sum_by_row_get xR i]
  exact hScalarApprox

-- ---------------------------------------------------------------------------
-- Row-wise matrix mean (axis=1)
-- ---------------------------------------------------------------------------

/-- Per-row forward error budget for the mean along the last axis of a matrix. The rounded row sum
`Σ̂` divided by the exact row length `n` is rounded once (`Context.divNat`), so the budget is one
half ulp of `Σ̂ / n` plus the row-sum budget divided by `n`. -/
def NF.meanRowBound {n : Nat} (eps : ℝ) (rowR : Tensor R [n]) : ℝ :=
  ulp β fexp
      (NF.toReal (Tensor.sum (α := R) (s := [n]) rowR) /
        (n : ℝ)) / 2 +
    sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR / (n : ℝ)

/-- Row-wise mean budget vector for an `m × n` runtime matrix. -/
def NF.meanRowBoundVec {m n : Nat} (eps : ℝ) (xR : Tensor R [m, n]) : Tensor ℝ [m] :=
  Tensor.dim fun i =>
    Tensor.scalar (meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i))

/-- Scalar certificate for one row mean: the rounded quotient `RN(Σ̂ / n)` approximates `Σ / n`
within `meanRowBound`. -/
theorem NF.approx_meanRow {n : Nat} (hn : 0 < n)
    {rowS : Tensor ℝ [n]} {rowR : Tensor R [n]} {eps : ℝ}
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal)
      rowS rowR eps) :
    abs (NF.toReal
          (Context.divNat (Tensor.sum (α := R) (s := [n]) rowR) n) -
        Tensor.sum (α := ℝ) (s := [n]) rowS / (n : ℝ)) ≤
      meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps rowR := by
  have hSum := ApproxTensor.sum (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) hx
  have hSumScalar :
      abs (NF.toReal (Tensor.sum (α := R) (s := [n]) rowR) -
          Tensor.sum (α := ℝ) (s := [n]) rowS) ≤
        sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := NF.toReal)).1 hSum
  have hnpos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hround := approx_ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (NF.toReal (Tensor.sum (α := R) (s := [n]) rowR) / (n : ℝ))
  have hdiff :
      abs (NF.toReal (Tensor.sum (α := R) (s := [n]) rowR) /
            (n : ℝ) - Tensor.sum (α := ℝ) (s := [n]) rowS / (n : ℝ)) ≤
        sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [n]) eps rowR / (n : ℝ) := by
    rw [← sub_div, abs_div, abs_of_pos hnpos]
    exact div_le_div_of_nonneg_right hSumScalar hnpos.le
  exact (abs_sub_le _ _ _).trans (add_le_add hround hdiff)

/-- The mean along axis `1` of an `m × n` matrix approximates the exact row means
within `linfNorm (meanRowBoundVec eps xR)`. Each entry is the budget `meanRowBound` of its
row. -/
theorem ApproxTensor.reduce_mean_rows
    {m n : Nat}
    {xS : Tensor ℝ [m, n]}
    {xR : Tensor R [m, n]}
    {eps : ℝ}
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps)
    (hRed : Shape.NonemptyAxis 1 ([m, n] : Shape)) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (Tensor.map (fun s => Context.divNat s n)
        (castShape (reduce (α := ℝ) .sum xS [⟨1, Nat.one_lt_two⟩] false)
          (Shape.reduce_singleton _ _)))
      (Tensor.map (fun s => Context.divNat s n)
        (castShape (reduce (α := R) .sum xR [⟨1, Nat.one_lt_two⟩] false)
          (Shape.reduce_singleton _ _)))
      (linfNorm (meanRowBoundVec (β := β) (fexp := fexp) (rnd := rnd) eps xR)) := by
  classical
  have hn : 0 < n := pos_of_row_axis hRed
  set boundVec : Tensor ℝ [m] :=
    meanRowBoundVec (β := β) (fexp := fexp) (rnd := rnd) eps xR with hboundVec
  have hBoundNonneg : 0 ≤ linfNorm boundVec := linfNorm_nonneg (t := boundVec)
  refine approxTensor_dim_of_forall
    (n := m) (s := .scalar)
    (eps := linfNorm boundVec) hBoundNonneg ?_
  intro i
  have hxRow :=
    unstack (α := R) (toSpec := NF.toReal)
      (xS := xS) (xR := xR) (eps := eps) hx i
  have hlocal := approx_meanRow (β := β) (fexp := fexp) (rnd := rnd) hn hxRow
  have hunstack : boundVec.unstack i =
      Tensor.scalar (meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i)) := by
    simp [hboundVec, meanRowBoundVec]
  have hle : meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i) ≤
      linfNorm boundVec := by
    have hcomponent := linfNorm_unstack_le (t := boundVec) i
    rw [hunstack] at hcomponent
    have habs : abs (meanRowBound (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i)) ≤
        linfNorm boundVec := by
      simpa [linfNorm, Numerics.MathFunctions.abs]
        using hcomponent
    exact le_trans (le_abs_self _) habs
  have hScalarApprox :
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (Tensor.scalar (Context.divNat (Tensor.sum (α := ℝ) (s := [n]) (xS.unstack i)) n))
        (Tensor.scalar (Context.divNat (Tensor.sum (α := R) (s := [n]) (xR.unstack i)) n))
        (linfNorm boundVec) :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := NF.toReal)).2
      (by simpa [abs_sub_comm, Context.divNat_real] using le_trans hlocal hle)
  erw [reduce_mean_by_row_get xS i, reduce_mean_by_row_get xR i]
  exact hScalarApprox

-- ---------------------------------------------------------------------------
-- Column-wise matrix sum (axis=0)
-- ---------------------------------------------------------------------------

/-- Extract column `j` from an `m × n` tensor. -/
def NF.column {α : Type*} [Storage α] {m n : Nat} (x : Tensor α [m, n]) (j : Fin n) :
    Tensor α [m] :=
  Tensor.dim fun i => (x.unstack i).unstack j

/-- Column-wise budget vector for an `m × n` runtime matrix, the axis-0 mirror of
`sumRowBoundVec`: entry `j` is the accumulated rounding budget of runtime column `j`. -/
def NF.sumColumnBoundVec {m n : Nat} (eps : ℝ) (xR : Tensor R [m, n]) : Tensor ℝ [n] :=
  Tensor.dim fun j =>
    Tensor.scalar (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := [m]) eps
      (column (m := m) (n := n) xR j))

/-- The sum along axis `0` approximates the exact column sums within
`linfNorm (sumColumnBoundVec eps xR)`. The budget reads the columns of `xR`, so the row and
column reductions each need their own bound. -/
theorem ApproxTensor.reduce_sum_columns
    {m n : Nat}
    {xS : Tensor ℝ [m, n]}
    {xR : Tensor R [m, n]}
    {eps : ℝ}
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (castShape (reduce (α := ℝ) .sum xS [⟨0, Nat.two_pos⟩] false)
        (Shape.reduce_singleton _ _))
      (castShape (reduce (α := R) .sum xR [⟨0, Nat.two_pos⟩] false)
        (Shape.reduce_singleton _ _))
      (linfNorm (sumColumnBoundVec (β := β) (fexp := fexp) (rnd := rnd) eps xR)) := by
  classical
  have hε : 0 ≤ eps := eps_nonneg (s := [m, n]) hx
  let boundVec := sumColumnBoundVec (β := β) (fexp := fexp) (rnd := rnd) eps xR
  have hBoundNonneg : 0 ≤ linfNorm boundVec := linfNorm_nonneg (t := boundVec)

  refine approxTensor_dim_of_forall
    (n := n) (s := .scalar)
    (xS := castShape (reduce (α := ℝ) .sum xS [⟨0, Nat.two_pos⟩] false)
      (Shape.reduce_singleton _ _))
    (xR := castShape (reduce (α := R) .sum xR [⟨0, Nat.two_pos⟩] false)
      (Shape.reduce_singleton _ _))
    (eps := linfNorm boundVec) hBoundNonneg ?_
  intro j
  have hcol :
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (column (m := m) (n := n) xS j) (column (m := m) (n := n) xR j) eps := by
    refine approxTensor_dim_of_forall
      (n := m) (s := .scalar)
      (xS := column (m := m) (n := n) xS j)
      (xR := column (m := m) (n := n) xR j)
      (eps := eps) hε ?_
    intro i
    have hrow :=
      unstack (α := R)
        (toSpec := NF.toReal) hx i
    have hij :=
      unstack (α := R)
        (toSpec := NF.toReal) hrow j
    simpa [column] using hij

  have hSum :=
    sum (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar)
      (xS := column (m := m) (n := n) xS j)
      (xR := column (m := m) (n := n) xR j) (eps := eps) hcol
  have hSumScalar :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := NF.toReal)).1 hSum
  have hle :
      sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar)
          eps (column (m := m) (n := n) xR j) ≤
        linfNorm boundVec := by
    have hcomponent := linfNorm_unstack_le (t := boundVec) j
    have habs :
        abs (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar)
          eps (column (m := m) (n := n) xR j)) ≤ linfNorm boundVec := by
      simpa [boundVec, sumColumnBoundVec, linfNorm,
        Numerics.MathFunctions.abs] using hcomponent
    exact le_trans (le_abs_self _) habs
  have herror : abs
      (NF.toReal
          (Tensor.sum (α := R) (s := .dim m .scalar) (column (m := m) (n := n) xR j)) -
        Tensor.sum (α := ℝ) (s := .dim m .scalar) (column (m := m) (n := n) xS j))
      ≤ linfNorm boundVec := le_trans hSumScalar hle
  have hScalarApprox :
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (Tensor.scalar
          (Tensor.sum (α := ℝ) (s := .dim m .scalar) (column (m := m) (n := n) xS j)))
        (Tensor.scalar
          (Tensor.sum (α := R) (s := .dim m .scalar) (column (m := m) (n := n) xR j)))
        (linfNorm boundVec) :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := NF.toReal)).2
      (by simpa [abs_sub_comm] using herror)
  have hEqS :=
    reduce_sum_by_column_get (α := ℝ) (m := m) (n := n) xS j
  have hEqR :=
    reduce_sum_by_column_get (α := R) (m := m) (n := n) xR j
  rw [hEqS, hEqR]
  simpa [column] using hScalarApprox

end

end
end TorchLean
