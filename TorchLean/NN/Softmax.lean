/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Algebra.BigOperators.Field
import TorchLean.Tensor.Elementwise.Real
public import TorchLean.Tensor.LinearAlgebra.Real
public import TorchLean.Tensor.Numeric.Real
public import TorchLean.NN.Gradients.Activation
public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Layers.Attention
public import TorchLean.Tensor.Context.Real

/-!
# Softmax analysis properties

Real-valued facts about the spec softmax operators. The vector kernels live in
`TorchLean.Tensor.Layers.Activation` and the axis-parametric operators in
`TorchLean.Tensor.Layers.Softmax`. The proofs need real analysis and finite sums, which is why
the module sits under `TorchLean/NN/`.

Theorems:

- `Softmax.item_maxVec`: over `ℝ` the maximum that stable softmax shifts by is the left fold of
  `max`.
- `getScalar_le_maxVec` and `exists_getScalar_eq_maxVec`: that maximum is an attained
  upper bound.
- `sub_maxVec_nonpos` and `exists_sub_maxVec_eq_zero`: shifted logits are nonpositive and
  one of them is zero.
- `maxShiftedExpVec_le_one` and `sum_maxShiftedExpVec_bounds`: shifted exponentials are at most
  `1`, and their sum lies between `1` and the axis length.
- `exists_vec_eq_div_sum`: the stable kernel divides positive weights by their sum.
- `vec_pos`, `vec_mem_unitInterval` and `Softmax.sum_vec`: every
  coordinate is strictly positive and lies in `[0,1]`, and a nonempty softmax vector sums to `1`.
- `sum_softmaxBackward`: the stable softmax VJP has coordinate sum zero.
- `abs_softmaxBackward_le`: upstream coordinates bounded by `G` give VJP
  coordinates bounded by `2G`, whatever the axis length.
- `sum_softmax_row`: every row of an axis-`1` matrix softmax sums to `1` when there is at least
  one key.
- `Softmax.getScalar_vec_eq_exp_div` and `LogSoftmax.getScalar_vec_eq_sub_log`: over `ℝ`
  the stable kernels agree coordinatewise with the textbook formulas, which connects them to the
  analytic `Vec.softmax` and `Vec.logSoftmax` of `TorchLean.NN.Softmax.FDeriv` and
  `TorchLean.NN.Softmax.Log`.
- `softmax_zero_vec`, `softmax_one_matrix` and `unstack_softmax_one`: on vectors and matrix rows the
  axis-parametric operators reduce to the vector kernels.

Everything is stated over `ℝ` because positivity of `exp` and division by a positive denominator
are what let the outputs be read as probabilities. Overflow and underflow in a float
implementation are outside these theorems: rounded subtraction, exponential evaluation and
denominator accumulation need separate bounds.
-/

@[expose] public section

open scoped BigOperators
open FloatLib.Numerics (MathFunctions)

noncomputable section

namespace TorchLean

open TorchLean.Tensor

/-! ## Stable max shift -/

/-- Over `ℝ` the maximum that stable softmax shifts by is the left fold of `max` from the first
coordinate. -/
theorem Softmax.item_maxVec {n : Nat} (t : Tensor ℝ [Nat.succ n]) :
    Tensor.item (maxVec t) =
      (List.finRange (Nat.succ n)).foldl (fun acc j => max acc (t.getScalar j))
        (t.getScalar ⟨0, Nat.succ_pos n⟩) := by
  simp only [maxVec, Tensor.item_scalar, Context.maximum_real]

/-- Every coordinate is bounded above by the exact maximum that stable softmax shifts by. -/
theorem Softmax.getScalar_le_maxVec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Tensor.getScalar t i ≤ Tensor.item (maxVec t) := by
  rw [item_maxVec]
  exact List.le_foldl_max_of_mem (List.finRange (Nat.succ n))
    (fun j => t.getScalar j)
    (acc := t.getScalar ⟨0, Nat.succ_pos n⟩) (i := i)
    (List.mem_finRange i)

/-- The maximum used by stable softmax is attained by an input coordinate. -/
theorem Softmax.exists_getScalar_eq_maxVec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ i, Tensor.getScalar t i = Tensor.item (maxVec t) := by
  rw [item_maxVec]
  let firstIndex : Fin (Nat.succ n) := ⟨0, Nat.succ_pos n⟩
  let value : Fin (Nat.succ n) → ℝ := fun i => t.getScalar i
  change ∃ i, value i =
    (List.finRange (Nat.succ n)).foldl (fun acc j => max acc (value j))
      (value firstIndex)
  rcases List.foldl_max_eq_init_or_mem (List.finRange (Nat.succ n)) value
      (value firstIndex) with hfirst | ⟨i, hi, hvalue⟩
  · exact ⟨firstIndex, hfirst.symm⟩
  · exact ⟨i, hvalue.symm⟩

/-- Every logit shifted by the implementation's maximum is nonpositive. -/
theorem Softmax.sub_maxVec_nonpos {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Tensor.getScalar
      (t - Tensor.full _ (maxVec t).item) i ≤ 0 := by
  have hle := getScalar_le_maxVec t i
  rw [show (t - Tensor.full _ (maxVec t).item) =
    Tensor.zipWith (· - ·) t (Tensor.full _ (maxVec t).item) by rfl]
  rw [Tensor.getScalar_zipWith]
  simpa using sub_nonpos.mpr hle

/-- At least one max-shifted logit is exactly zero. -/
theorem Softmax.exists_sub_maxVec_eq_zero {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ i, Tensor.getScalar
      (t - Tensor.full _ (maxVec t).item) i = 0 := by
  rcases exists_getScalar_eq_maxVec t with ⟨i, hi⟩
  refine ⟨i, ?_⟩
  rw [show (t - Tensor.full _ (maxVec t).item) =
    Tensor.zipWith (· - ·) t (Tensor.full _ (maxVec t).item) by rfl]
  rw [Tensor.getScalar_zipWith]
  simpa using sub_eq_zero.mpr hi

/-- Exponentiating a max-shifted real logit produces a value at most one. -/
theorem Softmax.maxShiftedExpVec_le_one {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Tensor.getScalar (maxShiftedExpVec t) i ≤ 1 := by
  have hshift := sub_maxVec_nonpos t i
  simpa [maxShiftedExpVec, Tensor.exp,
    MathFunctions.exp_real] using (Real.exp_le_one_iff.mpr hshift)

/-- Max-shifted real exponentials remain strictly positive. -/
theorem Softmax.maxShiftedExpVec_pos {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    0 < Tensor.getScalar (maxShiftedExpVec t) i := by
  simpa [maxShiftedExpVec, Tensor.exp,
    MathFunctions.exp_real] using
    Real.exp_pos (Tensor.getScalar
      (t - Tensor.full _ (maxVec t).item) i)

/-- One max-shifted exponential is exactly one, because the maximum is attained. -/
theorem Softmax.exists_maxShiftedExpVec_eq_one {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ i, Tensor.getScalar (maxShiftedExpVec t) i = 1 := by
  rcases exists_sub_maxVec_eq_zero t with ⟨i, hzero⟩
  refine ⟨i, ?_⟩
  simpa [maxShiftedExpVec, Tensor.exp,
    MathFunctions.exp_real] using (Real.exp_eq_one_iff _).mpr hzero

/-- For a vector of length `n + 1`, the stable softmax denominator lies in `[1, n + 1]`.

The lower bound rules out division by zero, and the upper bound holds because every shifted
exponential is at most one.
-/
theorem Softmax.sum_maxShiftedExpVec_bounds {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    1 ≤ Tensor.sum (maxShiftedExpVec t) ∧
      Tensor.sum (maxShiftedExpVec t) ≤ Nat.succ n := by
  classical
  let ex := maxShiftedExpVec t
  have hpos : ∀ i, 0 ≤ Tensor.getScalar ex i :=
    fun i => le_of_lt (maxShiftedExpVec_pos t i)
  have hle : ∀ i, Tensor.getScalar ex i ≤ 1 := fun i => maxShiftedExpVec_le_one t i
  rcases exists_maxShiftedExpVec_eq_one t with ⟨witness, hwitness⟩
  rw [Tensor.sum_vec]
  constructor
  · calc
      1 = Tensor.getScalar ex witness := hwitness.symm
      _ ≤ ∑ i, Tensor.getScalar ex i :=
        Finset.single_le_sum (fun i _ => hpos i) (Finset.mem_univ witness)
  · calc
      (∑ i, Tensor.getScalar ex i) ≤ ∑ _i : Fin (Nat.succ n), (1 : ℝ) := by
        exact Finset.sum_le_sum fun i _ => hle i
      _ = Nat.succ n := by simp

/-! ## Normalized coordinates -/

/-- Coordinate equation for the stable vector softmax: coordinate `i` is the `i`-th max-shifted
exponential divided by the sum of all of them.

The proofs in this file unfold the stable kernel only through this lemma, so none of them depends
on how the shared normalizer is implemented. -/
theorem Softmax.getScalar_vec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Tensor.getScalar (vec (α := ℝ) (n := Nat.succ n) t) i =
      Tensor.getScalar (maxShiftedExpVec t) i /
        Tensor.sum (maxShiftedExpVec t) := by
  change Tensor.getScalar
      (Tensor.zipWith (· / ·) (maxShiftedExpVec t)
        (Tensor.full _ (Tensor.sum (maxShiftedExpVec t)))) i = _
  rw [Tensor.getScalar_zipWith]
  simp

/--
The stable vector softmax divides positive weights by their sum.

Those weights are the max-shifted exponentials. The positivity, range and normalization proofs
below use only this description of the kernel.
-/
theorem Softmax.exists_vec_eq_div_sum {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ weights : Fin (Nat.succ n) → ℝ,
      (∀ i, 0 < weights i) ∧
      ∀ i,
        Tensor.getScalar (vec (α := ℝ)
            (n := Nat.succ n) t) i =
          weights i / ∑ j, weights j := by
  refine ⟨fun j => Tensor.getScalar (maxShiftedExpVec t) j,
    fun j => maxShiftedExpVec_pos t j, fun i => ?_⟩
  rw [getScalar_vec, Tensor.sum_vec]

/-- Coordinate formula for the max-shifted exponentials. -/
theorem Softmax.getScalar_maxShiftedExpVec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (j : Fin (Nat.succ n)) :
    Tensor.getScalar (maxShiftedExpVec t) j =
      Real.exp (Tensor.getScalar t j - Tensor.item (maxVec t)) := by
  simp only [maxShiftedExpVec, Tensor.exp, Tensor.sub_def,
    Tensor.getScalar_map, Tensor.getScalar_zipWith]
  simp [MathFunctions.exp_real]

/-- The stable softmax denominator is `∑ⱼ exp(xⱼ)` times `exp(-m)`, where `m` is the maximum it
shifts by. -/
theorem Softmax.sum_maxShiftedExpVec {n : Nat} (t : Tensor ℝ [Nat.succ n]) :
    Tensor.sum (maxShiftedExpVec t) =
      (∑ j, Real.exp (Tensor.getScalar t j)) *
        Real.exp (-Tensor.item (maxVec t)) := by
  rw [Tensor.sum_vec, Finset.sum_mul]
  refine Finset.sum_congr rfl ?_
  intro j _
  rw [getScalar_maxShiftedExpVec, Real.exp_sub, Real.exp_neg, div_eq_mul_inv]

/-- Over `ℝ`, the stable max-shifted softmax agrees coordinatewise with the textbook formula
`exp(xᵢ) / ∑ⱼ exp(xⱼ)`.

Analytic developments work with the unshifted formula through this lemma, while the spec keeps
its overflow-safe implementation. -/
theorem Softmax.getScalar_vec_eq_exp_div {n : Nat} (t : Tensor ℝ [n]) (i : Fin n) :
    Tensor.getScalar (vec (α := ℝ) t) i =
      Real.exp (Tensor.getScalar t i) /
        ∑ j, Real.exp (Tensor.getScalar t j) := by
  cases n with
  | zero => exact i.elim0
  | succ n =>
    rw [getScalar_vec, sum_maxShiftedExpVec, getScalar_maxShiftedExpVec,
      Real.exp_sub, Real.exp_neg, div_eq_mul_inv]
    exact mul_div_mul_right _ _ (inv_ne_zero (Real.exp_ne_zero _))

/-- Over `ℝ`, the stable log-softmax agrees coordinatewise with `xᵢ - log ∑ⱼ exp(xⱼ)`. -/
theorem LogSoftmax.getScalar_vec_eq_sub_log {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Tensor.getScalar (vec (α := ℝ)
        (n := Nat.succ n) t) i =
      Tensor.getScalar t i -
        Real.log (∑ j, Real.exp (Tensor.getScalar t j)) := by
  let pivot : Fin (Nat.succ n) := Softmax.pivot t
  let m := t.getScalar pivot
  let shifted := t - Tensor.full _ m
  let tail : Tensor ℝ [Nat.succ n] := Tensor.ofFn fun j =>
    if j = pivot then 0 else MathFunctions.exp (shifted.getScalar j)
  have hshift (j : Fin (Nat.succ n)) : shifted.getScalar j = t.getScalar j - m := by
    simp [shifted, sub_def]
  have hpartition : 1 + Tensor.sum tail =
      ∑ j, Real.exp (t.getScalar j - m) := by
    rw [Tensor.sum_vec]
    simp only [tail, Tensor.getScalar_ofFn, hshift, MathFunctions.exp]
    calc
      1 + ∑ j, (if j = pivot then 0 else Real.exp (t.getScalar j - m)) =
          ∑ j, ((if j = pivot then 1 else 0) +
            (if j = pivot then 0 else Real.exp (t.getScalar j - m))) := by
        rw [Finset.sum_add_distrib]
        simp
      _ = _ := by
        apply Finset.sum_congr rfl
        intro j _
        by_cases h : j = pivot <;> simp [h, m]
  have hweight : (∑ j, Real.exp (t.getScalar j - m)) =
      (∑ j, Real.exp (t.getScalar j)) * Real.exp (-m) := by
    simp_rw [sub_eq_add_neg, Real.exp_add]
    rw [Finset.sum_mul]
  have hsumPos : 0 < ∑ j, Real.exp (Tensor.getScalar t j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  have hlog : Context.log1p (Tensor.sum tail) =
      Real.log (∑ j, Real.exp (Tensor.getScalar t j)) -
        m := by
    rw [Context.log1p_eq_log_one_add, hpartition, hweight,
      Real.log_mul (ne_of_gt hsumPos) (Real.exp_ne_zero _),
      Real.log_exp, sub_eq_add_neg]
  have hfinite : Context.isInf m = false := by simp [Context.isInf]
  change Tensor.getScalar
    (shifted - (Tensor.full _ (if Context.isInf m then shifted.getScalar pivot
      else Context.log1p (Tensor.sum tail)))) i = _
  simp only [hfinite, Bool.false_eq_true, ↓reduceIte, sub_def, Tensor.getScalar_zipWith,
    hshift, hlog, Tensor.getScalar_full]
  ring

/-! ## Axis softmax on vectors and matrices

The axis-parametric operators move the selected axis to the innermost position and back. A vector
(axis `0`) and the key axis of a matrix (axis `1`) need no permutation, so there the operators
reduce definitionally to the vector kernels. Downstream files use these lemmas and never see how
the permutation bookkeeping is implemented. Two real identities for the cancellation-resistant
log-softmax VJP and JVP kernels come first. -/

/-- Over `ℝ`, the cancellation-resistant log-softmax VJP is `dY - exp y * ∑ⱼ dYⱼ` for every `y`,
even when `exp y` does not sum to one. -/
theorem LogSoftmax.vecBackward_eq {n : Nat} (y dY : Tensor ℝ [n]) :
    vecBackward y dY =
      dY - (Tensor.exp y *
        Tensor.full _ (Tensor.sum dY)) := by
  cases n with
  | zero =>
      apply Tensor.ext_vector
      intro i
      exact i.elim0
  | succ n =>
      let pivot : Fin (Nat.succ n) := Softmax.pivot y
      let others : Tensor ℝ [Nat.succ n] := Tensor.ofFn fun i =>
        if i = pivot then 0 else dY.getScalar i
      have hsum : Tensor.sum dY =
          dY.getScalar pivot + Tensor.sum others := by
        simp only [Tensor.sum_vec, others, Tensor.getScalar_ofFn]
        calc
          (∑ i, dY.getScalar i) = ∑ i, ((if i = pivot then dY.getScalar pivot else 0) +
              (if i = pivot then 0 else dY.getScalar i)) := by
            apply Finset.sum_congr rfl
            intro i _
            by_cases h : i = pivot <;> simp [h]
          _ = _ := by rw [Finset.sum_add_distrib]; simp
      apply Tensor.ext_vector
      intro i
      simp only [vecBackward, Tensor.getScalar_ofFn]
      change (if i = pivot then
          -(Context.expm1 (y.getScalar i)) * dY.getScalar i -
            MathFunctions.exp (y.getScalar i) * Tensor.sum others
        else dY.getScalar i - MathFunctions.exp (y.getScalar i) *
          Tensor.sum dY) = _
      simp only [sub_def, Tensor.getScalar_zipWith, Tensor.getScalar_mul,
        Tensor.exp, Tensor.getScalar_map, Tensor.getScalar_full]
      split_ifs with hi
      · rw [Context.expm1_eq_exp_sub_one, hsum, hi]
        simp only [MathFunctions.exp]
        ring
      · rfl

/-- Over `ℝ`, the log-softmax JVP kernel, which splits off the pivot's complementary probability,
is `dx` minus the broadcast dot product `dot (exp y) dx` for every `y`, even when `exp y` does
not sum to one. -/
theorem LogSoftmax.innermostJVP_eq {n : Nat} (y dx : Tensor ℝ [n]) :
    innermostJVP y dx =
      dx - Tensor.full _ (dot (Tensor.exp y) dx) := by
  classical
  cases n with
  | zero =>
      apply Tensor.ext_vector
      intro i
      exact i.elim0
  | succ n =>
      let pivot : Fin (n + 1) := Softmax.pivot y
      let others : Tensor ℝ [n + 1] := Tensor.ofFn fun i =>
        if i = pivot then 0 else MathFunctions.exp (y.getScalar i) * dx.getScalar i
      have hsum : dot (Tensor.exp y) dx =
          MathFunctions.exp (y.getScalar pivot) * dx.getScalar pivot + Tensor.sum others := by
        simp only [dot, Tensor.sum_vec, Tensor.getScalar_mul,
          Tensor.exp, Tensor.getScalar_map, others, Tensor.getScalar_ofFn]
        calc
          (∑ i, MathFunctions.exp (y.getScalar i) * dx.getScalar i) =
              ∑ i, ((if i = pivot then
                MathFunctions.exp (y.getScalar pivot) * dx.getScalar pivot else 0) +
                (if i = pivot then 0 else
                  MathFunctions.exp (y.getScalar i) * dx.getScalar i)) := by
            apply Finset.sum_congr rfl
            intro i _
            by_cases h : i = pivot <;> simp [h]
          _ = _ := by rw [Finset.sum_add_distrib]; simp
      apply Tensor.ext_vector
      intro i
      change ((dx - Tensor.full _ (dx.getScalar pivot)) -
        (Tensor.full _ (Context.expm1 (y.getScalar pivot) * dx.getScalar pivot +
            Tensor.sum others))).getScalar i = _
      rw [hsum, Context.expm1_eq_exp_sub_one]
      simp only [sub_def, Tensor.getScalar_zipWith,
        Tensor.getScalar_full, MathFunctions.exp]
      ring

/-- Axis-`0` softmax on a vector is the vector kernel. -/
theorem Tensor.softmax_zero_vec {n : Nat} (t : Tensor ℝ [n]) :
    Tensor.softmax t ⟨0, Nat.one_pos⟩ = Softmax.vec t := rfl

/-- Axis-`0` softmax backward on a vector is the vector kernel. -/
theorem Tensor.softmaxBackward_zero_vec {n : Nat} (x dY : Tensor ℝ [n]) :
    Tensor.softmaxBackward (α := ℝ) (s := [n]) x dY ⟨0, Nat.one_pos⟩ =
      Softmax.innermostBackward x dY := rfl

/-- Axis-`0` log-softmax on a vector is the vector kernel. -/
theorem Tensor.logSoftmax_zero_vec {n : Nat} (t : Tensor ℝ [n]) :
    Tensor.logSoftmax t ⟨0, Nat.one_pos⟩ = LogSoftmax.vec t := rfl

/-- Axis-`0` log-softmax backward on a vector is the vector kernel. -/
theorem Tensor.logSoftmaxBackward_zero_vec {n : Nat} (y dY : Tensor ℝ [n]) :
    Tensor.logSoftmaxBackward (α := ℝ) (s := [n]) y dY ⟨0, Nat.one_pos⟩ =
      LogSoftmax.innermostBackward y dY := rfl

/-- Axis-`1` softmax on a matrix is the row-wise innermost kernel. -/
theorem Tensor.softmax_one_matrix {m n : Nat} (A : Tensor ℝ [m, n]) :
    Tensor.softmax (α := ℝ) (s := [m, n]) A ⟨1, Shape.AxisInBounds.proof⟩ =
      Softmax.innermost A := rfl

/-- Row `i` of an axis-`1` matrix softmax is the vector softmax of row `i`. -/
theorem unstack_softmax_one {m n : Nat} (A : Tensor ℝ [m, n]) (i : Fin m) :
    Tensor.unstack (Tensor.softmax (α := ℝ) (s := [m, n]) A ⟨1, Shape.AxisInBounds.proof⟩) i =
      Softmax.vec (Tensor.unstack A i) := by
  rw [softmax_one_matrix]
  exact Softmax.unstack_innermost_matrix A i

/-! ## Probability-simplex properties -/

/-- Every coordinate of a nonempty real softmax vector is strictly positive. -/
theorem Softmax.vec_pos {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    0 < Tensor.getScalar (vec (α := ℝ)
        (n := Nat.succ n) t) i := by
  classical
  rcases exists_vec_eq_div_sum t with ⟨weights, hpos, hcoord⟩
  rw [hcoord i]
  exact div_pos (hpos i) (Finset.sum_pos (fun j _ => hpos j) Finset.univ_nonempty)

/-- The entries of a nonempty real softmax vector sum to `1`. -/
theorem Softmax.sum_vec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    Tensor.sum (vec (α := ℝ)
        (n := Nat.succ n) t) = 1 := by
  classical
  rcases exists_vec_eq_div_sum t with ⟨weights, hpos, hcoord⟩
  rw [Tensor.sum_vec]
  simp_rw [hcoord]
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (Finset.sum_pos (fun j _ => hpos j) Finset.univ_nonempty))

/-- Every coordinate of a nonempty real softmax vector lies in the closed unit interval. -/
theorem Softmax.vec_mem_unitInterval {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Tensor.getScalar (vec (α := ℝ) (n := Nat.succ n) t) i
      ∈ Set.Icc 0 1 := by
  classical
  let y := vec (α := ℝ) (n := Nat.succ n) t
  have hpos : ∀ j, 0 < Tensor.getScalar y j := fun j => vec_pos t j
  have hsum : ∑ j, Tensor.getScalar y j = 1 := by
    simpa [Tensor.sum_vec] using sum_vec t
  constructor
  · exact le_of_lt (hpos i)
  · calc
      Tensor.getScalar y i ≤ ∑ j, Tensor.getScalar y j :=
        Finset.single_le_sum (fun j _ => le_of_lt (hpos j)) (Finset.mem_univ i)
      _ = 1 := hsum

/-! ## Backward conservation -/

/-- Coordinate formula for the innermost stable softmax VJP: with
`y = vec x`, the `i`-th input gradient is `yᵢ (dYᵢ - ∑ⱼ yⱼ dYⱼ)`.
Both theorems below start from it. -/
theorem Softmax.getScalar_innermostBackward {n : Nat}
    (x dY : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Tensor.getScalar (innermostBackward x dY) i =
      Tensor.getScalar (vec x) i *
        (Tensor.getScalar dY i -
          ∑ j, Tensor.getScalar (vec x) j *
            Tensor.getScalar dY j) := by
  let pivot : Fin (Nat.succ n) := Softmax.pivot x
  let anchor := dY.getScalar pivot
  let y := vec x
  let centered := dY - Tensor.full _ anchor
  have hcoord (j : Fin (Nat.succ n)) : centered.getScalar j = dY.getScalar j - anchor := by
    simp [centered, sub_def]
  have hmass : (∑ j, y.getScalar j) = 1 := by
    simpa [Tensor.sum_vec, y] using sum_vec x
  have hs : Tensor.sum (centered * y) =
      (∑ j, y.getScalar j * dY.getScalar j) - anchor := by
    rw [Tensor.sum_vec]
    simp_rw [Tensor.getScalar_mul, hcoord, mul_comm (_ - anchor), mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hmass, one_mul]
  change (y * (centered - (Tensor.full _ (Tensor.sum
      (centered * y))))).getScalar i = _
  rw [Tensor.getScalar_mul, sub_def, Tensor.getScalar_zipWith, hcoord, hs]
  simp only [Tensor.getScalar_full]
  change y.getScalar i * (_ - _) = y.getScalar i * _
  ring

/-- The stable softmax VJP is tangent to the probability simplex: its coordinates sum to zero.

`Tensor.softmaxBackward` along axis `0` is the vector VJP that the spec and tape layers use, and
the sum vanishes because the stable forward weights sum to one. The theorem is stated for this
tensor definition, which is separate from the analytic `EuclideanSpace` presentation of the same
derivative. -/
theorem Tensor.sum_softmaxBackward {n : Nat}
    (x dY : Tensor ℝ [Nat.succ n]) :
    Tensor.sum
      (Tensor.softmaxBackward (α := ℝ) (s := [Nat.succ n]) x dY ⟨0, Nat.one_pos⟩) = 0 := by
  have hy : (∑ i, Tensor.getScalar (Softmax.vec x) i) = 1 := by
    simpa [sum_vec] using Softmax.sum_vec x
  -- `∑ᵢ yᵢ (dYᵢ - S) = S - (∑ᵢ yᵢ) S = 0` because the forward weights sum to one.
  rw [softmaxBackward_zero_vec, sum_vec]
  simp_rw [Softmax.getScalar_innermostBackward, mul_sub, Finset.sum_sub_distrib,
    ← Finset.sum_mul, hy, one_mul, sub_self]

/-- Coordinatewise bound for the concrete stable softmax VJP.

If every upstream coordinate has magnitude at most `G`, each input-gradient coordinate has
magnitude at most `2G`. The estimate does not grow with the axis length because the softmax output
is a nonnegative vector of total mass one. -/
theorem Tensor.abs_softmaxBackward_le {n : Nat}
    (x dY : Tensor ℝ [Nat.succ n]) (G : ℝ)
    (hdY : ∀ i, |Tensor.getScalar dY i| ≤ G) (i : Fin (Nat.succ n)) :
    |Tensor.getScalar
      (Tensor.softmaxBackward (α := ℝ) (s := [Nat.succ n]) x dY ⟨0, Nat.one_pos⟩) i| ≤
        2 * G := by
  rw [softmaxBackward_zero_vec, Softmax.getScalar_innermostBackward]
  set y := Softmax.vec (α := ℝ) (n := Nat.succ n) x
  set S := ∑ j, Tensor.getScalar y j * Tensor.getScalar dY j
  have hyPos : ∀ j, 0 < Tensor.getScalar y j := fun j => Softmax.vec_pos x j
  have hySum : (∑ j, Tensor.getScalar y j) = 1 := by
    simpa [y, sum_vec] using Softmax.sum_vec x
  -- The weighted mean `S` of the upstream coordinates is bounded by `G`.
  have hSAbs : |S| ≤ G := by
    calc
      |S| ≤ ∑ j, |Tensor.getScalar y j * Tensor.getScalar dY j| :=
        Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, Tensor.getScalar y j * |Tensor.getScalar dY j| := by
        refine Finset.sum_congr rfl ?_
        intro j _
        rw [abs_mul, abs_of_pos (hyPos j)]
      _ ≤ ∑ j, Tensor.getScalar y j * G :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hdY j) (le_of_lt (hyPos j))
      _ = G := by rw [← Finset.sum_mul, hySum, one_mul]
  have hyLeOne : Tensor.getScalar y i ≤ 1 :=
    (Softmax.vec_mem_unitInterval x i).2
  have hdiff : |Tensor.getScalar dY i - S| ≤ 2 * G := by
    calc
      |Tensor.getScalar dY i - S| ≤ |Tensor.getScalar dY i| + |S| :=
        abs_sub _ _
      _ ≤ G + G := add_le_add (hdY i) hSAbs
      _ = 2 * G := by ring
  rw [abs_mul, abs_of_pos (hyPos i)]
  calc
    Tensor.getScalar y i * |Tensor.getScalar dY i - S| ≤
        1 * |Tensor.getScalar dY i - S| :=
      mul_le_mul_of_nonneg_right hyLeOne (abs_nonneg _)
    _ ≤ 1 * (2 * G) := mul_le_mul_of_nonneg_left hdiff zero_le_one
    _ = 2 * G := one_mul _

/-- Each row of the axis-`1` softmax of a score matrix sums to `1` when there is at least one
key. -/
theorem Tensor.sum_softmax_row {nQ nK : Nat} (hK : nK ≠ 0)
    (scores : Tensor ℝ [nQ, nK]) (i : Fin nQ) :
    Tensor.sum
        (Tensor.unstack
          (Tensor.softmax (α := ℝ) (s := [nQ, nK]) scores ⟨1, Shape.AxisInBounds.proof⟩) i)
      = 1 := by
  cases nK with
  | zero => exact (hK rfl).elim
  | succ nK' =>
      rw [unstack_softmax_one]
      exact Softmax.sum_vec (Tensor.unstack scores i)

/-- Coordinate formula for the weighted softmax Jacobian, without a normalization assumption. -/
theorem Softmax.getScalar_backwardFromWeights {n : Nat}
    (weights gradient : Tensor ℝ [n]) (i : Fin n) :
    (backwardFromWeights weights gradient).getScalar i =
      weights.getScalar i *
        (gradient.getScalar i - ∑ j, weights.getScalar j * gradient.getScalar j) := by
  cases n with
  | zero => exact Fin.elim0 i
  | succ n =>
      simp only [backwardFromWeights, Bool.false_eq_true, ↓reduceIte,
        Tensor.getScalar_mul, sub_def, Tensor.getScalar_zipWith,
        Tensor.getScalar_full, Tensor.sum_vec]
      congr 2
      exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- For normalized weights or an all-zero masked row, the weighted softmax VJP, centered or not,
equals the uncentered VJP of the upstream gradient shifted by any constant `anchor`.

Because the anchor is arbitrary, it may come from the rounded execution and need not sit where
the exact weights peak. Numerical bounds can then charge the centering subtraction that actually
ran, without assuming that both executions pick the same coordinate.
-/
theorem Softmax.backwardFromWeights_shift {n : Nat}
    (weights gradient : Tensor ℝ [n]) (anchor : ℝ)
    (hmass : (∑ j, weights.getScalar j) = 1 ∨ ∀ j, weights.getScalar j = 0)
    (center : Bool := false) :
    backwardFromWeights weights gradient (center := center) =
      backwardFromWeights weights
        (gradient - Tensor.full _ anchor) := by
  cases n with
  | zero =>
      apply Tensor.ext_vector
      intro i
      exact Fin.elim0 i
  | succ n =>
      let formula (a : ℝ) := weights *
        ((gradient - Tensor.full _ a) -
          (Tensor.full _ (Tensor.sum
            ((gradient - Tensor.full _ a) * weights))))
      have hformula (a : ℝ) : formula a =
          backwardFromWeights weights gradient := by
        apply Tensor.ext_vector
        intro i
        rw [getScalar_backwardFromWeights]
        simp only [formula, Tensor.getScalar_mul, sub_def,
          Tensor.getScalar_zipWith, Tensor.getScalar_full,
          Tensor.sum_vec]
        rcases hmass with hmass | hzero
        · simp_rw [sub_mul]
          rw [Finset.sum_sub_distrib, ← Finset.mul_sum, hmass, mul_one]
          have hcomm : (∑ j, gradient.getScalar j * weights.getScalar j) =
              ∑ j, weights.getScalar j * gradient.getScalar j :=
            Finset.sum_congr rfl fun j _ => mul_comm _ _
          rw [hcomm]
          ring
        · rw [hzero i]
          simp
      have hright :
          backwardFromWeights weights
            (gradient - Tensor.full _ anchor) = formula anchor := by
        rfl
      rw [hright, hformula anchor]
      cases center with
      | false => rfl
      | true =>
          let pivot : Fin (Nat.succ n) := Softmax.pivot weights
          exact hformula (gradient.getScalar pivot)

end TorchLean
