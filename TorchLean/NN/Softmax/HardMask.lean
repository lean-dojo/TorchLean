/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Algebra.BigOperators.Field
public import TorchLean.NN.Softmax.Correctness
public import TorchLean.Tensor.Layers.Attention

/-!
# The derivative of Boolean-masked softmax

The mask is fixed while the scores vary. A row with at least one allowed key normalizes the
exponentials of its allowed scores, and a row with every key blocked is the constant zero function.
One derivative formula, `diag(w) - w wᵀ`, covers both cases because a blocked key has weight zero.

The implementation subtracts a maximum before exponentiating. Over the reals that common factor
cancels, and the proofs differentiate the resulting quotient. Ties for the maximum therefore need
no hypothesis: the maximum belongs to the stable evaluation, and the function being differentiated
is smooth even where the maximizing key changes.
-/

@[expose] public section

namespace TorchLean.HardMaskedSoftmax

open TorchLean.Tensor
open scoped BigOperators

noncomputable section

open Vec (inner_eq_sum_mul)

/-- Numerator of coordinate `i`: `exp (x i)` for an allowed key and `0` for a blocked one. -/
def numerator {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) (i : Fin n) : ℝ :=
  if mask.getScalar i then Real.exp (x i) else 0

/-- Sum of all numerators, positive if and only if the mask allows at least one key. -/
def denominator {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) : ℝ :=
  ∑ i, numerator mask x i

/-- Real masked-softmax weights `numerator / denominator`. A fully blocked row computes `0 / 0`,
which is `0` in Lean, so it gives the zero row. -/
def weights {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) : Vec n :=
  WithLp.toLp 2 fun i => numerator mask x i / denominator mask x

/-- If a step returns `none` exactly when its input is `none` and the index is blocked, a fold over
`indices` returns `none` if and only if it starts at `none` and every index is blocked.

Only the presence of a maximum matters here. Its value and the order in which allowed entries
are visited disappear when the common exponential factor is cancelled below. -/
private theorem scan_eq_none {n : Nat} (mask : Tensor Bool [n])
    (step : Option ℝ → Fin n → Option ℝ)
    (hstep : ∀ best i, step best i = none ↔ best = none ∧ mask.getScalar i = false)
    (indices : List (Fin n)) (initial : Option ℝ) :
    indices.foldl step initial = none ↔
      initial = none ∧ ∀ i ∈ indices, mask.getScalar i = false := by
  induction indices generalizing initial with
  | nil => simp
  | cons i indices ih =>
      simp [List.foldl_cons, ih, hstep, and_assoc]

/-- Removing the stable shift gives the ordinary exponential quotient on the allowed keys. -/
theorem _root_.TorchLean.Softmax.getScalar_hardMaskedVec {n : Nat}
    (scores : Tensor ℝ [n]) (mask : Tensor Bool [n]) (i : Fin n) :
    (Softmax.hardMaskedVec scores mask).getScalar i =
      numerator mask (getScalarE scores) i / denominator mask (getScalarE scores) := by
  classical
  have hnone :
      Softmax.hardMaskedMax? scores mask = none ↔ ∀ j, mask.getScalar j = false := by
    unfold Softmax.hardMaskedMax?
    calc
      _ ↔ (none : Option ℝ) = none ∧
          ∀ j ∈ List.finRange n, mask.getScalar j = false := by
        apply scan_eq_none
        intro best j
        cases best <;> cases mask.getScalar j <;> simp
      _ ↔ _ := by simp
  unfold Softmax.hardMaskedVec
  split
  · rename_i h
    have hall := hnone.mp h
    simp [numerator, hall]
  · rename_i shift h
    have hterm (j : Fin n) :
        (if mask.getScalar j then Real.exp (scores.getScalar j - shift) else 0) =
          numerator mask (getScalarE scores) j * Real.exp (-shift) := by
      cases hj : mask.getScalar j <;>
        simp [numerator, hj, Real.exp_sub, Real.exp_neg, div_eq_mul_inv]
    change (_ / _ : Tensor ℝ [n]).getScalar i = _
    rw [div_def, getScalar_zipWith]
    simp only [getScalar_zipWith, Tensor.sum_vec, Tensor.getScalar_full,
      MathFunctions.exp_real]
    simp_rw [hterm]
    rw [← Finset.sum_mul]
    exact mul_div_mul_right _ _ (Real.exp_ne_zero (-shift))

/-- As Euclidean vectors, the stable spec and the smooth real formula `weights` are equal. -/
theorem _root_.TorchLean.Softmax.getScalarE_hardMaskedVec {n : Nat} (scores : Tensor ℝ [n])
    (mask : Tensor Bool [n]) :
    getScalarE (Softmax.hardMaskedVec scores mask) =
      weights mask (getScalarE scores) := by
  ext i
  exact Softmax.getScalar_hardMaskedVec scores mask i

/-- Row Jacobian `diag(w) - w wᵀ` with `w = weights mask x`, zero on a fully blocked row. -/
def derivative {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) :
    Vec n →L[ℝ] Vec n :=
  (EuclideanSpace.equiv (Fin n) ℝ).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi fun i =>
      weights mask x i • (EuclideanSpace.proj (𝕜 := ℝ) i - innerSL ℝ (weights mask x)))

/-- Coordinate `i` of the derivative is `wᵢ * (dxᵢ - ∑ⱼ wⱼ * dxⱼ)`, the weighted centering that
the spec JVP and VJP perform. -/
@[simp] theorem derivative_apply {n : Nat} (mask : Tensor Bool [n]) (x dx : Vec n)
    (i : Fin n) :
    derivative mask x dx i =
      weights mask x i * (dx i - ∑ j, weights mask x j * dx j) := by
  simp [derivative, smul_eq_mul, inner_eq_sum_mul]

/-- If the mask allows at least one key, the denominator is strictly positive. -/
theorem denominator_pos {n : Nat} (mask : Tensor Bool [n]) (x : Vec n)
    (h : ∃ i, mask.getScalar i = true) : 0 < denominator mask x := by
  obtain ⟨i, hi⟩ := h
  apply Finset.sum_pos'
  · intro j _
    cases hj : mask.getScalar j <;> simp [numerator, hj, (Real.exp_pos _).le]
  · exact ⟨i, Finset.mem_univ i, by simpa [numerator, hi] using Real.exp_pos (x i)⟩

/-- Numerator `i` has derivative `numerator mask x i • EuclideanSpace.proj i` because the mask is
fixed. -/
private theorem hasFDerivAt_numerator {n : Nat} (mask : Tensor Bool [n]) (x : Vec n)
    (i : Fin n) :
    HasFDerivAt (fun y => numerator mask y i)
      (numerator mask x i • EuclideanSpace.proj (𝕜 := ℝ) i) x := by
  cases hi : mask.getScalar i
  · simpa [numerator, hi] using (hasFDerivAt_const (𝕜 := ℝ) (0 : ℝ) x)
  · simpa [numerator, hi] using Vec.hasFDerivAt_exp_apply x i

/-- For a fixed Boolean mask, `weights mask` has Fréchet derivative `derivative mask x` at every
score vector `x`.

A fully blocked row is the constant zero function and is handled before the quotient rule, so the
proof never divides by a zero denominator or differentiates at a masked `-∞`. -/
theorem hasFDerivAt_weights {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) :
    HasFDerivAt (weights mask) (derivative mask x) x := by
  classical
  by_cases hallowed : ∃ i, mask.getScalar i = true
  · have hdenom : denominator mask x ≠ 0 := ne_of_gt (denominator_pos mask x hallowed)
    let sumDerivative : Vec n →L[ℝ] ℝ :=
      ∑ j, numerator mask x j • EuclideanSpace.proj (𝕜 := ℝ) j
    have hsum : HasFDerivAt (denominator mask) sumDerivative x := by
      change HasFDerivAt (fun y : Vec n => ∑ j, numerator mask y j)
        (∑ j, numerator mask x j • EuclideanSpace.proj (𝕜 := ℝ) j) x
      exact HasFDerivAt.fun_sum (u := (Finset.univ : Finset (Fin n)))
        (fun j _ => hasFDerivAt_numerator mask x j)
    have hinv := (hasFDerivAt_inv (𝕜 := ℝ) hdenom).comp x hsum
    have hcoord (i : Fin n) :
        HasFDerivAt (fun y => weights mask y i)
          (weights mask x i • (EuclideanSpace.proj (𝕜 := ℝ) i - innerSL ℝ (weights mask x)))
          x := by
      have hmul := (hasFDerivAt_numerator mask x i).mul hinv
      have hmap :
          numerator mask x i •
              ((ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ)
                (-(denominator mask x ^ 2)⁻¹)).comp sumDerivative) +
            (denominator mask x)⁻¹ • (numerator mask x i • EuclideanSpace.proj (𝕜 := ℝ) i) =
          weights mask x i • (EuclideanSpace.proj (𝕜 := ℝ) i - innerSL ℝ (weights mask x)) := by
        ext dx
        simp only [_root_.add_apply, smul_apply,
          ContinuousLinearMap.comp_apply, ContinuousLinearMap.smulRight_apply,
          one_apply_eq_self, sub_apply, EuclideanSpace.coe_proj, innerSL_apply_apply,
          inner_eq_sum_mul, smul_eq_mul, weights, PiLp.toLp_apply, sumDerivative,
          sum_apply, div_mul_eq_mul_div]
        rw [← Finset.sum_div]
        field_simp [hdenom]; ring
      have hfun :
          (fun y => weights mask y i) =
            (fun y => numerator mask y i) * ((fun y => y⁻¹) ∘ denominator mask) := by
        funext y
        simp only [weights, PiLp.toLp_apply, div_eq_mul_inv,
          Pi.mul_apply, Function.comp_apply]
      exact (hmul.congr_of_eventuallyEq hfun.eventuallyEq).congr_fderiv hmap
    have hpi := (hasFDerivAt_pi (𝕜 := ℝ)
      (φ := fun i y => weights mask y i)
      (φ' := fun i =>
        weights mask x i • (EuclideanSpace.proj (𝕜 := ℝ) i - innerSL ℝ (weights mask x)))
      (x := x)).2 hcoord
    have hfun :
        weights mask = (EuclideanSpace.equiv (Fin n) ℝ).symm ∘ (fun y i => weights mask y i) := by
      funext y
      ext i
      rfl
    have hcomp :=
      (EuclideanSpace.equiv (Fin n) ℝ).symm.toContinuousLinearMap.hasFDerivAt.comp x hpi
    exact hcomp.congr_of_eventuallyEq hfun.eventuallyEq
  · have hall (i : Fin n) : mask.getScalar i = false := by
      cases hi : mask.getScalar i
      · rfl
      · exact (hallowed ⟨i, hi⟩).elim
    have hw : weights mask = fun _ => (0 : Vec n) := by
      funext y
      ext i
      simp [weights, numerator, hall]
    have hd : derivative mask x = 0 := by
      ext dx i
      simp [derivative_apply, hw]
    rw [hw, hd]
    exact hasFDerivAt_const (0 : Vec n) x

/-- Read as a map on Euclidean vectors, the stable implementation `Softmax.hardMaskedVec`
has derivative `derivative mask x` at every `x`. -/
theorem _root_.TorchLean.Softmax.hasFDerivAt_hardMaskedVec {n : Nat} (mask : Tensor Bool [n])
    (x : Vec n) :
    HasFDerivAt
      (fun y => getScalarE (Softmax.hardMaskedVec (ofFnE y) mask))
      (derivative mask x) x := by
  simpa only [Softmax.getScalarE_hardMaskedVec, getScalarE_ofFnE] using hasFDerivAt_weights mask x

/-- For either value of `center`, the backward helper `Softmax.backwardFromWeights` on the
hard-masked weights applies `derivative mask x` to the upstream gradient. -/
theorem backward_eq_derivative {n : Nat} (scores gradient : Tensor ℝ [n])
    (mask : Tensor Bool [n]) (center : Bool := false) :
    getScalarE (Softmax.backwardFromWeights
      (Softmax.hardMaskedVec scores mask) gradient (center := center)) =
        derivative mask (getScalarE scores) (getScalarE gradient) := by
  have hw (i : Fin n) :
      (Softmax.hardMaskedVec scores mask).getScalar i =
        weights mask (getScalarE scores) i :=
    Softmax.getScalar_hardMaskedVec scores mask i
  have hmass : (∑ j, (Softmax.hardMaskedVec scores mask).getScalar j) = 1 ∨
      ∀ j, (Softmax.hardMaskedVec scores mask).getScalar j = 0 := by
    by_cases hallowed : ∃ j, mask.getScalar j = true
    · left
      simp_rw [hw]
      change (∑ j, numerator mask (getScalarE scores) j /
        denominator mask (getScalarE scores)) = 1
      rw [← Finset.sum_div]
      exact div_self (ne_of_gt (denominator_pos mask (getScalarE scores) hallowed))
    · right
      have hall (j : Fin n) : mask.getScalar j = false := by
        cases hj : mask.getScalar j
        · rfl
        · exact (hallowed ⟨j, hj⟩).elim
      intro j
      simp [hw, weights, numerator, hall]
  have hzero : gradient - Tensor.full _ (0 : ℝ) = gradient := by
    apply Tensor.ext_vector
    intro i
    simp [sub_def]
  rw [Softmax.backwardFromWeights_shift _ gradient 0 hmass center, hzero]
  ext i
  simp only [getScalarE_ofLp, Softmax.getScalar_backwardFromWeights,
    hw, derivative_apply]

/-- The row derivative is self-adjoint, including rows whose weights are all zero. -/
theorem inner_derivative {n : Nat} (mask : Tensor Bool [n]) (x dx gradient : Vec n) :
    inner ℝ (derivative mask x dx) gradient =
      inner ℝ dx (derivative mask x gradient) := by
  rw [inner_eq_sum_mul, inner_eq_sum_mul]
  simp_rw [derivative_apply]
  let w := weights mask x
  have hleft :
      (∑ i, w i * (dx i - ∑ j, w j * dx j) * gradient i) =
        (∑ i, w i * dx i * gradient i) -
          (∑ j, w j * dx j) * ∑ i, w i * gradient i := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  have hright :
      (∑ i, dx i * (w i * (gradient i - ∑ j, w j * gradient j))) =
        (∑ i, w i * dx i * gradient i) -
          (∑ j, w j * gradient j) * ∑ i, w i * dx i := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hleft, hright]
  ring

/-- The backward helper that attention calls is the adjoint of the Fréchet derivative of the stable
forward spec, applied to the upstream gradient.

An algebraic JVP/VJP pairing would only relate two formulas to each other. Here
`Softmax.hasFDerivAt_hardMaskedVec` supplies the analytic derivative, so the helper is tied to the
forward function itself. -/
theorem backward_eq_fderiv {n : Nat} (scores gradient : Tensor ℝ [n])
    (mask : Tensor Bool [n]) (center : Bool := false) :
    getScalarE (Softmax.backwardFromWeights
      (Softmax.hardMaskedVec scores mask) gradient (center := center)) =
      (fderiv ℝ
        (fun x : Vec n => getScalarE (Softmax.hardMaskedVec (ofFnE x) mask))
        (getScalarE scores)).adjoint (getScalarE gradient) := by
  rw [(Softmax.hasFDerivAt_hardMaskedVec mask (getScalarE scores)).fderiv,
    backward_eq_derivative (center := center)]
  apply ext_inner_left ℝ
  intro dx
  rw [ContinuousLinearMap.adjoint_inner_right]
  exact (inner_derivative mask (getScalarE scores) dx (getScalarE gradient)).symm

end

end TorchLean.HardMaskedSoftmax
