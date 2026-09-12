/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.SoftmaxSpec
public import NN.Spec.Layers.Attention

/-!
# The derivative of Boolean-masked softmax

The mask is fixed while the scores vary. A row with an allowed key is a normalized sum of
exponentials over those keys; a row without one is the constant zero function. These are separate
cases of the same derivative formula, because a blocked key has exactly zero weight.

The implementation subtracts a maximum before exponentiating. We first cancel that common factor
over the reals, then differentiate the resulting quotient. In particular, ties for the maximum do
not require a hypothesis: the maximum is an implementation detail of the stable evaluation, and
the function being differentiated is smooth even where that maximum changes.
-/

@[expose] public section

namespace Proofs.Autograd.HardMaskedSoftmax

open Spec TorchLean
open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- An allowed logit contributes its exponential; a blocked logit contributes zero. -/
def numerator {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) (i : Fin n) : ℝ :=
  if mask.getScalar i then Real.exp (x i) else 0

/-- The sum is positive exactly when the fixed mask has an allowed key. -/
def denominator {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) : ℝ :=
  ∑ i, numerator mask x i

/-- The real formula for the weights, including the zero row when every key is blocked. -/
def weights {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) : Vec n :=
  softmaxVecOfFun fun i => numerator mask x i / denominator mask x

/-- The maximum scan returns `none` precisely when it has never encountered an allowed entry.

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
theorem getScalar_hardMaskedSoftmaxVecSpec {n : Nat}
    (scores : Tensor ℝ [n]) (mask : Tensor Bool [n]) (i : Fin n) :
    (Spec.hardMaskedSoftmaxVecSpec scores mask).getScalar i =
      numerator mask (getScalarE scores) i / denominator mask (getScalarE scores) := by
  classical
  have hnone :
      Spec.hardMaskedMax? scores mask = none ↔ ∀ j, mask.getScalar j = false := by
    unfold Spec.hardMaskedMax?
    calc
      _ ↔ (none : Option ℝ) = none ∧
          ∀ j ∈ List.finRange n, mask.getScalar j = false := by
        apply scan_eq_none
        intro best j
        cases best <;> cases mask.getScalar j <;> simp
      _ ↔ _ := by simp
  unfold Spec.hardMaskedSoftmaxVecSpec
  split
  · rename_i h
    have hall := hnone.mp h
    simp [numerator, hall, Spec.replicate]
  · rename_i shift h
    have hterm (j : Fin n) :
        (if mask.getScalar j then Real.exp (scores.getScalar j - shift) else 0) =
          numerator mask (getScalarE scores) j * Real.exp (-shift) := by
      cases hj : mask.getScalar j <;>
        simp [numerator, hj, Real.exp_sub, Real.exp_neg, div_eq_mul_inv]
    change (divSpec _ _).getScalar i = _
    rw [divSpec, getScalar_map2Spec]
    simp only [getScalar_map2Spec, Spec.sum_spec_vec, Spec.getScalar_replicate,
      Tensor.item_scalar, Proofs.mathfunc_exp_eq_rexp]
    simp_rw [hterm]
    rw [← Finset.sum_mul]
    exact mul_div_mul_right _ _ (Real.exp_ne_zero (-shift))

/-- The implementation and the smooth real presentation have the same vector coordinates. -/
theorem spec_eq_weights {n : Nat} (scores : Tensor ℝ [n]) (mask : Tensor Bool [n]) :
    getScalarE (Spec.hardMaskedSoftmaxVecSpec scores mask) =
      weights mask (getScalarE scores) := by
  ext i
  exact getScalar_hardMaskedSoftmaxVecSpec scores mask i

/-- The row Jacobian is `diag(w) - w wᵀ`. This definition accepts the zero row as well. -/
def derivative {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) :
    Vec n →L[ℝ] Vec n :=
  (euclideanEquiv n).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi fun i =>
      weights mask x i • (evalCLM i - dotCLM (weights mask x)))

/-- Reading one coordinate exposes the weighted centering performed by the spec JVP and VJP. -/
@[simp] theorem derivative_apply {n : Nat} (mask : Tensor Bool [n]) (x dx : Vec n)
    (i : Fin n) :
    derivative mask x dx i =
      weights mask x i * (dx i - ∑ j, weights mask x j * dx j) := by
  simp [derivative, euclideanEquiv, smul_eq_mul]

/-- A nonempty allowed set makes the exponential denominator strictly positive. -/
theorem denominator_pos {n : Nat} (mask : Tensor Bool [n]) (x : Vec n)
    (h : ∃ i, mask.getScalar i = true) : 0 < denominator mask x := by
  obtain ⟨i, hi⟩ := h
  apply Finset.sum_pos'
  · intro j _
    cases hj : mask.getScalar j <;> simp [numerator, hj, (Real.exp_pos _).le]
  · exact ⟨i, Finset.mem_univ i, by simpa [numerator, hi] using Real.exp_pos (x i)⟩

/-- Differentiating one numerator never differentiates the mask. -/
private theorem hasFDerivAt_numerator {n : Nat} (mask : Tensor Bool [n]) (x : Vec n)
    (i : Fin n) :
    HasFDerivAt (fun y => numerator mask y i) (numerator mask x i • evalCLM i) x := by
  cases hi : mask.getScalar i
  · simpa [numerator, hi] using (hasFDerivAt_const (𝕜 := ℝ) (0 : ℝ) x)
  · simpa [numerator, hi] using ((evalCLM i).hasFDerivAt (x := x)).exp

/-- Hard-masked softmax is differentiable for every score vector and every fixed Boolean mask.

An all-false row is handled before applying the quotient rule. Consequently the proof neither
divides by its zero denominator nor invents a derivative at a masked infinity. -/
theorem hasFDerivAt_weights {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) :
    HasFDerivAt (weights mask) (derivative mask x) x := by
  classical
  by_cases hallowed : ∃ i, mask.getScalar i = true
  · have hdenom : denominator mask x ≠ 0 := ne_of_gt (denominator_pos mask x hallowed)
    let sumDerivative : Vec n →L[ℝ] ℝ :=
      ∑ j, numerator mask x j • evalCLM j
    have hsum : HasFDerivAt (denominator mask) sumDerivative x := by
      change HasFDerivAt (fun y : Vec n => ∑ j, numerator mask y j)
        (∑ j, numerator mask x j • evalCLM j) x
      exact HasFDerivAt.fun_sum (u := (Finset.univ : Finset (Fin n)))
        (fun j _ => hasFDerivAt_numerator mask x j)
    have hinv := (hasFDerivAt_inv (𝕜 := ℝ) hdenom).comp x hsum
    have hcoord (i : Fin n) :
        HasFDerivAt (fun y => weights mask y i)
          (weights mask x i • (evalCLM i - dotCLM (weights mask x))) x := by
      have hmul := (hasFDerivAt_numerator mask x i).mul hinv
      have hmap :
          numerator mask x i •
              ((ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ)
                (-(denominator mask x ^ 2)⁻¹)).comp sumDerivative) +
            (denominator mask x)⁻¹ • (numerator mask x i • evalCLM i) =
          weights mask x i • (evalCLM i - dotCLM (weights mask x)) := by
        ext dx
        simp only [_root_.add_apply, smul_apply,
          ContinuousLinearMap.comp_apply, ContinuousLinearMap.smulRight_apply,
          one_apply_eq_self, sub_apply, evalCLM_apply,
          dotCLM_apply, smul_eq_mul, weights, softmaxVecOfFun_apply, sumDerivative,
          sum_apply, div_mul_eq_mul_div]
        rw [← Finset.sum_div]
        field_simp [hdenom]; ring
      have hfun :
          (fun y => weights mask y i) =
            (fun y => numerator mask y i) * ((fun y => y⁻¹) ∘ denominator mask) := by
        funext y
        simp only [weights, softmaxVecOfFun_apply, div_eq_mul_inv,
          Pi.mul_apply, Function.comp_apply]
      exact (hmul.congr_of_eventuallyEq hfun.eventuallyEq).congr_fderiv hmap
    have hpi := (hasFDerivAt_pi (𝕜 := ℝ)
      (φ := fun i y => weights mask y i)
      (φ' := fun i => weights mask x i • (evalCLM i - dotCLM (weights mask x)))
      (x := x)).2 hcoord
    have hfun :
        weights mask = (euclideanEquiv n).symm ∘ (fun y i => weights mask y i) := by
      funext y
      ext i
      rfl
    have hcomp := (euclideanEquiv n).symm.toContinuousLinearMap.hasFDerivAt.comp x hpi
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

/-- This derivative theorem is about the actual stable tensor implementation. -/
theorem hasFDerivAt_spec {n : Nat} (mask : Tensor Bool [n]) (x : Vec n) :
    HasFDerivAt
      (fun y => getScalarE (Spec.hardMaskedSoftmaxVecSpec (ofFnE y) mask))
      (derivative mask x) x := by
  simpa only [spec_eq_weights, getScalarE_ofFnE] using hasFDerivAt_weights mask x

/-- The implementation's weighted-centering helper evaluates the derivative itself. -/
theorem backward_eq_derivative {n : Nat} (scores gradient : Tensor ℝ [n])
    (mask : Tensor Bool [n]) :
    getScalarE (Spec.softmaxBackwardFromWeightsSpec
      (Spec.hardMaskedSoftmaxVecSpec scores mask) gradient) =
        derivative mask (getScalarE scores) (getScalarE gradient) := by
  have hw (i : Fin n) :
      (Spec.hardMaskedSoftmaxVecSpec scores mask).getScalar i =
        weights mask (getScalarE scores) i :=
    getScalar_hardMaskedSoftmaxVecSpec scores mask i
  ext i
  simp only [getScalarE_ofLp, Spec.softmaxBackwardFromWeightsSpec,
    Spec.getScalar_mul_spec, subSpec, getScalar_map2Spec, Spec.sum_spec_vec,
    Spec.getScalar_replicate, Tensor.item_scalar, hw, derivative_apply]
  congr 1
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  rw [mul_comm]

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

/-- The concrete backward helper is the adjoint of the derivative of the concrete forward.

This is stronger than an algebraic JVP/VJP pairing: `hasFDerivAt_spec` supplies the analytic
derivative, and the equality below identifies the helper called by attention with its adjoint. -/
theorem backward_eq_adjoint_fderiv {n : Nat} (scores gradient : Tensor ℝ [n])
    (mask : Tensor Bool [n]) :
    getScalarE (Spec.softmaxBackwardFromWeightsSpec
      (Spec.hardMaskedSoftmaxVecSpec scores mask) gradient) =
      (fderiv ℝ
        (fun x : Vec n => getScalarE (Spec.hardMaskedSoftmaxVecSpec (ofFnE x) mask))
        (getScalarE scores)).adjoint (getScalarE gradient) := by
  rw [(hasFDerivAt_spec mask (getScalarE scores)).fderiv, backward_eq_derivative]
  apply ext_inner_left ℝ
  intro dx
  rw [ContinuousLinearMap.adjoint_inner_right]
  exact (inner_derivative mask (getScalarE scores) dx (getScalarE gradient)).symm

end

end Proofs.Autograd.HardMaskedSoftmax
