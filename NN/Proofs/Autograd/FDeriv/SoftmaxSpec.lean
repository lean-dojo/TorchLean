/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Softmax
public import NN.Proofs.Autograd.FDeriv.LogSoftmax
public import NN.Spec.Layers.Loss

/-!
# Softmax and log-softmax: spec definitions versus analytic presentation

`NN.Proofs.Autograd.FDeriv.Softmax` and `NN.Proofs.Autograd.FDeriv.LogSoftmax` prove
Fréchet-derivative facts about `softmaxVec` and `logSoftmaxVec`, which are stated directly on
Euclidean vectors. The specification layer instead defines `Activation.softmaxVecSpec` and
`Activation.logSoftmaxVecSpec` on tensors, using the numerically stable max-shifted form.

This file closes that gap over `ℝ`:

- `getScalarE_softmaxVecSpec` and `getScalarE_logSoftmaxVecSpec` identify the spec kernels with
  the analytic definitions after vectorization;
- `hasFDerivAt_softmaxSpec_vec` and `hasFDerivAt_logSoftmaxSpec_vec` transfer differentiability
  to `Activation.softmaxSpec 0` and `Activation.logSoftmaxSpec 0`;
- `softmaxFDerivCorrect` and `logSoftmaxFDerivCorrect` package `Spec.softmaxOp 0` and
  `Spec.logSoftmaxOp 0` as `OpSpecFDerivCorrect`, so their backward rules are proved to be the
  adjoint of the true derivative (`softmaxBackwardSpec_eq_vjp`, `logSoftmaxBackwardSpec_eq_vjp`).
- `hasFDerivAt_distillSpec_vec` identifies the cancellation-resistant student pullback with
  the derivative of the teacher-logit loss, with the teacher held fixed.
- `hasFDerivAt_crossEntropyLogitsSpec_vec` and `hasFDerivAt_crossEntropyTargets_vec` prove
  the logits and target-weight derivatives of the weighted loss. Targets need not sum to one.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec _root_.TorchLean
open _root_.TorchLean _root_.TorchLean.Tensor

open scoped BigOperators
open scoped _root_.Autograd

noncomputable section

/-! ## Softmax -/

/-- The spec softmax kernel is the analytic `softmaxVec` after vectorization. -/
theorem getScalarE_softmaxVecSpec {n : Nat} (t : Tensor ℝ [n]) :
    getScalarE (Activation.softmaxVecSpec (α := ℝ) (n := n) t) =
      softmaxVec (n := n) (getScalarE t) := by
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      ext i
      rw [getScalarE_ofLp, Proofs.getScalar_softmaxVecSpec_eq_exp_div]
      simp [softmaxVec, sumExp]

/-- `softmaxVecSpec` on a vector built from Euclidean coordinates. -/
theorem softmaxVecSpec_ofFnE {n : Nat} (x : Vec n) :
    Activation.softmaxVecSpec (α := ℝ) (n := n) (ofFnE x) = ofFnE (softmaxVec (n := n) x) := by
  calc
    Activation.softmaxVecSpec (α := ℝ) (n := n) (ofFnE x)
        = ofFnE (getScalarE (Activation.softmaxVecSpec (α := ℝ) (n := n) (ofFnE x))) := by
          rw [ofFnE_getScalarE]
    _ = ofFnE (softmaxVec (n := n) x) := by
          rw [getScalarE_softmaxVecSpec, getScalarE_ofFnE]

/-- The vectorized forward map of `Spec.softmaxOp 0` on vectors is `softmaxVec`. -/
theorem softmaxSpec_zero_forwardVec {n : Nat} :
    (fun xV : Vec n => getScalarE (Activation.softmaxSpec (α := ℝ) (s := [n]) 0 (ofFnE xV))) =
      softmaxVec (n := n) := by
  funext xV
  rw [Proofs.softmaxSpec_zero_vec, getScalarE_softmaxVecSpec, getScalarE_ofFnE]

/-- Axis-`0` spec softmax on vectors is Fréchet-differentiable with derivative
`softmaxDerivCLM`. -/
theorem hasFDerivAt_softmaxSpec_vec {n : Nat} (xV : Vec n) :
    HasFDerivAt
      (fun xV : Vec n => getScalarE (Activation.softmaxSpec (α := ℝ) (s := [n]) 0 (ofFnE xV)))
      (softmaxDerivCLM (n := n) xV) xV := by
  rw [softmaxSpec_zero_forwardVec]
  exact hasFDerivAt_softmaxVec xV

/-- The spec softmax VJP is the analytic `softmaxJvp` (the softmax Jacobian is symmetric). -/
theorem getScalarE_softmaxBackwardSpec {n : Nat} (x dY : Tensor ℝ [n]) :
    getScalarE (Activation.softmaxBackwardSpec (α := ℝ) (s := [n]) 0 x dY) =
      softmaxJvp (n := n) (getScalarE x) (getScalarE dY) := by
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      have hy : ∀ i, TorchLean.Tensor.getScalar (Activation.softmaxVecSpec x) i =
          softmaxVec (n := Nat.succ n) (getScalarE x) i := by
        intro i
        rw [← getScalarE_softmaxVecSpec]
        rfl
      rw [Proofs.softmaxBackwardSpec_zero_vec]
      ext i
      rw [getScalarE_ofLp, Proofs.getScalar_softmaxInnermostBackwardSpec]
      simp [hy, softmaxJvp, dotCLM_apply, getScalarE_ofLp]

/-- `Spec.softmaxOp 0` on vectors with its analytic JVP and the VJP/JVP adjointness law. -/
def softmaxCorrect (n : Nat) : OpSpecCorrect (.dim n .scalar) (.dim n .scalar) where
  op := Spec.softmaxOp (α := ℝ) (s := [n]) 0
  jvp := fun x dx => ofFnE (softmaxJvp (n := n) (getScalarE x) (getScalarE dx))
  correct := by
    intro x dx δ
    change Spec.dot (ofFnE (softmaxJvp (n := n) (getScalarE x) (getScalarE dx))) δ =
      Spec.dot dx (Activation.softmaxBackwardSpec (α := ℝ) (s := [n]) 0 x δ)
    rw [dot_eq_inner_vec, dot_eq_inner_vec, getScalarE_ofFnE, getScalarE_softmaxBackwardSpec]
    exact inner_softmaxJvp_comm (n := n) (getScalarE x) (getScalarE dx) (getScalarE δ)

/-- `Spec.softmaxOp 0` on vectors is analytically correct: its JVP is the Fréchet derivative. -/
def softmaxFDerivCorrect (n : Nat) : OpSpecFDerivCorrect n n where
  correct := softmaxCorrect n
  deriv := softmaxDerivCLM (n := n)
  hasFDerivAt := fun xV => hasFDerivAt_softmaxSpec_vec xV
  jvp_eq := by
    intro xV dxV
    change getScalarE (ofFnE (softmaxJvp (n := n) (getScalarE (ofFnE xV))
      (getScalarE (ofFnE dxV)))) = _
    rw [getScalarE_ofFnE, getScalarE_ofFnE, getScalarE_ofFnE, softmaxJvp_eq_deriv]

/-- The spec softmax backward is the vector-Jacobian product of the spec softmax forward. -/
theorem softmaxBackwardSpec_eq_vjp {n : Nat} (x δ : Tensor ℝ [n]) :
    getScalarE (Activation.softmaxBackwardSpec (α := ℝ) (s := [n]) 0 x δ) =
      VJP[fun xV : Vec n => getScalarE (Activation.softmaxSpec (α := ℝ) (s := [n]) 0 (ofFnE xV)),
        getScalarE x] (getScalarE δ) :=
  (softmaxFDerivCorrect n).backward_eq_adjoint_fderiv x δ

/-! ## Log-softmax -/

/-- The spec log-softmax kernel is the analytic `logSoftmaxVec` after vectorization. -/
theorem getScalarE_logSoftmaxVecSpec {n : Nat} (t : Tensor ℝ [n]) :
    getScalarE (Activation.logSoftmaxVecSpec (α := ℝ) (n := n) t) =
      logSoftmaxVec (n := n) (getScalarE t) := by
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      ext i
      rw [getScalarE_ofLp, Proofs.getScalar_logSoftmaxVecSpec_eq_sub_log]
      simp [logSoftmaxVec, sumExp]

/-- The vectorized forward map of `Spec.logSoftmaxOp 0` on vectors is `logSoftmaxVec`. -/
theorem logSoftmaxSpec_zero_forwardVec {n : Nat} :
    (fun xV : Vec n =>
      getScalarE (Activation.logSoftmaxSpec (α := ℝ) (s := [n]) 0 (ofFnE xV))) =
      logSoftmaxVec (n := n) := by
  funext xV
  rw [Proofs.logSoftmaxSpec_zero_vec, getScalarE_logSoftmaxVecSpec, getScalarE_ofFnE]

/-- Axis-`0` spec log-softmax on vectors is Fréchet-differentiable with derivative
`logSoftmaxDerivCLM`. -/
theorem hasFDerivAt_logSoftmaxSpec_vec {n : Nat} (xV : Vec n) :
    HasFDerivAt
      (fun xV : Vec n =>
        getScalarE (Activation.logSoftmaxSpec (α := ℝ) (s := [n]) 0 (ofFnE xV)))
      (logSoftmaxDerivCLM (n := n) xV) xV := by
  rw [logSoftmaxSpec_zero_forwardVec]
  exact hasFDerivAt_logSoftmaxVec xV

/-- Exponentiating the spec log-softmax output recovers the analytic softmax coordinates. -/
theorem getScalar_expSpec_logSoftmaxVecSpec {n : Nat} (x : Tensor ℝ [Nat.succ n])
    (i : Fin (Nat.succ n)) :
    TorchLean.Tensor.getScalar (expSpec (Activation.logSoftmaxVecSpec x)) i =
      softmaxVec (n := Nat.succ n) (getScalarE x) i := by
  have hsumPos : 0 < ∑ j, Real.exp (TorchLean.Tensor.getScalar x j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  rw [expSpec, TorchLean.Tensor.getScalar_map, Proofs.getScalar_logSoftmaxVecSpec_eq_sub_log,
    mathfunc_exp_eq_rexp, Real.exp_sub, Real.exp_log hsumPos]
  simp [softmaxVec, sumExp]

/-- The typed graph's compensated log-softmax JVP agrees with the analytic derivative. -/
theorem getScalarE_logSoftmaxJvpSpec {n : Nat} (x dx : Tensor ℝ [n]) :
    getScalarE (Activation.logSoftmaxJvpSpec 0 (Activation.logSoftmaxSpec 0 x) dx) =
      logSoftmaxJvp (getScalarE x) (getScalarE dx) := by
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      change getScalarE (Activation.Internal.logSoftmaxInnermostJvpSpec
        (Activation.logSoftmaxVecSpec x) dx) = _
      rw [Proofs.logSoftmaxInnermostJvpSpec_eq]
      ext i
      simp only [getScalarE_ofLp, subSpec, TorchLean.Tensor.getScalar_map2Spec,
        Spec.replicate, TorchLean.Tensor.getScalar_const, TorchLean.Tensor.item_scalar]
      simp only [dotSpec, Spec.sum_spec_vec, Spec.getScalar_mul_spec,
        getScalar_expSpec_logSoftmaxVecSpec]
      simp [logSoftmaxJvp, dotCLM_apply, getScalarE_ofLp]

/-- The spec log-softmax VJP, evaluated on the recomputed forward output, is the analytic
`logSoftmaxVjp`. -/
theorem getScalarE_logSoftmaxBackwardSpec {n : Nat} (x dY : Tensor ℝ [n]) :
    getScalarE (Activation.logSoftmaxBackwardSpec (α := ℝ) (s := [n]) 0
        (Activation.logSoftmaxSpec (α := ℝ) (s := [n]) 0 x) dY) =
      logSoftmaxVjp (n := n) (getScalarE x) (getScalarE dY) := by
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      have hsum : TorchLean.Tensor.sumSpec dY = ∑ j, getScalarE dY j := by
        rw [Spec.sum_spec_vec]
        rfl
      rw [Proofs.logSoftmaxBackwardSpec_zero_vec, Proofs.logSoftmaxSpec_zero_vec]
      change getScalarE (Activation.Internal.logSoftmaxVecBackwardSpec
        (Activation.logSoftmaxVecSpec x) dY) = _
      rw [Proofs.logSoftmaxVecBackwardSpec_eq]
      ext i
      rw [getScalarE_ofLp, show subSpec dY _ = TorchLean.Tensor.map2Spec (· - ·) dY _ by rfl,
        TorchLean.Tensor.getScalar_map2Spec, Spec.getScalar_mul_spec,
        getScalar_expSpec_logSoftmaxVecSpec, hsum]
      simp [logSoftmaxVjp, Spec.replicate]

/-- `Spec.logSoftmaxOp 0` on vectors with its analytic JVP and the VJP/JVP adjointness law. -/
def logSoftmaxCorrect (n : Nat) : OpSpecCorrect (.dim n .scalar) (.dim n .scalar) where
  op := Spec.logSoftmaxOp (α := ℝ) (s := [n]) 0
  jvp := fun x dx => ofFnE (logSoftmaxJvp (n := n) (getScalarE x) (getScalarE dx))
  correct := by
    intro x dx δ
    change Spec.dot (ofFnE (logSoftmaxJvp (n := n) (getScalarE x) (getScalarE dx))) δ =
      Spec.dot dx (Activation.logSoftmaxBackwardSpec (α := ℝ) (s := [n]) 0
        (Activation.logSoftmaxSpec (α := ℝ) (s := [n]) 0 x) δ)
    rw [dot_eq_inner_vec, dot_eq_inner_vec, getScalarE_ofFnE, getScalarE_logSoftmaxBackwardSpec]
    exact inner_logSoftmaxJvp_vjp (n := n) (getScalarE x) (getScalarE dx) (getScalarE δ)

/-- `Spec.logSoftmaxOp 0` on vectors is analytically correct: its JVP is the Fréchet
derivative. -/
def logSoftmaxFDerivCorrect (n : Nat) : OpSpecFDerivCorrect n n where
  correct := logSoftmaxCorrect n
  deriv := logSoftmaxDerivCLM (n := n)
  hasFDerivAt := fun xV => hasFDerivAt_logSoftmaxSpec_vec xV
  jvp_eq := by
    intro xV dxV
    change getScalarE (ofFnE (logSoftmaxJvp (n := n) (getScalarE (ofFnE xV))
      (getScalarE (ofFnE dxV)))) = _
    rw [getScalarE_ofFnE, getScalarE_ofFnE, getScalarE_ofFnE, logSoftmaxJvp_eq_deriv]

/-- The spec log-softmax backward is the vector-Jacobian product of the spec log-softmax
forward. -/
theorem logSoftmaxBackwardSpec_eq_vjp {n : Nat} (x δ : Tensor ℝ [n]) :
    getScalarE (Activation.logSoftmaxBackwardSpec (α := ℝ) (s := [n]) 0
        (Activation.logSoftmaxSpec (α := ℝ) (s := [n]) 0 x) δ) =
      VJP[fun xV : Vec n =>
          getScalarE (Activation.logSoftmaxSpec (α := ℝ) (s := [n]) 0 (ofFnE xV)),
        getScalarE x] (getScalarE δ) :=
  (logSoftmaxFDerivCorrect n).backward_eq_adjoint_fderiv x δ

/-! ## Weighted logits cross-entropy -/

private theorem crossEntropy_vector_terms_eq {n : Nat} (x target : Tensor ℝ [n]) :
    CrossEntropy.vectorTerms x target = mulSpec target (Activation.logSoftmaxVecSpec x) := by
  cases n with
  | zero =>
      apply Tensor.ext_vector
      intro i
      exact i.elim0
  | succ n =>
      apply Tensor.ext_vector
      intro i
      simp only [CrossEntropy.vectorTerms, getScalar_ofFn, sub_self, beq_self_eq_true, ite_true,
        mulSpec, getScalar_map2Spec]

/-- The overflow-aware weighted loss preserves the real cross-entropy formula. This is a real
identity, not bitwise equivalence between the two floating-point evaluation schedules. -/
theorem crossEntropyLogitsSpec_eq_dot {n : Nat} (x target : Tensor ℝ [n]) :
    crossEntropyLogitsSpec 0 x target =
      -(dotCLM (getScalarE target)) (getScalarE (Activation.logSoftmaxSpec 0 x)) := by
  change -sumSpec (CrossEntropy.vectorTerms x target) / ((1 : Nat) : ℝ) = _
  rw [crossEntropy_vector_terms_eq, Nat.cast_one, div_one, Proofs.logSoftmaxSpec_zero_vec]
  simp only [Spec.sum_spec_vec, mulSpec, getScalar_map2Spec, dotCLM_apply, getScalarE_ofLp]

private theorem getScalarE_crossEntropy_gradient {n : Nat} (x : Vec n)
    (target : Tensor ℝ [n]) :
    getScalarE (crossEntropyLogitsDerivSpec 0 (ofFnE x) target) =
      logSoftmaxVjp x (-getScalarE target) := by
  have same := getScalarE_logSoftmaxBackwardSpec (ofFnE x)
    (Tensor.map (fun value => -value) target)
  simp only [crossEntropyLogitsDerivSpec, reduceIte,
    show axisMeanDenom [n] 0 = 1 from rfl, Nat.cast_one, div_one]
  have hscale (t : Tensor ℝ [n]) : scaleSpec t 1 = t := by
    apply Tensor.ext_vector
    intro i
    simp [scaleSpec]
  rw [hscale]
  have hneg : getScalarE (Tensor.map (fun value => -value) target) = -getScalarE target := by
    ext i
    simp [getScalarE_ofLp]
  simpa only [getScalarE_ofFnE, hneg] using same

/-- The shared logits pullback differentiates the actual weighted loss, including unnormalized
and zero target weights. No probability-normalization hypothesis is needed. -/
theorem hasFDerivAt_crossEntropyLogitsSpec_vec {n : Nat} (x : Vec n)
    (target : Tensor ℝ [n]) :
    HasFDerivAt (fun logits : Vec n => crossEntropyLogitsSpec 0 (ofFnE logits) target)
      (dotCLM (getScalarE (crossEntropyLogitsDerivSpec 0 (ofFnE x) target))) x := by
  have h := ((dotCLM (getScalarE target)).hasFDerivAt.comp x
    (hasFDerivAt_logSoftmaxSpec_vec x)).neg
  have hderiv : -((dotCLM (getScalarE target)).comp (logSoftmaxDerivCLM x)) =
      dotCLM (getScalarE (crossEntropyLogitsDerivSpec 0 (ofFnE x) target)) := by
    rw [getScalarE_crossEntropy_gradient]
    ext dx
    simp only [neg_apply, ContinuousLinearMap.comp_apply]
    rw [← logSoftmaxJvp_eq_deriv]
    have adjoint := inner_logSoftmaxJvp_vjp x dx (-getScalarE target)
    simp only [inner_eq_sum_mul, PiLp.neg_apply, mul_neg, Finset.sum_neg_distrib] at adjoint
    simpa only [dotCLM_apply, mul_comm] using adjoint
  convert h.congr_fderiv hderiv using 1
  funext logits
  exact crossEntropyLogitsSpec_eq_dot (ofFnE logits) target

/-- Target weights remain differentiable, including at zero. The numerical implementation must
not turn a zero-weight forward shortcut into a zero target derivative. -/
theorem hasFDerivAt_crossEntropyTargets_vec {n : Nat} (logits : Tensor ℝ [n])
    (target : Vec n) :
    HasFDerivAt (fun weights : Vec n => crossEntropyLogitsSpec 0 logits (ofFnE weights))
      (dotCLM (getScalarE (CrossEntropy.targetGradient 0 logits))) target := by
  have hgradient : getScalarE (CrossEntropy.targetGradient 0 logits) =
      -getScalarE (Activation.logSoftmaxSpec 0 logits) := by
    change getScalarE (Tensor.map (fun value => -value / ((1 : Nat) : ℝ))
      (CrossEntropy.vectorTerms logits (Tensor.full [n] 1))) = _
    rw [crossEntropy_vector_terms_eq]
    ext i
    simp [getScalarE_ofLp, Proofs.logSoftmaxSpec_zero_vec, mulSpec]
  have hderiv : dotCLM (getScalarE (CrossEntropy.targetGradient 0 logits)) =
      -(dotCLM (getScalarE (Activation.logSoftmaxSpec 0 logits))) := by
    rw [hgradient]
    ext direction
    simp [dotCLM_apply, PiLp.neg_apply, neg_mul]
  rw [hderiv]
  have h := (dotCLM (getScalarE (Activation.logSoftmaxSpec 0 logits))).hasFDerivAt.neg
    (x := target)
  convert h using 1
  funext weights
  rw [crossEntropyLogitsSpec_eq_dot, getScalarE_ofFnE]
  change -(dotCLM weights (getScalarE (Activation.logSoftmaxSpec 0 logits))) =
    -(dotCLM (getScalarE (Activation.logSoftmaxSpec 0 logits))) weights
  simp only [dotCLM_apply, mul_comm]

/-! ## Distillation -/

/-- The logit-difference gradient is the usual student-minus-teacher probability difference
over the reals. Both numerical branches and the residual correction preserve this identity. -/
theorem distillDerivSpec_eq_sub_softmax {n : Nat} (student teacher : Tensor ℝ [n + 1]) :
    Spec.distillDerivSpec 0 student teacher =
      subSpec (Activation.softmaxVecSpec student) (Activation.softmaxVecSpec teacher) := by
  classical
  let p := Activation.softmaxVecSpec teacher
  let q := Activation.softmaxVecSpec student
  let delta := subSpec student teacher
  let mean := reduceDim sumSpec 0 (mulSpec p delta)
  let d := subSpec delta (broadcastAfterSum [n + 1] 0 mean)
  let correction := Tensor.map Activation.Math.log1pSpec
    (reduceDim sumSpec 0 (mulSpec p (Tensor.map Activation.Math.expm1Spec d)))
  let ratio := subSpec d (broadcastAfterSum [n + 1] 0 correction)
  let u := addSpec
    (map2Spec (fun weight r => if MathFunctions.abs r < 1 then
      weight * Activation.Math.expm1Spec r else 0) p ratio)
    (map2Spec (fun r difference => if MathFunctions.abs r < 1 then 0 else difference)
      ratio (subSpec q p))
  let a := ∑ j, Real.exp (teacher.getScalar j)
  let b := ∑ j, Real.exp (student.getScalar j)
  let factor := Real.exp (-mean.item) / a
  let z := ∑ j, p.getScalar j * Real.exp (d.getScalar j)
  have ha : 0 < a := Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  have hp : ∀ j, 0 < p.getScalar j := fun j => Proofs.softmax_vec_spec_pos teacher j
  have hz : 0 < z := Finset.sum_pos (fun j _ => mul_pos (hp j) (Real.exp_pos _))
    Finset.univ_nonempty
  have hf : factor ≠ 0 := div_ne_zero (Real.exp_ne_zero _) (ne_of_gt ha)
  have hmassP : (∑ j, p.getScalar j) = 1 := by
    simpa [Spec.sum_spec_vec, p] using Proofs.sum_spec_softmax_vec_spec teacher
  have hmassQ : (∑ j, q.getScalar j) = 1 := by
    simpa [Spec.sum_spec_vec, q] using Proofs.sum_spec_softmax_vec_spec student
  have hbroadcast (v : Tensor ℝ []) (i : Fin (n + 1)) :
      (broadcastAfterSum [n + 1] 0 v).getScalar i = v.item := by
    simp [broadcastAfterSum, Tensor.getScalar]
  have hd (i : Fin (n + 1)) :
      d.getScalar i = student.getScalar i - teacher.getScalar i - mean.item := by
    simp [d, delta, subSpec, hbroadcast]
  -- Factor the same positive normalization out of every teacher-weighted exponential.
  have hweight (i : Fin (n + 1)) :
      p.getScalar i * Real.exp (d.getScalar i) = Real.exp (student.getScalar i) * factor := by
    rw [show p.getScalar i = Real.exp (teacher.getScalar i) / a from
      Proofs.getScalar_softmaxVecSpec_eq_exp_div teacher i, hd,
      Real.exp_sub, Real.exp_sub]
    dsimp [factor]
    rw [Real.exp_neg]
    field_simp [Real.exp_ne_zero, ne_of_gt ha]
  have hfactor : z = b * factor := by
    dsimp only [z]
    simp_rw [hweight]
    exact (Finset.sum_mul _ _ _).symm
  have hc : correction.item = Real.log z := by
    change Activation.Math.log1pSpec
      (sumSpec (mulSpec p (Tensor.map Activation.Math.expm1Spec d))) = _
    rw [Proofs.log1p_spec_eq_log_one_add, Spec.sum_spec_vec]
    simp_rw [Spec.getScalar_mul_spec, Tensor.getScalar_map,
      Proofs.expm1_spec_eq_exp_sub_one, mul_sub, mul_one]
    rw [Finset.sum_sub_distrib, hmassP]
    congr 1
    dsimp only [z]
    ring
  have hratio (i : Fin (n + 1)) :
      p.getScalar i * Real.exp (ratio.getScalar i) = q.getScalar i := by
    have hr : ratio.getScalar i = d.getScalar i - correction.item := by
      simp [ratio, subSpec, hbroadcast]
    rw [hr, hc, Real.exp_sub, Real.exp_log hz, ← mul_div_assoc, hweight,
      hfactor, show q.getScalar i = Real.exp (student.getScalar i) / b from
        Proofs.getScalar_softmaxVecSpec_eq_exp_div student i]
    exact mul_div_mul_right _ _ hf
  have hu (i : Fin (n + 1)) : u.getScalar i = q.getScalar i - p.getScalar i := by
    simp only [u, addSpec, subSpec, Tensor.getScalar_map2Spec]
    split_ifs with h
    · simp only [Proofs.expm1_spec_eq_exp_sub_one, mul_sub, mul_one, add_zero]
      rw [hratio]
    · simp
  have hsum : sumSpec u = 0 := by
    rw [Spec.sum_spec_vec]
    simp_rw [hu]
    rw [Finset.sum_sub_distrib, hmassQ, hmassP, sub_self]
  change scaleSpec (subSpec u (mulSpec q
    (broadcastAfterSum [n + 1] 0 (reduceDim sumSpec 0 u))))
      (1 / (Spec.axisMeanDenom [n + 1] 0 : ℝ)) = subSpec q p
  apply Tensor.ext_vector
  intro i
  have hsumItem : (reduceDim sumSpec 0 u).item = 0 := hsum
  simp [scaleSpec, subSpec, mulSpec, hbroadcast, hsumItem,
    show Spec.axisMeanDenom [n + 1] 0 = 1 from rfl, hu]

private theorem distill_spec_eq_dot {n : Nat} (student teacher : Tensor ℝ [n + 1]) :
    Spec.distillSpec 0 student teacher =
      -(dotCLM (getScalarE (Activation.softmaxVecSpec teacher)))
        (getScalarE (Activation.logSoftmaxSpec 0 student)) := by
  classical
  unfold Spec.distillSpec Spec.meanOverAxisSlices
  rw [Proofs.softmaxSpec_zero_vec]
  simp only [show Spec.axisMeanDenom [n + 1] 0 = 1 from rfl, Nat.cast_one, div_one,
    Spec.sum_spec_vec, dotCLM_apply]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  simp only [Tensor.getScalar_map2Spec, getScalarE_ofLp]
  rw [ite_eq_right]
  simpa only [beq_iff_eq] using ne_of_gt (Proofs.softmax_vec_spec_pos teacher i)

/-- With teacher logits fixed, `distillDerivSpec` is the Fréchet derivative of the actual
distillation loss on nonempty class vectors. This connects the compensated pullback to the loss,
rather than just identifying it with another gradient formula. -/
theorem hasFDerivAt_distillSpec_vec {n : Nat} (student : Vec (n + 1))
    (teacher : Tensor ℝ [n + 1]) :
    HasFDerivAt (fun x : Vec (n + 1) => Spec.distillSpec 0 (ofFnE x) teacher)
      (dotCLM (getScalarE (Spec.distillDerivSpec 0 (ofFnE student) teacher))) student := by
  classical
  let p := getScalarE (Activation.softmaxVecSpec teacher)
  have h := ((dotCLM p).hasFDerivAt.comp student
    (hasFDerivAt_logSoftmaxSpec_vec student)).neg
  have hderiv : -((dotCLM p).comp (logSoftmaxDerivCLM student)) =
      dotCLM (getScalarE (Spec.distillDerivSpec 0 (ofFnE student) teacher)) := by
    ext dx
    simp only [neg_apply, ContinuousLinearMap.comp_apply]
    rw [← logSoftmaxJvp_eq_deriv, distillDerivSpec_eq_sub_softmax]
    have hp : (∑ i, p i) = 1 := by
      simpa [p, getScalarE_ofLp, Spec.sum_spec_vec] using
        Proofs.sum_spec_softmax_vec_spec teacher
    simp only [dotCLM_apply, logSoftmaxJvp, vecOfFun_apply, getScalarE_ofLp,
      subSpec, Tensor.getScalar_map2Spec]
    change -(∑ i, p i * (dx i - ∑ j, softmaxVec student j * dx j)) =
      ∑ i, (getScalarE (Activation.softmaxVecSpec (ofFnE student)) i - p i) * dx i
    rw [getScalarE_softmaxVecSpec, getScalarE_ofFnE]
    simp_rw [mul_sub, sub_mul]
    rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, ← Finset.sum_mul, hp, one_mul]
    ring
  convert h.congr_fderiv hderiv using 1
  funext x
  exact distill_spec_eq_dot (ofFnE x) teacher

end
end Autograd
end Proofs
