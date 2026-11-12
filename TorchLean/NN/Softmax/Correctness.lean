/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Softmax
public import TorchLean.NN.Softmax.Log
public import TorchLean.Tensor.Layers.Loss.Axis

/-!
# Softmax and log-softmax: spec definitions versus analytic presentation

`TorchLean.NN.Softmax.FDeriv` and `TorchLean.NN.Softmax.Log` prove Fréchet-derivative facts about
`Vec.softmax` and `Vec.logSoftmax`, which are stated on Euclidean vectors. The specification layer
defines `Softmax.vec` and `LogSoftmax.vec` on tensors, in the numerically stable max-shifted form.

Over `ℝ`, the results below close that gap:

- `Softmax.getScalarE_vec` and `LogSoftmax.getScalarE_vec` identify the spec kernels with
  the analytic definitions after vectorization.
- `Tensor.hasFDerivAt_softmax` and `Tensor.hasFDerivAt_logSoftmax` transfer differentiability
  to `Tensor.softmax` and `Tensor.logSoftmax` along axis `0`.
- `FDerivOp.softmax` and `FDerivOp.logSoftmax` package the arms of `Op.softmax` and
  `Op.logSoftmax` along axis `0` as `FDerivOp`, so their backward rules are proved to be
  the adjoint of the true derivative (`softmaxBackward_eq_fderiv`, `logSoftmaxBackward_eq_fderiv`).
- `hasFDerivAt_distillLoss` identifies the cancellation-resistant student pullback with
  the derivative of the teacher-logit loss, with the teacher held fixed.
- `CrossEntropy.vectorLogProbs_real` and `CrossEntropy.logProbs_real` identify the
  log-probabilities of cross-entropy, in both the one-level and the two-level form, with the
  log-softmax. `CrossEntropy.log_two_level` is the identity behind the two-level normalizer.
- `CrossEntropy.hasFDerivAt_logits` and `CrossEntropy.hasFDerivAt_target` prove
  the logits and target-weight derivatives of the weighted loss. Targets need not sum to one.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open Tensor
open Rep (dot)

open scoped BigOperators

noncomputable section

open Vec

/-! ## Softmax -/

/-- The spec softmax kernel is the analytic `Vec.softmax` after vectorization. -/
theorem Softmax.getScalarE_vec {n : Nat} (t : Tensor ℝ [n]) :
    getScalarE (vec (α := ℝ) (n := n) t) =
      Vec.softmax (n := n) (getScalarE t) := by
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      ext i
      rw [getScalarE_ofLp, Softmax.getScalar_vec_eq_exp_div]
      simp [Vec.softmax, sumExp]

/-- The vectorized forward map of `Tensor.softmax` along axis `0` on vectors is `Vec.softmax`. -/
theorem Tensor.getScalarE_softmax_ofFnE {n : Nat} :
    (fun xV : Vec n => getScalarE (Tensor.softmax (s := [n]) (ofFnE xV) ⟨0, Nat.one_pos⟩)) =
      Vec.softmax (n := n) := by
  funext xV
  rw [Tensor.softmax_zero_vec, Softmax.getScalarE_vec, getScalarE_ofFnE]

/-- Axis-`0` spec softmax on vectors is Fréchet-differentiable with derivative
`softmaxDerivCLM`. -/
theorem Tensor.hasFDerivAt_softmax {n : Nat} (xV : Vec n) :
    HasFDerivAt
      (fun xV : Vec n => getScalarE (Tensor.softmax (s := [n]) (ofFnE xV) ⟨0, Nat.one_pos⟩))
      (softmaxDerivCLM (n := n) xV) xV := by
  rw [getScalarE_softmax_ofFnE]
  exact Vec.hasFDerivAt_softmax xV

/-- The spec softmax VJP is the analytic `softmaxJVP` (the softmax Jacobian is symmetric). -/
theorem Tensor.getScalarE_softmaxBackward {n : Nat} (x dY : Tensor ℝ [n]) :
    getScalarE (Tensor.softmaxBackward (α := ℝ) (s := [n]) x dY ⟨0, Nat.one_pos⟩) =
      softmaxJVP (n := n) (getScalarE x) (getScalarE dY) := by
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      have hy : ∀ i, Tensor.getScalar (Softmax.vec x) i =
          Vec.softmax (n := Nat.succ n) (getScalarE x) i := by
        intro i
        rw [← Softmax.getScalarE_vec]
        rfl
      rw [Tensor.softmaxBackward_zero_vec]
      ext i
      rw [getScalarE_ofLp, Softmax.getScalar_innermostBackward]
      simp [hy, softmaxJVP, inner_eq_sum_mul, getScalarE_ofLp]

/-- The softmax arm along axis `0` on vectors with its analytic JVP and the VJP/JVP adjointness
law. -/
def AdjointOp.softmax (n : Nat) : Tape.AdjointOp ℝ (.dim n .scalar) (.dim n .scalar) where
  op :=
    { forward := fun x => Tensor.softmax x ⟨0, Nat.one_pos⟩
      backward := fun x g => Tensor.softmaxBackward x g ⟨0, Nat.one_pos⟩ }
  jvp := fun x dx => ofFnE (softmaxJVP (n := n) (getScalarE x) (getScalarE dx))
  isAdjointPair := by
    intro x dx δ
    change Rep.dot (ofFnE (softmaxJVP (n := n) (getScalarE x) (getScalarE dx))) δ =
      Rep.dot dx (Tensor.softmaxBackward (α := ℝ) (s := [n]) x δ ⟨0, Nat.one_pos⟩)
    rw [dot_eq_inner_getScalarE, dot_eq_inner_getScalarE, getScalarE_ofFnE,
      getScalarE_softmaxBackward]
    exact inner_softmaxJVP_comm (n := n) (getScalarE x) (getScalarE dx) (getScalarE δ)

/-- The softmax arm along axis `0` on vectors is analytically correct: its JVP is the Fréchet
derivative. -/
def FDerivOp.softmax (n : Nat) : FDerivOp n n where
  toAdjointOp := AdjointOp.softmax n
  deriv := softmaxDerivCLM (n := n)
  hasFDerivAt := fun xV => Tensor.hasFDerivAt_softmax xV
  jvp_eq := by
    intro xV dxV
    change getScalarE (ofFnE (softmaxJVP (n := n) (getScalarE (ofFnE xV))
      (getScalarE (ofFnE dxV)))) = _
    rw [getScalarE_ofFnE, getScalarE_ofFnE, getScalarE_ofFnE, softmaxJVP_eq_deriv]

/-- The spec softmax backward is the vector-Jacobian product of the spec softmax forward. -/
theorem Tensor.softmaxBackward_eq_fderiv {n : Nat} (x δ : Tensor ℝ [n]) :
    getScalarE (Tensor.softmaxBackward (α := ℝ) (s := [n]) x δ ⟨0, Nat.one_pos⟩) =
      (fderiv ℝ (fun xV : Vec n => getScalarE (Tensor.softmax (s := [n]) (ofFnE xV)
        ⟨0, Nat.one_pos⟩)) (getScalarE x)).adjoint (getScalarE δ) :=
  (FDerivOp.softmax n).backward_eq_fderiv x δ

/-! ## Log-softmax -/

/-- The spec log-softmax kernel is the analytic `Vec.logSoftmax` after vectorization. -/
theorem LogSoftmax.getScalarE_vec {n : Nat} (t : Tensor ℝ [n]) :
    getScalarE (vec (α := ℝ) (n := n) t) =
      Vec.logSoftmax (n := n) (getScalarE t) := by
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      ext i
      rw [getScalarE_ofLp, LogSoftmax.getScalar_vec_eq_sub_log]
      simp [Vec.logSoftmax, sumExp]

/-- The vectorized forward map of `Tensor.logSoftmax` along axis `0` on vectors is
`Vec.logSoftmax`. -/
theorem Tensor.getScalarE_logSoftmax_ofFnE {n : Nat} :
    (fun xV : Vec n =>
      getScalarE (Tensor.logSoftmax (α := ℝ) (s := [n]) (ofFnE xV) ⟨0, Nat.one_pos⟩)) =
      Vec.logSoftmax (n := n) := by
  funext xV
  rw [Tensor.logSoftmax_zero_vec, LogSoftmax.getScalarE_vec, getScalarE_ofFnE]

/-- Axis-`0` spec log-softmax on vectors is Fréchet-differentiable with derivative
`logSoftmaxDerivCLM`. -/
theorem Tensor.hasFDerivAt_logSoftmax {n : Nat} (xV : Vec n) :
    HasFDerivAt
      (fun xV : Vec n =>
        getScalarE (Tensor.logSoftmax (α := ℝ) (s := [n]) (ofFnE xV) ⟨0, Nat.one_pos⟩))
      (logSoftmaxDerivCLM (n := n) xV) xV := by
  rw [getScalarE_logSoftmax_ofFnE]
  exact Vec.hasFDerivAt_logSoftmax xV

/-- Exponentiating the spec log-softmax output recovers the analytic softmax coordinates. -/
theorem LogSoftmax.getScalar_exp_vec {n : Nat} (x : Tensor ℝ [Nat.succ n])
    (i : Fin (Nat.succ n)) :
    Tensor.getScalar (Tensor.exp (vec x)) i =
      Vec.softmax (n := Nat.succ n) (getScalarE x) i := by
  have hsumPos : 0 < ∑ j, Real.exp (Tensor.getScalar x j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  rw [Tensor.exp, Tensor.getScalar_map, LogSoftmax.getScalar_vec_eq_sub_log,
    MathFunctions.exp_real, Real.exp_sub, Real.exp_log hsumPos]
  simp [Vec.softmax, sumExp]

/-- The typed graph's compensated log-softmax JVP agrees with the analytic derivative. -/
theorem Tensor.getScalarE_logSoftmaxJVP {n : Nat} (x dx : Tensor ℝ [n]) :
    getScalarE (Tensor.logSoftmaxJVP (Tensor.logSoftmax x ⟨0, Nat.one_pos⟩) dx ⟨0, Nat.one_pos⟩) =
      Vec.logSoftmaxJVP (getScalarE x) (getScalarE dx) := by
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      change getScalarE (LogSoftmax.innermostJVP
        (LogSoftmax.vec x) dx) = _
      rw [LogSoftmax.innermostJVP_eq]
      ext i
      simp only [getScalarE_ofLp, sub_def, Tensor.getScalar_zipWith,
        Tensor.getScalar_full]
      simp only [dot, sum_vec, getScalar_mul,
        LogSoftmax.getScalar_exp_vec]
      simp [Vec.logSoftmaxJVP, inner_eq_sum_mul, getScalarE_ofLp]

/-- The spec log-softmax VJP, evaluated on the recomputed forward output, is the analytic
`logSoftmaxVJP`. -/
theorem Tensor.getScalarE_logSoftmaxBackward {n : Nat} (x dY : Tensor ℝ [n]) :
    getScalarE (Tensor.logSoftmaxBackward (Tensor.logSoftmax x ⟨0, Nat.one_pos⟩) dY
        ⟨0, Nat.one_pos⟩) =
      logSoftmaxVJP (n := n) (getScalarE x) (getScalarE dY) := by
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      have hsum : Tensor.sum dY = ∑ j, getScalarE dY j := by
        rw [sum_vec]
        rfl
      rw [Tensor.logSoftmaxBackward_zero_vec, Tensor.logSoftmax_zero_vec]
      change getScalarE (LogSoftmax.vecBackward
        (LogSoftmax.vec x) dY) = _
      rw [LogSoftmax.vecBackward_eq]
      ext i
      rw [getScalarE_ofLp, show (dY - _) = Tensor.zipWith (· - ·) dY _ by rfl,
        Tensor.getScalar_zipWith, getScalar_mul,
        LogSoftmax.getScalar_exp_vec, hsum]
      simp [logSoftmaxVJP]

/-- The log-softmax arm along axis `0` on vectors with its analytic JVP and the VJP/JVP
adjointness law. -/
def AdjointOp.logSoftmax (n : Nat) :
    Tape.AdjointOp ℝ (.dim n .scalar) (.dim n .scalar) where
  op :=
    { forward := fun x => Tensor.logSoftmax x ⟨0, Nat.one_pos⟩
      backward := fun x g =>
        Tensor.logSoftmaxBackward (Tensor.logSoftmax x ⟨0, Nat.one_pos⟩) g ⟨0, Nat.one_pos⟩ }
  jvp := fun x dx => ofFnE (Vec.logSoftmaxJVP (n := n) (getScalarE x) (getScalarE dx))
  isAdjointPair := by
    intro x dx δ
    change Rep.dot (ofFnE (Vec.logSoftmaxJVP (n := n) (getScalarE x) (getScalarE dx))) δ =
      Rep.dot dx
        (Tensor.logSoftmaxBackward (Tensor.logSoftmax x ⟨0, Nat.one_pos⟩) δ ⟨0, Nat.one_pos⟩)
    rw [dot_eq_inner_getScalarE, dot_eq_inner_getScalarE, getScalarE_ofFnE,
      getScalarE_logSoftmaxBackward]
    exact inner_logSoftmaxJVP_vjp (n := n) (getScalarE x) (getScalarE dx) (getScalarE δ)

/-- The log-softmax arm along axis `0` on vectors is analytically correct: its JVP is the
Fréchet derivative. -/
def FDerivOp.logSoftmax (n : Nat) : FDerivOp n n where
  toAdjointOp := AdjointOp.logSoftmax n
  deriv := logSoftmaxDerivCLM (n := n)
  hasFDerivAt := fun xV => Tensor.hasFDerivAt_logSoftmax xV
  jvp_eq := by
    intro xV dxV
    change getScalarE (ofFnE (Vec.logSoftmaxJVP (n := n) (getScalarE (ofFnE xV))
      (getScalarE (ofFnE dxV)))) = _
    rw [getScalarE_ofFnE, getScalarE_ofFnE, getScalarE_ofFnE, logSoftmaxJVP_eq_deriv]

/-- The spec log-softmax backward is the vector-Jacobian product of the spec log-softmax
forward. -/
theorem Tensor.logSoftmaxBackward_eq_fderiv {n : Nat} (x δ : Tensor ℝ [n]) :
    getScalarE (Tensor.logSoftmaxBackward (Tensor.logSoftmax x ⟨0, Nat.one_pos⟩) δ
        ⟨0, Nat.one_pos⟩) =
      (fderiv ℝ (fun xV : Vec n =>
          getScalarE (Tensor.logSoftmax (α := ℝ) (s := [n]) (ofFnE xV) ⟨0, Nat.one_pos⟩))
        (getScalarE x)).adjoint (getScalarE δ) :=
  (FDerivOp.logSoftmax n).backward_eq_fderiv x δ

/-! ## Weighted logits cross-entropy -/

/-- The two-level normalizer over the reals: for `q ≠ p`, `log (1 + exp (x q - x p + log (1 + I)))`
with `I = ∑_{j ∉ {p, q}} exp (x j - x q)` is `log (∑ j, exp (x j)) - x p`. -/
theorem CrossEntropy.log_two_level {n : Nat} (x : Fin n → ℝ) {p q : Fin n} (hq : q ≠ p) :
    Real.log (1 + Real.exp (x q - x p + Real.log
        (1 + ∑ j, if j = p ∨ j = q then 0 else Real.exp (x j - x q)))) =
      Real.log (∑ j, Real.exp (x j)) - x p := by
  have hI : 0 ≤ ∑ j, (if j = p ∨ j = q then 0 else Real.exp (x j - x q)) :=
    Finset.sum_nonneg fun j _ => by split_ifs <;> positivity
  have hsplit : ∑ j, (if j = p then 0 else Real.exp (x j - x q)) =
      1 + ∑ j, (if j = p ∨ j = q then 0 else Real.exp (x j - x q)) := by
    have hpt (j : Fin n) : (if j = p then 0 else Real.exp (x j - x q)) =
        (if j = q then 1 else 0) + (if j = p ∨ j = q then 0 else Real.exp (x j - x q)) := by
      by_cases hjp : j = p
      · simp [hjp, hq.symm]
      · by_cases hjq : j = q
        · subst hjq
          simp [hjp]
        · simp [hjp, hjq]
    rw [Finset.sum_congr rfl fun j _ => hpt j, Finset.sum_add_distrib]
    simp
  have hscale : Real.exp (x q - x p) * ∑ j, (if j = p then 0 else Real.exp (x j - x q)) =
      ∑ j, if j = p then 0 else Real.exp (x j - x p) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    split_ifs
    · rw [mul_zero]
    · rw [← Real.exp_add]
      congr 1
      ring
  have htail : 1 + ∑ j, (if j = p then 0 else Real.exp (x j - x p)) =
      ∑ j, Real.exp (x j - x p) := by
    have hpt (j : Fin n) : Real.exp (x j - x p) =
        (if j = p then 1 else 0) + (if j = p then 0 else Real.exp (x j - x p)) := by
      by_cases hjp : j = p
      · subst hjp
        simp
      · simp [hjp]
    rw [Finset.sum_congr rfl fun j _ => hpt j, Finset.sum_add_distrib]
    simp
  have hweight : ∑ j, Real.exp (x j - x p) = (∑ j, Real.exp (x j)) * Real.exp (-x p) := by
    simp_rw [sub_eq_add_neg, Real.exp_add]
    rw [Finset.sum_mul]
  have hS : 0 < ∑ j, Real.exp (x j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨p, Finset.mem_univ p⟩
  rw [Real.exp_add, Real.exp_log (by linarith), ← hsplit, hscale, htail, hweight,
    Real.log_mul hS.ne' (Real.exp_ne_zero _), Real.log_exp, sub_eq_add_neg]

/-- Over the reals, `vectorLogProbs` is the stable log-softmax `LogSoftmax.vec`, in both its
one-level and its two-level branch. -/
theorem CrossEntropy.vectorLogProbs_real {n : Nat} (t : Tensor ℝ [n]) :
    vectorLogProbs t = LogSoftmax.vec t := by
  match n with
  | 0 => rfl
  | n + 1 =>
      apply Tensor.ext_vector
      intro i
      simp only [vectorLogProbs]
      split
      · rename_i htwo
        rw [LogSoftmax.getScalar_vec_eq_sub_log]
        have h := log_two_level (fun j => t.getScalar j) (runnerUp_ne_of_twoLevelAt htwo)
        simp only [Tensor.sub_def, Tensor.getScalar_zipWith, Tensor.getScalar_full,
          Context.log1p_eq_log_one_add, MathFunctions.exp, Tensor.sum_vec, Tensor.getScalar_ofFn]
        rw [h]
        ring
      · rfl

/-- Over the reals, `logProbs x dim` is the log-softmax of `x` along `dim`. -/
theorem CrossEntropy.logProbs_real {s : Shape} (x : Tensor ℝ s) (dim : Fin s.rank) :
    logProbs x dim = Tensor.logSoftmax x dim := by
  apply Tensor.Rep.ext
  intro c
  rw [logProbs_apply, Tensor.logSoftmax_apply, vectorLogProbs_real]

/-- Over the reals, the weighted terms of one class vector are the target times the vector
log-softmax of the logits. -/
theorem CrossEntropy.vectorTerms_real {n : Nat} (x t : Tensor ℝ [n]) :
    CrossEntropy.vectorTerms x t = t * LogSoftmax.vec x := by
  cases n with
  | zero =>
      apply Tensor.ext_vector
      intro i
      exact i.elim0
  | succ n =>
      apply Tensor.ext_vector
      intro i
      simp only [CrossEntropy.vectorTerms, vectorLogProbs_real, getScalar_ofFn, sub_self,
        beq_self_eq_true, ite_true, mul_def, getScalar_zipWith]

/-- Over `ℝ`, the overflow-aware weighted loss equals the cross-entropy formula, the negated dot
product of `target` with the log-softmax of `x`. Rounded evaluations of the two schedules can
differ. -/
theorem CrossEntropy.logits_eq_dot {n : Nat} (x target : Tensor ℝ [n]) :
    Tensor.crossEntropy .logits x target ⟨0, Shape.AxisInBounds.proof⟩ =
      -(innerSL ℝ (getScalarE target)) (getScalarE (Tensor.logSoftmax x ⟨0, Nat.one_pos⟩)) := by
  change -Tensor.sum (CrossEntropy.vectorTerms x target) / ((1 : Nat) : ℝ) = _
  rw [vectorTerms_real, Nat.cast_one, div_one, Tensor.logSoftmax_zero_vec]
  simp only [Tensor.sum_vec, mul_def, getScalar_zipWith, innerSL_apply_apply, inner_eq_sum_mul,
    getScalarE_ofLp]

private theorem CrossEntropy.getScalarE_backward {n : Nat} (x : Vec n)
    (target : Tensor ℝ [n]) :
    getScalarE (Tensor.crossEntropyBackward .logits (ofFnE x) target ⟨0, Nat.one_pos⟩) =
      logSoftmaxVJP x (-getScalarE target) := by
  have same := getScalarE_logSoftmaxBackward (ofFnE x)
    (Tensor.map (fun value => -value) target)
  simp only [Tensor.crossEntropyBackward, logProbs_real, reduceIte,
    show axisMeanDenom [n] ⟨0, Shape.AxisInBounds.proof⟩ = 1 from rfl, Context.divNat_real,
    Nat.cast_one, div_one]
  have hid (t : Tensor ℝ [n]) : Tensor.map (fun value => value) t = t := by
    apply Tensor.ext_vector
    intro i
    simp [Tensor.getScalar_map]
  rw [hid]
  have hneg : getScalarE (Tensor.map (fun value => -value) target) = -getScalarE target := by
    ext i
    simp [getScalarE_ofLp]
  simpa only [getScalarE_ofFnE, hneg] using same

/-- The shared logits pullback `Tensor.crossEntropyBackward` gives the Fréchet derivative of the
weighted loss for any target weights, including unnormalized and zero ones, with no
probability-normalization hypothesis. -/
theorem CrossEntropy.hasFDerivAt_logits {n : Nat} (x : Vec n)
    (target : Tensor ℝ [n]) :
    HasFDerivAt (fun logits : Vec n =>
        Tensor.crossEntropy .logits (ofFnE logits) target ⟨0, Shape.AxisInBounds.proof⟩)
      (innerSL ℝ (getScalarE (Tensor.crossEntropyBackward .logits (ofFnE x) target ⟨0,
          Nat.one_pos⟩))) x := by
  have h := ((innerSL ℝ (getScalarE target)).hasFDerivAt.comp x
    (Tensor.hasFDerivAt_logSoftmax x)).neg
  have hderiv : -((innerSL ℝ (getScalarE target)).comp (logSoftmaxDerivCLM x)) =
      innerSL ℝ (getScalarE (Tensor.crossEntropyBackward .logits (ofFnE x) target ⟨0,
          Nat.one_pos⟩)) := by
    rw [getScalarE_backward]
    ext dx
    simp only [neg_apply, ContinuousLinearMap.comp_apply]
    rw [← logSoftmaxJVP_eq_deriv]
    have adjoint := inner_logSoftmaxJVP_vjp x dx (-getScalarE target)
    simp only [inner_eq_sum_mul, PiLp.neg_apply, mul_neg, Finset.sum_neg_distrib] at adjoint
    simpa only [innerSL_apply_apply, inner_eq_sum_mul, mul_comm] using adjoint
  convert h.congr_fderiv hderiv using 1
  funext logits
  exact logits_eq_dot (ofFnE logits) target

/-- The loss is differentiable in the target weights at every point, zero weights included, with
gradient `Tensor.crossEntropyTargetBackward`, the negated log-softmax of the logits. A zero-weight
shortcut in the forward pass must therefore not zero that derivative. -/
theorem CrossEntropy.hasFDerivAt_target {n : Nat} (logits : Tensor ℝ [n])
    (target : Vec n) :
    HasFDerivAt (fun weights : Vec n =>
        Tensor.crossEntropy .logits logits (ofFnE weights) ⟨0, Shape.AxisInBounds.proof⟩)
      (innerSL ℝ (getScalarE (Tensor.crossEntropyTargetBackward logits ⟨0, Nat.one_pos⟩)))
      target := by
  have hgradient : getScalarE (Tensor.crossEntropyTargetBackward logits ⟨0, Nat.one_pos⟩) =
      -getScalarE (Tensor.logSoftmax logits ⟨0, Shape.AxisInBounds.proof⟩) := by
    change getScalarE (Tensor.map (fun value => -value / ((1 : Nat) : ℝ))
      (CrossEntropy.vectorTerms logits (Tensor.full [n] 1))) = _
    rw [vectorTerms_real, Tensor.logSoftmax_zero_vec]
    ext i
    simp [getScalarE_ofLp, mul_def]
  have hderiv :
      innerSL ℝ (getScalarE (Tensor.crossEntropyTargetBackward logits ⟨0, Nat.one_pos⟩)) =
        -(innerSL ℝ (getScalarE (Tensor.logSoftmax logits ⟨0, Shape.AxisInBounds.proof⟩))) := by
    rw [hgradient, map_neg]
  rw [hderiv]
  have h := (innerSL ℝ (getScalarE (Tensor.logSoftmax logits ⟨0, Nat.one_pos⟩))).hasFDerivAt.neg
    (x := target)
  convert h using 1
  funext weights
  rw [logits_eq_dot, getScalarE_ofFnE]
  change
    -(innerSL ℝ weights (getScalarE (Tensor.logSoftmax logits ⟨0, Shape.AxisInBounds.proof⟩))) =
      -(innerSL ℝ (getScalarE (Tensor.logSoftmax logits ⟨0, Shape.AxisInBounds.proof⟩))) weights
  simp only [innerSL_apply_apply, inner_eq_sum_mul, mul_comm]

/-! ## Distillation -/

/-- The logit-difference gradient is the usual student-minus-teacher probability difference
over the reals. Both numerical branches and the residual correction preserve this identity. -/
theorem Tensor.distillLossBackward_eq_sub {n : Nat} (student teacher : Tensor ℝ [n + 1]) :
    distillLossBackward ⟨0, Shape.AxisInBounds.proof⟩ student teacher =
      Softmax.vec student -
          Softmax.vec teacher := by
  classical
  let p := Softmax.vec teacher
  let q := Softmax.vec student
  let delta := student - teacher
  let mean : Tensor ℝ [] :=
    castShape (reduce .sum (p * delta) [⟨0, Shape.AxisInBounds.proof⟩] false)
      (Shape.reduce_singleton _ _)
  let d := delta - Tensor.broadcast mean [n + 1] _ (Shape.eraseIdx_broadcastDims [n + 1] 0)
  let correction : Tensor ℝ [] := Tensor.map Context.log1p
    (castShape (reduce .sum (p * Tensor.map Context.expm1 d)
      [⟨0, Shape.AxisInBounds.proof⟩] false) (Shape.reduce_singleton _ _))
  let ratio := d -
    Tensor.broadcast correction [n + 1] _ (Shape.eraseIdx_broadcastDims [n + 1] 0)
  let u := (zipWith (fun weight r => if MathFunctions.abs r < 1 then
      weight * Context.expm1 r else 0) p ratio) +
    (zipWith (fun r difference => if MathFunctions.abs r < 1 then 0 else difference)
      ratio (q - p))
  let a := ∑ j, Real.exp (teacher.getScalar j)
  let b := ∑ j, Real.exp (student.getScalar j)
  let factor := Real.exp (-mean.item) / a
  let z := ∑ j, p.getScalar j * Real.exp (d.getScalar j)
  have ha : 0 < a := Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  have hp : ∀ j, 0 < p.getScalar j := fun j => Softmax.vec_pos teacher j
  have hz : 0 < z := Finset.sum_pos (fun j _ => mul_pos (hp j) (Real.exp_pos _))
    Finset.univ_nonempty
  have hf : factor ≠ 0 := div_ne_zero (Real.exp_ne_zero _) (ne_of_gt ha)
  have hmassP : (∑ j, p.getScalar j) = 1 := by
    simpa [sum_vec, p] using Softmax.sum_vec teacher
  have hmassQ : (∑ j, q.getScalar j) = 1 := by
    simpa [sum_vec, q] using Softmax.sum_vec student
  have hbroadcast (v : Tensor ℝ []) :
      Tensor.broadcast v [n + 1] _ (Shape.eraseIdx_broadcastDims [n + 1] 0) =
        Tensor.full [n + 1] v.item :=
    Tensor.Rep.ext fun c => by
      erw [Tensor.broadcast_apply]
  have hd (i : Fin (n + 1)) :
      d.getScalar i = student.getScalar i - teacher.getScalar i - mean.item := by
    simp [d, delta, sub_def, ↓hbroadcast]
  -- Factor the same positive normalization out of every teacher-weighted exponential.
  have hweight (i : Fin (n + 1)) :
      p.getScalar i * Real.exp (d.getScalar i) = Real.exp (student.getScalar i) * factor := by
    rw [show p.getScalar i = Real.exp (teacher.getScalar i) / a from
      Softmax.getScalar_vec_eq_exp_div teacher i, hd,
      Real.exp_sub, Real.exp_sub]
    dsimp [factor]
    rw [Real.exp_neg]
    field_simp [Real.exp_ne_zero, ne_of_gt ha]
  have hfactor : z = b * factor := by
    dsimp only [z]
    simp_rw [hweight]
    exact (Finset.sum_mul _ _ _).symm
  have hc : correction.item = Real.log z := by
    rw [item_map]
    erw [castShape_reduce_sum_vector, item_scalar]
    rw [Context.log1p_eq_log_one_add, sum_vec]
    simp_rw [getScalar_mul, Tensor.getScalar_map,
      Context.expm1_eq_exp_sub_one, mul_sub, mul_one]
    rw [Finset.sum_sub_distrib, hmassP]
    congr 1
    dsimp only [z]
    ring
  have hratio (i : Fin (n + 1)) :
      p.getScalar i * Real.exp (ratio.getScalar i) = q.getScalar i := by
    have hr : ratio.getScalar i = d.getScalar i - correction.item := by
      simp [ratio, sub_def, ↓hbroadcast]
    rw [hr, hc, Real.exp_sub, Real.exp_log hz, ← mul_div_assoc, hweight,
      hfactor, show q.getScalar i = Real.exp (student.getScalar i) / b from
        Softmax.getScalar_vec_eq_exp_div student i]
    exact mul_div_mul_right _ _ hf
  have hu (i : Fin (n + 1)) : u.getScalar i = q.getScalar i - p.getScalar i := by
    simp only [u, add_def, sub_def, Tensor.getScalar_zipWith]
    split_ifs with h
    · simp only [Context.expm1_eq_exp_sub_one, mul_sub, mul_one, add_zero]
      rw [hratio]
    · simp
  have hsum : Tensor.sum u = 0 := by
    rw [sum_vec]
    simp_rw [hu]
    rw [Finset.sum_sub_distrib, hmassQ, hmassP, sub_self]
  change Tensor.map (Context.divNat · (axisMeanDenom [n + 1] ⟨0, Shape.AxisInBounds.proof⟩))
    (u - (q * (Tensor.broadcast (castShape (reduce .sum u [⟨0, Shape.AxisInBounds.proof⟩] false)
      (Shape.reduce_singleton _ _) : Tensor ℝ []) [n + 1] _
        (Shape.eraseIdx_broadcastDims [n + 1] 0)))) = q - p
  rw [show axisMeanDenom [n + 1] ⟨0, Shape.AxisInBounds.proof⟩ = 1 from rfl]
  apply Tensor.ext_vector
  intro i
  have hsumItem : Tensor.item (castShape (reduce .sum u [⟨0, Shape.AxisInBounds.proof⟩] false)
      (Shape.reduce_singleton _ _) : Tensor ℝ []) = 0 := by
    erw [castShape_reduce_sum_vector, item_scalar]
    exact hsum
  simp [sub_def, mul_def, ↓hbroadcast, ↓hsumItem, hu]

private theorem Tensor.distillLoss_eq_dot {n : Nat} (student teacher : Tensor ℝ [n + 1]) :
    distillLoss ⟨0, Shape.AxisInBounds.proof⟩ student teacher =
      -(innerSL ℝ (getScalarE (Softmax.vec teacher)))
        (getScalarE (Tensor.logSoftmax student ⟨0, Shape.AxisInBounds.proof⟩)) := by
  classical
  unfold distillLoss meanOverAxisSlices
  rw [Tensor.softmax_zero_vec]
  simp only [show axisMeanDenom [n + 1] ⟨0, Shape.AxisInBounds.proof⟩ = 1 from rfl,
    Context.divNat_real, Nat.cast_one, div_one, sum_vec, innerSL_apply_apply,
    inner_eq_sum_mul]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  simp only [Tensor.getScalar_zipWith, getScalarE_ofLp]
  rw [ite_eq_right]
  simpa only [beq_iff_eq] using ne_of_gt (Softmax.vec_pos teacher i)

/-- With teacher logits fixed, `distillLossBackward` is the Fréchet derivative of the distillation
loss itself on nonempty class vectors, which ties the compensated pullback to the loss it
differentiates. -/
theorem Tensor.hasFDerivAt_distillLoss {n : Nat} (student : Vec (n + 1))
    (teacher : Tensor ℝ [n + 1]) :
    HasFDerivAt (fun x : Vec (n + 1) => distillLoss ⟨0, Shape.AxisInBounds.proof⟩ (ofFnE x)
        teacher)
      (innerSL ℝ (getScalarE (distillLossBackward ⟨0, Shape.AxisInBounds.proof⟩
        (ofFnE student) teacher))) student := by
  classical
  let p := getScalarE (Softmax.vec teacher)
  have h := ((innerSL ℝ p).hasFDerivAt.comp student
    (hasFDerivAt_logSoftmax student)).neg
  have hderiv : -((innerSL ℝ p).comp (logSoftmaxDerivCLM student)) =
      innerSL ℝ (getScalarE (distillLossBackward ⟨0, Shape.AxisInBounds.proof⟩
        (ofFnE student) teacher)) := by
    ext dx
    simp only [neg_apply, ContinuousLinearMap.comp_apply]
    rw [← logSoftmaxJVP_eq_deriv, distillLossBackward_eq_sub]
    have hp : (∑ i, p i) = 1 := by
      simpa [p, getScalarE_ofLp, sum_vec] using
        Softmax.sum_vec teacher
    simp only [innerSL_apply_apply, inner_eq_sum_mul, Vec.logSoftmaxJVP,
      getScalarE_ofLp, sub_def, Tensor.getScalar_zipWith]
    change -(∑ i, p i * (dx i - ∑ j, Vec.softmax student j * dx j)) =
      ∑ i, (getScalarE (Softmax.vec (ofFnE student)) i - p i) * dx i
    rw [Softmax.getScalarE_vec, getScalarE_ofFnE]
    simp_rw [mul_sub, sub_mul]
    rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, ← Finset.sum_mul, hp, one_mul]
    ring
  convert h.congr_fderiv hderiv using 1
  funext x
  exact distillLoss_eq_dot (ofFnE x) teacher

end
end TorchLean
