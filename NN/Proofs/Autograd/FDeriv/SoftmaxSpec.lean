/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Softmax
public import NN.Proofs.Autograd.FDeriv.LogSoftmax

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
      have hs : TorchLean.Tensor.sumSpec (mulSpec dY (Activation.softmaxVecSpec x)) =
          dotCLM (n := Nat.succ n) (softmaxVec (getScalarE x)) (getScalarE dY) := by
        rw [Spec.sum_spec_vec, dotCLM_apply]
        refine Finset.sum_congr rfl ?_
        intro j _
        rw [Spec.getScalar_mul_spec, hy, mul_comm]
        rfl
      rw [Proofs.softmaxBackwardSpec_zero_vec]
      change getScalarE (mulSpec (Activation.softmaxVecSpec x)
        (subSpec dY (Spec.replicate (Tensor.scalar
          (TorchLean.Tensor.sumSpec (mulSpec dY (Activation.softmaxVecSpec x))))))) = _
      ext i
      rw [getScalarE_ofLp, Spec.getScalar_mul_spec, hs, hy]
      change _ = softmaxVec (getScalarE x) i * (getScalarE dY i - _)
      congr 1
      rw [show subSpec dY _ = TorchLean.Tensor.map2Spec (· - ·) dY _ by rfl,
        TorchLean.Tensor.getScalar_map2Spec]
      simp [Spec.replicate]

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
  rw [expSpec, TorchLean.Tensor.getScalar_mapSpec, Proofs.getScalar_logSoftmaxVecSpec_eq_sub_log,
    mathfunc_exp_eq_rexp, Real.exp_sub, Real.exp_log hsumPos]
  simp [softmaxVec, sumExp]

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
      change getScalarE (subSpec dY (mulSpec (expSpec (Activation.logSoftmaxVecSpec x))
        (Spec.replicate (Tensor.scalar (TorchLean.Tensor.sumSpec dY))))) = _
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

end
end Autograd
end Proofs
