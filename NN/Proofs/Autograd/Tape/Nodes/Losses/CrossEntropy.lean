/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Reductions
public import NN.Proofs.Autograd.Tape.Nodes.Softmax
public import Mathlib.Analysis.InnerProductSpace.Calculus

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

/-!
# Cross Entropy

One-hot cross entropy over the last axis, written as `target · log_softmax(logits)`.
-/

/-- Cross-entropy loss for logits and one-hot targets of shape `(m×n)`.

Forward:
`-(1/m) * ⟪target, log_softmax_last(logits)⟫`

This matches the common PyTorch `cross_entropy` convention with one-hot targets,
using `logSoftmax` on logits (numerically stable versus `log(softmax)` for floats; here
$\mathbb R$).
-/
def crossEntropyOneHotLast {Γ : List Shape} {m n : Nat}
    (logits target : Idx Γ (.dim m (.dim n .scalar))) : Node Γ Shape.scalar :=
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Spec.Shape.size s = m * n := by simp [s, Spec.Shape.size]
  let c : ℝ := (1 : ℝ) / (m : ℝ)
  Node.ofFn (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      let xMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logits xV)
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let logp : Vec (m * n) := LogSoftmaxLastAxis.forwardMN (m := m) (n := n) xMN
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
        (-c) * inner ℝ tMN logp)
    (jvp := fun xV dxV =>
      let xMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logits xV)
      let dxMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logits dxV)
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let dtMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target dxV)
      let logp : Vec (m * n) := LogSoftmaxLastAxis.forwardMN (m := m) (n := n) xMN
      let dlogp : Vec (m * n) := LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
        (-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp))
    (vjp := fun xV δV =>
      let i0 : Fin (Spec.Shape.size Shape.scalar) := ⟨0, by simp [Spec.Shape.size]⟩
      let δ0 : ℝ := δV i0
      let xMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logits xV)
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let logp : Vec (m * n) := LogSoftmaxLastAxis.forwardMN (m := m) (n := n) xMN
      let scale : ℝ := (-c) * δ0
      let dLogits : Vec (m * n) := scale • LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN tMN
      let dTarget : Vec (m * n) := scale • logp
      CtxVec.single (Γ := Γ) (s := s) logits (castVec hsz.symm dLogits) +
        CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget))
    (correct_inner := by
      intro xV dxV δV
      classical
      let s : Shape := .dim m (.dim n .scalar)
      let hsz : Spec.Shape.size s = m * n := by simp [s, Spec.Shape.size]
      let c : ℝ := (1 : ℝ) / (m : ℝ)
      let i0 : Fin (Spec.Shape.size Shape.scalar) := ⟨0, by simp [Spec.Shape.size]⟩
      let δ0 : ℝ := δV i0
      let xMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logits xV)
      let dxMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logits dxV)
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let dtMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target dxV)
      let logp : Vec (m * n) := LogSoftmaxLastAxis.forwardMN (m := m) (n := n) xMN
      let dlogp : Vec (m * n) := LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN
      let scale : ℝ := (-c) * δ0
      let dLogits : Vec (m * n) := scale • LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN tMN
      let dTarget : Vec (m * n) := scale • logp
      have hL :
          inner ℝ (vecOfFun (n := Spec.Shape.size Shape.scalar)
            (fun _ => (-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp))) δV
            =
          ((-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp)) * δ0 := by
        convert
          inner_scalarVec_left (a := (-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp)) (δ := δV)
          using 1
      have hA :
          inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logits (castVec hsz.symm dLogits)) =
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) logits dxV) (castVec hsz.symm dLogits) := by
        simpa using (CtxVec.inner_get_single (Γ := Γ) (s := s) logits dxV (castVec hsz.symm
          dLogits))
      have hB :
          inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget)) =
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) := by
        simpa using (CtxVec.inner_get_single (Γ := Γ) (s := s) target dxV (castVec hsz.symm
          dTarget))
      have hAc :
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) logits dxV) (castVec hsz.symm dLogits) =
            inner ℝ dxMN dLogits := by
        -- move to the flattened `(m*n)` space via `castVec` isometry
        have h := inner_castVec_castVec (h := hsz)
          (x := CtxVec.get (Γ := Γ) (s := s) logits dxV)
          (y := castVec hsz.symm dLogits)
        -- simplify the double cast on the RHS
        simpa [dxMN] using h.symm
      have hBc :
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) =
            inner ℝ dtMN dTarget := by
        have h := inner_castVec_castVec (h := hsz)
          (x := CtxVec.get (Γ := Γ) (s := s) target dxV)
          (y := castVec hsz.symm dTarget)
        simpa [dtMN] using h.symm
      have hsoft :
          inner ℝ tMN dlogp = inner ℝ dxMN (LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN tMN) :=
            by
        -- swap the inner, then use the rowwise log-softmax adjointness lemma
        have h := LogSoftmaxLastAxis.inner_jvpMN_vjp (m := m) (n := n) (x := xMN) (dx := dxMN) (δ :=
          tMN)
        -- h : ⟪dlogp, tMN⟫ = ⟪dxMN, vjpMN xMN tMN⟫
        simpa [dlogp, real_inner_comm] using h
      have hAterm :
          inner ℝ dxMN dLogits = scale * inner ℝ dxMN (LogSoftmaxLastAxis.vjpMN (m := m) (n := n)
            xMN tMN) := by
        simp [dLogits, scale, inner_smul_right]
      have hBterm :
          inner ℝ dtMN dTarget = scale * inner ℝ dtMN logp := by
        simp [dTarget, scale, inner_smul_right]
      -- combine
      calc
        inner ℝ
            (vecOfFun (n := Spec.Shape.size Shape.scalar)
              (fun _ => (-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp)))
            δV
            =
          ((-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp)) * δ0 := hL
        _ =
          scale * (inner ℝ tMN dlogp + inner ℝ dtMN logp) := by
            simp [scale, mul_assoc, mul_left_comm, mul_comm]
        _ =
          scale * inner ℝ dxMN (LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN tMN) + scale * inner
            ℝ dtMN logp := by
            simp [hsoft, mul_add]
        _ =
          inner ℝ dxMN dLogits + inner ℝ dtMN dTarget := by
            simp [hAterm, hBterm]
        _ =
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) logits dxV) (castVec hsz.symm dLogits) +
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) := by
            simp [hAc, hBc]
        _ =
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := s) logits (castVec hsz.symm dLogits) +
                CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget)) := by
            -- fold back through `CtxVec.single` and `inner_add_right`
            simp [inner_add_right, hA, hB])

/-- `NodeFDerivCorrect` for `crossEntropyOneHotLast` (one-hot targets; last-axis reduction). -/
def crossEntropyOneHotLastFderiv {Γ : List Shape} {m n : Nat}
    (logits target : Idx Γ (.dim m (.dim n .scalar))) :
    NodeFDerivCorrect (crossEntropyOneHotLast (Γ := Γ) (m := m) (n := n) logits target) := by
  classical
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Spec.Shape.size s = m * n := by simp [s, Spec.Shape.size]
  let logitsMN : CtxVec Γ → Vec (m * n) :=
    fun xV => castVec hsz (CtxVec.get (Γ := Γ) (s := s) logits xV)
  let targetMN : CtxVec Γ → Vec (m * n) :=
    fun xV => castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
  let logitsMNCLM : CtxVec Γ →L[ℝ] Vec (m * n) :=
    (Graph.castCLM (h := hsz)).comp (CtxVec.getCLM (Γ := Γ) (s := s) logits)
  let targetMNCLM : CtxVec Γ →L[ℝ] Vec (m * n) :=
    (Graph.castCLM (h := hsz)).comp (CtxVec.getCLM (Γ := Γ) (s := s) target)
  let c : ℝ := (1 : ℝ) / (m : ℝ)
  refine
    { deriv := fun xV =>
        let logpDeriv : CtxVec Γ →L[ℝ] Vec (m * n) :=
          (LogSoftmaxLastAxis.derivMN (m := m) (n := n) (logitsMN xV)).comp logitsMNCLM
        let innerDeriv : CtxVec Γ →L[ℝ] ℝ :=
          (fderivInnerCLM ℝ (targetMN xV, LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (logitsMN
            xV))).comp
            (targetMNCLM.prod logpDeriv)
        vecScalarCLM.comp ((-c) • innerDeriv)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    -- `logitsMN` and `targetMN` are linear.
    have hlogits : HasFDerivAt logitsMN logitsMNCLM xV := by
      have h := logitsMNCLM.hasFDerivAt (x := xV)
      have hfun : logitsMN = fun x => logitsMNCLM x := by
        funext x
        simp [logitsMN, logitsMNCLM, CtxVec.getCLM_apply, Graph.castCLM]
      exact h.congr_of_eventuallyEq hfun.eventuallyEq
    have htarget : HasFDerivAt targetMN targetMNCLM xV := by
      have h := targetMNCLM.hasFDerivAt (x := xV)
      have hfun : targetMN = fun x => targetMNCLM x := by
        funext x
        simp [targetMN, targetMNCLM, CtxVec.getCLM_apply, Graph.castCLM]
      exact h.congr_of_eventuallyEq hfun.eventuallyEq

    -- `logp = log_softmax_last(logitsMN)` derivative.
    have hlogp :
        HasFDerivAt (LogSoftmaxLastAxis.forwardMN (m := m) (n := n))
          (LogSoftmaxLastAxis.derivMN (m := m) (n := n) (logitsMN xV)) (logitsMN xV) :=
      LogSoftmaxLastAxis.hasFDerivAt_forwardMN (m := m) (n := n) (logitsMN xV)
    have hlogpComp :
        HasFDerivAt (fun x : CtxVec Γ => LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (logitsMN
          x))
          ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) (logitsMN xV)).comp logitsMNCLM) xV :=
      hlogp.comp xV hlogits

    -- Inner-product derivative.
    have hinter :
        HasFDerivAt (fun x : CtxVec Γ =>
            inner ℝ (targetMN x) (LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (logitsMN x)))
          ((fderivInnerCLM ℝ (targetMN xV, LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (logitsMN
            xV))).comp
            (targetMNCLM.prod ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) (logitsMN xV)).comp
              logitsMNCLM))) xV :=
      HasFDerivAt.inner (𝕜 := ℝ) (hf := htarget) (hg := hlogpComp)

    have hscaled := hinter.const_smul (-c)

    -- Wrap scalar into `Vec 1` using `vecScalarCLM`.
    have hwrap :
        HasFDerivAt (fun x : CtxVec Γ =>
            vecScalarCLM ((-c) • inner ℝ (targetMN x)
              (LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (logitsMN x))))
          (vecScalarCLM.comp ((-c) •
            (fderivInnerCLM ℝ (targetMN xV, LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (logitsMN
              xV))).comp
              (targetMNCLM.prod ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) (logitsMN xV)).comp
                logitsMNCLM)))) xV := by
      have hlin : HasFDerivAt (fun r : ℝ => vecScalarCLM r) vecScalarCLM
          ((-c) • inner ℝ (targetMN xV) (LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (logitsMN
            xV))) :=
        vecScalarCLM.hasFDerivAt (x := (-c) • inner ℝ (targetMN xV)
          (LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (logitsMN xV)))
      exact hlin.comp xV hscaled

    -- Identify `forwardVec` with the wrapped expression.
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := Shape.scalar)
            (crossEntropyOneHotLast (Γ := Γ) (m := m) (n := n) logits target))
          =
        fun x : CtxVec Γ =>
          vecScalarCLM ((-c) • inner ℝ (targetMN x)
            (LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (logitsMN x))) := by
      funext x
      ext i
      -- both sides reduce to the same scalar, packaged into `Vec 1`.
      let xMN : Vec (m * n) := logitsMN x
      let tMN : Vec (m * n) := targetMN x
      let logp : Vec (m * n) := LogSoftmaxLastAxis.forwardMN (m := m) (n := n) xMN
      have hL :
          (Node.forwardVec (Γ := Γ) (τ := Shape.scalar)
              (crossEntropyOneHotLast (Γ := Γ) (m := m) (n := n) logits target) x).ofLp i
            =
          (-c) * inner ℝ tMN logp := by
        simp only [crossEntropyOneHotLast, Node.forwardVec_ofFn]
        rw [vecOfFun_ofLp]
      have hR :
          (vecScalarCLM ((-c) • inner ℝ tMN logp)).ofLp i = (-c) * inner ℝ tMN logp := by
        simp [smul_eq_mul]
      simp [xMN, tMN, logp, hL]
    exact hwrap.congr_of_eventuallyEq hEq.eventuallyEq

  · intro xV dxV
    let xMN : Vec (m * n) := logitsMN xV
    let dxMN : Vec (m * n) := logitsMN dxV
    let tMN : Vec (m * n) := targetMN xV
    let dtMN : Vec (m * n) := targetMN dxV
    let logp : Vec (m * n) := LogSoftmaxLastAxis.forwardMN (m := m) (n := n) xMN
    let dlogp : Vec (m * n) := LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN
    -- Compute the scalar derivative via `fderivInnerCLM_apply`, then use `jvpMN_eq_derivMN`.
    have hjvp :
        LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN =
          (LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN :=
      LogSoftmaxLastAxis.jvpMN_eq_derivMN (m := m) (n := n) xMN dxMN
    have hdxLogits : logitsMNCLM dxV = dxMN := by
      simp [logitsMNCLM, logitsMN, dxMN, CtxVec.getCLM_apply, Graph.castCLM]
    have hdxTarget : targetMNCLM dxV = dtMN := by
      simp [targetMNCLM, targetMN, dtMN, CtxVec.getCLM_apply, Graph.castCLM]
    ext i
    -- LHS: node JVP scalar
    have hL :
        (Node.jvpVec (Γ := Γ) (τ := Shape.scalar)
            (crossEntropyOneHotLast (Γ := Γ) (m := m) (n := n) logits target) xV dxV).ofLp i
          =
        (-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp) := by
      simp only [crossEntropyOneHotLast, Node.jvpVec_ofFn]
      rw [vecOfFun_ofLp]
    -- RHS: derivative CLM applied to `dxV` (scalar packaged into `Vec 1`)
    let logpDeriv : CtxVec Γ →L[ℝ] Vec (m * n) :=
      (LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN).comp logitsMNCLM
    let innerDeriv : CtxVec Γ →L[ℝ] ℝ :=
      (fderivInnerCLM ℝ (tMN, logp)).comp (targetMNCLM.prod logpDeriv)
    have hlogpDeriv :
        logpDeriv dxV = (LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN := by
      simp [logpDeriv, ContinuousLinearMap.comp_apply, hdxLogits]
    have hinnerDeriv :
        innerDeriv dxV = inner ℝ tMN ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN) +
          inner ℝ dtMN logp := by
      -- `fderivInnerCLM_apply` gives the explicit bilinear formula
      simp [innerDeriv, ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply,
        hdxTarget, hlogpDeriv, fderivInnerCLM_apply]
    have hR :
        ((vecScalarCLM.comp ((-c) • innerDeriv)) dxV).ofLp i =
          (-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp) := by
      -- turn `•` into multiplication on scalars, and rewrite `derivMN` as `jvpMN`
      calc
        ((vecScalarCLM.comp ((-c) • innerDeriv)) dxV).ofLp i
            = ((-c) • innerDeriv dxV) := by
                simp []
        _ = (-c) * innerDeriv dxV := by simp [smul_eq_mul]
        _ = (-c) * (inner ℝ tMN ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN) + inner ℝ
          dtMN logp) := by
              simp [hinnerDeriv]
        _ = (-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp) := by
              have hderiv :
                  (LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN = dlogp := by
                simpa [dlogp] using hjvp.symm
              simp [hderiv]
    calc
      (Node.jvpVec (Γ := Γ) (τ := Shape.scalar)
            (crossEntropyOneHotLast (Γ := Γ) (m := m) (n := n) logits target) xV dxV).ofLp i
          =
          (-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp) := hL
      _ = ((vecScalarCLM.comp ((-c) • innerDeriv)) dxV).ofLp i := hR.symm

-- ---------------------------------------------------------------------------
-- Loss: negative log-likelihood (one-hot targets; log-probs input; mean over batch)
-- ---------------------------------------------------------------------------

end TapeNodes

end

end Autograd
end Proofs
