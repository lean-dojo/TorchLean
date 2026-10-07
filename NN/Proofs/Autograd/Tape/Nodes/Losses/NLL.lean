/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Reductions
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Negative Log-Likelihood

Negative log-likelihood for one-hot targets when the input is already log-probabilities.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

/-- Negative log-likelihood loss for log-probabilities and one-hot targets of shape `(m×n)`.

Forward:
`-(1/m) * ⟪target, logProbs⟫`

This is the natural primitive loss that `crossEntropy` reduces to after `logSoftmax`.
-/
def nllOneHotLast {Γ : List Shape} {m n : Nat}
    (logProbs target : Idx Γ (.dim m (.dim n .scalar))) : Node Γ Shape.scalar :=
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Spec.Shape.size s = m * n := by simp [s, Spec.Shape.size]
  let c : ℝ := (1 : ℝ) / (m : ℝ)
  Node.ofFn (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let lpMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
        (-c) * inner ℝ tMN lpMN)
    (jvp := fun xV dxV =>
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let dtMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target dxV)
      let lpMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      let dlpMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs dxV)
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
        (-c) * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN))
    (vjp := fun xV δV =>
      let i0 : Fin (Spec.Shape.size Shape.scalar) := ⟨0, by simp [Spec.Shape.size]⟩
      let δ0 : ℝ := δV i0
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let lpMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      let scale : ℝ := (-c) * δ0
      let dLogProbs : Vec (m * n) := scale • tMN
      let dTarget : Vec (m * n) := scale • lpMN
      CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs) +
        CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget))
    (correct_inner := by
      intro xV dxV δV
      simp only [inner_scalarVec_left, inner_add_right, CtxVec.inner_get_single]
      rw [← inner_castVec_left hsz, ← inner_castVec_left hsz]
      simp only [inner_smul_right]
      rw [real_inner_comm (castVec hsz (CtxVec.get logProbs dxV))]
      ring)

/-- `NodeFDerivCorrect` for `nllOneHotLast` (negative log-likelihood with one-hot targets). -/
def nllOneHotLastFderiv {Γ : List Shape} {m n : Nat}
    (logProbs target : Idx Γ (.dim m (.dim n .scalar))) :
    NodeFDerivCorrect (nllOneHotLast (Γ := Γ) (m := m) (n := n) logProbs target) := by
  classical
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Spec.Shape.size s = m * n := by simp [s, Spec.Shape.size]
  let logpMN : CtxVec Γ → Vec (m * n) := fun xV => castVec hsz (CtxVec.get (Γ := Γ) (s := s)
    logProbs xV)
  let targetMN : CtxVec Γ → Vec (m * n) := fun xV => castVec hsz (CtxVec.get (Γ := Γ) (s := s)
    target xV)
  let logpMNCLM : CtxVec Γ →L[ℝ] Vec (m * n) := (Graph.castCLM hsz).comp (CtxVec.getCLM (Γ := Γ) (s
    := s) logProbs)
  let targetMNCLM : CtxVec Γ →L[ℝ] Vec (m * n) := (Graph.castCLM hsz).comp (CtxVec.getCLM (Γ := Γ)
    (s := s) target)
  let c : ℝ := (1 : ℝ) / (m : ℝ)
  refine
    { deriv := fun xV =>
        vecScalarCLM.comp
          ((-c) •
            (fderivInnerCLM ℝ (targetMN xV, logpMN xV)).comp
              (targetMNCLM.prod logpMNCLM))
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hlogp0 :
        HasFDerivAt (fun x : CtxVec Γ =>
            (Graph.castCLM hsz) (CtxVec.getCLM (Γ := Γ) (s := s) logProbs x)) logpMNCLM xV :=
      logpMNCLM.hasFDerivAt (x := xV)
    have hlogpEq :
        (fun x : CtxVec Γ =>
            (Graph.castCLM hsz) (CtxVec.getCLM (Γ := Γ) (s := s) logProbs x)) = logpMN := by
      funext x
      simp [logpMN, CtxVec.getCLM_apply, Graph.castCLM]
    have hlogp : HasFDerivAt logpMN logpMNCLM xV :=
      hlogp0.congr_of_eventuallyEq hlogpEq.symm.eventuallyEq

    have htarget0 :
        HasFDerivAt (fun x : CtxVec Γ =>
            (Graph.castCLM hsz) (CtxVec.getCLM (Γ := Γ) (s := s) target x)) targetMNCLM xV :=
      targetMNCLM.hasFDerivAt (x := xV)
    have htargetEq :
        (fun x : CtxVec Γ =>
            (Graph.castCLM hsz) (CtxVec.getCLM (Γ := Γ) (s := s) target x)) = targetMN := by
      funext x
      simp [targetMN, CtxVec.getCLM_apply, Graph.castCLM]
    have htarget : HasFDerivAt targetMN targetMNCLM xV :=
      htarget0.congr_of_eventuallyEq htargetEq.symm.eventuallyEq
    have hinter :
        HasFDerivAt (fun x => inner ℝ (targetMN x) (logpMN x))
          ((fderivInnerCLM ℝ (targetMN xV, logpMN xV)).comp (targetMNCLM.prod logpMNCLM)) xV := by
      simpa using (HasFDerivAt.inner (𝕜 := ℝ) (hf := htarget) (hg := hlogp))
    have hscaled :
        HasFDerivAt (fun x => (-c) • inner ℝ (targetMN x) (logpMN x))
          ((-c) • (fderivInnerCLM ℝ (targetMN xV, logpMN xV)).comp (targetMNCLM.prod logpMNCLM)) xV
            :=
      hinter.const_smul (-c)
    have hwrap :
        HasFDerivAt (fun x : CtxVec Γ => vecScalarCLM ((-c) • inner ℝ (targetMN x) (logpMN x)))
          (vecScalarCLM.comp ((-c) • (fderivInnerCLM ℝ (targetMN xV, logpMN xV)).comp
            (targetMNCLM.prod logpMNCLM))) xV := by
      have hlin : HasFDerivAt (fun r : ℝ => vecScalarCLM r) vecScalarCLM
          ((-c) • inner ℝ (targetMN xV) (logpMN xV)) :=
        vecScalarCLM.hasFDerivAt (x := (-c) • inner ℝ (targetMN xV) (logpMN xV))
      exact hlin.comp xV hscaled
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := Shape.scalar)
            (nllOneHotLast (Γ := Γ) (m := m) (n := n) logProbs target))
          =
        fun x : CtxVec Γ => vecScalarCLM ((-c) • inner ℝ (targetMN x) (logpMN x)) := by
      funext x
      ext i
      let tMN : Vec (m * n) := targetMN x
      let lpMN : Vec (m * n) := logpMN x
      have hL :
          (Node.forwardVec (Γ := Γ) (τ := Shape.scalar)
              (nllOneHotLast (Γ := Γ) (m := m) (n := n) logProbs target) x).ofLp i
            =
          (-c) * inner ℝ tMN lpMN := by
        simp only [nllOneHotLast, Node.forwardVec_ofFn]
        rw [vecOfFun_ofLp]
      have hR :
          (vecScalarCLM ((-c) • inner ℝ tMN lpMN)).ofLp i = (-c) * inner ℝ tMN lpMN := by
        simp [smul_eq_mul]
      simp [tMN, lpMN, hL]
    exact hwrap.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext i
    let tMN : Vec (m * n) := targetMN xV
    let dtMN : Vec (m * n) := targetMN dxV
    let lpMN : Vec (m * n) := logpMN xV
    let dlpMN : Vec (m * n) := logpMN dxV
    have hL :
        (Node.jvpVec (Γ := Γ) (τ := Shape.scalar)
            (nllOneHotLast (Γ := Γ) (m := m) (n := n) logProbs target) xV dxV).ofLp i
          =
        (-c) * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN) := by
      simp only [nllOneHotLast, Node.jvpVec_ofFn]
      rw [vecOfFun_ofLp]
    let D : CtxVec Γ →L[ℝ] ℝ :=
      (-c) • (fderivInnerCLM ℝ (tMN, lpMN)).comp (targetMNCLM.prod logpMNCLM)
    have hD :
        D dxV = (-c) * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN) := by
      simp [D, ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply, fderivInnerCLM_apply,
        tMN, dtMN, lpMN, dlpMN, logpMN, targetMN, logpMNCLM, targetMNCLM, Graph.castCLM,
          CtxVec.getCLM_apply,
        castVec, smul_eq_mul]
    have hR : ((vecScalarCLM.comp D) dxV).ofLp i = D dxV := by
      simp [ContinuousLinearMap.comp_apply]
    calc
      (Node.jvpVec (Γ := Γ) (τ := Shape.scalar)
          (nllOneHotLast (Γ := Γ) (m := m) (n := n) logProbs target) xV dxV).ofLp i
          =
        (-c) * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN) := hL
      _ = D dxV := hD.symm
      _ = ((vecScalarCLM.comp D) dxV).ofLp i := hR.symm

end TapeNodes

end

end Autograd
end Proofs
