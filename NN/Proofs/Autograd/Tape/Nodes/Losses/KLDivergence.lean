/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Reductions
public import Mathlib.Analysis.InnerProductSpace.Calculus
public import NN.Proofs.Autograd.FDeriv.Elementwise

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

/-!
# KL Divergence

Batchmean KL-divergence for log-probability inputs and probability targets.
-/

/-- KL-divergence loss for `logProbs` and `target` probabilities of shape `(m×n)`.

Forward (batchmean reduction):
`(1/m) * Σ_{i,j} target[i,j] * (log(target[i,j]) - logProbs[i,j])`

This matches PyTorch `KLDivLoss` / `F.kl_div` with:
- `input` = log-probabilities,
- `target` = probabilities (not log-target),
- `reduction="batchmean"`.

We use the `Real.log`/`x⁻¹` derivative spec, so the node's VJP is correct on points
where `target` entries are nonzero.
-/
def klDivLast {Γ : List Shape} {m n : Nat}
    (logProbs target : Idx Γ (.dim m (.dim n .scalar))) : Node Γ Shape.scalar :=
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Spec.Shape.size s = m * n := by simp [s, Spec.Shape.size]
  let c : ℝ := (1 : ℝ) / (m : ℝ)
  Node.ofFn (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      let q : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let lp : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      let logq : Vec (m * n) := elemwiseVec (n := m * n) (f := Real.log) q
      let rhs : Vec (m * n) := logq - lp
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
        c * inner ℝ q rhs)
    (jvp := fun xV dxV =>
      let q : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let dq : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target dxV)
      let lp : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      let dlp : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs dxV)
      let logq : Vec (m * n) := elemwiseVec (n := m * n) (f := Real.log) q
      let dlogq : Vec (m * n) := vecOfFun (n := m * n) fun i => dq i * (q i)⁻¹
      let rhs : Vec (m * n) := logq - lp
      let drhs : Vec (m * n) := dlogq - dlp
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
        c * (inner ℝ dq rhs + inner ℝ q drhs))
    (vjp := fun xV δV =>
      let i0 : Fin (Spec.Shape.size Shape.scalar) := ⟨0, by simp [Spec.Shape.size]⟩
      let δ0 : ℝ := δV i0
      let q : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let lp : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      let logq : Vec (m * n) := elemwiseVec (n := m * n) (f := Real.log) q
      let rhs : Vec (m * n) := logq - lp
      let qInvMul : Vec (m * n) := vecOfFun (n := m * n) fun i => q i * (q i)⁻¹
      let scale : ℝ := c * δ0
      let dLogProbs : Vec (m * n) := vecOfFun (n := m * n) fun i => scale * (-q i)
      let dTarget : Vec (m * n) := vecOfFun (n := m * n) fun i => scale * (rhs i + qInvMul i)
      CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs) +
        CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget))
    (correct_inner := by
      intro xV dxV δV
      classical
      let s : Shape := .dim m (.dim n .scalar)
      let hsz : Spec.Shape.size s = m * n := by simp [s, Spec.Shape.size]
      let c : ℝ := (1 : ℝ) / (m : ℝ)
      let i0 : Fin (Spec.Shape.size Shape.scalar) := ⟨0, by simp [Spec.Shape.size]⟩
      let δ0 : ℝ := δV i0
      let q : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let dq : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target dxV)
      let lp : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      let dlp : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs dxV)
      let logq : Vec (m * n) := elemwiseVec (n := m * n) (f := Real.log) q
      let dlogq : Vec (m * n) := vecOfFun (n := m * n) fun i => dq i * (q i)⁻¹
      let rhs : Vec (m * n) := logq - lp
      let drhs : Vec (m * n) := dlogq - dlp
      let qInvMul : Vec (m * n) := vecOfFun (n := m * n) fun i => q i * (q i)⁻¹
      let scale : ℝ := c * δ0
      let dLogProbs : Vec (m * n) := vecOfFun (n := m * n) fun i => scale * (-q i)
      let dTarget : Vec (m * n) := vecOfFun (n := m * n) fun i => scale * (rhs i + qInvMul i)
      have hL :
          inner ℝ
              (vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
                c * (inner ℝ dq rhs + inner ℝ q drhs))
              δV
            =
          (c * (inner ℝ dq rhs + inner ℝ q drhs)) * δ0 := by
        convert
          inner_scalarVec_left (a := c * (inner ℝ dq rhs + inner ℝ q drhs)) (δ := δV)
          using 1
      have hA :
          inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs)) =
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) logProbs dxV) (castVec hsz.symm dLogProbs) := by
        simpa using
          (CtxVec.inner_get_single (Γ := Γ) (s := s) logProbs dxV (castVec hsz.symm dLogProbs))
      have hB :
          inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget)) =
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) := by
        simpa using
          (CtxVec.inner_get_single (Γ := Γ) (s := s) target dxV (castVec hsz.symm dTarget))
      have hAc :
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) logProbs dxV) (castVec hsz.symm dLogProbs) =
            inner ℝ dlp dLogProbs := by
        have h :=
          inner_castVec_castVec (h := hsz)
            (x := CtxVec.get (Γ := Γ) (s := s) logProbs dxV)
            (y := castVec hsz.symm dLogProbs)
        simpa [dlp] using h.symm
      have hBc :
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) =
            inner ℝ dq dTarget := by
        have h :=
          inner_castVec_castVec (h := hsz)
            (x := CtxVec.get (Γ := Γ) (s := s) target dxV)
            (y := castVec hsz.symm dTarget)
        simpa [dq] using h.symm
      have hAterm : inner ℝ dlp dLogProbs = scale * (- inner ℝ q dlp) := by
        simp [dLogProbs, scale, inner_eq_sum_mul, vecOfFun, mul_assoc, mul_left_comm,
          Finset.mul_sum, Finset.sum_neg_distrib, real_inner_comm]
      have hBterm :
          inner ℝ dq dTarget = scale * (inner ℝ dq rhs + inner ℝ q dlogq) := by
        -- expand `dTarget = scale • (rhs + qInvMul)` and use `qInvMul` to express `inner q dlogq`
        have hmul : inner ℝ q dlogq = inner ℝ dq qInvMul := by
          simp [inner_eq_sum_mul, dlogq, qInvMul, vecOfFun, mul_assoc, mul_comm]
        have hdTarget : dTarget = scale • (rhs + qInvMul) := by
          ext i
          simp [dTarget, scale, vecOfFun, smul_eq_mul, mul_add, mul_assoc]
        calc
          inner ℝ dq dTarget
              =
            inner ℝ dq (scale • (rhs + qInvMul)) := by
              simp [hdTarget]
          _ =
            scale * inner ℝ dq (rhs + qInvMul) := by
              -- avoid expanding `smul_add` before applying `inner_smul_right`
              simpa [smul_eq_mul] using (inner_smul_right (x := dq) (y := rhs + qInvMul) (r :=
                scale))
          _ =
            scale * (inner ℝ dq rhs + inner ℝ dq qInvMul) := by
              simp [inner_add_right, mul_add]
          _ =
            scale * (inner ℝ dq rhs + inner ℝ q dlogq) := by
              simp [hmul]
      have hqd :
          inner ℝ q drhs = inner ℝ q dlogq - inner ℝ q dlp := by
        simp [drhs, sub_eq_add_neg, inner_add_right, inner_neg_right]
      have hSubst :
          scale * (inner ℝ dq rhs + inner ℝ q dlogq - inner ℝ q dlp) =
            inner ℝ dlp dLogProbs + inner ℝ dq dTarget := by
        calc
          scale * (inner ℝ dq rhs + inner ℝ q dlogq - inner ℝ q dlp)
              =
            scale * (inner ℝ dq rhs + inner ℝ q dlogq) + scale * (-inner ℝ q dlp) := by
              simp [sub_eq_add_neg, mul_add, add_assoc]
          _ =
            scale * (-inner ℝ q dlp) + scale * (inner ℝ dq rhs + inner ℝ q dlogq) := by
              ac_rfl
          _ =
            inner ℝ dlp dLogProbs + inner ℝ dq dTarget := by
              have hAterm' : scale * (-inner ℝ q dlp) = inner ℝ dlp dLogProbs := by
                simpa using hAterm.symm
              have hBterm' : scale * (inner ℝ dq rhs + inner ℝ q dlogq) = inner ℝ dq dTarget := by
                simpa using hBterm.symm
              calc
                scale * (-inner ℝ q dlp) + scale * (inner ℝ dq rhs + inner ℝ q dlogq)
                    =
                  inner ℝ dlp dLogProbs + scale * (inner ℝ dq rhs + inner ℝ q dlogq) := by
                    simp [hAterm']
                _ =
                  inner ℝ dlp dLogProbs + inner ℝ dq dTarget := by
                    simp [hBterm']
      calc
        inner ℝ
            (vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
              c * (inner ℝ dq rhs + inner ℝ q drhs))
            δV
            =
          (c * (inner ℝ dq rhs + inner ℝ q drhs)) * δ0 := hL
        _ =
          scale * (inner ℝ dq rhs + inner ℝ q drhs) := by
            simp [scale, mul_assoc, mul_left_comm, mul_comm]
        _ =
          scale * (inner ℝ dq rhs + inner ℝ q dlogq - inner ℝ q dlp) := by
            simp [hqd, sub_eq_add_neg, add_assoc]
        _ =
          inner ℝ dlp dLogProbs + inner ℝ dq dTarget := by
            exact hSubst
        _ =
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) logProbs dxV) (castVec hsz.symm dLogProbs) +
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) := by
            have h1 :
                inner ℝ dlp dLogProbs + inner ℝ dq dTarget =
                  inner ℝ (CtxVec.get (Γ := Γ) (s := s) logProbs dxV) (castVec hsz.symm dLogProbs) +
                    inner ℝ dq dTarget := by
              simpa using congrArg (fun t => t + inner ℝ dq dTarget) hAc.symm
            have h2 :
                inner ℝ (CtxVec.get (Γ := Γ) (s := s) logProbs dxV) (castVec hsz.symm dLogProbs) +
                    inner ℝ dq dTarget =
                  inner ℝ (CtxVec.get (Γ := Γ) (s := s) logProbs dxV) (castVec hsz.symm dLogProbs) +
                    inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) :=
                      by
              simpa using
                congrArg
                  (fun t =>
                    inner ℝ (CtxVec.get (Γ := Γ) (s := s) logProbs dxV) (castVec hsz.symm dLogProbs)
                      + t)
                  hBc.symm
            exact h1.trans h2
        _ =
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs) +
                CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget)) := by
            -- rewrite each term using `inner_get_single`, then combine with additivity
            have hA' :
                inner ℝ (CtxVec.get (Γ := Γ) (s := s) logProbs dxV) (castVec hsz.symm dLogProbs) +
                    inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget)
                  =
                inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs))
                  +
                    inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) :=
                      by
              simpa using
                congrArg
                  (fun t =>
                    t +
                      inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget))
                  hA.symm
            have hB' :
                inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs))
                  +
                    inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget)
                  =
                inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs))
                  +
                    inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget))
                      := by
              simpa using
                congrArg
                  (fun t =>
                    inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm
                      dLogProbs)) + t)
                  hB.symm
            calc
              inner ℝ (CtxVec.get (Γ := Γ) (s := s) logProbs dxV) (castVec hsz.symm dLogProbs) +
                  inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget)
                  =
                inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs))
                  +
                    inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) :=
                      hA'
              _ =
                inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs))
                  +
                  inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget)) :=
                    hB'
              _ =
                inner ℝ dxV
                    (CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs) +
                      CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget)) := by
                simp [inner_add_right])

/-- Pointwise `NodeFDerivCorrectAt` for `klDivLast`, assuming `target` entries are nonzero. -/
def klDivLastFderivAt {Γ : List Shape} {m n : Nat}
    (logProbs target : Idx Γ (.dim m (.dim n .scalar))) (xV : CtxVec Γ)
    (ht :
      ∀ i : Fin (Spec.Shape.size (.dim m (.dim n .scalar))),
        CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) target xV i ≠ 0) :
    NodeFDerivCorrectAt (klDivLast (Γ := Γ) (m := m) (n := n) logProbs target) xV := by
  classical
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Spec.Shape.size s = m * n := by simp [s, Spec.Shape.size]
  let qMN : CtxVec Γ → Vec (m * n) := fun x => castVec hsz (CtxVec.get (Γ := Γ) (s := s) target x)
  let lpMN : CtxVec Γ → Vec (m * n) := fun x => castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs
    x)
  let qMNCLM : CtxVec Γ →L[ℝ] Vec (m * n) := (Graph.castCLM hsz).comp (CtxVec.getCLM (Γ := Γ) (s :=
    s) target)
  let lpMNCLM : CtxVec Γ →L[ℝ] Vec (m * n) := (Graph.castCLM hsz).comp (CtxVec.getCLM (Γ := Γ) (s :=
    s) logProbs)
  let c : ℝ := (1 : ℝ) / (m : ℝ)
  let logDerivVecCLM : Vec (m * n) →L[ℝ] Vec (m * n) :=
    elemwiseDerivCLM (n := m * n) (f' := fun x : ℝ => x⁻¹) (qMN xV)
  have hq0 :
      HasFDerivAt qMN qMNCLM xV := by
    have h0 : HasFDerivAt (fun x : CtxVec Γ => qMNCLM x) qMNCLM xV := qMNCLM.hasFDerivAt (x := xV)
    have hEq : qMN = fun x : CtxVec Γ => qMNCLM x := by
      funext x
      simp [qMN, qMNCLM, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply, Graph.castCLM]
    exact h0.congr_of_eventuallyEq hEq.eventuallyEq
  have hlp0 :
      HasFDerivAt lpMN lpMNCLM xV := by
    have h0 : HasFDerivAt (fun x : CtxVec Γ => lpMNCLM x) lpMNCLM xV := lpMNCLM.hasFDerivAt (x :=
      xV)
    have hEq : lpMN = fun x : CtxVec Γ => lpMNCLM x := by
      funext x
      simp [lpMN, lpMNCLM, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply, Graph.castCLM]
    exact h0.congr_of_eventuallyEq hEq.eventuallyEq

  have hlogq0 :
      HasFDerivAt (elemwiseVec (n := m * n) (f := Real.log)) logDerivVecCLM (qMN xV) := by
    -- pointwise `Real.log` derivative via the given nonzero hypothesis
    refine hasFDerivAt_elemwiseVec_at (n := m * n) (x := qMN xV) (f := Real.log) (f' := fun x =>
      x⁻¹) ?_
    intro i
    have : (qMN xV) i ≠ 0 := by
      -- `castVec` is just a reindexing of entries
      simpa [qMN] using ht (Fin.cast hsz.symm i)
    exact Real.hasDerivAt_log this
  let logqMN : CtxVec Γ → Vec (m * n) := fun x => elemwiseVec (n := m * n) (f := Real.log) (qMN x)
  let logqDeriv : CtxVec Γ →L[ℝ] Vec (m * n) := logDerivVecCLM.comp qMNCLM
  have hlogq :
      HasFDerivAt logqMN logqDeriv xV := by
    -- compose the vector-log derivative with `qMN`
    have hcomp := hlogq0.comp xV hq0
    refine hcomp.congr_of_eventuallyEq ?_
    exact Filter.Eventually.of_forall fun _ => rfl

  let rhsMN : CtxVec Γ → Vec (m * n) := fun x => logqMN x - lpMN x
  let rhsDeriv : CtxVec Γ →L[ℝ] Vec (m * n) := logqDeriv - lpMNCLM
  have hrhs :
      HasFDerivAt rhsMN rhsDeriv xV := by
    have hsub := hlogq.sub hlp0
    refine hsub.congr_of_eventuallyEq ?_
    exact Filter.Eventually.of_forall fun _ => rfl

  have hinter :
      HasFDerivAt (fun x => inner ℝ (qMN x) (rhsMN x))
        ((fderivInnerCLM ℝ (qMN xV, rhsMN xV)).comp (qMNCLM.prod rhsDeriv)) xV := by
    simpa using (HasFDerivAt.inner (𝕜 := ℝ) (hf := hq0) (hg := hrhs))

  have hscaled :
      HasFDerivAt (fun x => c • inner ℝ (qMN x) (rhsMN x))
        (c • (fderivInnerCLM ℝ (qMN xV, rhsMN xV)).comp (qMNCLM.prod rhsDeriv)) xV :=
    hinter.const_smul c
  have hwrap :
      HasFDerivAt (fun x : CtxVec Γ => vecScalarCLM (c • inner ℝ (qMN x) (rhsMN x)))
        (vecScalarCLM.comp (c • (fderivInnerCLM ℝ (qMN xV, rhsMN xV)).comp (qMNCLM.prod rhsDeriv)))
          xV := by
    have hlin : HasFDerivAt (fun r : ℝ => vecScalarCLM r) vecScalarCLM (c • inner ℝ (qMN xV) (rhsMN
      xV)) :=
      vecScalarCLM.hasFDerivAt (x := c • inner ℝ (qMN xV) (rhsMN xV))
    exact hlin.comp xV hscaled

  refine
    { deriv := vecScalarCLM.comp (c • (fderivInnerCLM ℝ (qMN xV, rhsMN xV)).comp (qMNCLM.prod
      rhsDeriv))
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · -- connect to the node's `forwardVec`
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := Shape.scalar)
            (klDivLast (Γ := Γ) (m := m) (n := n) logProbs target))
          =
        fun x : CtxVec Γ => vecScalarCLM (c • inner ℝ (qMN x) (rhsMN x)) := by
      funext x
      ext i
      simp [klDivLast, Node.forwardVec_ofFn, qMN, lpMN, logqMN, rhsMN, c, s,
        smul_eq_mul, mul_assoc, mul_comm]
    exact hwrap.congr_of_eventuallyEq hEq.eventuallyEq
  · intro dxV
    ext i
    let q : Vec (m * n) := qMN xV
    let dq : Vec (m * n) := qMN dxV
    let lp : Vec (m * n) := lpMN xV
    let dlp : Vec (m * n) := lpMN dxV
    let logq : Vec (m * n) := logqMN xV
    let dlogq : Vec (m * n) := vecOfFun (n := m * n) fun j => dq j * (q j)⁻¹
    let rhs : Vec (m * n) := logq - lp
    let drhs : Vec (m * n) := dlogq - dlp
    have hjvp :
        (Node.jvpVec (Γ := Γ) (τ := Shape.scalar)
            (klDivLast (Γ := Γ) (m := m) (n := n) logProbs target) xV dxV).ofLp i
          =
        c * (inner ℝ dq rhs + inner ℝ q drhs) := by
      have hscalar :
          ((EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin (Spec.Shape.size Shape.scalar))).symm
              (fun _ => c * (inner ℝ dq rhs + inner ℝ q drhs))).ofLp i
            =
          c * (inner ℝ dq rhs + inner ℝ q drhs) := by
        convert
          euclideanEquiv_symm_ofLp
            (n := Spec.Shape.size Shape.scalar)
            (f := fun _ : Fin (Spec.Shape.size Shape.scalar) =>
              c * (inner ℝ dq rhs + inner ℝ q drhs))
            (i := i) using 1
      simp only [klDivLast, Node.jvpVec_ofFn]
      simpa [qMN, lpMN, logqMN, c, s, q, dq, lp, dlp, logq, dlogq, rhs, drhs,
        vecOfFun, Spec.Shape.size] using hscalar
    let D : CtxVec Γ →L[ℝ] ℝ := c • (fderivInnerCLM ℝ (qMN xV, rhsMN xV)).comp (qMNCLM.prod
      rhsDeriv)
    have hD :
        D dxV = c * (inner ℝ dq rhs + inner ℝ q drhs) := by
      have hq : qMNCLM dxV = dq := by
        simp [qMNCLM, qMN, dq, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply, Graph.castCLM]
      have hlp : lpMNCLM dxV = dlp := by
        simp [lpMNCLM, lpMN, dlp, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply,
          Graph.castCLM]
      have hlog : logqDeriv dxV = dlogq := by
        ext j
        simp [logqDeriv, logDerivVecCLM, hq, elemwiseDerivCLM, vecOfFun, dlogq, q, dq,
          mul_comm]
      have hrhsDeriv : rhsDeriv dxV = drhs := by
        simp [rhsDeriv, drhs, hlog, hlp, sub_eq_add_neg]
      have hrhsMN : rhsMN xV = rhs := by
        simp [rhsMN, rhs, logqMN, lpMN, logq, lp, sub_eq_add_neg]
      -- now unfold the derivative of `inner` and the linear maps feeding it
      simp [D, smul_eq_mul, ContinuousLinearMap.comp_apply,
        ContinuousLinearMap.prod_apply, fderivInnerCLM_apply, hq, hrhsDeriv, hrhsMN,
        q, dq, rhs, drhs, add_comm]
    have hR : ((vecScalarCLM.comp D) dxV).ofLp i = D dxV := by
      simp [ContinuousLinearMap.comp_apply]
    calc
      (Node.jvpVec (Γ := Γ) (τ := Shape.scalar)
          (klDivLast (Γ := Γ) (m := m) (n := n) logProbs target) xV dxV).ofLp i
          =
        c * (inner ℝ dq rhs + inner ℝ q drhs) := hjvp
      _ = D dxV := hD.symm
      _ = ((vecScalarCLM.comp D) dxV).ofLp i := hR.symm

end TapeNodes

end

end Autograd
end Proofs
