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
# Binary Cross-Entropy With Logits

Stable BCE-with-logits loss and its tape-level derivative proof.
-/

/-- Binary cross-entropy with logits for same-shaped logits/targets.

Forward (mean reduction over all entries):
`(1/N) * Σ_i (softplus(logits_i) - target_i * logits_i)`

This matches PyTorch's `BCEWithLogitsLoss` with `reduction="mean"`,
and uses the stable identity `BCEWithLogits(x,t) = softplus(x) - t*x`.
-/
def bceWithLogits {Γ : List Shape} {s : Shape}
    (logits target : Idx Γ s) : Node Γ Shape.scalar :=
  let n : Nat := Spec.Shape.size s
  let c : ℝ := (1 : ℝ) / (n : ℝ)
  Node.ofFn (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      let x : Vec n := CtxVec.get (Γ := Γ) (s := s) logits xV
      let t : Vec n := CtxVec.get (Γ := Γ) (s := s) target xV
      let sp : Vec n := elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)) x
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
        c * (sumCLM (n := n) (sp - (vecOfFun (n := n) fun i => t i * x i))))
    (jvp := fun xV dxV =>
      let x : Vec n := CtxVec.get (Γ := Γ) (s := s) logits xV
      let dx : Vec n := CtxVec.get (Γ := Γ) (s := s) logits dxV
      let t : Vec n := CtxVec.get (Γ := Γ) (s := s) target xV
      let dt : Vec n := CtxVec.get (Γ := Γ) (s := s) target dxV
      let sp' : Vec n := elemwiseVec (n := n) (f := Activation.Math.softplusDerivSpec (α := ℝ)) x
      -- d/dx softplus(x) = sigmoid(x)
      let dsp : Vec n := vecOfFun (n := n) fun i => dx i * sp' i
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
        c * (sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i))))
    (vjp := fun xV δV =>
      let i0 : Fin (Spec.Shape.size Shape.scalar) := ⟨0, by simp [Spec.Shape.size]⟩
      let δ0 : ℝ := δV i0
      let x : Vec n := CtxVec.get (Γ := Γ) (s := s) logits xV
      let t : Vec n := CtxVec.get (Γ := Γ) (s := s) target xV
      let sp' : Vec n := elemwiseVec (n := n) (f := Activation.Math.softplusDerivSpec (α := ℝ)) x
      let scale : ℝ := c * δ0
      let dLogits : Vec n := vecOfFun (n := n) fun i => scale * (sp' i - t i)
      let dTarget : Vec n := vecOfFun (n := n) fun i => scale * (-x i)
      CtxVec.single (Γ := Γ) (s := s) logits dLogits +
        CtxVec.single (Γ := Γ) (s := s) target dTarget)
    (correct_inner := by
      intro xV dxV δV
      classical
      let n : Nat := Spec.Shape.size s
      let c : ℝ := (1 : ℝ) / (n : ℝ)
      let i0 : Fin (Spec.Shape.size Shape.scalar) := ⟨0, by simp [Spec.Shape.size]⟩
      let δ0 : ℝ := δV i0
      let x : Vec n := CtxVec.get (Γ := Γ) (s := s) logits xV
      let dx : Vec n := CtxVec.get (Γ := Γ) (s := s) logits dxV
      let t : Vec n := CtxVec.get (Γ := Γ) (s := s) target xV
      let dt : Vec n := CtxVec.get (Γ := Γ) (s := s) target dxV
      let sp' : Vec n := elemwiseVec (n := n) (f := Activation.Math.softplusDerivSpec (α := ℝ)) x
      let dsp : Vec n := vecOfFun (n := n) fun i => dx i * sp' i
      let scale : ℝ := c * δ0
      let dLogits : Vec n := vecOfFun (n := n) fun i => scale * (sp' i - t i)
      let dTarget : Vec n := vecOfFun (n := n) fun i => scale * (-x i)
      let jvpOut : Vec (Spec.Shape.size Shape.scalar) :=
        vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
          c * sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i))
      have hL :
          inner ℝ jvpOut δV =
            (c * sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i))) * δ0
              := by
        convert
          inner_scalarVec_left
            (a := c * sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i)))
            (δ := δV) using 1
      have hA :
          inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logits dLogits) =
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) logits dxV) dLogits := by
        simpa using (CtxVec.inner_get_single (Γ := Γ) (s := s) logits dxV dLogits)
      have hB :
          inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) target dTarget) =
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) dTarget := by
        simpa using (CtxVec.inner_get_single (Γ := Γ) (s := s) target dxV dTarget)
      have hAterm :
          inner ℝ dx dLogits = scale * (inner ℝ sp' dx - inner ℝ t dx) := by
        have hdLogits : dLogits = scale • (sp' - t) := by
          ext i
          simp [dLogits, vecOfFun, smul_eq_mul, sub_eq_add_neg, mul_add]
        -- `⟪dx, scale·(sp' - t)⟫ = scale·(⟪sp', dx⟫ - ⟪t, dx⟫)`
        calc
          inner ℝ dx dLogits
              =
            inner ℝ dx (scale • (sp' - t)) := by
              simp [hdLogits]
          _ =
            scale * inner ℝ dx (sp' - t) := by
              simp [inner_smul_right]
          _ =
            scale * (inner ℝ dx sp' - inner ℝ dx t) := by
              simp [sub_eq_add_neg, inner_add_right, inner_neg_right, mul_add]
          _ =
            scale * (inner ℝ sp' dx - inner ℝ t dx) := by
              simp [real_inner_comm]
      have hBterm :
          inner ℝ dt dTarget = scale * (- inner ℝ dt x) := by
        have hdTarget : dTarget = scale • (-x) := by
          ext i
          simp [dTarget, vecOfFun, smul_eq_mul]
        calc
          inner ℝ dt dTarget
              =
            inner ℝ dt (scale • (-x)) := by
              simp [hdTarget]
          _ =
            scale * inner ℝ dt (-x) := by
              simp [inner_smul_right]
          _ =
            scale * (- inner ℝ dt x) := by
              simp [inner_neg_right]
      have hsum :
          sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i))
            =
          inner ℝ sp' dx - inner ℝ dt x - inner ℝ t dx := by
        -- `sum` over coordinates of `dx*sig - (dt*x + t*dx)`
        simp [sumCLM_apply, dsp, sp', elemwiseVec, inner_eq_sum_mul, vecOfFun,
          sub_eq_add_neg, add_left_comm, add_comm, mul_comm,
          Finset.sum_add_distrib, Finset.sum_neg_distrib]
      calc
        inner ℝ jvpOut δV
            =
          (c * sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i))) * δ0 :=
            hL
        _ =
          scale * (inner ℝ sp' dx - inner ℝ dt x - inner ℝ t dx) := by
            have hmul :
                (c * sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i))) *
                  δ0
                  =
                scale * sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i))
                  := by
              -- rearrange multiplications; avoid `simp` lemmas that introduce disjunctions
              simp [scale, mul_assoc, mul_left_comm, mul_comm]
            calc
              (c * sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i))) *
                δ0
                  =
                scale * sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i))
                  := hmul
              _ =
                scale * (inner ℝ sp' dx - inner ℝ dt x - inner ℝ t dx) := by
                  simp [hsum]
        _ =
          inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logits dLogits) +
            inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) target dTarget) := by
            -- rewrite `CtxVec.get` projections to our named `dx`/`dt`
            have hx : inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logits dLogits) = inner ℝ dx
              dLogits := by
              simpa [dx] using hA
            have ht : inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) target dTarget) = inner ℝ dt
              dTarget := by
              simpa [dt] using hB
            have hx' :
                inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logits dLogits) =
                  scale * (inner ℝ sp' dx - inner ℝ t dx) := by
              simpa [hx] using hAterm
            have ht' :
                inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) target dTarget) =
                  scale * (- inner ℝ dt x) := by
              simpa [ht] using hBterm
            -- rearrange the scalar algebra
            have hsplit :
                scale * (inner ℝ sp' dx - inner ℝ dt x - inner ℝ t dx) =
                  scale * (inner ℝ sp' dx - inner ℝ t dx) + scale * (- inner ℝ dt x) := by
              simp [sub_eq_add_neg, add_assoc, add_left_comm, add_comm, mul_add]
            calc
              scale * (inner ℝ sp' dx - inner ℝ dt x - inner ℝ t dx)
                  =
                scale * (inner ℝ sp' dx - inner ℝ t dx) + scale * (- inner ℝ dt x) := hsplit
              _ =
                inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) logits dLogits) +
                  inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) target dTarget) := by
                  simp [hx', ht']
        _ =
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := s) logits dLogits +
                CtxVec.single (Γ := Γ) (s := s) target dTarget) := by
            simp [inner_add_right])

/-- `NodeFDerivCorrect` for `bceWithLogits` (binary cross-entropy with logits). -/
def bceWithLogitsFderiv {Γ : List Shape} {s : Shape} (logits target : Idx Γ s) :
    NodeFDerivCorrect (bceWithLogits (Γ := Γ) (s := s) logits target) := by
  classical
  let n : Nat := Spec.Shape.size s
  let logitsV : CtxVec Γ → Vec n := fun xV => CtxVec.get (Γ := Γ) (s := s) logits xV
  let targetV : CtxVec Γ → Vec n := fun xV => CtxVec.get (Γ := Γ) (s := s) target xV
  let logitsCLM : CtxVec Γ →L[ℝ] Vec n := CtxVec.getCLM (Γ := Γ) (s := s) logits
  let targetCLM : CtxVec Γ →L[ℝ] Vec n := CtxVec.getCLM (Γ := Γ) (s := s) target
  let c : ℝ := (1 : ℝ) / (n : ℝ)
  refine
    { deriv := fun xV =>
        let spDeriv : CtxVec Γ →L[ℝ] Vec n :=
          (elemwiseDerivCLM (n := n) (f' := Activation.Math.softplusDerivSpec (α := ℝ)) (logitsV
            xV)).comp
            logitsCLM
        let sumSpDeriv : CtxVec Γ →L[ℝ] ℝ := (sumCLM (n := n)).comp spDeriv
        let innerDeriv : CtxVec Γ →L[ℝ] ℝ :=
          (fderivInnerCLM ℝ (targetV xV, logitsV xV)).comp (targetCLM.prod logitsCLM)
        vecScalarCLM.comp (c • (sumSpDeriv - innerDeriv))
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hlogits0 :
        HasFDerivAt (fun x : CtxVec Γ => logitsCLM x) logitsCLM xV :=
      logitsCLM.hasFDerivAt (x := xV)
    have hlogitsEq : (fun x : CtxVec Γ => logitsCLM x) = logitsV := by
      funext x
      dsimp [logitsCLM, logitsV]
      exact CtxVec.getCLM_apply (Γ := Γ) (s := s) logits x
    have hlogits : HasFDerivAt logitsV logitsCLM xV :=
      hlogits0.congr_of_eventuallyEq hlogitsEq.symm.eventuallyEq

    have htarget0 :
        HasFDerivAt (fun x : CtxVec Γ => targetCLM x) targetCLM xV :=
      targetCLM.hasFDerivAt (x := xV)
    have htargetEq : (fun x : CtxVec Γ => targetCLM x) = targetV := by
      funext x
      dsimp [targetCLM, targetV]
      exact CtxVec.getCLM_apply (Γ := Γ) (s := s) target x
    have htarget : HasFDerivAt targetV targetCLM xV :=
      htarget0.congr_of_eventuallyEq htargetEq.symm.eventuallyEq

    have hsoftplus :
        HasFDerivAt (elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)))
          (elemwiseDerivCLM (n := n) (f' := Activation.Math.softplusDerivSpec (α := ℝ)) (logitsV
            xV))
          (logitsV xV) :=
      hasFDerivAt_elemwiseVec (n := n) (x := logitsV xV)
        (f := Activation.Math.softplusSpec (α := ℝ))
        (f' := Activation.Math.softplusDerivSpec (α := ℝ))
        (fun z => Proofs.softplus_deriv_correct (x := z))
    have hsp :
        HasFDerivAt (fun x => elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ))
          (logitsV x))
          ((elemwiseDerivCLM (n := n) (f' := Activation.Math.softplusDerivSpec (α := ℝ)) (logitsV
            xV)).comp logitsCLM)
          xV := by
      have hcomp := hsoftplus.comp xV hlogits
      refine hcomp.congr_of_eventuallyEq ?_
      exact Filter.Eventually.of_forall fun _ => rfl
    have hsumSp :
        HasFDerivAt (fun x => sumCLM (n := n)
              (elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)) (logitsV x)))
          ((sumCLM (n := n)).comp
              ((elemwiseDerivCLM (n := n) (f' := Activation.Math.softplusDerivSpec (α := ℝ))
                (logitsV xV)).comp logitsCLM))
          xV := by
      have hsum :
          HasFDerivAt (fun v : Vec n => sumCLM (n := n) v) (sumCLM (n := n))
            (elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)) (logitsV xV)) :=
        (sumCLM (n := n)).hasFDerivAt (x := elemwiseVec (n := n) (f := Activation.Math.softplusSpec
          (α := ℝ)) (logitsV xV))
      exact hsum.comp xV hsp

    have hinter :
        HasFDerivAt (fun x => inner ℝ (targetV x) (logitsV x))
          ((fderivInnerCLM ℝ (targetV xV, logitsV xV)).comp (targetCLM.prod logitsCLM)) xV := by
      simpa using (HasFDerivAt.inner (𝕜 := ℝ) (hf := htarget) (hg := hlogits))

    have hdiff :
        HasFDerivAt (fun x => sumCLM (n := n)
              (elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)) (logitsV x)) -
              inner ℝ (targetV x) (logitsV x))
          (((sumCLM (n := n)).comp
              ((elemwiseDerivCLM (n := n) (f' := Activation.Math.softplusDerivSpec (α := ℝ))
                (logitsV xV)).comp logitsCLM)) -
            ((fderivInnerCLM ℝ (targetV xV, logitsV xV)).comp (targetCLM.prod logitsCLM))) xV :=
      hsumSp.sub hinter

    have hscaled :
        HasFDerivAt (fun x => c •
              (sumCLM (n := n)
                (elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)) (logitsV x)) -
                inner ℝ (targetV x) (logitsV x)))
          (c •
            (((sumCLM (n := n)).comp
                ((elemwiseDerivCLM (n := n) (f' := Activation.Math.softplusDerivSpec (α := ℝ))
                  (logitsV xV)).comp logitsCLM)) -
              ((fderivInnerCLM ℝ (targetV xV, logitsV xV)).comp (targetCLM.prod logitsCLM)))) xV :=
      hdiff.const_smul c

    have hwrap :
        HasFDerivAt
          (fun x : CtxVec Γ =>
            vecScalarCLM
              (c • (sumCLM (n := n)
                (elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)) (logitsV x)) -
                inner ℝ (targetV x) (logitsV x))))
          (vecScalarCLM.comp
            (c •
              (((sumCLM (n := n)).comp
                  ((elemwiseDerivCLM (n := n) (f' := Activation.Math.softplusDerivSpec (α := ℝ))
                    (logitsV xV)).comp logitsCLM)) -
                ((fderivInnerCLM ℝ (targetV xV, logitsV xV)).comp (targetCLM.prod logitsCLM))))) xV
                  := by
      have hlin :
          HasFDerivAt (fun r : ℝ => vecScalarCLM r) vecScalarCLM
            (c • (sumCLM (n := n)
              (elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)) (logitsV xV)) -
              inner ℝ (targetV xV) (logitsV xV))) :=
        vecScalarCLM.hasFDerivAt (x := c • (sumCLM (n := n)
          (elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)) (logitsV xV)) -
          inner ℝ (targetV xV) (logitsV xV)))
      exact hlin.comp xV hscaled

    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := Shape.scalar)
            (bceWithLogits (Γ := Γ) (s := s) logits target))
          =
        fun x : CtxVec Γ =>
          vecScalarCLM (c •
            ((sumCLM (n := n))
                (elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)) (logitsV x)) -
              inner ℝ (targetV x) (logitsV x))) := by
      funext x
      ext i
      -- Expand the node forward definition, then rewrite `sumCLM` and `inner` into explicit sums.
      simp [bceWithLogits, Node.forwardVec_ofFn, logitsV, targetV, c, sumCLM_apply,
        elemwiseVec, inner_eq_sum_mul, vecOfFun, sub_eq_add_neg,
        smul_eq_mul, mul_assoc, mul_comm, add_comm,
        Finset.sum_add_distrib, Finset.sum_neg_distrib, n]
    exact hwrap.congr_of_eventuallyEq hEq.eventuallyEq

  · intro xV dxV
    ext i
    let x : Vec n := logitsV xV
    let dx : Vec n := logitsV dxV
    let t : Vec n := targetV xV
    let dt : Vec n := targetV dxV
    let sp' : Vec n := elemwiseVec (n := n) (f := Activation.Math.softplusDerivSpec (α := ℝ)) x
    have hjvp :
        (Node.jvpVec (Γ := Γ) (τ := Shape.scalar)
            (bceWithLogits (Γ := Γ) (s := s) logits target) xV dxV).ofLp i
          =
        c * sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j) -
            (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)) := by
      have hscalar :
          ((EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin (Spec.Shape.size Shape.scalar))).symm
              (fun _ =>
                c * sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j) -
                  (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)))).ofLp i
            =
          c * sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j) -
            (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)) := by
        convert
          euclideanEquiv_symm_ofLp
            (n := Spec.Shape.size Shape.scalar)
            (f := fun _ : Fin (Spec.Shape.size Shape.scalar) =>
              c * sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j) -
                (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)))
            (i := i) using 1
      simp only [bceWithLogits, Node.jvpVec_ofFn]
      simpa [x, dx, t, dt, sp', logitsV, targetV, c, elemwiseVec, vecOfFun,
        Spec.Shape.size, n] using hscalar
    let spDeriv : CtxVec Γ →L[ℝ] Vec n :=
      (elemwiseDerivCLM (n := n) (f' := Activation.Math.softplusDerivSpec (α := ℝ)) x).comp
        logitsCLM
    let sumSpDeriv : CtxVec Γ →L[ℝ] ℝ := (sumCLM (n := n)).comp spDeriv
    let innerDeriv : CtxVec Γ →L[ℝ] ℝ :=
      (fderivInnerCLM ℝ (t, x)).comp (targetCLM.prod logitsCLM)
    let D : CtxVec Γ →L[ℝ] ℝ := c • (sumSpDeriv - innerDeriv)
    have hD :
        D dxV =
          c * sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j) -
              (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)) := by
      have hlogits : logitsCLM dxV = dx := by
        dsimp [logitsCLM, logitsV, dx]
        exact CtxVec.getCLM_apply (Γ := Γ) (s := s) logits dxV
      have htarget : targetCLM dxV = dt := by
        dsimp [targetCLM, targetV, dt]
        exact CtxVec.getCLM_apply (Γ := Γ) (s := s) target dxV
      have hspDeriv :
          spDeriv dxV = vecOfFun (n := n) (fun j => dx j * sp' j) := by
        -- expand `elemwiseDerivCLM` coordinatewise
        simp [spDeriv, ContinuousLinearMap.comp_apply, elemwiseDerivCLM, elemwiseVec, vecOfFun,
          hlogits, sp', dx]
      have hsumSp :
          sumSpDeriv dxV = sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j)) := by
        simp [sumSpDeriv, ContinuousLinearMap.comp_apply, hspDeriv]
      have hinter :
          innerDeriv dxV = inner ℝ dt x + inner ℝ t dx := by
        -- `fderivInnerCLM_apply` yields the bilinear derivative formula; reorder with
        -- commutativity.
        simp [innerDeriv, ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply,
          fderivInnerCLM_apply, htarget, hlogits, x, dx, t, dt, add_comm]
      -- combine: `sumSp - inner = sum(A) - (sum(B) + sum(C)) = sum(A - (B+C))`
      let A : Vec n := vecOfFun (n := n) fun j => dx j * sp' j
      let B : Vec n := vecOfFun (n := n) fun j => dt j * x j
      let C : Vec n := vecOfFun (n := n) fun j => t j * dx j
      have hinnerB : inner ℝ dt x = sumCLM (n := n) B := by
        simp [B, inner_eq_sum_mul, sumCLM_apply, vecOfFun, mul_comm]
      have hinnerC : inner ℝ t dx = sumCLM (n := n) C := by
        simp [C, inner_eq_sum_mul, sumCLM_apply, vecOfFun, mul_comm]
      have hsumABC :
          sumCLM (n := n) A - (inner ℝ dt x + inner ℝ t dx) =
            sumCLM (n := n) (A - (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)) := by
        -- unfold `A` and use linearity of `sumCLM` plus the inner-as-sum facts
        simp [A, B, C, hinnerB, hinnerC, sumCLM_apply, vecOfFun,
          sub_eq_add_neg, add_comm, mul_comm,
          Finset.sum_add_distrib, Finset.sum_neg_distrib]
      calc
        D dxV
            = c * (sumSpDeriv dxV - innerDeriv dxV) := by
                simp [D, smul_eq_mul]
        _ = c * (sumCLM (n := n) A - (inner ℝ dt x + inner ℝ t dx)) := by
              simp [hsumSp, hinter, A]
        _ = c * sumCLM (n := n) (A - (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)) := by
              simp [hsumABC]
        _ = c * sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j) -
              (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)) := by
              simp [A]
    have hR : ((vecScalarCLM.comp D) dxV).ofLp i = D dxV := by
      simp [ContinuousLinearMap.comp_apply]
    calc
      (Node.jvpVec (Γ := Γ) (τ := Shape.scalar)
          (bceWithLogits (Γ := Γ) (s := s) logits target) xV dxV).ofLp i
          =
        c * sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j) -
            (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)) := hjvp
      _ = D dxV := hD.symm
      _ = ((vecScalarCLM.comp D) dxV).ofLp i := hR.symm

-- ---------------------------------------------------------------------------
-- Loss: KL divergence (log-probs input, probs target; last axis; mean over batch)
-- ---------------------------------------------------------------------------

end TapeNodes

end

end Autograd
end Proofs
