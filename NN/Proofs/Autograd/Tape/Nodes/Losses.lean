/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Reductions

/-!
# Loss tape nodes

Differentiable loss nodes used by training and verification examples: MSE, cross entropy with one-hot
targets, negative log likelihood, BCE-with-logits, and KL divergence.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

-- ---------------------------------------------------------------------------
-- Loss: scalar mean squared error
-- ---------------------------------------------------------------------------

/-- Mean-squared-error loss node: `c * ‖yhat - target‖^2`, with `c = 1 / size(s)`. -/
def mseLoss {Γ : List Shape} {s : Shape} (yhat target : Idx Γ s) : Node Γ Shape.scalar :=
  let n : Nat := Shape.size s
  let c : ℝ := (1 : ℝ) / (n : ℝ)
  Node.ofVec (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      vecOfFun (n := Shape.size Shape.scalar) fun _ =>
        c * ‖(CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)‖ ^ 2)
    (jvp := fun xV dxV =>
      let diff := (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
      let ddiff := (CtxVec.get (Γ := Γ) (s := s) yhat dxV) - (CtxVec.get (Γ := Γ) (s := s) target
        dxV)
      vecOfFun (n := Shape.size Shape.scalar) fun _ => c * (2 * inner ℝ diff ddiff))
    (vjp := fun xV δV =>
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
      let δ0 : ℝ := δV i0
      let diff := (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
      let scale : ℝ := δ0 * (2 * c)
      let dYhat : Vec (Shape.size s) := scale • diff
      let dTarget : Vec (Shape.size s) := -dYhat
      CtxVec.single (Γ := Γ) (s := s) yhat dYhat + CtxVec.single (Γ := Γ) (s := s) target dTarget)
    (correct_inner := by
      intro xV dxV δV
      classical
      let n : Nat := Shape.size s
      let c : ℝ := (1 : ℝ) / (n : ℝ)
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
      let δ0 : ℝ := δV i0
      let diff := (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
      let dy := (CtxVec.get (Γ := Γ) (s := s) yhat dxV)
      let dt := (CtxVec.get (Γ := Γ) (s := s) target dxV)
      let ddiff := dy - dt
      let scale : ℝ := δ0 * (2 * c)
      let dYhat : Vec (Shape.size s) := scale • diff
      let dTarget : Vec (Shape.size s) := -dYhat
      have hL :
          inner ℝ (vecOfFun (n := Shape.size Shape.scalar) (fun _ => c * (2 * inner ℝ diff ddiff)))
            δV
            =
          (c * (2 * inner ℝ diff ddiff)) * δ0 := by
        convert inner_scalarVec_left (a := c * (2 * inner ℝ diff ddiff)) (δ := δV) using 1
      have hA :
          inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) yhat dYhat) =
            inner ℝ dy dYhat := by
        simpa using (CtxVec.inner_get_single (Γ := Γ) (s := s) yhat dxV dYhat)
      have hB :
          inner ℝ dxV (CtxVec.single (Γ := Γ) (s := s) target dTarget) =
            inner ℝ dt dTarget := by
        simpa using (CtxVec.inner_get_single (Γ := Γ) (s := s) target dxV dTarget)
      have hR :
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := s) yhat dYhat + CtxVec.single (Γ := Γ) (s := s) target
                dTarget)
            =
          (inner ℝ dy dYhat) + (inner ℝ dt dTarget) := by
        simp [inner_add_right, hA, hB]
      -- Now simplify the RHS using `dTarget = -dYhat` and `ddiff = dy - dt`.
      have hR' :
          (inner ℝ dy dYhat) + (inner ℝ dt dTarget) =
            scale * inner ℝ ddiff diff := by
        have hdd : inner ℝ ddiff diff = inner ℝ dy diff - inner ℝ dt diff := by
          simp [ddiff, inner_sub_left]
        -- unfold `dYhat`/`dTarget`, and reduce to a ring identity
        simp [dYhat, dTarget, hdd, inner_smul_right, inner_neg_right, sub_eq_add_neg]
        ring
      -- Relate `scale * ⟪ddiff,diff⟫` to LHS form.
      have hfinal :
          (c * (2 * inner ℝ diff ddiff)) * δ0 = scale * inner ℝ ddiff diff := by
        simp [scale, mul_assoc, mul_left_comm, mul_comm, real_inner_comm]
      -- Finish.
      calc
        inner ℝ (vecOfFun (n := Shape.size Shape.scalar) (fun _ => c * (2 * inner ℝ diff ddiff))) δV
            = (c * (2 * inner ℝ diff ddiff)) * δ0 := hL
        _ = scale * inner ℝ ddiff diff := hfinal
        _ = inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := s) yhat dYhat + CtxVec.single (Γ := Γ) (s := s) target
                dTarget) := by
              simp [hR, hR'] )

/-- `NodeFDerivCorrect` for `mse_loss`. -/
def mseLossFderiv {Γ : List Shape} {s : Shape} (yhat target : Idx Γ s) :
    NodeFDerivCorrect (mseLoss (Γ := Γ) (s := s) yhat target) := by
  classical
  let n : Nat := Shape.size s
  let c : ℝ := (1 : ℝ) / (n : ℝ)
  let diffDeriv : CtxVec Γ →L[ℝ] Vec (Shape.size s) :=
    (CtxVec.getCLM (Γ := Γ) (s := s) yhat) - (CtxVec.getCLM (Γ := Γ) (s := s) target)
  refine
    { deriv := fun xV =>
        let diffV : Vec (Shape.size s) :=
          (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
        vecScalarCLM.comp (c • (2 • (innerSL ℝ diffV)).comp diffDeriv)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    -- `get` projections are CLMs.
    have hgetY :
        HasFDerivAt (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) yhat x)
          (CtxVec.getCLM (Γ := Γ) (s := s) yhat) xV := by
      have h := (CtxVec.getCLM (Γ := Γ) (s := s) yhat).hasFDerivAt (x := xV)
      have hfun :
          (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) yhat x)
            =
          fun x : CtxVec Γ => (CtxVec.getCLM (Γ := Γ) (s := s) yhat) x := by
        funext x
        exact (CtxVec.getCLM_apply (Γ := Γ) (s := s) yhat x).symm
      exact h.congr_of_eventuallyEq hfun.eventuallyEq
    have hgetT :
        HasFDerivAt (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) target x)
          (CtxVec.getCLM (Γ := Γ) (s := s) target) xV := by
      have h := (CtxVec.getCLM (Γ := Γ) (s := s) target).hasFDerivAt (x := xV)
      have hfun :
          (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) target x)
            =
          fun x : CtxVec Γ => (CtxVec.getCLM (Γ := Γ) (s := s) target) x := by
        funext x
        exact (CtxVec.getCLM_apply (Γ := Γ) (s := s) target x).symm
      exact h.congr_of_eventuallyEq hfun.eventuallyEq
    have hdiff :
        HasFDerivAt
          (fun x : CtxVec Γ =>
            (CtxVec.get (Γ := Γ) (s := s) yhat x) - (CtxVec.get (Γ := Γ) (s := s) target x))
          diffDeriv xV := by
      simpa [diffDeriv] using hgetY.sub hgetT
    -- `‖diff ·‖^2` and scale by `c`.
    have hsq :
        HasFDerivAt
          (fun x : CtxVec Γ =>
            ‖(CtxVec.get (Γ := Γ) (s := s) yhat x) - (CtxVec.get (Γ := Γ) (s := s) target x)‖ ^ 2)
          (2 • (innerSL ℝ ((CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s)
            target xV))).comp diffDeriv)
          xV := by
      simpa using hdiff.norm_sq
    have hscaled :=
      (hsq.const_smul c)
    -- wrap scalar into `Vec 1` using `vecScalarCLM`.
    let g : CtxVec Γ → ℝ :=
      fun x =>
        c • (‖(CtxVec.get (Γ := Γ) (s := s) yhat x) - (CtxVec.get (Γ := Γ) (s := s) target x)‖ ^ 2)
    have hwrap :
        HasFDerivAt (fun x : CtxVec Γ => vecScalarCLM (g x))
          (vecScalarCLM.comp (c • (2 • (innerSL ℝ ((CtxVec.get (Γ := Γ) (s := s) yhat xV) -
              (CtxVec.get (Γ := Γ) (s := s) target xV))).comp diffDeriv))) xV := by
      have hlin : HasFDerivAt (fun r : ℝ => vecScalarCLM r) vecScalarCLM (g xV) :=
        vecScalarCLM.hasFDerivAt (x := g xV)
      exact hlin.comp xV hscaled
    -- identify `forwardVec` with the wrapped form.
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := Shape.scalar) (mseLoss (Γ := Γ) (s := s) yhat target))
          =
        fun x : CtxVec Γ => vecScalarCLM (g x) := by
      funext x
      ext i
      fin_cases i
      let k0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
      -- Reduce to `r = (vecScalarCLM r).ofLp 0`, then close by `vecScalarCLM_ofLp`.
      have hbase :
          (↑(Shape.size s))⁻¹ *
              ‖CtxVec.get (Γ := Γ) (s := s) yhat x - CtxVec.get (Γ := Γ) (s := s) target x‖ ^ 2 =
            (vecScalarCLM
                  ((↑(Shape.size s))⁻¹ *
                    ‖CtxVec.get (Γ := Γ) (s := s) yhat x - CtxVec.get (Γ := Γ) (s := s) target x‖ ^
                      2)).ofLp
              k0 := by
        simp
      -- Now unfold both sides to this canonical scalar.
      simpa [mseLoss, Node.forwardVec_ofVec, g, c, n, Shape.size, smul_eq_mul, div_eq_mul_inv]
        using hbase
    exact hwrap.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    let diffV : Vec (Shape.size s) :=
      (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
    let D0 : CtxVec Γ →L[ℝ] ℝ :=
      (c • (2 • (innerSL ℝ diffV)).comp diffDeriv)
    let ddiffV : Vec (Shape.size s) :=
      (CtxVec.get (Γ := Γ) (s := s) yhat dxV) - (CtxVec.get (Γ := Γ) (s := s) target dxV)
    have hdiffDeriv : diffDeriv dxV = ddiffV := by
      simp [diffDeriv, ddiffV, CtxVec.getCLM_apply, ContinuousLinearMap.sub_apply]
    -- Avoid expanding `inner` on differences; both sides are the same scalar packaged into `Vec 1`.
    have hD0 :
        D0 dxV = c * (2 * inner ℝ diffV ddiffV) := by
      -- `innerSL` is `y ↦ ⟪diffV,y⟫`
      simp [D0, hdiffDeriv, innerSL_apply_apply, ContinuousLinearMap.comp_apply,
        ContinuousLinearMap.smul_apply,
        smul_eq_mul, mul_left_comm]
    -- Finish by extensionality on `Vec 1`.
    ext i
    -- both sides are constant in `i`; rewrite the RHS scalar via `hD0`.
    have hL :
        (Node.jvpVec (Γ := Γ) (τ := Shape.scalar) (mseLoss (Γ := Γ) (s := s) yhat target) xV
          dxV).ofLp i =
          c * (2 * inner ℝ diffV ddiffV) := by
      simp [mseLoss, Node.jvpVec_ofVec, diffV, ddiffV, Shape.size, c, n, div_eq_mul_inv]
    have hR : ((vecScalarCLM.comp D0) dxV).ofLp i = D0 dxV := by
      simp [ContinuousLinearMap.comp_apply]
    calc
      (Node.jvpVec (Γ := Γ) (τ := Shape.scalar) (mseLoss (Γ := Γ) (s := s) yhat target) xV
        dxV).ofLp i
          = c * (2 * inner ℝ diffV ddiffV) := hL
      _ = D0 dxV := hD0.symm
      _ = ((vecScalarCLM.comp D0) dxV).ofLp i := hR.symm

-- ---------------------------------------------------------------------------
-- Loss: cross entropy (one-hot targets; last-axis softmax; mean over batch)
-- ---------------------------------------------------------------------------

/-- Cross-entropy loss for logits and one-hot targets of shape `(m×n)`.

Forward:
`-(1/m) * ⟪target, log_softmax_last(logits)⟫`

This matches the common PyTorch `cross_entropy` convention with one-hot targets,
using `log_softmax` on logits (numerically stable vs `log(softmax)` for floats; here ℝ).
-/
def crossEntropyOneHotLast {Γ : List Shape} {m n : Nat}
    (logits target : Idx Γ (.dim m (.dim n .scalar))) : Node Γ Shape.scalar :=
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Shape.size s = m * n := by simp [s, Shape.size]
  let c : ℝ := (1 : ℝ) / (m : ℝ)
  Node.ofVec (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      let xMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logits xV)
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let logp : Vec (m * n) := LogSoftmaxLastAxis.forwardMN (m := m) (n := n) xMN
      vecOfFun (n := Shape.size Shape.scalar) fun _ =>
        (-c) * inner ℝ tMN logp)
    (jvp := fun xV dxV =>
      let xMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logits xV)
      let dxMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logits dxV)
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let dtMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target dxV)
      let logp : Vec (m * n) := LogSoftmaxLastAxis.forwardMN (m := m) (n := n) xMN
      let dlogp : Vec (m * n) := LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN
      vecOfFun (n := Shape.size Shape.scalar) fun _ =>
        (-c) * (inner ℝ tMN dlogp + inner ℝ dtMN logp))
    (vjp := fun xV δV =>
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
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
      let hsz : Shape.size s = m * n := by simp [s, Shape.size]
      let c : ℝ := (1 : ℝ) / (m : ℝ)
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
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
          inner ℝ (vecOfFun (n := Shape.size Shape.scalar) (fun _ => (-c) * (inner ℝ tMN dlogp +
            inner ℝ dtMN logp))) δV
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
            (vecOfFun (n := Shape.size Shape.scalar) (fun _ => (-c) * (inner ℝ tMN dlogp + inner ℝ
              dtMN logp)))
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

set_option maxHeartbeats 2000000 in
/-- `NodeFDerivCorrect` for `cross_entropy_one_hot_last` (one-hot targets; last-axis reduction). -/
def crossEntropyOneHotLastFderiv {Γ : List Shape} {m n : Nat}
    (logits target : Idx Γ (.dim m (.dim n .scalar))) :
    NodeFDerivCorrect (crossEntropyOneHotLast (Γ := Γ) (m := m) (n := n) logits target) := by
  classical
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Shape.size s = m * n := by simp [s, Shape.size]
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
        simp [crossEntropyOneHotLast, Node.forwardVec_ofVec, xMN, tMN, logp, logitsMN, targetMN,
          c,
          s, Shape.size]
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
      simp [crossEntropyOneHotLast, Node.jvpVec_ofVec, xMN, dxMN, tMN, dtMN, logp, dlogp,
        logitsMN, targetMN, c, s, Shape.size]
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

/-- Negative log-likelihood loss for log-probabilities and one-hot targets of shape `(m×n)`.

Forward:
`-(1/m) * ⟪target, logProbs⟫`

This is the natural primitive loss that `cross_entropy` reduces to after `log_softmax`.
-/
def nllOneHotLast {Γ : List Shape} {m n : Nat}
    (logProbs target : Idx Γ (.dim m (.dim n .scalar))) : Node Γ Shape.scalar :=
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Shape.size s = m * n := by simp [s, Shape.size]
  let c : ℝ := (1 : ℝ) / (m : ℝ)
  Node.ofVec (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let lpMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      vecOfFun (n := Shape.size Shape.scalar) fun _ =>
        (-c) * inner ℝ tMN lpMN)
    (jvp := fun xV dxV =>
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let dtMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target dxV)
      let lpMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      let dlpMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs dxV)
      vecOfFun (n := Shape.size Shape.scalar) fun _ =>
        (-c) * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN))
    (vjp := fun xV δV =>
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
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
      classical
      let s : Shape := .dim m (.dim n .scalar)
      let hsz : Shape.size s = m * n := by simp [s, Shape.size]
      let c : ℝ := (1 : ℝ) / (m : ℝ)
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
      let δ0 : ℝ := δV i0
      let tMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let dtMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target dxV)
      let lpMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      let dlpMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs dxV)
      let scale : ℝ := (-c) * δ0
      let dLogProbs : Vec (m * n) := scale • tMN
      let dTarget : Vec (m * n) := scale • lpMN
      have hL :
          inner ℝ (vecOfFun (n := Shape.size Shape.scalar)
                (fun _ => (-c) * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN))) δV
            =
          ((-c) * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN)) * δ0 := by
        convert
          inner_scalarVec_left (a := (-c) * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN)) (δ := δV)
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
            inner ℝ dlpMN dLogProbs := by
        have h :=
          inner_castVec_castVec (h := hsz)
            (x := CtxVec.get (Γ := Γ) (s := s) logProbs dxV)
            (y := castVec hsz.symm dLogProbs)
        simpa [dlpMN] using h.symm
      have hBc :
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) =
            inner ℝ dtMN dTarget := by
        have h :=
          inner_castVec_castVec (h := hsz)
            (x := CtxVec.get (Γ := Γ) (s := s) target dxV)
            (y := castVec hsz.symm dTarget)
        simpa [dtMN] using h.symm
      have hAterm : inner ℝ dlpMN dLogProbs = scale * inner ℝ tMN dlpMN := by
        -- use commutativity to match the `inner` order in the JVP
        simp [dLogProbs, scale, inner_smul_right, real_inner_comm, mul_assoc]
      have hBterm : inner ℝ dtMN dTarget = scale * inner ℝ dtMN lpMN := by
        simp [dTarget, scale, inner_smul_right]
      calc
        inner ℝ
            (vecOfFun (n := Shape.size Shape.scalar)
              (fun _ => (-c) * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN)))
            δV
            =
          ((-c) * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN)) * δ0 := hL
        _ =
          scale * (inner ℝ tMN dlpMN + inner ℝ dtMN lpMN) := by
            simp [scale, mul_assoc, mul_left_comm, mul_comm]
        _ =
          scale * inner ℝ tMN dlpMN + scale * inner ℝ dtMN lpMN := by
            simp [mul_add]
        _ =
          inner ℝ dlpMN dLogProbs + inner ℝ dtMN dTarget := by
            simp [hAterm, hBterm]
        _ =
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) logProbs dxV) (castVec hsz.symm dLogProbs) +
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) target dxV) (castVec hsz.symm dTarget) := by
            simp [hAc, hBc]
        _ =
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := s) logProbs (castVec hsz.symm dLogProbs) +
                CtxVec.single (Γ := Γ) (s := s) target (castVec hsz.symm dTarget)) := by
            simp [inner_add_right, hA, hB])

/-- `NodeFDerivCorrect` for `nll_one_hot_last` (negative log-likelihood with one-hot targets). -/
def nllOneHotLastFderiv {Γ : List Shape} {m n : Nat}
    (logProbs target : Idx Γ (.dim m (.dim n .scalar))) :
    NodeFDerivCorrect (nllOneHotLast (Γ := Γ) (m := m) (n := n) logProbs target) := by
  classical
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Shape.size s = m * n := by simp [s, Shape.size]
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
        simp [nllOneHotLast, Node.forwardVec_ofVec, tMN, lpMN, logpMN, targetMN, c,
          s, Shape.size]
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
      simp [nllOneHotLast, Node.jvpVec_ofVec, tMN, dtMN, lpMN, dlpMN, logpMN, targetMN, c,
        s, Shape.size]
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

-- ---------------------------------------------------------------------------
-- Loss: BCE with logits (mean over all entries)
-- ---------------------------------------------------------------------------

/-- Binary cross-entropy with logits for same-shaped logits/targets.

Forward (mean reduction over all entries):
`(1/N) * Σ_i (softplus(logits_i) - target_i * logits_i)`

This matches PyTorch's `BCEWithLogitsLoss` with `reduction="mean"`,
and uses the stable identity `BCEWithLogits(x,t) = softplus(x) - t*x`.
-/
def bceWithLogits {Γ : List Shape} {s : Shape}
    (logits target : Idx Γ s) : Node Γ Shape.scalar :=
  let n : Nat := Shape.size s
  let c : ℝ := (1 : ℝ) / (n : ℝ)
  Node.ofVec (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      let x : Vec n := CtxVec.get (Γ := Γ) (s := s) logits xV
      let t : Vec n := CtxVec.get (Γ := Γ) (s := s) target xV
      let sp : Vec n := elemwiseVec (n := n) (f := Activation.Math.softplusSpec (α := ℝ)) x
      vecOfFun (n := Shape.size Shape.scalar) fun _ =>
        c * (sumCLM (n := n) (sp - (vecOfFun (n := n) fun i => t i * x i))))
    (jvp := fun xV dxV =>
      let x : Vec n := CtxVec.get (Γ := Γ) (s := s) logits xV
      let dx : Vec n := CtxVec.get (Γ := Γ) (s := s) logits dxV
      let t : Vec n := CtxVec.get (Γ := Γ) (s := s) target xV
      let dt : Vec n := CtxVec.get (Γ := Γ) (s := s) target dxV
      let sp' : Vec n := elemwiseVec (n := n) (f := Activation.Math.softplusDerivSpec (α := ℝ)) x
      -- d/dx softplus(x) = sigmoid(x)
      let dsp : Vec n := vecOfFun (n := n) fun i => dx i * sp' i
      vecOfFun (n := Shape.size Shape.scalar) fun _ =>
        c * (sumCLM (n := n) (dsp - (vecOfFun (n := n) fun i => dt i * x i + t i * dx i))))
    (vjp := fun xV δV =>
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
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
      let n : Nat := Shape.size s
      let c : ℝ := (1 : ℝ) / (n : ℝ)
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
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
      let jvpOut : Vec (Shape.size Shape.scalar) :=
        vecOfFun (n := Shape.size Shape.scalar) fun _ =>
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

/-- `NodeFDerivCorrect` for `bce_with_logits` (binary cross-entropy with logits). -/
def bceWithLogitsFderiv {Γ : List Shape} {s : Shape} (logits target : Idx Γ s) :
    NodeFDerivCorrect (bceWithLogits (Γ := Γ) (s := s) logits target) := by
  classical
  let n : Nat := Shape.size s
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
      simpa using (hsoftplus.comp xV hlogits)
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
      simp [bceWithLogits, Node.forwardVec_ofVec, logitsV, targetV, c, sumCLM_apply,
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
          ((EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin (Shape.size Shape.scalar))).symm
              (fun _ =>
                c * sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j) -
                  (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)))).ofLp i
            =
          c * sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j) -
            (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)) := by
        convert
          euclideanEquiv_symm_ofLp
            (n := Shape.size Shape.scalar)
            (f := fun _ : Fin (Shape.size Shape.scalar) =>
              c * sumCLM (n := n) (vecOfFun (n := n) (fun j => dx j * sp' j) -
                (vecOfFun (n := n) fun j => dt j * x j + t j * dx j)))
            (i := i) using 1
      simpa [bceWithLogits, Node.jvpVec_ofVec, x, dx, t, dt, sp', logitsV, targetV, c,
        elemwiseVec, vecOfFun, Shape.size, n] using hscalar
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
                simp [D, ContinuousLinearMap.smul_apply, ContinuousLinearMap.sub_apply, smul_eq_mul]
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
  let hsz : Shape.size s = m * n := by simp [s, Shape.size]
  let c : ℝ := (1 : ℝ) / (m : ℝ)
  Node.ofVec (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      let q : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) target xV)
      let lp : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) logProbs xV)
      let logq : Vec (m * n) := elemwiseVec (n := m * n) (f := Real.log) q
      let rhs : Vec (m * n) := logq - lp
      vecOfFun (n := Shape.size Shape.scalar) fun _ =>
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
      vecOfFun (n := Shape.size Shape.scalar) fun _ =>
        c * (inner ℝ dq rhs + inner ℝ q drhs))
    (vjp := fun xV δV =>
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
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
      let hsz : Shape.size s = m * n := by simp [s, Shape.size]
      let c : ℝ := (1 : ℝ) / (m : ℝ)
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
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
              (vecOfFun (n := Shape.size Shape.scalar) fun _ =>
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
            (vecOfFun (n := Shape.size Shape.scalar) fun _ =>
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

/-- Pointwise `NodeFDerivCorrectAt` for `kl_div_last`, assuming `target` entries are nonzero. -/
def klDivLastFderivAt {Γ : List Shape} {m n : Nat}
    (logProbs target : Idx Γ (.dim m (.dim n .scalar))) (xV : CtxVec Γ)
    (ht :
      ∀ i : Fin (Shape.size (.dim m (.dim n .scalar))),
        CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) target xV i ≠ 0) :
    NodeFDerivCorrectAt (klDivLast (Γ := Γ) (m := m) (n := n) logProbs target) xV := by
  classical
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Shape.size s = m * n := by simp [s, Shape.size]
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
    simpa [logqMN, logqDeriv] using (hlogq0.comp xV hq0)

  let rhsMN : CtxVec Γ → Vec (m * n) := fun x => logqMN x - lpMN x
  let rhsDeriv : CtxVec Γ →L[ℝ] Vec (m * n) := logqDeriv - lpMNCLM
  have hrhs :
      HasFDerivAt rhsMN rhsDeriv xV := by
    simpa [rhsMN, rhsDeriv] using hlogq.sub hlp0

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
      simp [klDivLast, Node.forwardVec_ofVec, qMN, lpMN, logqMN, rhsMN, c, s,
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
          ((EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin (Shape.size Shape.scalar))).symm
              (fun _ => c * (inner ℝ dq rhs + inner ℝ q drhs))).ofLp i
            =
          c * (inner ℝ dq rhs + inner ℝ q drhs) := by
        convert
          euclideanEquiv_symm_ofLp
            (n := Shape.size Shape.scalar)
            (f := fun _ : Fin (Shape.size Shape.scalar) => c * (inner ℝ dq rhs + inner ℝ q drhs))
            (i := i) using 1
      simpa [klDivLast, Node.jvpVec_ofVec, qMN, lpMN, logqMN, c, s,
        q, dq, lp, dlp, logq, dlogq, rhs, drhs, vecOfFun, Shape.size] using hscalar
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
      simp [D, ContinuousLinearMap.smul_apply, smul_eq_mul, ContinuousLinearMap.comp_apply,
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
