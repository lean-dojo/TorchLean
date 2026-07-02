/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Reductions
public import NN.Spec.Layers.Loss

@[expose] public section

namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

/-!
# Mean-Squared Error

Tape node and Fréchet derivative proof for scalar mean-squared error.
-/

-- ---------------------------------------------------------------------------
-- Loss: scalar mean squared error
-- ---------------------------------------------------------------------------

/--
Mean-squared-error loss node: `c * ‖yhat - target‖^2`, with
`c = 1 / Spec.meanDenom s`.

For nonempty shapes this is the usual `1 / Shape.size s`. For the empty shape case, the scalar loss
API is totalized with denominator `1`, matching `Spec.mseSpec` and the IR evaluator.
-/
def mseLoss {Γ : List Shape} {s : Shape} (yhat target : Idx Γ s) : Node Γ Shape.scalar :=
  let n : Nat := Spec.meanDenom s
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
      let n : Nat := Spec.meanDenom s
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
  let n : Nat := Spec.meanDenom s
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
      have hsub := hgetY.sub hgetT
      refine hsub.congr_of_eventuallyEq ?_
      exact Filter.Eventually.of_forall fun _ => rfl
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
      calc
        ((Node.forwardVec (Γ := Γ) (τ := Shape.scalar)
              (mseLoss (Γ := Γ) (s := s) yhat target)) x).ofLp ⟨0, by simp [Shape.size]⟩
            =
          (↑(Spec.meanDenom s))⁻¹ *
            ‖CtxVec.get (Γ := Γ) (s := s) yhat x -
              CtxVec.get (Γ := Γ) (s := s) target x‖ ^ 2 := by
            simp [mseLoss, Node.forwardVec_ofVec, Shape.size, div_eq_mul_inv]
        _ = (vecScalarCLM (g x)).ofLp ⟨0, by simp [Shape.size]⟩ := by
            simp [g, c, n, smul_eq_mul, div_eq_mul_inv]
    exact hwrap.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    let diffV : Vec (Shape.size s) :=
      (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
    let D0 : CtxVec Γ →L[ℝ] ℝ :=
      (c • (2 • (innerSL ℝ diffV)).comp diffDeriv)
    let ddiffV : Vec (Shape.size s) :=
      (CtxVec.get (Γ := Γ) (s := s) yhat dxV) - (CtxVec.get (Γ := Γ) (s := s) target dxV)
    have hdiffDeriv : diffDeriv dxV = ddiffV := by
      simp [diffDeriv, ddiffV, CtxVec.getCLM_apply]
    -- Avoid expanding `inner` on differences; both sides are the same scalar packaged into `Vec 1`.
    have hD0 :
        D0 dxV = c * (2 * inner ℝ diffV ddiffV) := by
      -- `innerSL` is `y ↦ ⟪diffV,y⟫`
      simp [D0, hdiffDeriv, innerSL_apply_apply, ContinuousLinearMap.comp_apply,
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

end TapeNodes

end

end Autograd
end Proofs
