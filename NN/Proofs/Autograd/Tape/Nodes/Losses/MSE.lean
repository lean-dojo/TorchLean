/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Reductions
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Mean-Squared Error

Tape node and Fréchet derivative proof for scalar mean-squared error.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

/--
Mean-squared-error loss node: `c * ‖yhat - target‖^2`, with
`c = 1 / TorchLean.Tensor.meanDenominator s`.

For nonempty shapes this is the usual `1 / Spec.Shape.size s`. For the empty shape case, the
scalar loss API is totalized with denominator `1`, matching `Spec.mseSpec` and the IR evaluator.
-/
def mseLoss {Γ : List Shape} {s : Shape} (yhat target : Idx Γ s) : Node Γ Shape.scalar :=
  let n : Nat := TorchLean.Tensor.meanDenominator s
  let c : ℝ := (1 : ℝ) / (n : ℝ)
  Node.ofFn (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ =>
        c * ‖(CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)‖ ^ 2)
    (jvp := fun xV dxV =>
      let diff := (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
      let ddiff := (CtxVec.get (Γ := Γ) (s := s) yhat dxV) - (CtxVec.get (Γ := Γ) (s := s) target
        dxV)
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ => c * (2 * inner ℝ diff ddiff))
    (vjp := fun xV δV =>
      let i0 : Fin (Spec.Shape.size Shape.scalar) := ⟨0, by simp [Spec.Shape.size]⟩
      let δ0 : ℝ := δV i0
      let diff := (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
      let scale : ℝ := δ0 * (2 * c)
      let dYhat : Vec (Spec.Shape.size s) := scale • diff
      let dTarget : Vec (Spec.Shape.size s) := -dYhat
      CtxVec.single (Γ := Γ) (s := s) yhat dYhat + CtxVec.single (Γ := Γ) (s := s) target dTarget)
    (correct_inner := by
      intro xV dxV δV
      simp only [inner_scalarVec_left, inner_add_right, CtxVec.inner_get_single,
        inner_smul_right, inner_neg_right, inner_sub_right]
      rw [real_inner_comm (CtxVec.get yhat dxV), real_inner_comm (CtxVec.get target dxV)]
      simp only [inner_sub_right]
      ring)

/-- `NodeFDerivCorrect` for `mseLoss`. -/
def mseLossFderiv {Γ : List Shape} {s : Shape} (yhat target : Idx Γ s) :
    NodeFDerivCorrect (mseLoss (Γ := Γ) (s := s) yhat target) := by
  classical
  let n : Nat := TorchLean.Tensor.meanDenominator s
  let c : ℝ := (1 : ℝ) / (n : ℝ)
  let diffDeriv : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size s) :=
    (CtxVec.getCLM (Γ := Γ) (s := s) yhat) - (CtxVec.getCLM (Γ := Γ) (s := s) target)
  refine
    { deriv := fun xV =>
        let diffV : Vec (Spec.Shape.size s) :=
          (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
        vecScalarCLM.comp (c • (2 • (innerSL ℝ diffV)).comp diffDeriv)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    -- `get` projections are CLMs.
    have hgetY :
        HasFDerivAt (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) yhat x)
          (CtxVec.getCLM (Γ := Γ) (s := s) yhat) xV :=
      CtxVec.hasFDerivAt_get yhat xV
    have hgetT :
        HasFDerivAt (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) target x)
          (CtxVec.getCLM (Γ := Γ) (s := s) target) xV :=
      CtxVec.hasFDerivAt_get target xV
    have hdiff :
        HasFDerivAt
          (fun x : CtxVec Γ =>
            (CtxVec.get (Γ := Γ) (s := s) yhat x) - (CtxVec.get (Γ := Γ) (s := s) target x))
          diffDeriv xV :=
      hgetY.sub hgetT
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
              (mseLoss (Γ := Γ) (s := s) yhat target)) x).ofLp ⟨0, by simp [Spec.Shape.size]⟩
            =
          (↑(TorchLean.Tensor.meanDenominator s))⁻¹ *
            ‖CtxVec.get (Γ := Γ) (s := s) yhat x -
              CtxVec.get (Γ := Γ) (s := s) target x‖ ^ 2 := by
            simp only [mseLoss, Node.forwardVec_ofFn]
            simp [Spec.Shape.size, div_eq_mul_inv]
        _ = (vecScalarCLM (g x)).ofLp ⟨0, by simp [Spec.Shape.size]⟩ := by
            simp [g, c, n, smul_eq_mul, div_eq_mul_inv]
    exact hwrap.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    let diffV : Vec (Spec.Shape.size s) :=
      (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
    let D0 : CtxVec Γ →L[ℝ] ℝ :=
      (c • (2 • (innerSL ℝ diffV)).comp diffDeriv)
    let ddiffV : Vec (Spec.Shape.size s) :=
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
      simp only [mseLoss, Node.jvpVec_ofFn]
      rw [vecOfFun_ofLp]
    have hR : ((vecScalarCLM.comp D0) dxV).ofLp i = D0 dxV := by
      simp [ContinuousLinearMap.comp_apply]
    calc
      (Node.jvpVec (Γ := Γ) (τ := Shape.scalar) (mseLoss (Γ := Γ) (s := s) yhat target) xV
        dxV).ofLp i
          = c * (2 * inner ℝ diffV ddiffV) := hL
      _ = D0 dxV := hD0.symm
      _ = ((vecScalarCLM.comp D0) dxV).ofLp i := hR.symm

end TapeNodes

end

end Autograd
end Proofs
