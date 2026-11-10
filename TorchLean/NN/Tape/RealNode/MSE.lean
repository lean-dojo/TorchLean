/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Reductions
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Mean squared error tape node

Tape node and Fréchet derivative certificate for the scalar mean squared error
`Tensor.meanSquaredError`. The node `RealNode.mseLoss` runs the rule of `Op.mseLoss` over `ℝ` on
two context entries, with the gradient `Tensor.meanSquaredErrorBackward`, and
`RealNode.mseLossFDeriv` proves that its JVP is the Fréchet derivative of its value.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open Shape TorchLean.Tensor

noncomputable section

open scoped BigOperators

/-- Vectorization maps `Tensor.scale a k` to `k • toVec a`. -/
private theorem Tensor.toVec_scale_eq_smul {s : Shape} (a : Tensor ℝ s) (k : ℝ) :
    toVec (Tensor.scale a k) = k • toVec a := by
  refine ext_inner_right ℝ fun w => ?_
  obtain ⟨W, rfl⟩ : ∃ W : Tensor ℝ s, w = toVec W :=
    ⟨ofVec w, (toVec_ofVec w).symm⟩
  rw [real_inner_smul_left, ← dot_eq_inner_toVec, ← dot_eq_inner_toVec,
    dot_scale_left, mul_comm]

/-- Vectorization maps tensor subtraction to vector subtraction. -/
private theorem Tensor.toVec_sub {s : Shape} (a b : Tensor ℝ s) :
    toVec (a - b) = toVec a - toVec b := by
  have h : a - b = a + Tensor.scale b (-1) := by
    apply Tensor.Rep.ext
    intro coordinate
    simp only [Tensor.add_apply, Rep.hSub_apply, Tensor.scale, Tensor.map_apply]
    ring
  rw [h, toVec_add, toVec_scale_eq_smul, neg_one_smul, sub_eq_add_neg]

/-- Vectorization maps `Tensor.neg a` to `-toVec a`. -/
private theorem Tensor.toVec_neg {s : Shape} (a : Tensor ℝ s) :
    toVec (Tensor.neg a) = -toVec a := by
  have h : Tensor.neg a = Tensor.scale a (-1) := by
    apply Tensor.Rep.ext
    intro coordinate
    simp only [Tensor.neg, Tensor.scale, Tensor.map_apply]
    ring
  rw [h, toVec_scale_eq_smul, neg_one_smul]

/-- A scalar tensor vectorizes to the constant one-coordinate vector. -/
private theorem Tensor.toVec_scalar_eq_toLp (a : ℝ) :
    toVec (Tensor.scalar a) = WithLp.toLp 2 fun _ : Fin (Shape.size Shape.scalar) => a := by
  apply PiLp.ext
  intro i
  exact toVec_scalar a i

/-- The item of the scalar tensor of a one-coordinate vector is that coordinate. -/
private theorem Tensor.item_ofVec (δ : Vec (Shape.size Shape.scalar)) :
    Tensor.item (ofVec (s := Shape.scalar) δ) = δ.ofLp ⟨0, by simp⟩ := by
  have h := toVec_scalar (ofVec (s := Shape.scalar) δ).item ⟨0, by simp⟩
  rw [Tensor.scalar_item, toVec_ofVec] at h
  exact h.symm

/-- Over `ℝ`, `Tensor.meanSquaredError a b` is `‖toVec a - toVec b‖ ^ 2 / meanDenominator s`. -/
private theorem Tensor.meanSquaredError_eq_norm_sq_div {s : Shape} (a b : Tensor ℝ s) :
    Tensor.meanSquaredError a b =
      ‖toVec a - toVec b‖ ^ 2 / (meanDenominator s : ℝ) := by
  rw [Tensor.meanSquaredError_of_isFinite a b (Context.isFinite_real _), Context.divNat_real,
    ← toVec_sub, ← real_inner_self_eq_norm_sq]
  congr 1
  exact (Tensor.dot_eq_rep_dot (a - b) (a - b)).trans (dot_eq_inner_toVec (a - b) (a - b))

/-- Over `ℝ`, `Tensor.meanSquaredErrorBackward a b` vectorizes to `(2 / n) • (toVec a - toVec b)`
for `n = meanDenominator s`. -/
private theorem Tensor.toVec_meanSquaredErrorBackward {s : Shape} (a b : Tensor ℝ s) :
    toVec (Tensor.meanSquaredErrorBackward a b) =
      (2 / (meanDenominator s : ℝ)) • (toVec a - toVec b) := by
  have h : Tensor.meanSquaredErrorBackward a b =
      Tensor.scale (a - b) (2 / (meanDenominator s : ℝ)) := by
    apply Tensor.Rep.ext
    intro coordinate
    simp only [Tensor.meanSquaredErrorBackward, Tensor.scale, Tensor.map_apply,
      Context.isFinite_real, ↓reduceIte, Context.divNat_real]
    ring
  rw [h, toVec_scale_eq_smul, toVec_sub]

/-- Over `ℝ`, summing `Tensor.meanSquaredErrorBackward a b * w` gives
`(2 / n) ⟪toVec a - toVec b, toVec w⟫` for `n = meanDenominator s`. -/
private theorem Tensor.sum_meanSquaredErrorBackward_mul {s : Shape} (a b w : Tensor ℝ s) :
    Tensor.sum (Tensor.meanSquaredErrorBackward a b * w) =
      2 / (meanDenominator s : ℝ) * inner ℝ (toVec a - toVec b) (toVec w) := by
  change Tensor.dot (Tensor.meanSquaredErrorBackward a b) w = _
  rw [Tensor.dot_eq_rep_dot, dot_eq_inner_toVec, toVec_meanSquaredErrorBackward,
    real_inner_smul_left]

namespace Tape.RealNode

open Vec

/--
Mean squared error loss node: the rule of `Op.mseLoss` over `ℝ` at the context entries `yhat` and
`target`. Its value is `Tensor.meanSquaredError yhat target`, and a cotangent `δ` pulls back to
`δ • Tensor.meanSquaredErrorBackward yhat target` at `yhat` and to its negation at `target`.
-/
def mseLoss {Γ : List Shape} {s : Shape} (yhat target : Idx Γ s) :
    LawfulNode (α := ℝ) Unit Γ Shape.scalar :=
  LawfulNode.ofFn (Γ := Γ) (τ := Shape.scalar)
    (f := fun xV =>
      toVec (Tensor.scalar (Tensor.meanSquaredError
        (ofVec (s := s) (CtxVec.get (Γ := Γ) (s := s) yhat xV))
        (ofVec (s := s) (CtxVec.get (Γ := Γ) (s := s) target xV)))))
    (jvp := fun xV dxV =>
      toVec (Tensor.scalar (Tensor.sum
        (Tensor.meanSquaredErrorBackward
            (ofVec (s := s) (CtxVec.get (Γ := Γ) (s := s) yhat xV))
            (ofVec (s := s) (CtxVec.get (Γ := Γ) (s := s) target xV)) *
          (ofVec (s := s) (CtxVec.get (Γ := Γ) (s := s) yhat dxV) -
            ofVec (s := s) (CtxVec.get (Γ := Γ) (s := s) target dxV))))))
    (vjp := fun xV δV =>
      let gradient := Tensor.scale
        (Tensor.meanSquaredErrorBackward
          (ofVec (s := s) (CtxVec.get (Γ := Γ) (s := s) yhat xV))
          (ofVec (s := s) (CtxVec.get (Γ := Γ) (s := s) target xV)))
        (Tensor.item (ofVec (s := Shape.scalar) δV))
      CtxVec.single (Γ := Γ) (s := s) yhat (toVec gradient) +
        CtxVec.single (Γ := Γ) (s := s) target (toVec (Tensor.neg gradient)))
    (correct_inner := by
      intro xV dxV δV
      simp only [Tensor.toVec_scalar_eq_toLp, inner_ofFun_scalar_left,
        Tensor.sum_meanSquaredErrorBackward_mul, Tensor.toVec_sub, Tensor.toVec_ofVec,
        inner_add_right, CtxVec.inner_get_single, Tensor.toVec_neg, Tensor.toVec_scale_eq_smul,
        Tensor.toVec_meanSquaredErrorBackward, Tensor.item_ofVec, inner_neg_right,
        real_inner_smul_right, inner_sub_left, inner_sub_right]
      rw [real_inner_comm (CtxVec.get yhat xV) (CtxVec.get yhat dxV),
        real_inner_comm (CtxVec.get target xV) (CtxVec.get yhat dxV),
        real_inner_comm (CtxVec.get yhat xV) (CtxVec.get target dxV),
        real_inner_comm (CtxVec.get target xV) (CtxVec.get target dxV)]
      ring)

/-- `FDerivCorrect` for `mseLoss`: at every `x` the forward map has Fréchet derivative
`dx ↦ (2 / n) ⟪get yhat x - get target x, get yhat dx - get target dx⟫` (a one-coordinate vector,
`n = meanDenominator s`), and the JVP of the node is that derivative. -/
def mseLossFDeriv {Γ : List Shape} {s : Shape} (yhat target : Idx Γ s) :
    FDerivCorrect (mseLoss (Γ := Γ) (s := s) yhat target).toNodeData () := by
  classical
  let n : Nat := Tensor.meanDenominator s
  let c : ℝ := (1 : ℝ) / (n : ℝ)
  let diffDeriv : CtxVec Γ →L[ℝ] Vec (Shape.size s) :=
    (CtxVec.getCLM (Γ := Γ) (s := s) yhat) - (CtxVec.getCLM (Γ := Γ) (s := s) target)
  refine
    { deriv := fun xV =>
        let diffV : Vec (Shape.size s) :=
          (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
        scalarCLM.comp (c • (2 • (innerSL ℝ diffV)).comp diffDeriv)
      hasFDerivAt := by
        intro xV
        -- The context projections have the bundled projections as derivatives.
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
        -- The squared norm of the difference, then the scaling by `c`.
        have hsq :
            HasFDerivAt
              (fun x : CtxVec Γ =>
                ‖(CtxVec.get (Γ := Γ) (s := s) yhat x) -
                  (CtxVec.get (Γ := Γ) (s := s) target x)‖ ^ 2)
              (2 • (innerSL ℝ ((CtxVec.get (Γ := Γ) (s := s) yhat xV) -
                (CtxVec.get (Γ := Γ) (s := s) target xV))).comp diffDeriv)
              xV := by
          simpa using hdiff.norm_sq
        have hscaled :=
          (hsq.const_smul c)
        -- `scalarCLM` puts the scalar into the one-coordinate vector.
        let g : CtxVec Γ → ℝ :=
          fun x =>
            c • (‖(CtxVec.get (Γ := Γ) (s := s) yhat x) -
              (CtxVec.get (Γ := Γ) (s := s) target x)‖ ^ 2)
        have hwrap :
            HasFDerivAt (fun x : CtxVec Γ => scalarCLM (g x))
              (scalarCLM.comp (c • (2 • (innerSL ℝ ((CtxVec.get (Γ := Γ) (s := s) yhat xV) -
                  (CtxVec.get (Γ := Γ) (s := s) target xV))).comp diffDeriv))) xV := by
          have hlin : HasFDerivAt (fun r : ℝ => scalarCLM r) scalarCLM (g xV) :=
            scalarCLM.hasFDerivAt (x := g xV)
          exact hlin.comp xV hscaled
        -- The forward map of the node is that wrapped scalar.
        have hEq :
            (NodeData.forwardVec (Γ := Γ) (τ := Shape.scalar)
              (mseLoss (Γ := Γ) (s := s) yhat target).toNodeData ())
              =
            fun x : CtxVec Γ => scalarCLM (g x) := by
          funext x
          apply PiLp.ext
          intro i
          simp only [mseLoss, LawfulNode.forwardVec_ofFn, Tensor.toVec_scalar,
            Tensor.meanSquaredError_eq_norm_sq_div, Tensor.toVec_ofVec, scalarCLM_apply, g, c,
            n, smul_eq_mul]
          ring
        exact hwrap.congr_of_eventuallyEq hEq.eventuallyEq
      jvp_eq := by
        intro xV dxV
        let diffV : Vec (Shape.size s) :=
          (CtxVec.get (Γ := Γ) (s := s) yhat xV) - (CtxVec.get (Γ := Γ) (s := s) target xV)
        let D0 : CtxVec Γ →L[ℝ] ℝ :=
          (c • (2 • (innerSL ℝ diffV)).comp diffDeriv)
        let ddiffV : Vec (Shape.size s) :=
          (CtxVec.get (Γ := Γ) (s := s) yhat dxV) - (CtxVec.get (Γ := Γ) (s := s) target dxV)
        have hdiffDeriv : diffDeriv dxV = ddiffV := by
          simp [diffDeriv, ddiffV, CtxVec.getCLM_apply]
        -- The derivative at `xV` sends `dxV` to `c * (2 * ⟪diffV, ddiffV⟫)`.
        have hD0 :
            D0 dxV = c * (2 * inner ℝ diffV ddiffV) := by
          -- `innerSL ℝ diffV` sends `y` to `⟪diffV, y⟫`.
          simp [D0, hdiffDeriv, innerSL_apply_apply, ContinuousLinearMap.comp_apply,
            smul_eq_mul, mul_left_comm]
        -- Compare the coordinates of the two one-coordinate vectors.
        apply PiLp.ext
        intro i
        have hL :
            (NodeData.jvpVec (Γ := Γ) (τ := Shape.scalar)
              (mseLoss (Γ := Γ) (s := s) yhat target).toNodeData () xV dxV).ofLp i =
              c * (2 * inner ℝ diffV ddiffV) := by
          simp only [mseLoss, LawfulNode.jvpVec_ofFn, Tensor.toVec_scalar,
            Tensor.sum_meanSquaredErrorBackward_mul, Tensor.toVec_sub, Tensor.toVec_ofVec, diffV,
            ddiffV, c, n]
          ring
        have hR : ((scalarCLM.comp D0) dxV).ofLp i = D0 dxV := by
          rw [ContinuousLinearMap.comp_apply, scalarCLM_apply]
        calc
          (NodeData.jvpVec (Γ := Γ) (τ := Shape.scalar)
            (mseLoss (Γ := Γ) (s := s) yhat target).toNodeData () xV dxV).ofLp i
              = c * (2 * inner ℝ diffV ddiffV) := hL
          _ = D0 dxV := hD0.symm
          _ = ((scalarCLM.comp D0) dxV).ofLp i := hR.symm }

end Tape.RealNode

end

end TorchLean
