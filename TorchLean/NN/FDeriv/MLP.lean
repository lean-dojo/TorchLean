/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.InnerProductSpace.Calculus
public import TorchLean.NN.FDeriv.Params
public import TorchLean.Tensor.Contract
public import TorchLean.Tensor.Elementwise.Real
public import TorchLean.Tensor.Layers.Loss

/-!
# Two-layer MLP with MSE loss

For the 2-layer MLP `TorchLean.Vec.mlp l1 l2` (Linear → ReLU → Linear) and the loss
`TorchLean.Tensor.meanSquaredError`, the backprop formulas, parameter gradients included, are
adjoints of the Fréchet derivative `fderiv` over `ℝ`. Weights vary in the inner product space
`Tensor ℝ [m, n]` with the Frobenius inner product, so their gradients are the outer products
`TorchLean.Linear.weightsDeriv`. Biases vary as Euclidean vectors `b` and enter a layer as
`ofFnE b`. The output gradient is `TorchLean.Tensor.meanSquaredErrorBackward`.

ReLU is not differentiable at 0, so the theorems for the first layer and for the input assume that
every pre-activation coordinate is nonzero. The gradient of a scalar loss `ℓ` at `y` is
`(fderiv ℝ ℓ y).adjoint 1`, the vector-Jacobian product with seed `1`.

## PyTorch correspondence / citations
- MLP building blocks: `torch.nn.Linear`, `torch.nn.functional.relu`.
  https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
  https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
- MSE loss reference behavior:
  https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
- Backward as the adjoint of the derivative is the Jacobian-transpose rule of PyTorch autograd:
  https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- `Vec.reluDerivCLM x` scales coordinate `i` by the ReLU derivative mask at `x i`. -/
theorem Vec.reluDerivCLM_apply {n : Nat} (x dx : Vec n) (i : Fin n) :
    (reluDerivCLM (n := n) x) dx i = dx i * Activation.reluDeriv (x i) := rfl

/-- `Vec.reluDerivCLM x` is symmetric for the Euclidean inner product, because it is a diagonal
scaling of `ℝⁿ`. -/
theorem Vec.reluDerivCLM_inner {n : Nat} (x dx δ : Vec n) :
    inner ℝ ((reluDerivCLM (n := n) x) dx) δ = inner ℝ dx ((reluDerivCLM (n := n) x) δ) := by
  classical
  simp [inner_eq_sum_mul, reluDerivCLM_apply, mul_assoc, mul_left_comm, mul_comm]

/-- `Vec.reluDerivCLM x` is self-adjoint: its adjoint sends `δ` to `reluDerivCLM x δ`. -/
theorem Vec.reluDerivCLM_adjoint_apply {n : Nat} (x δ : Vec n) :
    (reluDerivCLM (n := n) x).adjoint δ = (reluDerivCLM (n := n) x) δ := by
  apply ext_inner_left ℝ
  intro dx
  rw [ContinuousLinearMap.adjoint_inner_right]
  exact reluDerivCLM_inner x dx δ

/-- On Euclidean vectors the input gradient `Linear.inputDeriv W δ` is the adjoint of the map
`x ↦ W x` applied to `δ`, so it computes `Wᵀ δ`. -/
theorem Linear.getScalarE_inputDeriv_eq_adjoint {inDim outDim : Nat}
    (W : Tensor ℝ [outDim, inDim]) (δ : Tensor ℝ [outDim]) :
    getScalarE (Linear.inputDeriv (inDim := inDim) (outDim := outDim) W δ)
      =
    (Matrix.toEuclideanLin (Rep.matrixEquiv ℝ outDim inDim W)).toContinuousLinearMap.adjoint
      (getScalarE δ) := by
  classical
  let A : Vec inDim →L[ℝ] Vec outDim :=
    (Matrix.toEuclideanLin (Rep.matrixEquiv ℝ outDim inDim W)).toContinuousLinearMap
  let u : Vec inDim := getScalarE (Tensor.contract δ W _ _ [inDim] .vecMat)
  let v : Vec inDim := A.adjoint (getScalarE δ)
  have hforall : ∀ dxV : Vec inDim, inner ℝ dxV u = inner ℝ dxV v := by
    intro dxV
    have hdot := Tensor.dot_contract_matVec (W := W) (dLdy := δ) (dx := ofFnE dxV)
    have hinner :
        inner ℝ (getScalarE δ) (getScalarE (Tensor.contract W (ofFnE dxV) _ _ [outDim] .matVec))
          =
        inner ℝ (getScalarE (Tensor.contract δ W _ _ [inDim] .vecMat)) dxV := by
      simpa [dot_eq_inner_getScalarE, getScalarE_ofFnE] using hdot
    have hAx : getScalarE (Tensor.contract W (ofFnE dxV) _ _ [outDim] .matVec) = A dxV := by
      simpa [A] using
        getScalarE_contract_matVec (m := outDim) (n := inDim) (A := W) (v := ofFnE dxV)
    calc
      inner ℝ dxV u = inner ℝ u dxV := real_inner_comm _ _
      _ = inner ℝ (getScalarE δ) (A dxV) := by
        have hsymm := hinner.symm
        rw [hAx] at hsymm
        exact hsymm
      _ = inner ℝ (A dxV) (getScalarE δ) := real_inner_comm _ _
      _ = inner ℝ dxV v :=
        (ContinuousLinearMap.adjoint_inner_right (A := A) (x := dxV) (y := getScalarE δ)).symm
  have huv : u = v := ext_inner_left ℝ hforall
  exact huv

/-- The MSE `y ↦ Tensor.meanSquaredError (ofFnE y) t` against a fixed target `t` has Fréchet
derivative `(2 / n) • innerSL ℝ (y - getScalarE t)` at `y`. -/
theorem Tensor.hasFDerivAt_mse {n : Nat} (t : Tensor ℝ [n]) (y : Vec n) :
    HasFDerivAt (fun y : Vec n => Tensor.meanSquaredError (ofFnE y) t)
      ((2 / (n : ℝ)) • (innerSL ℝ (y - getScalarE t))) y := by
  have hsub : ∀ y : Vec n, getScalarE (ofFnE y - t) = y - getScalarE t := by
    intro y
    ext i
    simp [getScalarE_ofLp, Tensor.sub_def, getScalar_ofFnE]
  -- Over `ℝ` the loss is `‖y - t‖² / n`, and the empty case `n = 0` divides by one.
  have hfun : (fun y : Vec n => Tensor.meanSquaredError (ofFnE y) t) =
      fun y : Vec n => ((n : ℝ)⁻¹) * ‖y - getScalarE t‖ ^ 2 := by
    funext y
    rw [Tensor.meanSquaredError_of_isFinite (ofFnE y) t (Context.isFinite_real _),
      Context.divNat_real]
    change Tensor.dot (ofFnE y - t) (ofFnE y - t) / ((Tensor.meanDenominator [n] : Nat) : ℝ) = _
    rw [Tensor.dot_eq_rep_dot, dot_eq_inner_getScalarE, hsub, real_inner_self_eq_norm_sq]
    rcases Nat.eq_zero_or_pos n with hn | hn
    · subst hn
      have h0 : y - getScalarE t = 0 := by
        ext i
        exact absurd i.isLt (Nat.not_lt_zero _)
      simp [h0]
    · rw [Tensor.meanDenominator_of_size_pos (s := [n]) (by rw [Shape.size_vector]; exact hn),
        Shape.size_vector, div_eq_inv_mul]
  rw [hfun]
  apply (((hasFDerivAt_id y).sub_const (getScalarE t)).norm_sq.const_mul (n : ℝ)⁻¹).congr_fderiv
  ext z
  simp [div_eq_mul_inv, mul_assoc, mul_comm]

/-- The adjoint of the MSE derivative `(2 / n) • innerSL ℝ (y - getScalarE t)` sends the seed `1`
to the MSE gradient `getScalarE (Tensor.meanSquaredErrorBackward (ofFnE y) t)`. -/
theorem Tensor.adjoint_mseDeriv_one {n : Nat} (t : Tensor ℝ [n]) (y : Vec n) :
    ((2 / (n : ℝ)) • innerSL ℝ (y - getScalarE t)).adjoint (1 : ℝ) =
      getScalarE (Tensor.meanSquaredErrorBackward (ofFnE y) t) := by
  have hinner : (ContinuousLinearMap.adjoint ((innerSL ℝ) (y - getScalarE t))) (1 : ℝ) =
      y - getScalarE t := by
    simpa using congrArg (fun f => f (1 : ℝ))
      (ContinuousLinearMap.adjoint_innerSL_apply (𝕜 := ℝ) (x := y - getScalarE t))
  have hgrad : getScalarE (Tensor.meanSquaredErrorBackward (ofFnE y) t) =
      (2 / (n : ℝ)) • (y - getScalarE t) := by
    ext i
    have hn : Tensor.meanDenominator [n] = n := by
      rw [Tensor.meanDenominator_of_size_pos (s := [n])
        (by rw [Shape.size_vector]; exact i.pos), Shape.size_vector]
    simp [Tensor.meanSquaredErrorBackward, Tensor.sub_def, hn]
    ring
  calc
    ((2 / (n : ℝ)) • innerSL ℝ (y - getScalarE t)).adjoint (1 : ℝ)
        = (2 / (n : ℝ)) • (innerSL ℝ (y - getScalarE t)).adjoint (1 : ℝ) := by
          simp
    _ = (2 / (n : ℝ)) • (y - getScalarE t) := by
          rw [hinner]
    _ = getScalarE (Tensor.meanSquaredErrorBackward (ofFnE y) t) := hgrad.symm

/-- The MSE gradient is the adjoint of the MSE derivative at `y` applied to the seed `1`:
`(fderiv ℝ ℓ y).adjoint 1 = getScalarE (Tensor.meanSquaredErrorBackward (ofFnE y) t)`. -/
theorem Tensor.mseGrad_eq_adjoint_fderiv {n : Nat} (t : Tensor ℝ [n]) (y : Vec n) :
    (fderiv ℝ (fun y : Vec n => Tensor.meanSquaredError (ofFnE y) t) y).adjoint (1 : ℝ) =
      getScalarE (Tensor.meanSquaredErrorBackward (ofFnE y) t) := by
  rw [(hasFDerivAt_mse t y).fderiv]
  exact adjoint_mseDeriv_one t y

/-- Reverse-mode chain rule for a scalar composite `g ∘ f`: the adjoint of its derivative applied to
the seed `1` is `f'.adjoint (g'.adjoint 1)`. -/
theorem adjoint_fderiv_comp_apply_one
    {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]
    {f : E → F} {g : F → ℝ}
    {f' : E →L[ℝ] F} {g' : F →L[ℝ] ℝ} {x : E}
    (hf : HasFDerivAt f f' x) (hg : HasFDerivAt g g' (f x)) :
    (fderiv ℝ (fun x => g (f x)) x).adjoint (1 : ℝ) = f'.adjoint (g'.adjoint (1 : ℝ)) := by
  change (fderiv ℝ (g ∘ f) x).adjoint (1 : ℝ) = _
  rw [(hg.comp x hf).fderiv, ContinuousLinearMap.adjoint_comp]
  rfl

section

variable {inDim hidDim outDim : Nat}
variable (l1 : Linear ℝ inDim hidDim) (l2 : Linear ℝ hidDim outDim)
variable (x : Vec inDim) (t : Tensor ℝ [outDim])

/-!
The intermediate values follow textbook backprop. The pre-activation `z₁ = W₁ x + b₁` is
`Vec.affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x`, the hidden
activation `a₁` is `Vec.relu z₁`, and the network output `y = W₂ a₁ + b₂` is `Vec.mlp l1 l2 x`.
A weight slice replaces the `weights` of its layer and is differentiated at that layer's weights.
A bias slice sets the bias of its layer to `ofFnE b` and is differentiated at `getScalarE` of that
layer's bias.
-/

/-- The network output has Fréchet derivative `Vec.matApplyLin a₁` with respect to the
second-layer weights `l2.weights`, the linear map that sends `dW2` to `dW2 a₁`. -/
theorem Vec.hasFDerivAt_mlp_wrt_W2 :
    HasFDerivAt (fun W2 : Tensor ℝ [outDim, hidDim] => mlp l1 { l2 with weights := W2 } x)
      (matApplyLin (m := outDim) (n := hidDim)
        (relu (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x)))
      l2.weights := by
  let L2 := matApplyLin (m := outDim) (n := hidDim)
    (relu (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x))
  have hfun :
      (fun W2 : Tensor ℝ [outDim, hidDim] => mlp l1 { l2 with weights := W2 } x) =
        fun W2 : Tensor ℝ [outDim, hidDim] => L2 W2 + getScalarE l2.bias := by
    funext W2
    rfl
  rw [hfun]
  exact L2.hasFDerivAt.add_const (getScalarE l2.bias)

/-- The gradient of the loss `Tensor.meanSquaredError (ofFnE y) t` with respect to `l2.weights` is
`∂L/∂W2 = (∂L/∂y) ⊗ a₁`, the outer product `Linear.weightsDeriv` of the output gradient
`Tensor.meanSquaredErrorBackward` and the hidden activation `a₁`. -/
theorem Vec.grad_W2_mse :
    (fderiv ℝ (fun W2 : Tensor ℝ [outDim, hidDim] =>
        Tensor.meanSquaredError (ofFnE (mlp l1 { l2 with weights := W2 } x)) t)
        l2.weights).adjoint (1 : ℝ)
      =
    Linear.weightsDeriv
      (ofFnE (relu (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x)))
      (Tensor.meanSquaredErrorBackward (ofFnE (mlp l1 l2 x)) t) := by
  have hcomp : (fderiv ℝ (fun W2 : Tensor ℝ [outDim, hidDim] =>
        Tensor.meanSquaredError (ofFnE (mlp l1 { l2 with weights := W2 } x)) t)
        l2.weights).adjoint (1 : ℝ) =
      (matApplyLin (m := outDim) (n := hidDim)
        (relu (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias)
          x))).adjoint
        (((2 / (outDim : ℝ)) • innerSL ℝ (mlp l1 l2 x - getScalarE t)).adjoint (1 : ℝ)) :=
    adjoint_fderiv_comp_apply_one
      (f := fun W2 : Tensor ℝ [outDim, hidDim] => mlp l1 { l2 with weights := W2 } x)
      (g := fun y : Vec outDim => Tensor.meanSquaredError (ofFnE y) t)
      (hasFDerivAt_mlp_wrt_W2 l1 l2 x) (Tensor.hasFDerivAt_mse t _)
  rw [hcomp, Tensor.adjoint_mseDeriv_one, matApplyLin_adjoint_apply, ofFnE_getScalarE]

/-- The gradient of the loss with respect to the second-layer bias is `∂L/∂b2 = ∂L/∂y`, the output
gradient `getScalarE (Tensor.meanSquaredErrorBackward (ofFnE y) t)`. -/
theorem Vec.grad_b2_mse :
    (fderiv ℝ (fun b2 : Vec outDim =>
        Tensor.meanSquaredError (ofFnE (mlp l1 { l2 with bias := ofFnE b2 } x)) t)
        (getScalarE l2.bias)).adjoint (1 : ℝ)
      =
    getScalarE (Tensor.meanSquaredErrorBackward (ofFnE (mlp l1 l2 x)) t) := by
  let W1 := Rep.matrixEquiv ℝ hidDim inDim l1.weights
  let A2 : Vec hidDim →L[ℝ] Vec outDim :=
    (Matrix.toEuclideanLin (Rep.matrixEquiv ℝ outDim hidDim l2.weights)).toContinuousLinearMap
  have hf : HasFDerivAt (fun b2 : Vec outDim => mlp l1 { l2 with bias := ofFnE b2 } x)
      (ContinuousLinearMap.id ℝ (Vec outDim)) (getScalarE l2.bias) := by
    have hfun : (fun b2 : Vec outDim => mlp l1 { l2 with bias := ofFnE b2 } x) =
        fun b2 : Vec outDim => A2 (relu (affine W1 (getScalarE l1.bias) x)) + b2 := by
      funext b2
      change A2 (relu (affine W1 (getScalarE l1.bias) x)) + getScalarE (ofFnE b2) =
        A2 (relu (affine W1 (getScalarE l1.bias) x)) + b2
      rw [getScalarE_ofFnE]
    rw [hfun]
    exact (hasFDerivAt_id (getScalarE l2.bias)).const_add _
  have hy : mlp l1 { l2 with bias := ofFnE (getScalarE l2.bias) } x = mlp l1 l2 x :=
    congrArg (fun b => mlp l1 { l2 with bias := b } x) (ofFnE_getScalarE l2.bias)
  have hcomp : (fderiv ℝ (fun b2 : Vec outDim =>
        Tensor.meanSquaredError (ofFnE (mlp l1 { l2 with bias := ofFnE b2 } x)) t)
        (getScalarE l2.bias)).adjoint (1 : ℝ) =
      (ContinuousLinearMap.id ℝ (Vec outDim)).adjoint
        (((2 / (outDim : ℝ)) • innerSL ℝ
          (mlp l1 { l2 with bias := ofFnE (getScalarE l2.bias) } x - getScalarE t)).adjoint
            (1 : ℝ)) :=
    adjoint_fderiv_comp_apply_one
      (f := fun b2 : Vec outDim => mlp l1 { l2 with bias := ofFnE b2 } x)
      (g := fun y : Vec outDim => Tensor.meanSquaredError (ofFnE y) t) hf
      (Tensor.hasFDerivAt_mse t _)
  rw [hcomp, Tensor.adjoint_mseDeriv_one, hy, ContinuousLinearMap.adjoint_id,
    ContinuousLinearMap.id_apply]

/-- When every coordinate of `z₁` is nonzero, the network output has Fréchet derivative
`∂y/∂b1 = W2 ∘ ReLU'(z₁)` with respect to the first-layer bias. -/
theorem Vec.hasFDerivAt_mlp_wrt_b1
    (hx : ∀ i : Fin hidDim,
      (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x) i ≠ 0) :
    HasFDerivAt (fun b1 : Vec hidDim => mlp { l1 with bias := ofFnE b1 } l2 x)
      ((Matrix.toEuclideanLin
          (Rep.matrixEquiv ℝ outDim hidDim l2.weights)).toContinuousLinearMap.comp
        (reluDerivCLM
          (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x)))
      (getScalarE l1.bias) := by
  let W1 := Rep.matrixEquiv ℝ hidDim inDim l1.weights
  let W2 := Rep.matrixEquiv ℝ outDim hidDim l2.weights
  let A1 : Vec inDim →L[ℝ] Vec hidDim := (Matrix.toEuclideanLin W1).toContinuousLinearMap
  have hfun : (fun b1 : Vec hidDim => mlp { l1 with bias := ofFnE b1 } l2 x) =
      affine W2 (getScalarE l2.bias) ∘ relu ∘ fun b1 : Vec hidDim => A1 x + b1 := by
    funext b1
    change affine W2 (getScalarE l2.bias) (relu (A1 x + getScalarE (ofFnE b1))) =
      affine W2 (getScalarE l2.bias) (relu (A1 x + b1))
    rw [getScalarE_ofFnE]
  have hlin : HasFDerivAt (fun b1 : Vec hidDim => A1 x + b1)
      (ContinuousLinearMap.id ℝ (Vec hidDim)) (getScalarE l1.bias) :=
    (hasFDerivAt_id (getScalarE l1.bias)).const_add _
  have hrelu : HasFDerivAt relu (reluDerivCLM (affine W1 (getScalarE l1.bias) x))
      (A1 x + getScalarE l1.bias) :=
    hasFDerivAt_relu _ hx
  have hlin2 := hasFDerivAt_affine W2 (getScalarE l2.bias) (relu (A1 x + getScalarE l1.bias))
  have hcomp := HasFDerivAt.comp
    (f := relu ∘ fun b1 : Vec hidDim => A1 x + b1)
    (getScalarE l1.bias) hlin2
    (HasFDerivAt.comp (f := fun b1 : Vec hidDim => A1 x + b1) (getScalarE l1.bias) hrelu hlin)
  rw [ContinuousLinearMap.comp_id] at hcomp
  rw [hfun]
  exact hcomp

/-- When every coordinate of `z₁` is nonzero, the gradient of the loss with respect to the
first-layer bias is `ReLU'(z₁) (W2ᵀ (∂L/∂y))`. -/
theorem Vec.grad_b1_mse
    (hx : ∀ i : Fin hidDim,
      (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x) i ≠ 0) :
    (fderiv ℝ (fun b1 : Vec hidDim =>
        Tensor.meanSquaredError (ofFnE (mlp { l1 with bias := ofFnE b1 } l2 x)) t)
        (getScalarE l1.bias)).adjoint (1 : ℝ)
      =
    (reluDerivCLM (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x))
      ((Matrix.toEuclideanLin
          (Rep.matrixEquiv ℝ outDim hidDim l2.weights)).toContinuousLinearMap.adjoint
        (getScalarE (Tensor.meanSquaredErrorBackward (ofFnE (mlp l1 l2 x)) t))) := by
  have hy : mlp { l1 with bias := ofFnE (getScalarE l1.bias) } l2 x = mlp l1 l2 x :=
    congrArg (fun b => mlp { l1 with bias := b } l2 x) (ofFnE_getScalarE l1.bias)
  have hcomp : (fderiv ℝ (fun b1 : Vec hidDim =>
        Tensor.meanSquaredError (ofFnE (mlp { l1 with bias := ofFnE b1 } l2 x)) t)
        (getScalarE l1.bias)).adjoint (1 : ℝ) =
      ((Matrix.toEuclideanLin
          (Rep.matrixEquiv ℝ outDim hidDim l2.weights)).toContinuousLinearMap.comp
        (reluDerivCLM
          (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x))).adjoint
        (((2 / (outDim : ℝ)) • innerSL ℝ
          (mlp { l1 with bias := ofFnE (getScalarE l1.bias) } l2 x - getScalarE t)).adjoint
            (1 : ℝ)) :=
    adjoint_fderiv_comp_apply_one
      (f := fun b1 : Vec hidDim => mlp { l1 with bias := ofFnE b1 } l2 x)
      (g := fun y : Vec outDim => Tensor.meanSquaredError (ofFnE y) t)
      (hasFDerivAt_mlp_wrt_b1 l1 l2 x hx) (Tensor.hasFDerivAt_mse t _)
  rw [hcomp, Tensor.adjoint_mseDeriv_one, hy, ContinuousLinearMap.adjoint_comp,
    ContinuousLinearMap.comp_apply, reluDerivCLM_adjoint_apply]

/-- When every coordinate of `z₁` is nonzero, the gradient of the loss with respect to the input
`x` is `W1ᵀ (ReLU'(z₁) (W2ᵀ (∂L/∂y)))`. -/
theorem Vec.grad_x_mse
    (hx : ∀ i : Fin hidDim,
      (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x) i ≠ 0) :
    (fderiv ℝ (fun x : Vec inDim => Tensor.meanSquaredError (ofFnE (mlp l1 l2 x)) t)
        x).adjoint (1 : ℝ)
      =
    (Matrix.toEuclideanLin
        (Rep.matrixEquiv ℝ hidDim inDim l1.weights)).toContinuousLinearMap.adjoint
      ((reluDerivCLM
          (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x))
        ((Matrix.toEuclideanLin
            (Rep.matrixEquiv ℝ outDim hidDim l2.weights)).toContinuousLinearMap.adjoint
          (getScalarE (Tensor.meanSquaredErrorBackward (ofFnE (mlp l1 l2 x)) t)))) := by
  have hcomp : (fderiv ℝ (fun x : Vec inDim => Tensor.meanSquaredError (ofFnE (mlp l1 l2 x)) t)
        x).adjoint (1 : ℝ) =
      (mlpDeriv l1 l2 x).adjoint
        (((2 / (outDim : ℝ)) • innerSL ℝ (mlp l1 l2 x - getScalarE t)).adjoint (1 : ℝ)) :=
    adjoint_fderiv_comp_apply_one (f := mlp l1 l2)
      (g := fun y : Vec outDim => Tensor.meanSquaredError (ofFnE y) t)
      (hasFDerivAt_mlp l1 l2 x hx) (Tensor.hasFDerivAt_mse t _)
  rw [hcomp, Tensor.adjoint_mseDeriv_one]
  simp only [mlpDeriv, ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_apply,
    reluDerivCLM_adjoint_apply]

/-- When every coordinate of `z₁` is nonzero, the network output has Fréchet derivative
`dW1 ↦ W2 (ReLU'(z₁) (dW1 x))` with respect to the first-layer weights. -/
theorem Vec.hasFDerivAt_mlp_wrt_W1
    (hx : ∀ i : Fin hidDim,
      (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x) i ≠ 0) :
    HasFDerivAt (fun W1 : Tensor ℝ [hidDim, inDim] => mlp { l1 with weights := W1 } l2 x)
      ((Matrix.toEuclideanLin
          (Rep.matrixEquiv ℝ outDim hidDim l2.weights)).toContinuousLinearMap.comp
        ((reluDerivCLM
            (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x)).comp
          (matApplyLin (m := hidDim) (n := inDim) x))) l1.weights := by
  let L1 := matApplyLin (m := hidDim) (n := inDim) x
  let W2 := Rep.matrixEquiv ℝ outDim hidDim l2.weights
  have hlin : HasFDerivAt (fun W1 : Tensor ℝ [hidDim, inDim] => L1 W1 + getScalarE l1.bias)
      L1 l1.weights :=
    L1.hasFDerivAt.add_const (getScalarE l1.bias)
  have hrelu : HasFDerivAt relu
      (reluDerivCLM (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x))
      (L1 l1.weights + getScalarE l1.bias) :=
    hasFDerivAt_relu _ hx
  have hlin2 :=
    hasFDerivAt_affine W2 (getScalarE l2.bias) (relu (L1 l1.weights + getScalarE l1.bias))
  exact HasFDerivAt.comp
    (f := relu ∘ fun W1 : Tensor ℝ [hidDim, inDim] => L1 W1 + getScalarE l1.bias)
    l1.weights hlin2
    (HasFDerivAt.comp (f := fun W1 : Tensor ℝ [hidDim, inDim] => L1 W1 + getScalarE l1.bias)
      l1.weights hrelu hlin)

/-- When every coordinate of `z₁` is nonzero, the gradient of the loss with respect to
`l1.weights` is the outer product `Linear.weightsDeriv` of the backpropagated hidden gradient
`ReLU'(z₁) (W2ᵀ (∂L/∂y))` and the input `x`. -/
theorem Vec.grad_W1_mse
    (hx : ∀ i : Fin hidDim,
      (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x) i ≠ 0) :
    (fderiv ℝ (fun W1 : Tensor ℝ [hidDim, inDim] =>
        Tensor.meanSquaredError (ofFnE (mlp { l1 with weights := W1 } l2 x)) t)
        l1.weights).adjoint (1 : ℝ)
      =
    Linear.weightsDeriv (ofFnE x)
      (ofFnE ((reluDerivCLM
          (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x))
        ((Matrix.toEuclideanLin
            (Rep.matrixEquiv ℝ outDim hidDim l2.weights)).toContinuousLinearMap.adjoint
          (getScalarE (Tensor.meanSquaredErrorBackward (ofFnE (mlp l1 l2 x)) t))))) := by
  have hcomp : (fderiv ℝ (fun W1 : Tensor ℝ [hidDim, inDim] =>
        Tensor.meanSquaredError (ofFnE (mlp { l1 with weights := W1 } l2 x)) t)
        l1.weights).adjoint (1 : ℝ) =
      ((Matrix.toEuclideanLin
          (Rep.matrixEquiv ℝ outDim hidDim l2.weights)).toContinuousLinearMap.comp
        ((reluDerivCLM
            (affine (Rep.matrixEquiv ℝ hidDim inDim l1.weights) (getScalarE l1.bias) x)).comp
          (matApplyLin (m := hidDim) (n := inDim) x))).adjoint
        (((2 / (outDim : ℝ)) • innerSL ℝ (mlp l1 l2 x - getScalarE t)).adjoint (1 : ℝ)) :=
    adjoint_fderiv_comp_apply_one
      (f := fun W1 : Tensor ℝ [hidDim, inDim] => mlp { l1 with weights := W1 } l2 x)
      (g := fun y : Vec outDim => Tensor.meanSquaredError (ofFnE y) t)
      (hasFDerivAt_mlp_wrt_W1 l1 l2 x hx) (Tensor.hasFDerivAt_mse t _)
  rw [hcomp, Tensor.adjoint_mseDeriv_one]
  simp only [ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_apply,
    reluDerivCLM_adjoint_apply, matApplyLin_adjoint_apply]

end

end
end TorchLean
