/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.InnerProductSpace.Calculus
public import NN.Proofs.Autograd.FDeriv.Params

/-!
# MlpMse

End-to-end analytic soundness for a first training target:

* 2-layer MLP (Linear → ReLU → Linear)
* MSE loss

We prove that the usual backprop formulas (including **parameter gradients**) coincide with the
adjoint of the Fréchet derivative (`fderiv`) over `ℝ`.

Notes:
- This is a spec-level (`ℝ`) theorem.
- ReLU is not differentiable at 0, so we assume a "no kinks" hypothesis on the pre-activation.
- Reverse-mode is naturally stated as a VJP theorem for vector outputs; scalar loss gradients are
  obtained by choosing `δ = 1` (equivalently `VJP[loss, y] 1 = ∇ loss y`).

## PyTorch correspondence / citations
- MLP building blocks: `torch.nn.Linear`, `torch.nn.functional.relu`.
  https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
  https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
- MSE loss reference behavior:
  https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
- “Backward equals adjoint of derivative” is exactly the Jacobian-transpose theorem for PyTorch
  autograd:
  https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec _root_.TorchLean
open _root_.TorchLean _root_.TorchLean.Tensor
open scoped BigOperators
open scoped _root_.Autograd

noncomputable section

-- ---------------------------------------------------------------------------
-- ReLU derivative map is self-adjoint (coordinatewise scaling on ℝⁿ)
-- ---------------------------------------------------------------------------

/-- Coordinate formula for `reluDerivCLM`: it scales each coordinate by `relu'(xᵢ)`. -/
theorem reluDerivCLM_apply {n : Nat} (x dx : Vec n) (i : Fin n) :
    (reluDerivCLM (n := n) x) dx i = dx i * Activation.Math.reluDerivSpec (x i) := rfl

/--
ReLU derivative map is self-adjoint w.r.t. the Euclidean inner product.

This is because it is a diagonal scaling map on `ℝⁿ`.
-/
theorem reluDerivCLM_inner {n : Nat} (x dx δ : Vec n) :
    inner ℝ ((reluDerivCLM (n := n) x) dx) δ = inner ℝ dx ((reluDerivCLM (n := n) x) δ) := by
  classical
  -- Expand inner products into coordinate sums and commute scalars.
  simp [inner_eq_sum_mul, reluDerivCLM_apply, mul_assoc, mul_left_comm, mul_comm]

/-- The adjoint of `reluDerivCLM` equals itself (self-adjoint operator). -/
theorem reluDerivCLM_adjoint_apply {n : Nat} (x δ : Vec n) :
    (reluDerivCLM (n := n) x).adjoint δ = (reluDerivCLM (n := n) x) δ := by
  apply ext_inner_left ℝ
  intro dx
  rw [ContinuousLinearMap.adjoint_inner_right]
  exact reluDerivCLM_inner x dx δ

-- ---------------------------------------------------------------------------
-- Linear layer: `vecMatMulSpec` corresponds to the adjoint on Euclidean vectors
-- ---------------------------------------------------------------------------

/--
The spec-level “input derivative” for a linear layer agrees with the Euclidean adjoint.

In words: the tensor expression for `∂(W x)/∂x` applied to an upstream `δ` is `Wᵀ δ`, and this is
exactly the adjoint of the CLM `x ↦ W x`.
-/
theorem getScalarE_linearInputDerivSpec_eq_adjoint
    {inDim outDim : Nat}
    (W : Tensor ℝ [outDim, inDim])
    (δ : Tensor ℝ [outDim]) :
    getScalarE (Spec.linearInputDerivSpec (inDim := inDim) (outDim := outDim) W δ)
      =
    (matCLM (m := outDim) (n := inDim) (tensorToMatrix (m := outDim) (n := inDim) W)).adjoint
      (getScalarE δ) := by
  classical
  -- Let `A x := W x` (as a continuous linear map on Euclidean vectors).
  let A : Vec inDim →L[ℝ] Vec outDim :=
    matCLM (m := outDim) (n := inDim) (tensorToMatrix (m := outDim) (n := inDim) W)
  let u : Vec inDim := getScalarE (vecMatMulSpec δ W)
  let v : Vec inDim := A.adjoint (getScalarE δ)

  have hforall : ∀ dxV : Vec inDim, inner ℝ dxV u = inner ℝ dxV v := by
    intro dxV
    -- Tensor-level adjointness (dot) for mat-vec vs vec-mat.
    have hdot :=
      dot_mat_linear_adjoint (inDim := inDim) (outDim := outDim)
        (W := W) (dLdy := δ) (dx := ofFnE dxV)
    -- Translate `dot` to `inner`.
    have hinner :
        inner ℝ (getScalarE δ) (getScalarE (Spec.matVecMulSpec W (ofFnE dxV)))
          =
        inner ℝ (getScalarE (vecMatMulSpec δ W)) dxV := by
      simpa [dot_eq_inner_vec, getScalarE_ofFnE] using hdot
    -- Identify the mat-vec output with `A dxV`.
    have hAx : getScalarE (Spec.matVecMulSpec W (ofFnE dxV)) = A dxV := by
      simpa [A] using
        (getScalarE_mat_vec_mul_spec (m := outDim) (n := inDim) (A := W) (v := ofFnE dxV))
    -- Use symmetry + the defining property of the adjoint.
    calc
      inner ℝ dxV u
          = inner ℝ u dxV := by simp [real_inner_comm]
      _ = inner ℝ (getScalarE δ) (A dxV) := by
            have htmp := hinner.symm
            -- Rewrite the mat-vec output to `A dxV` explicitly (avoid simp rewriting order issues).
            rw [hAx] at htmp
            simpa [u] using htmp
      _ = inner ℝ (A dxV) (getScalarE δ) := by simp [real_inner_comm]
      _ = inner ℝ dxV v := by
            simpa [v] using
              (ContinuousLinearMap.adjoint_inner_right (A := A) (x := dxV) (y := getScalarE δ)).symm

  -- A vector is determined by its inner products with every tangent.
  have huv : u = v := ext_inner_left ℝ hforall
  simpa [u, v, Spec.linearInputDerivSpec] using huv

-- ---------------------------------------------------------------------------
-- 2-layer MLP in Mat/Vec form
-- ---------------------------------------------------------------------------

/-- Affine map using `Mat` parameters (same as `affine`, but with `Mat` instead of `Matrix`). -/
def affineMat {inDim outDim : Nat} (W : Mat outDim inDim) (b : Vec outDim) : Vec inDim → Vec outDim
  :=
  affine (inDim := inDim) (outDim := outDim) (toMatrix (m := outDim) (n := inDim) W) b

/--
2-layer MLP in Euclidean `Vec` form, parameterized by `Mat` weights and `Vec` biases.

This is the same computation as `NN.Proofs.Autograd.FDeriv.Core`’s `mlpVec`, but set up for
parameter-gradient proofs where the parameter space is a Hilbert space.
-/
def mlpVecMat {inDim hidDim outDim : Nat}
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim) : Vec inDim → Vec outDim :=
  fun x =>
    let z1 := affineMat (inDim := inDim) (outDim := hidDim) W1 b1 x
    let a1 := reluVec (n := hidDim) z1
    affineMat (inDim := hidDim) (outDim := outDim) W2 b2 a1

/--
Mean-squared error loss (MSE) against a fixed target `t`:

`mse t y = (1/n) * ‖y - t‖²`.
-/
def mse {n : Nat} (t : Vec n) : Vec n → ℝ :=
  fun y => ((n : ℝ)⁻¹) * ‖y - t‖ ^ 2

/--
Gradient of MSE with respect to `y`:

`∇_y mse(t)(y) = (2/n) * (y - t)`.
-/
def mseGrad {n : Nat} (y t : Vec n) : Vec n :=
  (2 / (n : ℝ)) • (y - t)

/-- Fréchet derivative of MSE, packaged as a continuous linear map `Vec n →L ℝ`. -/
theorem hasFDerivAt_mse {n : Nat} (t y : Vec n) :
    HasFDerivAt (mse (n := n) t) ((2 / (n : ℝ)) • (innerSL ℝ (y - t))) y := by
  apply (((hasFDerivAt_id y).sub_const t).norm_sq.const_mul (n : ℝ)⁻¹).congr_fderiv
  ext z
  simp [div_eq_mul_inv, mul_assoc, mul_comm]

/-- The adjoint of the MSE derivative, applied to the seed `1`, is the gradient `mseGrad y t`. -/
theorem adjoint_mseDeriv_one {n : Nat} (t y : Vec n) :
    ((2 / (n : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ) = mseGrad (n := n) y t := by
  have hinner : (ContinuousLinearMap.adjoint ((innerSL ℝ) (y - t))) (1 : ℝ) = y - t := by
    simpa using congrArg (fun f => f (1 : ℝ))
      (ContinuousLinearMap.adjoint_innerSL_apply (𝕜 := ℝ) (x := y - t))
  calc
    ((2 / (n : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)
        = (2 / (n : ℝ)) • (innerSL ℝ (y - t)).adjoint (1 : ℝ) := by
          simp
    _ = (2 / (n : ℝ)) • (y - t) := by
          rw [hinner]
    _ = mseGrad (n := n) y t := rfl

/--
The VJP of MSE at `y` with upstream seed `1` equals the usual gradient `mseGrad y t`.

This is the scalar-loss specialization: for scalar loss `ℓ`, the gradient is `(fderiv ℓ)† 1`.
-/
theorem mseGrad_eq_adjoint_fderiv {n : Nat} (t y : Vec n) :
    VJP[mse (n := n) t, y] (1 : ℝ) = mseGrad (n := n) y t := by
  change (fderiv ℝ (mse t) y).adjoint (1 : ℝ) = _
  rw [(hasFDerivAt_mse t y).fderiv]
  exact adjoint_mseDeriv_one t y

/--
Convenience lemma: adjoint of the derivative of a scalar composition, applied to seed `1`.

For scalar loss `g ∘ f`, this is the reverse-mode “chain rule” in adjoint form.
-/
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

-- ---------------------------------------------------------------------------
-- Scalar-loss gradient theorems (inputs + parameters)
-- ---------------------------------------------------------------------------

section

variable {inDim hidDim outDim : Nat}

variable (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
variable (W2 : Mat outDim hidDim) (b2 : Vec outDim)
variable (x : Vec inDim) (t : Vec outDim)

/-!
We name the intermediate activations so the gradient statements read like textbook backprop:
`mlpPreActivation` is `z₁ = W₁ x + b₁`, `mlpHidden` is `a₁ = relu z₁`, and `mlpOutput` is the
network output `y = W₂ a₁ + b₂`.
-/

/-- Pre-activation `W1 x + b1` (the ReLU differentiability hypothesis is stated on it). -/
def mlpPreActivation : Vec hidDim :=
  affineMat (inDim := inDim) (outDim := hidDim) W1 b1 x

/-- Hidden activation `relu (W1 x + b1)`. -/
def mlpHidden : Vec hidDim :=
  reluVec (n := hidDim) (mlpPreActivation W1 b1 x)

/-- Network output `mlpVecMat W1 b1 W2 b2 x`. -/
def mlpOutput : Vec outDim :=
  mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x

/--
Fréchet derivative of the network output with respect to the second-layer weights `W2`.

Informally: `∂y/∂W2` is the linear map `dW2 ↦ dW2 a1`.
-/
theorem hasFDerivAt_mlp_wrt_W2 :
    HasFDerivAt (fun W2 : Mat outDim hidDim => mlpVecMat W1 b1 W2 b2 x)
      (matApplyLin (m := outDim) (n := hidDim) (mlpHidden W1 b1 x)) W2 := by
  -- The only dependence on `W2` is the final affine map `W2 ↦ W2 a1 + b2`, which is the
  -- continuous linear map `matApplyLin a1` plus a constant.
  have hfun :
      (fun W2 : Mat outDim hidDim => mlpVecMat W1 b1 W2 b2 x) =
        fun W2 : Mat outDim hidDim =>
          (matApplyLin (m := outDim) (n := hidDim) (mlpHidden W1 b1 x)) W2 + b2 := by
    funext W2
    rfl
  rw [hfun]
  exact (matApplyLin (m := outDim) (n := hidDim) (mlpHidden W1 b1 x)).hasFDerivAt.add_const b2

/--
Closed-form gradient of scalar loss `mse t (mlpVecMat …)` with respect to `W2`.

Result: `∂L/∂W2 = (∂L/∂y) ⊗ a1`, i.e. outer product of the output gradient and hidden activation.
-/
theorem grad_W2_mse :
    (fderiv ℝ (fun W2 : Mat outDim hidDim => mse (n := outDim) t (mlpVecMat W1 b1 W2 b2 x))
        W2).adjoint (1 : ℝ)
      =
    outer (m := outDim) (n := hidDim) (mseGrad (n := outDim) (mlpOutput W1 b1 W2 b2 x) t)
      (mlpHidden W1 b1 x) := by
  -- Compose `mse` with the `W2`-slice of the network.
  let f : Mat outDim hidDim → Vec outDim := fun W2 => mlpVecMat W1 b1 W2 b2 x
  let y : Vec outDim := f W2
  have hf : HasFDerivAt f (matApplyLin (m := outDim) (n := hidDim) (mlpHidden W1 b1 x)) W2 :=
    hasFDerivAt_mlp_wrt_W2 W1 b1 W2 b2 x
  have hg : HasFDerivAt (mse (n := outDim) t) ((2 / (outDim : ℝ)) • innerSL ℝ (y - t)) y :=
    hasFDerivAt_mse (n := outDim) t y
  have hcomp :=
    adjoint_fderiv_comp_apply_one (f := f) (g := mse (n := outDim) t) (x := W2) hf hg
  have hδ := adjoint_mseDeriv_one (n := outDim) t y
  calc
    (fderiv ℝ (fun W2 => mse (n := outDim) t (f W2)) W2).adjoint (1 : ℝ)
        = (matApplyLin (m := outDim) (n := hidDim) (mlpHidden W1 b1 x)).adjoint
            (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) := by
            simpa [f, y] using hcomp
    _ = (matApplyLin (m := outDim) (n := hidDim) (mlpHidden W1 b1 x)).adjoint
          (mseGrad (n := outDim) y t) := by
          rw [hδ]
    _ = outer (m := outDim) (n := hidDim) (mseGrad (n := outDim) y t) (mlpHidden W1 b1 x) :=
          matApplyLin_adjoint_apply (mlpHidden W1 b1 x) (mseGrad (n := outDim) y t)

/--
Closed-form gradient of the scalar loss with respect to the second-layer bias `b2`.

Result: `∂L/∂b2 = ∂L/∂y`.
-/
theorem grad_b2_mse :
    (fderiv ℝ (fun b2 : Vec outDim => mse (n := outDim) t (mlpVecMat W1 b1 W2 b2 x))
        b2).adjoint (1 : ℝ)
      =
    mseGrad (n := outDim) (mlpVecMat W1 b1 W2 b2 x) t := by
  let f : Vec outDim → Vec outDim := fun b2 => mlpVecMat W1 b1 W2 b2 x
  let y : Vec outDim := f b2
  have hf : HasFDerivAt f (1 : Vec outDim →L[ℝ] Vec outDim) b2 := by
    -- `b2 ↦ (W2 a1) + b2` is affine with derivative `1`.
    dsimp [f, mlpVecMat, mlpPreActivation, affineMat, affine]
    -- Remaining `let`-binders are constant in `b2`.
    change HasFDerivAt
      (fun b2 : Vec outDim =>
        (matCLM (m := outDim) (n := hidDim) (toMatrix W2))
          (reluVec (n := hidDim) (mlpPreActivation W1 b1 x)) + b2)
      (ContinuousLinearMap.id ℝ (Vec outDim)) b2
    simpa using (HasFDerivAt.const_add (c := (matCLM (m := outDim) (n := hidDim) (toMatrix W2))
        (reluVec (n := hidDim) (mlpPreActivation W1 b1 x))) (hasFDerivAt_id b2))
  have hg : HasFDerivAt (mse (n := outDim) t) ((2 / (outDim : ℝ)) • innerSL ℝ (y - t)) y :=
    hasFDerivAt_mse (n := outDim) t y
  have hcomp :=
    adjoint_fderiv_comp_apply_one (f := f) (g := mse (n := outDim) t) (x := b2) hf hg
  have hδ := adjoint_mseDeriv_one (n := outDim) t y
  calc
    (fderiv ℝ (fun b2 => mse (n := outDim) t (f b2)) b2).adjoint (1 : ℝ)
        = (1 : Vec outDim →L[ℝ] Vec outDim).adjoint (((2 / (outDim : ℝ)) • innerSL ℝ (y -
          t)).adjoint (1 : ℝ)) := by
            simpa [f, y] using hcomp
    _ = (1 : Vec outDim →L[ℝ] Vec outDim).adjoint (mseGrad (n := outDim) y t) := by
          rw [hδ]
    _ = mseGrad (n := outDim) y t := by
          -- `simp` doesn't unfold `1` to `ContinuousLinearMap.id` in this context.
          change ((ContinuousLinearMap.id ℝ (Vec outDim)).adjoint (mseGrad (n := outDim) y t)) =
            mseGrad (n := outDim) y t
          rw [ContinuousLinearMap.adjoint_id]
          rfl

/--
Fréchet derivative of the network output with respect to the first-layer bias `b1`,
under the ReLU “no kinks” hypothesis.

Informally: `∂y/∂b1 = W2 ∘ ReLU'(z1)`.
-/
theorem hasFDerivAt_mlp_wrt_b1 (hx : ∀ i : Fin hidDim, (mlpPreActivation W1 b1 x) i ≠ 0) :
    HasFDerivAt (fun b1 : Vec hidDim => mlpVecMat W1 b1 W2 b2 x)
      ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
        (reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))) b1 := by
  -- `b1 ↦ affine W2 b2 (relu (affine W1 b1 x))`
  let z1V : Vec hidDim := mlpPreActivation W1 b1 x
  have hlin :
      HasFDerivAt (fun b1 : Vec hidDim => affineMat (inDim := inDim) (outDim := hidDim) W1 b1 x)
        (1 : Vec hidDim →L[ℝ] Vec hidDim) b1 := by
    -- `b1 ↦ const + b1`
    dsimp [affineMat, affine]
    change HasFDerivAt
      (fun b1 : Vec hidDim => (matCLM (m := hidDim) (n := inDim) (toMatrix W1)) x + b1)
      (ContinuousLinearMap.id ℝ (Vec hidDim)) b1
    simpa using (HasFDerivAt.const_add (c := (matCLM (m := hidDim) (n := inDim) (toMatrix W1)) x)
      (hasFDerivAt_id b1))
  have hrelu :
      HasFDerivAt (reluVec (n := hidDim)) (reluDerivCLM (n := hidDim) z1V) z1V :=
    hasFDerivAt_reluVec (n := hidDim) (x := z1V) (hx := by
      intro i
      simpa [z1V] using hx i)
  have hlin2 :
      HasFDerivAt (affineMat (inDim := hidDim) (outDim := outDim) W2 b2)
        (matCLM (m := outDim) (n := hidDim) (toMatrix W2)) (reluVec z1V) :=
    hasFDerivAt_affine (inDim := hidDim) (outDim := outDim) (W := toMatrix W2) (b := b2) (x :=
      reluVec z1V)
  have hcomp1 := hrelu.comp b1 hlin
  have hcomp2 := hlin2.comp b1 hcomp1
  have hlinId :
      (matCLM (m := outDim) (n := hidDim) (toMatrix W2) ∘SL
          reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x) ∘SL
            (1 : Vec hidDim →L[ℝ] Vec hidDim)) =
        (matCLM (m := outDim) (n := hidDim) (toMatrix W2) ∘SL
          reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x)) := by
    ext db1
    simp [ContinuousLinearMap.comp_apply]
  change HasFDerivAt
    (fun b1 : Vec hidDim =>
      affine (inDim := hidDim) (outDim := outDim) (toMatrix W2) b2
        (reluVec (n := hidDim)
          (affine (inDim := inDim) (outDim := hidDim) (toMatrix W1) b1 x)))
    ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
      (reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))) b1
  simpa [mlpVecMat, affineMat, z1V, mlpPreActivation, Function.comp_def,
    ContinuousLinearMap.comp_assoc] using hcomp2.congr_fderiv hlinId

/--
Closed-form gradient of the scalar loss with respect to the first-layer bias `b1`
(under the ReLU “no kinks” hypothesis).
-/
theorem grad_b1_mse (hx : ∀ i : Fin hidDim, (mlpPreActivation W1 b1 x) i ≠ 0) :
    (fderiv ℝ (fun b1 : Vec hidDim => mse (n := outDim) t (mlpVecMat W1 b1 W2 b2 x))
        b1).adjoint (1 : ℝ)
      =
    (reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))
      ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint
        (mseGrad (n := outDim) (mlpOutput W1 b1 W2 b2 x) t)) := by
  let f : Vec hidDim → Vec outDim := fun b1 => mlpVecMat W1 b1 W2 b2 x
  let y : Vec outDim := f b1
  have hf : HasFDerivAt f
      ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
        (reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))) b1 :=
    hasFDerivAt_mlp_wrt_b1 W1 b1 W2 b2 x hx
  have hg : HasFDerivAt (mse (n := outDim) t) ((2 / (outDim : ℝ)) • innerSL ℝ (y - t)) y :=
    hasFDerivAt_mse (n := outDim) t y
  have hcomp :=
    adjoint_fderiv_comp_apply_one (f := f) (g := mse (n := outDim) t) (x := b1) hf hg
  have hδ := adjoint_mseDeriv_one (n := outDim) t y
  calc
    (fderiv ℝ (fun b1 => mse (n := outDim) t (f b1)) b1).adjoint (1 : ℝ)
        = ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
            (reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))).adjoint
            (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) := by
            simpa [f, y] using hcomp
    _ = ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
            (reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))).adjoint
            (mseGrad (n := outDim) y t) := by
          rw [hδ]
    _ = (reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x)).adjoint
          ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y t))
            := by
          simp [ContinuousLinearMap.adjoint_comp]
    _ = (reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))
          ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y t))
            := by
          simp [reluDerivCLM_adjoint_apply]

/--
Fréchet derivative of the network output with respect to the input `x`,
under the ReLU “no kinks” hypothesis.

Informally: `∂y/∂x = W2 ∘ ReLU'(z1) ∘ W1`.
-/
theorem hasFDerivAt_mlp_wrt_x (hx : ∀ i : Fin hidDim, (mlpPreActivation W1 b1 x) i ≠ 0) :
    HasFDerivAt (mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2)
      ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
        ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x)).comp
          (matCLM (m := hidDim) (n := inDim) (toMatrix W1)))) x := by
  -- Same proof as `hasFDerivAt_mlpVec`, just with parameters in `Mat`.
  let z1V : Vec hidDim := mlpPreActivation W1 b1 x
  have hlin1 :
      HasFDerivAt (affineMat (inDim := inDim) (outDim := hidDim) W1 b1)
        (matCLM (m := hidDim) (n := inDim) (toMatrix W1)) x :=
    hasFDerivAt_affine (inDim := inDim) (outDim := hidDim) (W := toMatrix W1) (b := b1) (x := x)
  have hrelu :
      HasFDerivAt (reluVec (n := hidDim)) (reluDerivCLM (n := hidDim) z1V) z1V :=
    hasFDerivAt_reluVec (n := hidDim) (x := z1V) (hx := by
      intro i
      simpa [z1V] using hx i)
  have hlin2 :
      HasFDerivAt (affineMat (inDim := hidDim) (outDim := outDim) W2 b2)
        (matCLM (m := outDim) (n := hidDim) (toMatrix W2)) (reluVec z1V) :=
    hasFDerivAt_affine (inDim := hidDim) (outDim := outDim) (W := toMatrix W2) (b := b2) (x :=
      reluVec z1V)
  have hcomp1 := hrelu.comp x hlin1
  have hcomp2 := hlin2.comp x hcomp1
  change HasFDerivAt
    (fun x : Vec inDim =>
      affine (inDim := hidDim) (outDim := outDim) (toMatrix W2) b2
        (reluVec (n := hidDim)
          (affine (inDim := inDim) (outDim := hidDim) (toMatrix W1) b1 x)))
    ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
      ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x)).comp
        (matCLM (m := hidDim) (n := inDim) (toMatrix W1)))) x
  simpa [mlpVecMat, affineMat, z1V, mlpPreActivation, Function.comp_def,
    ContinuousLinearMap.comp_assoc] using hcomp2

/--
Closed-form gradient of the scalar loss with respect to the input `x`,
under the ReLU “no kinks” hypothesis.
-/
theorem grad_x_mse (hx : ∀ i : Fin hidDim, (mlpPreActivation W1 b1 x) i ≠ 0) :
    (fderiv ℝ (fun x : Vec inDim => mse (n := outDim) t (mlpVecMat W1 b1 W2 b2 x))
        x).adjoint (1 : ℝ)
      =
    (matCLM (m := hidDim) (n := inDim) (toMatrix W1)).adjoint
      ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))
        ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint
          (mseGrad (n := outDim) (mlpOutput W1 b1 W2 b2 x) t))) := by
  let f : Vec inDim → Vec outDim :=
    mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2
  let y : Vec outDim := f x
  let f' : Vec inDim →L[ℝ] Vec outDim :=
    (matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
      ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x)).comp
        (matCLM (m := hidDim) (n := inDim) (toMatrix W1)))
  have hf : HasFDerivAt f f' x := hasFDerivAt_mlp_wrt_x W1 b1 W2 b2 x hx
  have hg : HasFDerivAt (mse (n := outDim) t) ((2 / (outDim : ℝ)) • innerSL ℝ (y - t)) y :=
    hasFDerivAt_mse (n := outDim) t y
  have hcomp := adjoint_fderiv_comp_apply_one (f := f) (g := mse (n := outDim) t) (x := x) hf hg
  have hδ := adjoint_mseDeriv_one (n := outDim) t y
  calc
    (fderiv ℝ (fun x => mse (n := outDim) t (f x)) x).adjoint (1 : ℝ)
        = f'.adjoint (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) := by
            simpa [f, y, f'] using hcomp
    _ = f'.adjoint (mseGrad (n := outDim) y t) := by
          rw [hδ]
    _ = (matCLM (m := hidDim) (n := inDim) (toMatrix W1)).adjoint
          ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x)).adjoint
            ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
              t))) := by
          simp [f', ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_assoc]
    _ = (matCLM (m := hidDim) (n := inDim) (toMatrix W1)).adjoint
          ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))
            ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
              t))) := by
          simp [reluDerivCLM_adjoint_apply]

/--
Fréchet derivative of the network output with respect to the first-layer weights `W1`,
under the ReLU “no kinks” hypothesis.

The derivative is linear in `W1` through the slice `dW1 ↦ dW1 x`, then propagated through
`ReLU'` and `W2`.
-/
theorem hasFDerivAt_mlp_wrt_W1 (hx : ∀ i : Fin hidDim, (mlpPreActivation W1 b1 x) i ≠ 0) :
    HasFDerivAt (fun W1 : Mat hidDim inDim => mlpVecMat W1 b1 W2 b2 x)
      ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
        ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x)).comp
          (matApplyLin (m := hidDim) (n := inDim) x))) W1 := by
  -- `W1 ↦ affine W2 b2 (relu (W1 x + b1))`
  let z1V : Vec hidDim := mlpPreActivation W1 b1 x
  have hlin :
    HasFDerivAt (fun W1 : Mat hidDim inDim => (matApplyLin (m := hidDim) (n := inDim) x) W1 + b1)
        (matApplyLin (m := hidDim) (n := inDim) x) W1 := by
    simpa using (HasFDerivAt.add_const (c := b1)
      (ContinuousLinearMap.hasFDerivAt (matApplyLin (m := hidDim) (n := inDim) x)))
  have hrelu :
      HasFDerivAt (reluVec (n := hidDim)) (reluDerivCLM (n := hidDim) z1V) z1V :=
    hasFDerivAt_reluVec (n := hidDim) (x := z1V) (hx := by
      intro i
      simpa [z1V] using hx i)
  have hlin2 :
      HasFDerivAt (affineMat (inDim := hidDim) (outDim := outDim) W2 b2)
        (matCLM (m := outDim) (n := hidDim) (toMatrix W2)) (reluVec z1V) :=
    hasFDerivAt_affine (inDim := hidDim) (outDim := outDim) (W := toMatrix W2) (b := b2) (x :=
      reluVec z1V)
  have hcomp1 := hrelu.comp W1 hlin
  have hcomp2 := hlin2.comp W1 hcomp1
  change HasFDerivAt
    (fun W1 : Mat hidDim inDim =>
      affine (inDim := hidDim) (outDim := outDim) (toMatrix W2) b2
        (reluVec (n := hidDim)
          ((matApplyLin (m := hidDim) (n := inDim) x) W1 + b1)))
    ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
      ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x)).comp
        (matApplyLin (m := hidDim) (n := inDim) x))) W1
  simpa [mlpVecMat, affineMat, z1V, mlpPreActivation, Function.comp_def,
    ContinuousLinearMap.comp_assoc] using hcomp2

/--
Closed-form gradient of the scalar loss with respect to `W1`
(under the ReLU “no kinks” hypothesis).

Result has the expected “outer product” form with backpropagated hidden gradient and input `x`.
-/
theorem grad_W1_mse (hx : ∀ i : Fin hidDim, (mlpPreActivation W1 b1 x) i ≠ 0) :
    (fderiv ℝ (fun W1 : Mat hidDim inDim => mse (n := outDim) t (mlpVecMat W1 b1 W2 b2 x))
        W1).adjoint (1 : ℝ)
      =
    outer (m := hidDim) (n := inDim)
      ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))
        ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint
          (mseGrad (n := outDim) (mlpOutput W1 b1 W2 b2 x) t)))
      x := by
  let f : Mat hidDim inDim → Vec outDim := fun W1 => mlpVecMat W1 b1 W2 b2 x
  let y : Vec outDim := f W1
  let f' : Mat hidDim inDim →L[ℝ] Vec outDim :=
    (matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
      ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x)).comp
        (matApplyLin (m := hidDim) (n := inDim) x))
  have hf : HasFDerivAt f f' W1 := hasFDerivAt_mlp_wrt_W1 W1 b1 W2 b2 x hx
  have hg : HasFDerivAt (mse (n := outDim) t) ((2 / (outDim : ℝ)) • innerSL ℝ (y - t)) y :=
    hasFDerivAt_mse (n := outDim) t y
  have hcomp := adjoint_fderiv_comp_apply_one (f := f) (g := mse (n := outDim) t) (x := W1) hf hg
  have hδ := adjoint_mseDeriv_one (n := outDim) t y
  calc
    (fderiv ℝ (fun W1 => mse (n := outDim) t (f W1)) W1).adjoint (1 : ℝ)
        = f'.adjoint (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) := by
            simpa [f, y, f'] using hcomp
    _ = f'.adjoint (mseGrad (n := outDim) y t) := by
          rw [hδ]
    _ = (matApplyLin (m := hidDim) (n := inDim) x).adjoint
          ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x)).adjoint
            ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
              t))) := by
          simp [f', ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_assoc]
    _ = (matApplyLin (m := hidDim) (n := inDim) x).adjoint
          ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))
            ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
              t))) := by
          simp [reluDerivCLM_adjoint_apply]
    _ = outer (m := hidDim) (n := inDim)
          ((reluDerivCLM (n := hidDim) (mlpPreActivation W1 b1 x))
            ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
              t))) x :=
          matApplyLin_adjoint_apply x _

end

end
end Autograd
end Proofs
