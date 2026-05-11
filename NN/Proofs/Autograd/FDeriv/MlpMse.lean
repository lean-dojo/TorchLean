/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Params
public import NN.Proofs.Autograd.Notation
public import NN.Spec.Models.Mlp

public import Mathlib.Analysis.InnerProductSpace.Calculus

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

open Spec
open Tensor
open scoped BigOperators
open scoped _root_.Autograd

noncomputable section

-- ---------------------------------------------------------------------------
-- Tensor ↔ Mat conversion (only for matrix-shaped tensors)
-- ---------------------------------------------------------------------------

/--
Convert a matrix-shaped tensor `W : Tensor ℝ (m×n)` into the Hilbert-space matrix type `Mat m n`.

This is used for parameter-gradient proofs, where the parameter space is a Hilbert space with the
Frobenius/L2 inner product.
-/
def toMatE {m n : Nat} (W : Tensor ℝ (.dim m (.dim n .scalar))) : Mat m n :=
  WithLp.toLp 2 fun i : Fin m =>
    WithLp.toLp 2 fun j : Fin n =>
      Spec.get2 W i j

/-- `toMatE` agrees with the coordinate-level view `tensorToMatrix` used by the FDeriv core. -/
lemma toMatrix_toMatE {m n : Nat} (W : Tensor ℝ (.dim m (.dim n .scalar))) :
    toMatrix (m := m) (n := n) (toMatE W) = tensorToMatrix (m := m) (n := n) W := by
  funext i j
  simp [toMatE, toMatrix, tensorToMatrix]

-- ---------------------------------------------------------------------------
-- ReLU derivative map is self-adjoint (coordinatewise scaling on ℝⁿ)
-- ---------------------------------------------------------------------------

/-- Coordinate formula for `reluDerivCLM`: it scales each coordinate by `relu'(xᵢ)`. -/
lemma reluDerivCLM_apply {n : Nat} (x dx : Vec n) (i : Fin n) :
    (reluDerivCLM (n := n) x) dx i = dx i * Activation.Math.reluDerivSpec (x i) := by
  -- `reluDerivCLM` is implemented by transporting the pointwise derivative map through
  -- `EuclideanSpace.equiv`.
  let e := EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)
  have :
      (e ((reluDerivCLM (n := n) x) dx)) i =
        (e dx) i * Activation.Math.reluDerivSpec ((e x) i) := by
    simp [reluDerivCLM, reluFunDeriv, e, ContinuousLinearMap.comp_apply,
      ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.proj_apply, smul_eq_mul]
  simpa [e] using this

/--
ReLU derivative map is self-adjoint w.r.t. the Euclidean inner product.

This is because it is a diagonal scaling map on `ℝⁿ`.
-/
lemma reluDerivCLM_inner {n : Nat} (x dx δ : Vec n) :
    inner ℝ ((reluDerivCLM (n := n) x) dx) δ = inner ℝ dx ((reluDerivCLM (n := n) x) δ) := by
  classical
  -- Expand inner products into coordinate sums and commute scalars.
  simp [inner_eq_sum_mul, reluDerivCLM_apply, mul_assoc, mul_left_comm, mul_comm]

/-- The adjoint of `reluDerivCLM` equals itself (self-adjoint operator). -/
lemma reluDerivCLM_adjoint_apply {n : Nat} (x δ : Vec n) :
    (reluDerivCLM (n := n) x).adjoint δ = (reluDerivCLM (n := n) x) δ := by
  classical
  let A := reluDerivCLM (n := n) x
  have hforall : ∀ dx : Vec n, inner ℝ dx (A.adjoint δ) = inner ℝ dx (A δ) := by
    intro dx
    calc
      inner ℝ dx (A.adjoint δ) = inner ℝ (A dx) δ := by
        simpa [A] using (ContinuousLinearMap.adjoint_inner_right (A := A) (x := dx) (y := δ))
      _ = inner ℝ dx (A δ) := by
        simpa [A] using (reluDerivCLM_inner (n := n) (x := x) (dx := dx) (δ := δ))

  set u := A.adjoint δ
  set v := A δ
  have h0 : inner ℝ (u - v) (u - v) = 0 := by
    have hEq := hforall (dx := (u - v))
    have : inner ℝ (u - v) u - inner ℝ (u - v) v = 0 := by
      simpa [sub_eq_zero] using congrArg (fun t => t - inner ℝ (u - v) v) hEq
    have hinnerSub :
        inner ℝ (u - v) (u - v) = inner ℝ (u - v) u - inner ℝ (u - v) v := by
      simpa using (inner_sub_right (x := u - v) (y := u) (z := v))
    exact hinnerSub.trans this
  have huv : u - v = 0 := (inner_self_eq_zero (𝕜 := ℝ) (x := (u - v))).1 h0
  have : u = v := sub_eq_zero.mp huv
  simpa [u, v, A] using this

-- ---------------------------------------------------------------------------
-- Linear layer: `vec_mat_mul_spec` corresponds to the adjoint on Euclidean vectors
-- ---------------------------------------------------------------------------

/--
The spec-level “input derivative” for a linear layer agrees with the Euclidean adjoint.

In words: the tensor expression for `∂(W x)/∂x` applied to an upstream `δ` is `Wᵀ δ`, and this is
exactly the adjoint of the CLM `x ↦ W x`.
-/
lemma toVecE_linear_input_deriv_spec_eq_adjoint
    {inDim outDim : Nat}
    (W : Tensor ℝ (.dim outDim (.dim inDim .scalar)))
    (δ : Tensor ℝ (.dim outDim .scalar)) :
    toVecE (Spec.linearInputDerivSpec (inDim := inDim) (outDim := outDim) W δ)
      =
    (matCLM (m := outDim) (n := inDim) (tensorToMatrix (m := outDim) (n := inDim) W)).adjoint
      (toVecE δ) := by
  classical
  -- Let `A x := W x` (as a continuous linear map on Euclidean vectors).
  let A : Vec inDim →L[ℝ] Vec outDim :=
    matCLM (m := outDim) (n := inDim) (tensorToMatrix (m := outDim) (n := inDim) W)
  let u : Vec inDim := toVecE (vecMatMulSpec δ W)
  let v : Vec inDim := A.adjoint (toVecE δ)

  have hforall : ∀ dxV : Vec inDim, inner ℝ dxV u = inner ℝ dxV v := by
    intro dxV
    -- Tensor-level adjointness (dot) for mat-vec vs vec-mat.
    have hdot :=
      dot_mat_linear_adjoint (inDim := inDim) (outDim := outDim)
        (W := W) (dLdy := δ) (dx := ofVecE dxV)
    -- Translate `dot` to `inner`.
    have hinner :
        inner ℝ (toVecE δ) (toVecE (Spec.matVecMulSpec W (ofVecE dxV)))
          =
        inner ℝ (toVecE (vecMatMulSpec δ W)) dxV := by
      simpa [dot_eq_inner_vec, toVecE_ofVecE] using hdot
    -- Identify the mat-vec output with `A dxV`.
    have hAx : toVecE (Spec.matVecMulSpec W (ofVecE dxV)) = A dxV := by
      simpa [A] using
        (toVecE_mat_vec_mul_spec (m := outDim) (n := inDim) (A := W) (v := ofVecE dxV))
    -- Use symmetry + the defining property of the adjoint.
    calc
      inner ℝ dxV u
          = inner ℝ u dxV := by simp [real_inner_comm]
      _ = inner ℝ (toVecE δ) (A dxV) := by
            have htmp := hinner.symm
            -- Rewrite the mat-vec output to `A dxV` explicitly (avoid simp rewriting order issues).
            rw [hAx] at htmp
            simpa [u] using htmp
      _ = inner ℝ (A dxV) (toVecE δ) := by simp [real_inner_comm]
      _ = inner ℝ dxV v := by
            simpa [v] using
              (ContinuousLinearMap.adjoint_inner_right (A := A) (x := dxV) (y := toVecE δ)).symm

  -- Nondegeneracy to conclude `u = v`.
  have h0 : inner ℝ (u - v) (u - v) = 0 := by
    have hEq := hforall (dxV := (u - v))
    have : inner ℝ (u - v) u - inner ℝ (u - v) v = 0 := by
      simpa [sub_eq_zero] using congrArg (fun t => t - inner ℝ (u - v) v) hEq
    have hinnerSub :
        inner ℝ (u - v) (u - v) = inner ℝ (u - v) u - inner ℝ (u - v) v := by
      simpa using (inner_sub_right (x := u - v) (y := u) (z := v))
    exact hinnerSub.trans this
  have huv : u - v = 0 := (inner_self_eq_zero (𝕜 := ℝ) (x := (u - v))).1 h0
  have : u = v := sub_eq_zero.mp huv
  -- Rewrite `u` back to `linear_input_deriv_spec`.
  simpa [u, v, Spec.linearInputDerivSpec] using this

-- ---------------------------------------------------------------------------
-- 2-layer MLP in Mat/Vec form
-- ---------------------------------------------------------------------------

/-- Affine map using `Mat` parameters (same as `affine`, but with `Mat` instead of `Matrix`). -/
def affineMat {inDim outDim : Nat} (W : Mat outDim inDim) (b : Vec outDim) : Vec inDim → Vec outDim
  :=
  affine (inDim := inDim) (outDim := outDim) (toMatrix (m := outDim) (n := inDim) W) b

/--
2-layer MLP in Euclidean `Vec` form, parameterized by `Mat` weights and `Vec` biases.

This is the same computation as `NN.Proofs.Autograd.FDeriv.Core`’s `mlpVec`, but set up for parameter-gradient
proofs where the parameter space is a Hilbert space.
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
lemma hasFDerivAt_mse {n : Nat} (t y : Vec n) :
    HasFDerivAt (mse (n := n) t) ((2 / (n : ℝ)) • (innerSL ℝ (y - t))) y := by
  have hsub : HasFDerivAt (fun y : Vec n => y - t) (1 : Vec n →L[ℝ] Vec n) y := by
    simpa using (hasFDerivAt_id y).sub_const t
  have hnorm : HasFDerivAt (fun z : Vec n => ‖z‖ ^ 2) (2 • innerSL ℝ (y - t)) (y - t) := by
    simpa using (hasStrictFDerivAt_norm_sq (x := (y - t)) (F := Vec n)).hasFDerivAt
  have hcomp : HasFDerivAt (fun y : Vec n => ‖y - t‖ ^ 2) (2 • innerSL ℝ (y - t)) y := by
    simpa using (hnorm.comp y hsub)
  have hscaled :
      HasFDerivAt (mse (n := n) t) (((n : ℝ)⁻¹) • (2 • innerSL ℝ (y - t))) y := by
    simpa [mse, mul_assoc] using (hcomp.const_mul ((n : ℝ)⁻¹))
  have hcoef :
      (((n : ℝ)⁻¹) • (2 • innerSL ℝ (y - t))) = ((2 / (n : ℝ)) • innerSL ℝ (y - t)) := by
    ext z
    simp [div_eq_mul_inv, mul_assoc, mul_comm]
  exact hscaled.congr_fderiv hcoef

/--
The VJP of MSE at `y` with upstream seed `1` equals the usual gradient `mseGrad y t`.

This is the scalar-loss specialization: for scalar loss `ℓ`, the gradient is `(fderiv ℓ)† 1`.
-/
lemma mseGrad_eq_adjoint_fderiv {n : Nat} (t y : Vec n) :
    VJP[mse (n := n) t, y] (1 : ℝ) = mseGrad (n := n) y t := by
  have hf : fderiv ℝ (mse (n := n) t) y = (2 / (n : ℝ)) • innerSL ℝ (y - t) := by
    simpa using (hasFDerivAt_mse (n := n) t y).fderiv
  calc
    (fderiv ℝ (mse (n := n) t) y).adjoint (1 : ℝ)
        = ((2 / (n : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ) := by
            simp [hf]
    _ = (2 / (n : ℝ)) • (innerSL ℝ (y - t)).adjoint (1 : ℝ) := by
          simp
    _ = (2 / (n : ℝ)) • (y - t) := by
          have hinner : (ContinuousLinearMap.adjoint ((innerSL ℝ) (y - t))) (1 : ℝ) = y - t := by
            simpa using congrArg (fun f => f (1 : ℝ))
              (ContinuousLinearMap.adjoint_innerSL_apply (𝕜 := ℝ) (x := y - t))
          rw [hinner]
    _ = mseGrad (n := n) y t := rfl

/--
Convenience lemma: adjoint of the derivative of a scalar composition, applied to seed `1`.

For scalar loss `g ∘ f`, this is the reverse-mode “chain rule” in adjoint form.
-/
lemma adjoint_fderiv_comp_apply_one
    {E F : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]
    {f : E → F} {g : F → ℝ}
    {f' : E →L[ℝ] F} {g' : F →L[ℝ] ℝ} {x : E}
    (hf : HasFDerivAt f f' x) (hg : HasFDerivAt g g' (f x)) :
    (fderiv ℝ (fun x => g (f x)) x).adjoint (1 : ℝ) = f'.adjoint (g'.adjoint (1 : ℝ)) := by
  have hcomp : HasFDerivAt (fun x => g (f x)) (g'.comp f') x := hg.comp x hf
  have hfderiv : fderiv ℝ (fun x => g (f x)) x = g'.comp f' := by
    simpa using hcomp.fderiv
  calc
    (fderiv ℝ (fun x => g (f x)) x).adjoint (1 : ℝ)
        = (ContinuousLinearMap.adjoint (g'.comp f')) (1 : ℝ) := by
            simp [hfderiv]
    _ = (ContinuousLinearMap.adjoint f').comp (ContinuousLinearMap.adjoint g') (1 : ℝ) := by
          simp [ContinuousLinearMap.adjoint_comp]
    _ = f'.adjoint (g'.adjoint (1 : ℝ)) := rfl

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
`z1` (pre-activation), `a1` (post-ReLU activation), and `y` (network output).
-/

/-- Pre-activation (used for the ReLU differentiability hypothesis). -/
def z1 : Vec hidDim :=
  affineMat (inDim := inDim) (outDim := hidDim) W1 b1 x

/-- Hidden activation `a1 = relu(z1)`. -/
def a1 : Vec hidDim :=
  reluVec (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)

/-- Network output `y = mlpVecMat W1 b1 W2 b2 x`. -/
def y : Vec outDim :=
  mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x

/--
Fréchet derivative of the network output with respect to the second-layer weights `W2`.

Informally: `∂y/∂W2` is the linear map `dW2 ↦ dW2 a1`.
-/
lemma hasFDerivAt_mlp_wrt_W2 :
    HasFDerivAt (fun W2 : Mat outDim hidDim => mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim
      := outDim) W1 b1 W2 b2 x)
      (matApplyLin (m := outDim) (n := hidDim) (a1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)) W2
        := by
  -- `W2 ↦ (W2 a1) + b2` (linear in `W2`).
  let a1V : Vec hidDim := a1 (inDim := inDim) (hidDim := hidDim) W1 b1 x
  -- Unfold the network slice; the only dependence on `W2` is the final affine.
  have hfun :
      (fun W2 : Mat outDim hidDim =>
          mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x)
        =
      fun W2 : Mat outDim hidDim => (matApplyLin (m := outDim) (n := hidDim) a1V) W2 + b2 := by
    funext W2
    -- `affineMat W2 b2 a1V = (toMatrix W2).mulVec a1V + b2`
    simp [mlpVecMat, affineMat, affine, a1V, a1, z1, matApplyLin, matApplyLM, matCLM]
    rfl
  -- Now apply the affine derivative rule.
  -- `W2 ↦ (matApplyLin a1V) W2` is a continuous linear map.
  simpa [hfun, a1V] using
    (HasFDerivAt.add_const (c := b2)
      (ContinuousLinearMap.hasFDerivAt (matApplyLin (m := outDim) (n := hidDim) a1V)))

/--
Closed-form gradient of scalar loss `mse t (mlpVecMat …)` with respect to `W2`.

Result: `∂L/∂W2 = (∂L/∂y) ⊗ a1`, i.e. outer product of the output gradient and hidden activation.
-/
lemma grad_W2_mse :
    (fderiv ℝ (fun W2 : Mat outDim hidDim =>
        mse (n := outDim) t (mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1
          W2 b2 x)) W2).adjoint
        (1 : ℝ)
      =
    outer (m := outDim) (n := hidDim)
      (mseGrad (n := outDim) (y (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2
        x) t)
      (a1 (inDim := inDim) (hidDim := hidDim) W1 b1 x) := by
  -- Compose `mse` with the `W2`-slice of the network.
  let f : Mat outDim hidDim → Vec outDim :=
    fun W2 => mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x
  let y : Vec outDim := f W2
  have hf : HasFDerivAt f
      (matApplyLin (m := outDim) (n := hidDim)
        (a1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)) W2 :=
    hasFDerivAt_mlp_wrt_W2 (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x
  have hg : HasFDerivAt (mse (n := outDim) t) ((2 / (outDim : ℝ)) • innerSL ℝ (y - t)) y :=
    hasFDerivAt_mse (n := outDim) t y
  have hcomp :=
    adjoint_fderiv_comp_apply_one (f := f) (g := mse (n := outDim) t) (x := W2) hf hg
  have hδ : (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) = mseGrad (n := outDim) y t
    := by
    -- `fderiv (mse t) y = (2/outDim) • innerSL (y - t)`.
    have : fderiv ℝ (mse (n := outDim) t) y = (2 / (outDim : ℝ)) • innerSL ℝ (y - t) := by
      simpa using (hasFDerivAt_mse (n := outDim) t y).fderiv
    simpa [Proofs.Autograd.vjp, Proofs.Autograd.jacobian, this] using
      (mseGrad_eq_adjoint_fderiv (n := outDim) t y)
  calc
    (fderiv ℝ (fun W2 => mse (n := outDim) t (f W2)) W2).adjoint (1 : ℝ)
        = (matApplyLin (m := outDim) (n := hidDim) (a1 (inDim := inDim) (hidDim := hidDim) W1 b1
          x)).adjoint
            (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) := by
            simpa [f, y] using hcomp
    _ = (matApplyLin (m := outDim) (n := hidDim) (a1 (inDim := inDim) (hidDim := hidDim) W1 b1
      x)).adjoint
          (mseGrad (n := outDim) y t) := by
          rw [hδ]
    _ = outer (m := outDim) (n := hidDim) (mseGrad (n := outDim) y t)
          (a1 (inDim := inDim) (hidDim := hidDim) W1 b1 x) := by
          simpa [a1] using
            (matApplyLin_adjoint_apply (m := outDim) (n := hidDim)
              (x := a1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)
              (δ := mseGrad (n := outDim) y t))

/--
Closed-form gradient of the scalar loss with respect to the second-layer bias `b2`.

Result: `∂L/∂b2 = ∂L/∂y`.
-/
lemma grad_b2_mse :
    (fderiv ℝ (fun b2 : Vec outDim =>
        mse (n := outDim) t (mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1
          W2 b2 x)) b2).adjoint
        (1 : ℝ)
      =
    mseGrad (n := outDim)
      (mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x) t := by
  let f : Vec outDim → Vec outDim :=
    fun b2 => mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x
  let y : Vec outDim := f b2
  have hf : HasFDerivAt f (1 : Vec outDim →L[ℝ] Vec outDim) b2 := by
    -- `b2 ↦ (W2 a1) + b2` is affine with derivative `1`.
    dsimp [f, mlpVecMat, z1, affineMat, affine]
    -- Remaining `let`-binders are constant in `b2`.
    simpa using (HasFDerivAt.const_add (c := (matCLM (m := outDim) (n := hidDim) (toMatrix W2))
        (reluVec (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))) (hasFDerivAt_id
          b2))
  have hg : HasFDerivAt (mse (n := outDim) t) ((2 / (outDim : ℝ)) • innerSL ℝ (y - t)) y :=
    hasFDerivAt_mse (n := outDim) t y
  have hcomp :=
    adjoint_fderiv_comp_apply_one (f := f) (g := mse (n := outDim) t) (x := b2) hf hg
  have hδ : (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) = mseGrad (n := outDim) y t
    := by
    have : fderiv ℝ (mse (n := outDim) t) y = (2 / (outDim : ℝ)) • innerSL ℝ (y - t) := by
      simpa using (hasFDerivAt_mse (n := outDim) t y).fderiv
    simpa [Proofs.Autograd.vjp, Proofs.Autograd.jacobian, this] using
      (mseGrad_eq_adjoint_fderiv (n := outDim) t y)
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
          simp

/--
Fréchet derivative of the network output with respect to the first-layer bias `b1`,
under the ReLU “no kinks” hypothesis.

Informally: `∂y/∂b1 = W2 ∘ ReLU'(z1)`.
-/
lemma hasFDerivAt_mlp_wrt_b1 (hx : ∀ i : Fin hidDim, (z1 (inDim := inDim) (hidDim := hidDim) W1 b1
  x) i ≠ 0) :
    HasFDerivAt (fun b1 : Vec hidDim => mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim :=
      outDim) W1 b1 W2 b2 x)
      ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
        (reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))) b1 := by
  -- `b1 ↦ affine W2 b2 (relu (affine W1 b1 x))`
  let z1V : Vec hidDim := z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x
  have hlin :
      HasFDerivAt (fun b1 : Vec hidDim => affineMat (inDim := inDim) (outDim := hidDim) W1 b1 x)
        (1 : Vec hidDim →L[ℝ] Vec hidDim) b1 := by
    -- `b1 ↦ const + b1`
    dsimp [affineMat, affine, z1]
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
  simpa [mlpVecMat, affineMat, z1V] using hcomp2

/--
Closed-form gradient of the scalar loss with respect to the first-layer bias `b1`
(under the ReLU “no kinks” hypothesis).
-/
lemma grad_b1_mse (hx : ∀ i : Fin hidDim, (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x) i ≠ 0) :
    (fderiv ℝ (fun b1 : Vec hidDim =>
        mse (n := outDim) t (mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1
          W2 b2 x)) b1).adjoint
        (1 : ℝ)
      =
    (reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))
      ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint
        (mseGrad (n := outDim) (y (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2
          x) t)) := by
  let f : Vec hidDim → Vec outDim :=
    fun b1 => mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x
  let y : Vec outDim := f b1
  have hf : HasFDerivAt f
      ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
        (reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))) b1 :=
    hasFDerivAt_mlp_wrt_b1 (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x hx
  have hg : HasFDerivAt (mse (n := outDim) t) ((2 / (outDim : ℝ)) • innerSL ℝ (y - t)) y :=
    hasFDerivAt_mse (n := outDim) t y
  have hcomp :=
    adjoint_fderiv_comp_apply_one (f := f) (g := mse (n := outDim) t) (x := b1) hf hg
  have hδ : (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) = mseGrad (n := outDim) y t
    := by
    have : fderiv ℝ (mse (n := outDim) t) y = (2 / (outDim : ℝ)) • innerSL ℝ (y - t) := by
      simpa using (hasFDerivAt_mse (n := outDim) t y).fderiv
    simpa [Proofs.Autograd.vjp, Proofs.Autograd.jacobian, this] using
      (mseGrad_eq_adjoint_fderiv (n := outDim) t y)
  calc
    (fderiv ℝ (fun b1 => mse (n := outDim) t (f b1)) b1).adjoint (1 : ℝ)
        = ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
            (reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))).adjoint
            (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) := by
            simpa [f, y] using hcomp
    _ = ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
            (reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))).adjoint
            (mseGrad (n := outDim) y t) := by
          rw [hδ]
    _ = (reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)).adjoint
          ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y t))
            := by
          simp [ContinuousLinearMap.adjoint_comp]
    _ = (reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))
          ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y t))
            := by
          simp [reluDerivCLM_adjoint_apply]

/--
Fréchet derivative of the network output with respect to the input `x`,
under the ReLU “no kinks” hypothesis.

Informally: `∂y/∂x = W2 ∘ ReLU'(z1) ∘ W1`.
-/
lemma hasFDerivAt_mlp_wrt_x (hx : ∀ i : Fin hidDim, (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)
  i ≠ 0) :
    HasFDerivAt (mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2)
      ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
        ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)).comp
          (matCLM (m := hidDim) (n := inDim) (toMatrix W1)))) x := by
  -- Same proof as `hasFDerivAt_mlpVec`, just with parameters in `Mat`.
  let z1V : Vec hidDim := z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x
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
  simpa [mlpVecMat, affineMat, z1V, ContinuousLinearMap.comp_assoc] using hcomp2

/--
Closed-form gradient of the scalar loss with respect to the input `x`,
under the ReLU “no kinks” hypothesis.
-/
lemma grad_x_mse (hx : ∀ i : Fin hidDim, (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x) i ≠ 0) :
    (fderiv ℝ (fun x : Vec inDim =>
        mse (n := outDim) t (mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1
          W2 b2 x)) x).adjoint
        (1 : ℝ)
      =
    (matCLM (m := hidDim) (n := inDim) (toMatrix W1)).adjoint
      ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))
        ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint
          (mseGrad (n := outDim) (y (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2
            b2 x) t))) := by
  let f : Vec inDim → Vec outDim :=
    mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2
  let y : Vec outDim := f x
  let f' : Vec inDim →L[ℝ] Vec outDim :=
    (matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
      ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)).comp
        (matCLM (m := hidDim) (n := inDim) (toMatrix W1)))
  have hf : HasFDerivAt f f' x :=
    hasFDerivAt_mlp_wrt_x (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x hx
  have hg : HasFDerivAt (mse (n := outDim) t) ((2 / (outDim : ℝ)) • innerSL ℝ (y - t)) y :=
    hasFDerivAt_mse (n := outDim) t y
  have hcomp := adjoint_fderiv_comp_apply_one (f := f) (g := mse (n := outDim) t) (x := x) hf hg
  have hδ : (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) = mseGrad (n := outDim) y t
    := by
    have : fderiv ℝ (mse (n := outDim) t) y = (2 / (outDim : ℝ)) • innerSL ℝ (y - t) := by
      simpa using (hasFDerivAt_mse (n := outDim) t y).fderiv
    simpa [Proofs.Autograd.vjp, Proofs.Autograd.jacobian, this] using
      (mseGrad_eq_adjoint_fderiv (n := outDim) t y)
  calc
    (fderiv ℝ (fun x => mse (n := outDim) t (f x)) x).adjoint (1 : ℝ)
        = f'.adjoint (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) := by
            simpa [f, y, f'] using hcomp
    _ = f'.adjoint (mseGrad (n := outDim) y t) := by
          rw [hδ]
    _ = (matCLM (m := hidDim) (n := inDim) (toMatrix W1)).adjoint
          ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)).adjoint
            ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
              t))) := by
          simp [f', ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_assoc]
    _ = (matCLM (m := hidDim) (n := inDim) (toMatrix W1)).adjoint
          ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))
            ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
              t))) := by
          simp [reluDerivCLM_adjoint_apply]

/--
Fréchet derivative of the network output with respect to the first-layer weights `W1`,
under the ReLU “no kinks” hypothesis.

The derivative is linear in `W1` through the slice `dW1 ↦ dW1 x`, then propagated through
`ReLU'` and `W2`.
-/
lemma hasFDerivAt_mlp_wrt_W1 (hx : ∀ i : Fin hidDim, (z1 (inDim := inDim) (hidDim := hidDim) W1 b1
  x) i ≠ 0) :
    HasFDerivAt (fun W1 : Mat hidDim inDim => mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim
      := outDim) W1 b1 W2 b2 x)
      ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
        ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)).comp
          (matApplyLin (m := hidDim) (n := inDim) x))) W1 := by
  -- `W1 ↦ affine W2 b2 (relu (W1 x + b1))`
  let z1V : Vec hidDim := z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x
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
  simpa [mlpVecMat, affineMat, z1V, z1] using hcomp2

/--
Closed-form gradient of the scalar loss with respect to `W1`
(under the ReLU “no kinks” hypothesis).

Result has the expected “outer product” form with backpropagated hidden gradient and input `x`.
-/
lemma grad_W1_mse (hx : ∀ i : Fin hidDim, (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x) i ≠ 0) :
    (fderiv ℝ (fun W1 : Mat hidDim inDim =>
        mse (n := outDim) t (mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1
          W2 b2 x)) W1).adjoint
        (1 : ℝ)
      =
    outer (m := hidDim) (n := inDim)
      ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))
        ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint
          (mseGrad (n := outDim) (y (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2
            b2 x) t)))
      x := by
  let f : Mat hidDim inDim → Vec outDim :=
    fun W1 => mlpVecMat (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x
  let y : Vec outDim := f W1
  let f' : Mat hidDim inDim →L[ℝ] Vec outDim :=
    (matCLM (m := outDim) (n := hidDim) (toMatrix W2)).comp
      ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)).comp
        (matApplyLin (m := hidDim) (n := inDim) x))
  have hf : HasFDerivAt f f' W1 :=
    hasFDerivAt_mlp_wrt_W1 (inDim := inDim) (hidDim := hidDim) (outDim := outDim) W1 b1 W2 b2 x hx
  have hg : HasFDerivAt (mse (n := outDim) t) ((2 / (outDim : ℝ)) • innerSL ℝ (y - t)) y :=
    hasFDerivAt_mse (n := outDim) t y
  have hcomp := adjoint_fderiv_comp_apply_one (f := f) (g := mse (n := outDim) t) (x := W1) hf hg
  have hδ : (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) = mseGrad (n := outDim) y t
    := by
    have : fderiv ℝ (mse (n := outDim) t) y = (2 / (outDim : ℝ)) • innerSL ℝ (y - t) := by
      simpa using (hasFDerivAt_mse (n := outDim) t y).fderiv
    simpa [Proofs.Autograd.vjp, Proofs.Autograd.jacobian, this] using
      (mseGrad_eq_adjoint_fderiv (n := outDim) t y)
  calc
    (fderiv ℝ (fun W1 => mse (n := outDim) t (f W1)) W1).adjoint (1 : ℝ)
        = f'.adjoint (((2 / (outDim : ℝ)) • innerSL ℝ (y - t)).adjoint (1 : ℝ)) := by
            simpa [f, y, f'] using hcomp
    _ = f'.adjoint (mseGrad (n := outDim) y t) := by
          rw [hδ]
    _ = (matApplyLin (m := hidDim) (n := inDim) x).adjoint
          ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x)).adjoint
            ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
              t))) := by
          simp [f', ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_assoc]
    _ = (matApplyLin (m := hidDim) (n := inDim) x).adjoint
          ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))
            ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
              t))) := by
          simp [reluDerivCLM_adjoint_apply]
    _ = outer (m := hidDim) (n := inDim)
          ((reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))
            ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
              t))) x := by
          simpa using
            (matApplyLin_adjoint_apply (m := hidDim) (n := inDim) (x := x)
              (δ := (reluDerivCLM (n := hidDim) (z1 (inDim := inDim) (hidDim := hidDim) W1 b1 x))
                ((matCLM (m := outDim) (n := hidDim) (toMatrix W2)).adjoint (mseGrad (n := outDim) y
                  t))))

end

end
end Autograd
end Proofs
