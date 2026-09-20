/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Core.RealCorrectness
public import NN.Proofs.Autograd.Core.Vectorization
public import NN.Proofs.Autograd.Notation
public import NN.Proofs.Gradients.Activation


/-!
# FDeriv Core

`HasFDerivAt`-level (analytic) soundness for the proved-correct autograd layer.

This file starts by connecting our tensor `dot` to the Euclidean-space inner product, then
proves a first end-to-end theorem for a 2-layer MLP (Linear → ReLU → Linear):

* the `OpSpec` reverse-mode `backward` computes the true analytic VJP,
  i.e. `backward x δ = VJP[f, x] δ` (after translating between tensors and vectors).

Notes:
- Everything here is over `ℝ` (spec-level exact arithmetic).
- ReLU is not differentiable at 0, so the theorems assume a "no kinks" hypothesis on the
  pre-activation vector.
- The tensor-output theorem shape is naturally VJP-based: for `f : ℝⁿ → ℝᵐ`, reverse-mode computes
  `δ ↦ (Df(x))ᵗ δ`. Scalar losses are the special case `m = 1` / `δ = 1`.

## PyTorch correspondence / citations
- Reverse-mode VJPs and Jacobian-transpose products are exactly what PyTorch’s backward computes.
  https://pytorch.org/docs/stable/autograd.html
- Linear layers and ReLU as used in the example are standard PyTorch building blocks.
  https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
  https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec _root_.TorchLean
open _root_.TorchLean _root_.TorchLean.Tensor
open Activation
open scoped BigOperators
open scoped _root_.Autograd

noncomputable section

/-- Abbreviation for the Euclidean-space equivalence `Vec n ≃ (Fin n → ℝ)`. -/
abbrev euclideanEquiv (n : Nat) := EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)

/-!
## Dot-product vs. Euclidean inner product

To connect `OpSpecCorrect` (stated with the tensor dot product) to `fderiv` and adjoints (stated
with Euclidean inner products), we prove that `Spec.dot` agrees with `inner` after vectorization.
-/

/-- `getScalarE` is defined via `EuclideanSpace.equiv`; this lemma exposes the underlying
coordinates. -/
@[simp] theorem euclideanEquiv_getScalarE {n : Nat} (t : Tensor ℝ [n]) :
    euclideanEquiv n (getScalarE t) = TorchLean.Tensor.getScalar t := by
  simpa [getScalarE, euclideanEquiv] using
    (ContinuousLinearEquiv.apply_symm_apply (euclideanEquiv n) (TorchLean.Tensor.getScalar t))

/--
For 1D scalar tensors, `Spec.dot` agrees with the Euclidean inner product on `Vec n`
after converting via `getScalarE`.
-/
theorem dot_eq_inner_vec {n : Nat} (a b : Tensor ℝ [n]) :
    Spec.dot a b = inner ℝ (getScalarE a) (getScalarE b) := by
  classical
  have hdot :
      Spec.dot a b =
        ∑ i : Fin n, TorchLean.Tensor.getScalar a i * TorchLean.Tensor.getScalar b i := by
    simpa using (Spec.dot_vec_eq_sum (a := a) (b := b))
  have hinter :
      inner ℝ (getScalarE a) (getScalarE b) = ∑ i : Fin n, (getScalarE a) i * (getScalarE b) i :=
    inner_eq_sum_mul (x := getScalarE a) (y := getScalarE b)
  calc
    Spec.dot a b
        = ∑ i : Fin n, TorchLean.Tensor.getScalar a i * TorchLean.Tensor.getScalar b i := hdot
    _ = inner ℝ (getScalarE a) (getScalarE b) := by
      simpa [getScalarE] using hinter.symm

/-- Coordinate formula for tensor addition under `TorchLean.Tensor.getScalar`. -/
theorem getScalar_add_spec_apply {n : Nat} (a b : Tensor ℝ [n]) (i : Fin n) :
    TorchLean.Tensor.getScalar (addSpec a b) i
      = TorchLean.Tensor.getScalar a i + TorchLean.Tensor.getScalar b i := by
  exact congrFun (Spec.getScalar_add_spec a b) i

/-- Vectorization commutes with tensor addition. -/
theorem getScalarE_add_spec {n : Nat} (a b : Tensor ℝ [n]) :
    getScalarE (addSpec a b) = getScalarE a + getScalarE b := by
  ext i
  simp [getScalarE_ofLp, getScalar_add_spec_apply]

/--
Vectorization commutes with elementwise mapping: `getScalarE (map_spec f t)` is `f` applied to each
coordinate of `TorchLean.Tensor.getScalar t`.
-/
theorem getScalarE_map_spec {n : Nat} (f : ℝ → ℝ) (t : Tensor ℝ [n]) :
    getScalarE (mapSpec (s := .dim n .scalar) f t) =
      (euclideanEquiv n).symm fun i => f (TorchLean.Tensor.getScalar t i) := by
  ext i
  simp [getScalarE_ofLp, mapSpec]

/--
Vectorization of `reluDerivSpec`: the derivative mask is ReLU’s scalar derivative applied
coordinatewise.
-/
theorem getScalarE_relu_deriv_spec {n : Nat} (t : Tensor ℝ [n]) :
    getScalarE (Activation.reluDerivSpec (α := ℝ) (s := .dim n .scalar) t)
      =
    (euclideanEquiv n).symm
      fun i => Activation.Math.reluDerivSpec (TorchLean.Tensor.getScalar t i) := by
  simpa [Activation.reluDerivSpec] using
    (getScalarE_map_spec (n := n) Activation.Math.reluDerivSpec t)

-- ---------------------------------------------------------------------------
-- Linear layer on Euclidean vectors
-- ---------------------------------------------------------------------------

/--
View a matrix-shaped tensor `W : Tensor ℝ (m×n)` as a Mathlib `Matrix (Fin m) (Fin n) ℝ`.

This is just the coordinate function `Spec.get2`.
-/
def tensorToMatrix {m n : Nat} (W : Tensor ℝ [m, n]) : Matrix (Fin m) (Fin n) ℝ
  :=
  fun i j => Spec.get2 W i j

/--
The matrix–vector multiplication map as a continuous linear map on Euclidean vectors.

This is the Euclidean-space version of the tensor op `matVecMulSpec`.
-/
def matCLM {m n : Nat} (W : Matrix (Fin m) (Fin n) ℝ) : (Vec n) →L[ℝ] (Vec m) := by
  classical
  let em := euclideanEquiv m
  let en := euclideanEquiv n
  let L : (Fin n → ℝ) →L[ℝ] (Fin m → ℝ) :=
    ⟨W.mulVecLin, LinearMap.continuous_of_finiteDimensional W.mulVecLin⟩
  exact em.symm.toContinuousLinearMap.comp (L.comp en.toContinuousLinearMap)

/--
Vectorization commutes with matrix–vector multiplication:
`getScalarE (mat_vec_mul_spec A v) = (matCLM (tensorToMatrix A)) (getScalarE v)`.
-/
theorem getScalarE_mat_vec_mul_spec {m n : Nat}
    (A : Tensor ℝ [m, n]) (v : Tensor ℝ [n]) :
    getScalarE (Spec.matVecMulSpec A v) =
      (matCLM (m := m) (n := n) (tensorToMatrix A)) (getScalarE v) := by
  classical
  apply (euclideanEquiv m).injective
  funext i
  -- Both sides are equal as `Fin m → ℝ`; match them via the coordinate sum.
  simpa [matCLM, tensorToMatrix, Matrix.mulVec, dotProduct, Matrix.mulVecLin_apply,
    euclideanEquiv_getScalarE,
    euclideanEquiv] using
    (Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec (A := A) (v := v) (i := i))

/--
Affine map `x ↦ W x + b` on Euclidean vectors.

This is the vector-space analogue of `Spec.linearSpec`.
-/
def affine {inDim outDim : Nat}
    (W : Matrix (Fin outDim) (Fin inDim) ℝ) (b : Vec outDim) :
    Vec inDim → Vec outDim :=
  fun x => (matCLM (m := outDim) (n := inDim) W) x + b

/-- `affine` is Fréchet-differentiable with derivative `W` (as a CLM), since it is linear +
  constant. -/
theorem hasFDerivAt_affine {inDim outDim : Nat}
    (W : Matrix (Fin outDim) (Fin inDim) ℝ) (b : Vec outDim) (x : Vec inDim) :
    HasFDerivAt (affine (inDim := inDim) (outDim := outDim) W b)
      (matCLM (m := outDim) (n := inDim) W) x := by
  have hlin :
      HasFDerivAt (fun x : Vec inDim => (matCLM (m := outDim) (n := inDim) W) x)
        (matCLM (m := outDim) (n := inDim) W) x :=
    ContinuousLinearMap.hasFDerivAt (matCLM (m := outDim) (n := inDim) W)
  change HasFDerivAt
    (fun x : Vec inDim => (matCLM (m := outDim) (n := inDim) W) x + b)
    (matCLM (m := outDim) (n := inDim) W) x
  simpa using hlin.add_const b

/--
Vectorization of `Spec.linearSpec` is the Euclidean affine map built from the same weights/bias.
-/
theorem getScalarE_linear_spec {inDim outDim : Nat}
    (l : Spec.LinearSpec ℝ inDim outDim) (x : Tensor ℝ [inDim]) :
    getScalarE (Spec.linearSpec (α := ℝ) l x)
      =
    affine (inDim := inDim) (outDim := outDim)
      (tensorToMatrix (m := outDim) (n := inDim) l.weights) (getScalarE l.bias) (getScalarE x) := by
  classical
  simp [Spec.linearSpec, affine, getScalarE_add_spec, getScalarE_mat_vec_mul_spec]

-- Coordinatewise ReLU on Euclidean vectors and its differentiability facts.

/-- Coordinatewise ReLU on `Fin n → ℝ` (function-space representation). -/
def reluFun {n : Nat} (x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => Activation.Math.reluSpec (x i)

/--
ReLU as a map on Euclidean vectors (coordinatewise `max x 0`).

This is the Euclidean-space analogue of `Spec.relu_op.forward`.
-/
def reluVec {n : Nat} (x : Vec n) : Vec n :=
  (euclideanEquiv n).symm (reluFun (n := n) ((euclideanEquiv n) x))

/--
Derivative candidate for the coordinatewise ReLU function on `Fin n → ℝ`, expressed as a diagonal
scaling map by the scalar derivative mask.
-/
def reluFunDeriv {n : Nat} (x : Fin n → ℝ) : (Fin n → ℝ) →L[ℝ] (Fin n → ℝ) :=
  by
    classical
    let pr := @ContinuousLinearMap.proj ℝ _ (Fin n) (fun _ : Fin n => ℝ) _ _ _
    exact ContinuousLinearMap.pi fun i : Fin n =>
      (pr i).smulRight (Activation.Math.reluDerivSpec (x i))

/-- Transport `reluFunDeriv` to `Vec n` via `EuclideanSpace.equiv`. -/
def reluDerivCLM {n : Nat} (x : Vec n) : Vec n →L[ℝ] Vec n :=
  (euclideanEquiv n).symm.toContinuousLinearMap.comp <|
    (reluFunDeriv (x := (euclideanEquiv n x))).comp (euclideanEquiv n).toContinuousLinearMap

/-!
ReLU is not differentiable at 0. We therefore assume a “no kinks” hypothesis that every coordinate
of `x` is nonzero.
-/
theorem hasFDerivAt_reluVec {n : Nat} (x : Vec n) (hx : ∀ i : Fin n, x i ≠ 0) :
    HasFDerivAt (reluVec (n := n)) (reluDerivCLM (n := n) x) x := by
  classical
  let xF : Fin n → ℝ := (euclideanEquiv n) x
  have hxF : ∀ i : Fin n, xF i ≠ 0 := by
    intro i
    simpa [xF] using hx i

  have hcoord :
      ∀ i : Fin n,
        HasFDerivAt (fun x : Fin n → ℝ => Activation.Math.reluSpec (x i))
          ((@ContinuousLinearMap.proj ℝ _ (Fin n) (fun _ : Fin n => ℝ) _ _ _ i).smulRight
              (Activation.Math.reluDerivSpec (xF i))) xF := by
    intro i
    have hrelu :
        HasDerivAt Activation.Math.reluSpec (Activation.Math.reluDerivSpec (xF i)) (xF i) :=
      Proofs.relu_deriv_correct (x := xF i) (h := hxF i)
    have hreluF :
        HasFDerivAt Activation.Math.reluSpec
          ((1 : ℝ →L[ℝ] ℝ).smulRight (Activation.Math.reluDerivSpec (xF i))) (xF i) :=
      hrelu.hasFDerivAt
    have happly :
        HasFDerivAt (fun x : Fin n → ℝ => x i)
          (@ContinuousLinearMap.proj ℝ _ (Fin n) (fun _ : Fin n => ℝ) _ _ _ i) xF :=
      hasFDerivAt_apply i xF
    have hcomp := hreluF.comp xF happly
    have hlin :
        ((1 : ℝ →L[ℝ] ℝ).smulRight (Activation.Math.reluDerivSpec (xF i))).comp
            (@ContinuousLinearMap.proj ℝ _ (Fin n) (fun _ : Fin n => ℝ) _ _ _ i)
          =
        (@ContinuousLinearMap.proj ℝ _ (Fin n) (fun _ : Fin n => ℝ) _ _ _ i).smulRight
            (Activation.Math.reluDerivSpec (xF i)) := by
      ext dx
      simp [ContinuousLinearMap.smulRight_apply]
    exact hcomp.congr_fderiv hlin

  have hReluFun : HasFDerivAt (reluFun (n := n)) (reluFunDeriv (x := xF)) xF := by
    -- Coordinatewise `relu` on function space.
    refine (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun (i : Fin n) => fun x : Fin n → ℝ => Activation.Math.reluSpec (x i))
        (φ' := fun (i : Fin n) =>
          (@ContinuousLinearMap.proj ℝ _ (Fin n) (fun _ : Fin n => ℝ) _ _ _ i).smulRight
            (Activation.Math.reluDerivSpec (xF i)))
        (x := xF)).2 ?_
    intro i
    simpa using hcoord i

  -- Transport the derivative back to `Vec n` via `euclideanEquiv n : Vec n ≃L Fin n → ℝ`.
  have he :
      HasFDerivAt (fun x : Vec n => (euclideanEquiv n) x)
        ((euclideanEquiv n).toContinuousLinearMap) x :=
    (ContinuousLinearMap.hasFDerivAt (euclideanEquiv n).toContinuousLinearMap)
  have hmid :
      HasFDerivAt (fun x : Vec n => reluFun (n := n) ((euclideanEquiv n) x))
        ((reluFunDeriv (x := xF)).comp (euclideanEquiv n).toContinuousLinearMap) x := by
    simpa [xF] using hReluFun.comp x he
  have he' :
      HasFDerivAt (fun y : Fin n → ℝ => (euclideanEquiv n).symm y)
        ((euclideanEquiv n).symm.toContinuousLinearMap)
        (reluFun (n := n) ((euclideanEquiv n) x)) :=
    (ContinuousLinearMap.hasFDerivAt (euclideanEquiv n).symm.toContinuousLinearMap)
  -- Compose `euclideanEquiv.symm ∘ reluFun ∘ euclideanEquiv`.
  change HasFDerivAt
    ((fun y : Fin n → ℝ => (euclideanEquiv n).symm y) ∘
      fun x : Vec n => reluFun (n := n) ((euclideanEquiv n) x))
    ((euclideanEquiv n).symm.toContinuousLinearMap ∘SL
      (reluFunDeriv (x := (euclideanEquiv n) x) ∘SL (euclideanEquiv n).toContinuousLinearMap)) x
  simpa [reluDerivCLM, xF, Function.comp] using he'.comp x hmid

-- ---------------------------------------------------------------------------
-- 2-layer MLP: Linear → ReLU → Linear
-- ---------------------------------------------------------------------------

/--
2-layer MLP forward map on Euclidean vectors:

`x ↦ affine W2 b2 (relu (affine W1 b1 x))`.
-/
def mlpVec {inDim hidDim outDim : Nat}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim) : Vec inDim → Vec outDim :=
  fun x =>
    let W1 := tensorToMatrix (m := hidDim) (n := inDim) l1.weights
    let b1 : Vec hidDim := getScalarE l1.bias
    let W2 := tensorToMatrix (m := outDim) (n := hidDim) l2.weights
    let b2 : Vec outDim := getScalarE l2.bias
    let z1 := affine (inDim := inDim) (outDim := hidDim) W1 b1 x
    let a1 := reluVec z1
    affine (inDim := hidDim) (outDim := outDim) W2 b2 a1

/--
Closed-form derivative (as a continuous linear map) of `mlpVec` at `x`.

This is the chain rule composition: `W2 ∘ ReLU'(z1) ∘ W1`.
-/
def mlpDeriv {inDim hidDim outDim : Nat}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim)
    (x : Vec inDim) : Vec inDim →L[ℝ] Vec outDim := by
  let W1 := tensorToMatrix (m := hidDim) (n := inDim) l1.weights
  let W2 := tensorToMatrix (m := outDim) (n := hidDim) l2.weights
  let b1 : Vec hidDim := getScalarE l1.bias
  let z1 := affine (inDim := inDim) (outDim := hidDim) W1 b1 x
  exact
    (matCLM (m := outDim) (n := hidDim) W2).comp
      ((reluDerivCLM (n := hidDim) z1).comp (matCLM (m := hidDim) (n := inDim) W1))

/--
Fréchet differentiability of the 2-layer MLP (Linear → ReLU → Linear) under a “no kinks” hypothesis.

Because ReLU is not differentiable at 0, we assume all pre-activation coordinates `z1ᵢ` are nonzero.
-/
theorem hasFDerivAt_mlpVec {inDim hidDim outDim : Nat}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim)
    (x : Vec inDim)
    (hx : ∀ i : Fin hidDim,
      let W1 := tensorToMatrix (m := hidDim) (n := inDim) l1.weights
      let b1 : Vec hidDim := getScalarE l1.bias
      (affine (inDim := inDim) (outDim := hidDim) W1 b1 x) i ≠ 0) :
    HasFDerivAt (mlpVec (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2)
      (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 x) x := by
  dsimp [mlpVec, mlpDeriv]
  let W1 := tensorToMatrix (m := hidDim) (n := inDim) l1.weights
  let b1 : Vec hidDim := getScalarE l1.bias
  let W2 := tensorToMatrix (m := outDim) (n := hidDim) l2.weights
  let b2 : Vec outDim := getScalarE l2.bias
  let z1 := affine (inDim := inDim) (outDim := hidDim) W1 b1 x

  have hlin1 :
      HasFDerivAt (affine (inDim := inDim) (outDim := hidDim) W1 b1)
        (matCLM (m := hidDim) (n := inDim) W1) x :=
    hasFDerivAt_affine (W := W1) (b := b1) x

  have hrelu :
      HasFDerivAt (reluVec (n := hidDim)) (reluDerivCLM (n := hidDim) z1) z1 :=
    hasFDerivAt_reluVec (x := z1) (n := hidDim) (hx := by
      intro i
      simpa [z1] using hx i)

  have hlin2 :
      HasFDerivAt (affine (inDim := hidDim) (outDim := outDim) W2 b2)
        (matCLM (m := outDim) (n := hidDim) W2) (reluVec z1) :=
    hasFDerivAt_affine (W := W2) (b := b2) (x := reluVec z1)

  have hcomp1 := hrelu.comp x hlin1
  have hcomp2 := hlin2.comp x hcomp1
  change HasFDerivAt
    ((affine (inDim := hidDim) (outDim := outDim) W2 b2) ∘
      (reluVec (n := hidDim)) ∘ (affine (inDim := inDim) (outDim := hidDim) W1 b1))
    (matCLM (m := outDim) (n := hidDim) W2 ∘SL
      reluDerivCLM (n := hidDim) z1 ∘SL matCLM (m := hidDim) (n := inDim) W1) x
  simpa [z1, ContinuousLinearMap.comp_assoc] using hcomp2

-- ---------------------------------------------------------------------------
-- Connect proved-correct `OpSpec.backward` to analytic VJP (adjoint of `fderiv`)
-- ---------------------------------------------------------------------------

/--
The spec-level MLP as a composed `Spec.OpSpec`:

`linear l1` then `relu` then `linear l2`.
-/
def mlpOp {inDim hidDim outDim : Nat}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim) :
    Spec.OpSpec ℝ (.dim inDim .scalar) (.dim outDim .scalar) :=
  Spec.OpSpec.compose (Spec.linearOp (α := ℝ) (inDim := inDim) (outDim := hidDim) l1)
    (Spec.OpSpec.compose
      (Spec.reluOp (α := ℝ) (s := .dim hidDim .scalar))
      (Spec.linearOp (α := ℝ) (inDim := hidDim) (outDim := outDim) l2))

/--
The proved-correct MLP `OpSpecCorrect`, built by composing the primitive correctness lemmas.

This provides the dot-level adjointness statement: `⟪JVP,δ⟫ = ⟪dx,VJP⟫`.
-/
def mlpCorrect {inDim hidDim outDim : Nat}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim) :
    OpSpecCorrect (.dim inDim .scalar) (.dim outDim .scalar) :=
  OpSpecCorrect.compose
    (linearCorrect (inDim := inDim) (outDim := hidDim) l1)
    (OpSpecCorrect.compose (reluCorrect (s := .dim hidDim .scalar))
      (linearCorrect (inDim := hidDim) (outDim := outDim) l2))

/--
Identify the `OpSpecCorrect` JVP for the MLP with the analytic derivative `mlpDeriv`,
after vectorizing tensors to Euclidean vectors.
-/
theorem getScalar_mlp_jvp {inDim hidDim outDim : Nat}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim)
    (x dx : Tensor ℝ [inDim]) :
    getScalarE ((mlpCorrect (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).jvp x dx)
      =
    (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 (getScalarE x))
      (getScalarE dx) := by
  classical
  -- Name the intermediate pre-activation and its Euclidean version.
  let z1T : Tensor ℝ [hidDim] := Spec.linearSpec (α := ℝ) l1 x
  let xV : Vec inDim := getScalarE x
  let dxV : Vec inDim := getScalarE dx
  let W1 := tensorToMatrix (m := hidDim) (n := inDim) l1.weights
  let W2 := tensorToMatrix (m := outDim) (n := hidDim) l2.weights
  let b1 : Vec hidDim := getScalarE l1.bias
  let z1V : Vec hidDim := affine (inDim := inDim) (outDim := hidDim) W1 b1 xV

  have hz1 : getScalarE z1T = z1V := by
    -- `getScalarE_linear_spec` is exactly the identification we need.
    simpa [z1T, z1V, xV, W1, b1] using
      (getScalarE_linear_spec (inDim := inDim) (outDim := hidDim) (l := l1) (x := x))

  -- Expand the JVP produced by `mlpCorrect`.
  have hjvp :
      (mlpCorrect (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).jvp x dx
        =
      Spec.matVecMulSpec l2.weights
        (mulSpec (Spec.matVecMulSpec l1.weights dx)
          (Activation.reluDerivSpec ((Spec.linearOp (α := ℝ) (inDim := inDim) (outDim := hidDim)
            l1).forward x))) := by
    simp [mlpCorrect, OpSpecCorrect.compose, linearCorrect, reluCorrect]

  -- Translate the inner "ReLU JVP" tensor to a Euclidean vector and match it to `reluDerivCLM`.
  let innerT : Tensor ℝ [hidDim] :=
    mulSpec (Spec.matVecMulSpec l1.weights dx)
      (Activation.reluDerivSpec ((Spec.linearOp (α := ℝ) (inDim := inDim) (outDim := hidDim)
        l1).forward x))

  have hInner :
      getScalarE innerT =
        (reluDerivCLM (n := hidDim) z1V)
          ((matCLM (m := hidDim) (n := inDim) W1) dxV) := by
    ext j
    have hdx1 :
        TorchLean.Tensor.getScalar (Spec.matVecMulSpec l1.weights dx) j =
          ((matCLM (m := hidDim) (n := inDim) W1) dxV).ofLp j := by
      have h :=
        congrArg (fun v : Vec hidDim => v.ofLp j)
          (getScalarE_mat_vec_mul_spec (m := hidDim) (n := inDim) (A := l1.weights) (v := dx))
      simpa [W1, dxV, getScalarE_ofLp] using h

    have hz1j : TorchLean.Tensor.getScalar z1T j = z1V.ofLp j := by
      have h := congrArg (fun v : Vec hidDim => v.ofLp j) hz1
      simpa [getScalarE_ofLp] using h

    have hrelu' :
        TorchLean.Tensor.getScalar
            (Activation.reluDerivSpec
              ((Spec.linearOp (α := ℝ) (inDim := inDim) (outDim := hidDim) l1).forward x)) j
          =
        Activation.Math.reluDerivSpec (z1V.ofLp j) := by
      have h :=
        congrArg (fun v : Vec hidDim => v.ofLp j)
          (getScalarE_relu_deriv_spec
            (n := hidDim)
            (t := ((Spec.linearOp (α := ℝ) (inDim := inDim) (outDim := hidDim) l1).forward x)))
      -- Replace `TorchLean.Tensor.getScalar (linear_spec l1 x) j` by the corresponding
      -- coordinate of `z1V`.
      simpa [Spec.linearOp, z1T, getScalarE_ofLp, hz1j] using h

    have hR :
        ((reluDerivCLM (n := hidDim) z1V)
              ((matCLM (m := hidDim) (n := inDim) W1) dxV)).ofLp j
          =
        ((matCLM (m := hidDim) (n := inDim) W1) dxV).ofLp j *
          Activation.Math.reluDerivSpec (z1V.ofLp j) := by
      -- Unfold the transported diagonal map and evaluate at coordinate `j`.
      simp [reluDerivCLM, reluFunDeriv, ContinuousLinearMap.comp_apply,
        ContinuousLinearMap.smulRight_apply, euclideanEquiv]

    -- Left side is the elementwise product `dx₁ ⊙ relu'(z₁)` in coordinate form.
    -- Right side is the same coordinate as computed by `reluDerivCLM`.
    simp [innerT, getScalarE_ofLp, Spec.getScalar_mul_spec, hdx1, hrelu', hR]

  -- Finish: translate the outer mat-vec and compare with `mlpDeriv`.
  -- First, rewrite the JVP `Tensor` as a mat-vec against `innerT`.
  have hjvp' :
      (mlpCorrect (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).jvp x dx
        =
      Spec.matVecMulSpec l2.weights innerT := by
    simpa [innerT] using hjvp

  -- Now both sides are `W2` applied to the same hidden vector.
  -- Use `getScalarE_mat_vec_mul_spec` for the tensor mat-vec, and unfold `mlpDeriv`.
  calc
    getScalarE ((mlpCorrect (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).jvp x dx)
        = (matCLM (m := outDim) (n := hidDim) W2) (getScalarE innerT) := by
            simpa [hjvp', W2] using
              (getScalarE_mat_vec_mul_spec (m := outDim) (n := hidDim) (A := l2.weights)
                (v := innerT))
    _ =
        (matCLM (m := outDim) (n := hidDim) W2)
          ((reluDerivCLM (n := hidDim) z1V) ((matCLM (m := hidDim) (n := inDim) W1) dxV)) := by
            simpa using congrArg (fun v : Vec hidDim => (matCLM (m := outDim) (n := hidDim) W2) v)
              hInner
    _ = (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV) dxV := by
            simp [mlpDeriv, xV, z1V, W1, W2, b1, ContinuousLinearMap.comp_apply]

/--
End-to-end analytic soundness for the 2-layer MLP `OpSpec`:

the `OpSpec.backward` returned by the spec-level reverse-mode rule equals the adjoint of the true
Fréchet derivative of the forward map (i.e. the analytic VJP), after vectorization.

This is the proof layer analogue of PyTorch’s claim that `loss.backward()` computes the correct VJP
for the composed model, assuming the primitive backward rules are correct.
-/
theorem mlp_backward_eq_adjoint_fderiv {inDim hidDim outDim : Nat}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim)
    (x : Tensor ℝ [inDim])
    (hx : ∀ i : Fin hidDim,
      let W1 := tensorToMatrix (m := hidDim) (n := inDim) l1.weights
      let b1 : Vec hidDim := getScalarE l1.bias
      (affine (inDim := inDim) (outDim := hidDim) W1 b1 (getScalarE x)) i ≠ 0) :
    ∀ δ : Tensor ℝ [outDim],
      getScalarE ((mlpOp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).backward x δ)
        =
      VJP[mlpVec (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2, getScalarE x]
        (getScalarE δ) := by
  intro δ
  classical
  let f := mlpVec (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2
  let xV : Vec inDim := getScalarE x
  have hf :
      HasFDerivAt f
        (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV) xV :=
    hasFDerivAt_mlpVec (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV hx

  -- Use the inner-product characterization of the adjoint; the `OpSpecCorrect` theorem gives the
  -- same characterization for the `OpSpec.backward` cotangent.
  have hdot :
      ∀ dxT : Tensor ℝ [inDim],
        Spec.dot ((mlpCorrect (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).jvp x
          dxT) δ
          =
        Spec.dot dxT ((mlpOp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).backward
          x δ) := by
    intro dxT
    simpa [mlpCorrect, mlpOp, OpSpecCorrect.compose, linearCorrect, reluCorrect] using
      (mlpCorrect (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).correct x dxT δ

  -- Convert `dot` to `inner` and rewrite the JVP using the analytic derivative.
  have hinner :
      ∀ dxV : Vec inDim,
        inner ℝ ((mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV) dxV)
          (getScalarE δ)
          =
        inner ℝ dxV (getScalarE ((mlpOp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1
          l2).backward x δ)) := by
    intro dxV
    -- Specialize `hdot` to `dxT := ofFnE dxV`, then translate from `dot` to `inner`.
    have hdot' := hdot (dxT := ofFnE dxV)
    have hinner' :
        inner ℝ (getScalarE ((mlpCorrect (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1
          l2).jvp x (ofFnE dxV)))
            (getScalarE δ)
          =
        inner ℝ (getScalarE (ofFnE dxV))
            (getScalarE
              ((mlpOp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).backward x δ))
        := by
      simpa [dot_eq_inner_vec] using hdot'
    -- Rewrite the JVP vector using `getScalar_mlp_jvp` and simplify `getScalarE (ofFnE dxV) = dxV`.
    have hjvpVec :
        getScalarE ((mlpCorrect (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).jvp x
          (ofFnE dxV))
          =
        (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV) dxV := by
      -- `getScalar_mlp_jvp` expects a tensor `dx`; apply it to `dx := ofFnE dxV`.
      simpa [xV] using
        (getScalar_mlp_jvp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 x
          (ofFnE dxV))
    have hinner'' := hinner'
    -- Replace the JVP vector, then simplify `getScalarE (ofFnE dxV)`.
    rw [hjvpVec] at hinner''
    simpa using hinner''

  -- Uniqueness: the element is determined by all inner products against `dxV`.
  -- Compare against the defining property of `adjoint`.
  have hadjoint :
      ∀ dxV : Vec inDim,
        inner ℝ ((mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV) dxV)
          (getScalarE δ)
          =
        inner ℝ dxV
          ((mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV).adjoint
            (getScalarE δ)) := by
    intro dxV
    -- Fundamental adjoint property: ⟪D x, y⟫ = ⟪x, D† y⟫.
    simpa using
      (ContinuousLinearMap.adjoint_inner_right
        (A := mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV)
        (x := dxV) (y := getScalarE δ)).symm

  -- Combine `hinner` and `hadjoint` to show the two candidates have equal inner products
  -- against all `dxV`, then conclude by `inner_self_eq_zero`.
  have hforall :
      ∀ dxV : Vec inDim,
        inner ℝ dxV (getScalarE ((mlpOp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1
          l2).backward x δ))
          =
        inner ℝ dxV
          ((mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV).adjoint
            (getScalarE δ)) := by
    intro dxV
    -- Both sides equal `inner ℝ ((D dxV)) δ`.
    exact (hinner dxV).symm.trans (hadjoint dxV)

  -- Nondegeneracy: if `inner dxV (u - v) = 0` for all dxV, then `u = v`.
  have :
      getScalarE ((mlpOp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).backward x δ)
        =
      (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV).adjoint
        (getScalarE δ) := by
    -- Let `e := u - v` and take `dxV := e`.
    set u :=
      getScalarE ((mlpOp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).backward x δ)
        with hu
    set v :=
      (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV).adjoint
        (getScalarE δ) with hv
    have h0 : inner ℝ (u - v) (u - v) = 0 := by
      have hEq := hforall (dxV := (u - v))
      -- Move to a `sub = 0` form and expand `inner (u-v) (u-v)` using bilinearity.
      have hSub : inner ℝ (u - v) u - inner ℝ (u - v) v = 0 := by
        simpa [sub_eq_zero] using congrArg (fun t => t - inner ℝ (u - v) v) hEq
      -- `⟪u - v, u - v⟫ = ⟪u - v, u⟫ - ⟪u - v, v⟫`.
      have hinnerSub :
          inner ℝ (u - v) (u - v) = inner ℝ (u - v) u - inner ℝ (u - v) v := by
        rw [inner_sub_right]
      exact hinnerSub.trans hSub
    have : u - v = 0 := (inner_self_eq_zero (𝕜 := ℝ) (x := (u - v))).1 h0
    simpa [hu, hv] using sub_eq_zero.mp this

  -- Replace the explicit derivative with `fderiv` using `hf`.
  have hfderiv :
      fderiv ℝ f xV = mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV := by
    simpa using hf.fderiv
  have hfderiv' :
      mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV = fderiv ℝ f xV := by
    simpa using hfderiv.symm
  calc
    getScalarE ((mlpOp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).backward x δ)
        =
      (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV).adjoint
        (getScalarE δ) := this
    _ =
      (fderiv ℝ f xV).adjoint (getScalarE δ) := by
        simpa using
          congrArg (fun D : Vec inDim →L[ℝ] Vec outDim => D.adjoint (getScalarE δ)) hfderiv'

end
end Autograd
end Proofs
