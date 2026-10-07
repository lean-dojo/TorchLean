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

This file starts by connecting our tensor `dot` to the Euclidean-space inner product, sets up the
generic coordinatewise calculus on Euclidean vectors (`elemwiseVec`, `elemwiseDerivCLM`), then
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

/-- Vectorization commutes with tensor addition. -/
theorem getScalarE_add_spec {n : Nat} (a b : Tensor ℝ [n]) :
    getScalarE (addSpec a b) = getScalarE a + getScalarE b := by
  ext i
  simp [getScalarE_ofLp, Spec.getScalar_add_spec]

/--
Vectorization commutes with elementwise mapping: `getScalarE (Tensor.map f t)` applies `f` to
each coordinate of `TorchLean.Tensor.getScalar t`.
-/
theorem getScalarE_map_spec {n : Nat} (f : ℝ → ℝ) (t : Tensor ℝ [n]) :
    getScalarE (Tensor.map (shape := .dim n .scalar) f t) =
      (euclideanEquiv n).symm fun i => f (TorchLean.Tensor.getScalar t i) := by
  ext i
  simp [getScalarE_ofLp]

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
def tensorToMatrix {m n : Nat} (W : Tensor ℝ [m, n]) : Matrix (Fin m) (Fin n) ℝ :=
  fun i j => Spec.get2 W i j

/--
The matrix–vector multiplication map as a continuous linear map on Euclidean vectors.

This is the Euclidean-space version of the tensor op `matVecMulSpec`.
-/
def matCLM {m n : Nat} (W : Matrix (Fin m) (Fin n) ℝ) : (Vec n) →L[ℝ] (Vec m) :=
  let L : (Fin n → ℝ) →L[ℝ] (Fin m → ℝ) :=
    ⟨W.mulVecLin, LinearMap.continuous_of_finiteDimensional W.mulVecLin⟩
  (euclideanEquiv m).symm.toContinuousLinearMap.comp
    (L.comp (euclideanEquiv n).toContinuousLinearMap)

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

-- ---------------------------------------------------------------------------
-- Generic coordinatewise calculus (`Vec n → Vec n`)
-- ---------------------------------------------------------------------------

/--
Apply a scalar function `f : ℝ → ℝ` coordinatewise to a vector.

This is the Euclidean-space analogue of the tensor-level `Tensor.map`.
-/
def elemwiseVec {n : Nat} (f : ℝ → ℝ) : Vec n → Vec n :=
  fun x => WithLp.toLp 2 fun i : Fin n => f (x.ofLp i)

/-- Coordinate evaluation as a continuous linear map on `Vec n`. -/
def evalCLM {n : Nat} (i : Fin n) : Vec n →L[ℝ] ℝ :=
  EuclideanSpace.proj (𝕜 := ℝ) i

/-- The coordinate evaluation functional reads coordinate `i`. -/
@[simp] theorem evalCLM_apply {n : Nat} (i : Fin n) (x : Vec n) :
    evalCLM (n := n) i x = x.ofLp i := rfl

/--
The derivative candidate for `elemwiseVec f` at a point `x`, built from a proposed scalar
derivative `f'`.

Concretely: `(elemwiseDerivCLM f' x) dx` has coordinates `i ↦ f'(xᵢ) * dxᵢ`.
-/
def elemwiseDerivCLM {n : Nat} (f' : ℝ → ℝ) (x : Vec n) : Vec n →L[ℝ] Vec n :=
  (euclideanEquiv n).symm.toContinuousLinearMap.comp <|
    ContinuousLinearMap.pi (fun i : Fin n =>
      ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
        (evalCLM (n := n) i) (f' (x.ofLp i)))

/--
`elemwiseVec f` is Fréchet differentiable at `x` as soon as `f` is differentiable at every
coordinate of `x`, with the diagonal derivative `elemwiseDerivCLM f' x`.

This is the form needed for maps such as ReLU whose scalar derivative exists only away from
finitely many points.
-/
theorem hasFDerivAt_elemwiseVec_at {n : Nat} {f f' : ℝ → ℝ} (x : Vec n)
    (hf : ∀ i : Fin n, HasDerivAt f (f' (x.ofLp i)) (x.ofLp i)) :
    HasFDerivAt (elemwiseVec (n := n) f) (elemwiseDerivCLM (n := n) f' x) x := by
  classical
  have hcoord :
      ∀ i : Fin n,
        HasFDerivAt (fun x : Vec n => f (x.ofLp i))
          (ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (evalCLM (n := n) i) (f' (x.ofLp i))) x := by
    intro i
    have hfF :
        HasFDerivAt f
          (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (1 : ℝ →L[ℝ] ℝ) (f' (x.ofLp i))) (x.ofLp i) :=
      (hf i).hasFDerivAt
    have happly : HasFDerivAt (fun x : Vec n => x.ofLp i) (evalCLM (n := n) i) x :=
      (evalCLM (n := n) i).hasFDerivAt
    have hlin :
        (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (1 : ℝ →L[ℝ] ℝ) (f' (x.ofLp i))).comp (evalCLM (n := n) i)
          =
        ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
          (evalCLM (n := n) i) (f' (x.ofLp i)) := by
      ext dx
      simp [ContinuousLinearMap.smulRight_apply]
    exact (hfF.comp x happly).congr_fderiv hlin

  have hFun :
      HasFDerivAt (fun x : Vec n => fun i : Fin n => f (x.ofLp i))
        (ContinuousLinearMap.pi (fun i : Fin n =>
          ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (evalCLM (n := n) i) (f' (x.ofLp i)))) x := by
    refine (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun i : Fin n => fun x : Vec n => f (x.ofLp i))
        (φ' := fun i : Fin n =>
          ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (evalCLM (n := n) i) (f' (x.ofLp i)))
        (x := x)).2 ?_
    intro i
    simpa using hcoord i
  have he' :
      HasFDerivAt (fun g : Fin n → ℝ => (euclideanEquiv n).symm g)
        ((euclideanEquiv n).symm.toContinuousLinearMap)
        (fun i : Fin n => f (x.ofLp i)) :=
    (ContinuousLinearMap.hasFDerivAt (euclideanEquiv n).symm.toContinuousLinearMap)
  have hcomp := he'.comp x hFun
  show HasFDerivAt (fun x : Vec n => WithLp.toLp 2 fun i : Fin n => f (x.ofLp i))
    (elemwiseDerivCLM (n := n) f' x) x
  simpa [elemwiseVec, elemwiseDerivCLM, euclideanEquiv, Function.comp_def,
    ContinuousLinearMap.comp_apply] using hcomp

/--
If `f` is differentiable everywhere with derivative `f'`, then `elemwiseVec f` is Fréchet
differentiable everywhere with derivative `elemwiseDerivCLM f'`.
-/
theorem hasFDerivAt_elemwiseVec {n : Nat} {f f' : ℝ → ℝ} (x : Vec n)
    (hf : ∀ z, HasDerivAt f (f' z) z) :
    HasFDerivAt (elemwiseVec (n := n) f) (elemwiseDerivCLM (n := n) f' x) x :=
  hasFDerivAt_elemwiseVec_at x fun i => hf (x.ofLp i)

-- ---------------------------------------------------------------------------
-- Coordinatewise ReLU on Euclidean vectors
-- ---------------------------------------------------------------------------

/--
ReLU as a map on Euclidean vectors (coordinatewise `max x 0`).

This is the Euclidean-space analogue of `Spec.relu_op.forward`.
-/
def reluVec {n : Nat} : Vec n → Vec n :=
  elemwiseVec Activation.Math.reluSpec

/-- Derivative of `reluVec` at `x`: the diagonal scaling by the scalar ReLU derivative mask. -/
def reluDerivCLM {n : Nat} (x : Vec n) : Vec n →L[ℝ] Vec n :=
  elemwiseDerivCLM Activation.Math.reluDerivSpec x

/--
ReLU is not differentiable at 0, so its Fréchet derivative is asserted under the “no kinks”
hypothesis that every coordinate of `x` is nonzero.
-/
theorem hasFDerivAt_reluVec {n : Nat} (x : Vec n) (hx : ∀ i : Fin n, x i ≠ 0) :
    HasFDerivAt (reluVec (n := n)) (reluDerivCLM (n := n) x) x :=
  hasFDerivAt_elemwiseVec_at (f := Activation.Math.reluSpec) (f' := Activation.Math.reluDerivSpec)
    x fun i => Proofs.relu_deriv_correct _ (hx i)

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
    (x : Vec inDim) : Vec inDim →L[ℝ] Vec outDim :=
  let W1 := tensorToMatrix (m := hidDim) (n := inDim) l1.weights
  let W2 := tensorToMatrix (m := outDim) (n := hidDim) l2.weights
  let b1 : Vec hidDim := getScalarE l1.bias
  let z1 := affine (inDim := inDim) (outDim := hidDim) W1 b1 x
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
          Activation.Math.reluDerivSpec (z1V.ofLp j) :=
      -- `reluDerivCLM` is a diagonal `elemwiseDerivCLM`; coordinate `j` reads off by definition.
      rfl

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
  let f := mlpVec l1 l2
  let xV := getScalarE x
  have hf : HasFDerivAt f (mlpDeriv l1 l2 xV) xV :=
    hasFDerivAt_mlpVec l1 l2 xV hx
  change getScalarE ((mlpOp l1 l2).backward x δ) =
    (fderiv ℝ f xV).adjoint (getScalarE δ)
  rw [hf.fderiv]
  apply ext_inner_left ℝ
  intro dxV
  rw [ContinuousLinearMap.adjoint_inner_right]
  have hdot :
      Spec.dot ((mlpCorrect l1 l2).jvp x (ofFnE dxV)) δ =
        Spec.dot (ofFnE dxV) ((mlpOp l1 l2).backward x δ) := by
    simpa only [mlpCorrect, mlpOp, OpSpecCorrect.compose, linearCorrect, reluCorrect] using
      (mlpCorrect l1 l2).correct x (ofFnE dxV) δ
  simpa only [dot_eq_inner_vec, getScalar_mlp_jvp, getScalarE_ofFnE] using hdot.symm

end
end Autograd
end Proofs
