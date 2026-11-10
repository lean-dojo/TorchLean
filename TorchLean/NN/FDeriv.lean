/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.InnerProductSpace.Adjoint
public import TorchLean.NN.AdjointOp
public import TorchLean.NN.Vec
public import TorchLean.NN.Gradients.Activation
public import TorchLean.Tensor.Contract

/-!
# FDeriv Core

Analytic (`HasFDerivAt`-level) soundness for the proved-correct autograd layer. A bridge from
the pairing `dot` to the Euclidean inner product and coordinatewise calculus on Euclidean vectors
(`elemwise`, `elemwiseDerivCLM`) lead up to a theorem for a 2-layer MLP (Linear → ReLU → Linear).
After translating between tensors and vectors, the reverse-mode `backward` of
`(AdjointOp.mlp l1 l2).op` is the adjoint derivative of `Vec.mlp l1 l2`, so
`backward x δ = (fderiv ℝ (Vec.mlp l1 l2) x).adjoint δ`. The theorem is about `backward` only:
`Vec.mlp` is defined on vectors, separately from the composed forward.

Everything here is over `ℝ`, the exact arithmetic of the specs. ReLU is not differentiable at 0,
so the ReLU and MLP theorems assume that no pre-activation coordinate is zero. For
`f : ℝⁿ → ℝᵐ`, reverse mode computes `δ ↦ (Df(x))ᵗ δ`, and a scalar loss is the case `m = 1`
with `δ = 1`.

## PyTorch correspondence / citations
- PyTorch's backward computes reverse-mode VJPs, which are Jacobian-transpose products.
  https://pytorch.org/docs/stable/autograd.html
- The linear layers and ReLU in the example are standard PyTorch building blocks.
  https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
  https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
-/

@[expose] public section

namespace TorchLean

open Tensor
open Rep (dot)
open scoped BigOperators

noncomputable section

open Vec

/-!
## Dot-product vs. Euclidean inner product

To connect `Tape.AdjointOp` (stated with the tensor pairing) to `fderiv` and adjoints
(stated with Euclidean inner products), we prove that the pairing `dot` agrees with `inner` after
vectorization.
-/

/-- `getScalarE` is defined via `EuclideanSpace.equiv`, so applying the equivalence returns the
tensor's coordinates. -/
@[simp] theorem equiv_getScalarE {n : Nat} (t : Tensor ℝ [n]) :
    EuclideanSpace.equiv (Fin n) ℝ (getScalarE t) = Tensor.getScalar t := by
  simpa [getScalarE] using
    (ContinuousLinearEquiv.apply_symm_apply (EuclideanSpace.equiv (Fin n) ℝ)
      (Tensor.getScalar t))

/-- For vectors, the pairing `dot` is the Euclidean inner product of the `getScalarE` images. -/
theorem Tensor.dot_eq_inner_getScalarE {n : Nat} (a b : Tensor ℝ [n]) :
    Rep.dot a b = inner ℝ (getScalarE a) (getScalarE b) := by
  classical
  have hdot :
      Rep.dot a b =
        ∑ i : Fin n, Tensor.getScalar a i * Tensor.getScalar b i := by
    simpa using (dot_vec_eq_sum (a := a) (b := b))
  have hinter :
      inner ℝ (getScalarE a) (getScalarE b) = ∑ i : Fin n, (getScalarE a) i * (getScalarE b) i :=
    inner_eq_sum_mul (x := getScalarE a) (y := getScalarE b)
  calc
    Rep.dot a b
        = ∑ i : Fin n, Tensor.getScalar a i * Tensor.getScalar b i := hdot
    _ = inner ℝ (getScalarE a) (getScalarE b) := by
      simpa [getScalarE] using hinter.symm

/-- Vectorization commutes with tensor addition. -/
theorem Tensor.getScalarE_add {n : Nat} (a b : Tensor ℝ [n]) :
    getScalarE (a + b) = getScalarE a + getScalarE b := by
  ext i
  simp [getScalarE_ofLp, getScalar_add]

/--
Vectorization commutes with elementwise mapping: `getScalarE (Tensor.map f t)` applies `f` to
each coordinate of `TorchLean.Tensor.getScalar t`.
-/
theorem Tensor.getScalarE_map {n : Nat} (f : ℝ → ℝ) (t : Tensor ℝ [n]) :
    getScalarE (Tensor.map (shape := .dim n .scalar) f t) =
      (EuclideanSpace.equiv (Fin n) ℝ).symm fun i => f (Tensor.getScalar t i) := by
  ext i
  simp [getScalarE_ofLp]

-- ---------------------------------------------------------------------------
-- Linear layer on Euclidean vectors
-- ---------------------------------------------------------------------------

/--
Vectorization commutes with matrix–vector multiplication: `Tensor.getScalarE` of
`Tensor.contract A v _ _ [m] .matVec` is `Matrix.toEuclideanLin (Tensor.Rep.matrixEquiv ℝ m n A)`
applied to `Tensor.getScalarE v`.
-/
theorem Tensor.getScalarE_contract_matVec {m n : Nat}
    (A : Tensor ℝ [m, n]) (v : Tensor ℝ [n]) :
    getScalarE (Tensor.contract A v _ _ [m] .matVec) =
      (Matrix.toEuclideanLin (Rep.matrixEquiv ℝ m n A)).toContinuousLinearMap
        (getScalarE v) := by
  classical
  apply (EuclideanSpace.equiv (Fin m) ℝ).injective
  funext i
  -- Both sides are equal as `Fin m → ℝ`. Match them via the coordinate sum.
  simpa [Rep.matrixEquiv_apply, get2_eq_apply, Matrix.mulVec, dotProduct,
    equiv_getScalarE] using
    (getScalar_contract_matVec_eq_sum (A := A) (v := v) (i := i))

/--
Affine map `x ↦ W x + b` on Euclidean vectors, the vector-space analogue of `Linear.forward`.
-/
def Vec.affine {inDim outDim : Nat}
    (W : Matrix (Fin outDim) (Fin inDim) ℝ) (b : Vec outDim) :
    Vec inDim → Vec outDim :=
  fun x => (Matrix.toEuclideanLin W).toContinuousLinearMap x + b

/-- `Vec.affine W b` has Fréchet derivative `(Matrix.toEuclideanLin W).toContinuousLinearMap` at
every point, because it is a continuous linear map plus a constant. -/
theorem Vec.hasFDerivAt_affine {inDim outDim : Nat}
    (W : Matrix (Fin outDim) (Fin inDim) ℝ) (b : Vec outDim) (x : Vec inDim) :
    HasFDerivAt (affine (inDim := inDim) (outDim := outDim) W b)
      (Matrix.toEuclideanLin W).toContinuousLinearMap x := by
  have hlin :
      HasFDerivAt (fun x : Vec inDim => (Matrix.toEuclideanLin W).toContinuousLinearMap x)
        (Matrix.toEuclideanLin W).toContinuousLinearMap x :=
    ContinuousLinearMap.hasFDerivAt (Matrix.toEuclideanLin W).toContinuousLinearMap
  change HasFDerivAt
    (fun x : Vec inDim => (Matrix.toEuclideanLin W).toContinuousLinearMap x + b)
    (Matrix.toEuclideanLin W).toContinuousLinearMap x
  simpa using hlin.add_const b

/--
Vectorization of `forward` is the Euclidean affine map built from the same weights/bias.
-/
theorem Linear.getScalarE_forward {inDim outDim : Nat}
    (l : Linear ℝ inDim outDim) (x : Tensor ℝ [inDim]) :
    getScalarE (forward (α := ℝ) l x)
      =
    affine (inDim := inDim) (outDim := outDim)
      (Rep.matrixEquiv ℝ outDim inDim l.weights) (getScalarE l.bias) (getScalarE x) := by
  classical
  simp [forward, affine, getScalarE_add, getScalarE_contract_matVec]

-- ---------------------------------------------------------------------------
-- Generic coordinatewise calculus (`Vec n → Vec n`)
-- ---------------------------------------------------------------------------

/--
Apply a scalar function `f : ℝ → ℝ` to each coordinate of a Euclidean vector, as `Tensor.map`
does for tensors.
-/
def Vec.elemwise {n : Nat} (f : ℝ → ℝ) : Vec n → Vec n :=
  fun x => WithLp.toLp 2 fun i : Fin n => f (x.ofLp i)

/--
The derivative candidate for `elemwise f` at a point `x`, built from a proposed scalar
derivative `f'`.

Concretely: `(elemwiseDerivCLM f' x) dx` has coordinates `i ↦ f'(xᵢ) * dxᵢ`.
-/
def Vec.elemwiseDerivCLM {n : Nat} (f' : ℝ → ℝ) (x : Vec n) : Vec n →L[ℝ] Vec n :=
  (EuclideanSpace.equiv (Fin n) ℝ).symm.toContinuousLinearMap.comp <|
    ContinuousLinearMap.pi (fun i : Fin n =>
      ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
        (EuclideanSpace.proj (𝕜 := ℝ) i) (f' (x.ofLp i)))

/--
If `f` has derivative `f' (x i)` at every coordinate `x i`, then `elemwise f` has the diagonal
Fréchet derivative `elemwiseDerivCLM f' x` at `x`.

Maps such as ReLU, whose scalar derivative fails to exist at finitely many points, need this
pointwise form.
-/
theorem Vec.hasFDerivAt_elemwise {n : Nat} {f f' : ℝ → ℝ} (x : Vec n)
    (hf : ∀ i : Fin n, HasDerivAt f (f' (x.ofLp i)) (x.ofLp i)) :
    HasFDerivAt (elemwise (n := n) f) (elemwiseDerivCLM (n := n) f' x) x := by
  classical
  have hcoord :
      ∀ i : Fin n,
        HasFDerivAt (fun x : Vec n => f (x.ofLp i))
          (ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (EuclideanSpace.proj (𝕜 := ℝ) i) (f' (x.ofLp i))) x := by
    intro i
    have hfF :
        HasFDerivAt f
          (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (1 : ℝ →L[ℝ] ℝ) (f' (x.ofLp i))) (x.ofLp i) :=
      (hf i).hasFDerivAt
    have happly : HasFDerivAt (fun x : Vec n => x.ofLp i) (EuclideanSpace.proj (𝕜 := ℝ) i) x :=
      (EuclideanSpace.proj (𝕜 := ℝ) i).hasFDerivAt
    have hlin :
        (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (1 : ℝ →L[ℝ] ℝ) (f' (x.ofLp i))).comp (EuclideanSpace.proj (𝕜 := ℝ) i)
          =
        ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
          (EuclideanSpace.proj (𝕜 := ℝ) i) (f' (x.ofLp i)) := by
      ext dx
      simp [ContinuousLinearMap.smulRight_apply]
    exact (hfF.comp x happly).congr_fderiv hlin

  have hFun :
      HasFDerivAt (fun x : Vec n => fun i : Fin n => f (x.ofLp i))
        (ContinuousLinearMap.pi (fun i : Fin n =>
          ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (EuclideanSpace.proj (𝕜 := ℝ) i) (f' (x.ofLp i)))) x := by
    refine (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun i : Fin n => fun x : Vec n => f (x.ofLp i))
        (φ' := fun i : Fin n =>
          ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (EuclideanSpace.proj (𝕜 := ℝ) i) (f' (x.ofLp i)))
        (x := x)).2 ?_
    intro i
    simpa using hcoord i
  have he' :
      HasFDerivAt (fun g : Fin n → ℝ => (EuclideanSpace.equiv (Fin n) ℝ).symm g)
        ((EuclideanSpace.equiv (Fin n) ℝ).symm.toContinuousLinearMap)
        (fun i : Fin n => f (x.ofLp i)) :=
    (ContinuousLinearMap.hasFDerivAt (EuclideanSpace.equiv (Fin n) ℝ).symm.toContinuousLinearMap)
  have hcomp := he'.comp x hFun
  show HasFDerivAt (fun x : Vec n => WithLp.toLp 2 fun i : Fin n => f (x.ofLp i))
    (elemwiseDerivCLM (n := n) f' x) x
  simpa [elemwise, elemwiseDerivCLM, Function.comp_def,
    ContinuousLinearMap.comp_apply] using hcomp

-- ---------------------------------------------------------------------------
-- Coordinatewise ReLU on Euclidean vectors
-- ---------------------------------------------------------------------------

/--
ReLU as a map on Euclidean vectors (coordinatewise `max x 0`).

This is the Euclidean-space analogue of `TorchLean.Tensor.relu`, which maps
`Activation.relu` over a tensor.
-/
def Vec.relu {n : Nat} : Vec n → Vec n :=
  elemwise Activation.relu

/-- Derivative of `relu` at `x`: the diagonal scaling by the scalar ReLU derivative mask. -/
def Vec.reluDerivCLM {n : Nat} (x : Vec n) : Vec n →L[ℝ] Vec n :=
  elemwiseDerivCLM Activation.reluDeriv x

/--
At a point `x` with no zero coordinate, where ReLU has no kink, `relu` has Fréchet derivative
`Vec.reluDerivCLM x`.
-/
theorem Vec.hasFDerivAt_relu {n : Nat} (x : Vec n) (hx : ∀ i : Fin n, x i ≠ 0) :
    HasFDerivAt (relu (n := n)) (reluDerivCLM (n := n) x) x :=
  hasFDerivAt_elemwise (f := Activation.relu) (f' := Activation.reluDeriv)
    x fun i => Activation.hasDerivAt_relu _ (hx i)

-- ---------------------------------------------------------------------------
-- 2-layer MLP: Linear → ReLU → Linear
-- ---------------------------------------------------------------------------

/--
2-layer MLP forward map on Euclidean vectors:

`x ↦ affine W2 b2 (relu (affine W1 b1 x))`.
-/
def Vec.mlp {inDim hidDim outDim : Nat}
    (l1 : Linear ℝ inDim hidDim)
    (l2 : Linear ℝ hidDim outDim) : Vec inDim → Vec outDim :=
  fun x =>
    let W1 := Rep.matrixEquiv ℝ hidDim inDim l1.weights
    let b1 : Vec hidDim := getScalarE l1.bias
    let W2 := Rep.matrixEquiv ℝ outDim hidDim l2.weights
    let b2 : Vec outDim := getScalarE l2.bias
    let z1 := affine (inDim := inDim) (outDim := hidDim) W1 b1 x
    let a1 := relu z1
    affine (inDim := hidDim) (outDim := outDim) W2 b2 a1

/--
Closed-form derivative of `mlp` at `x` as a continuous linear map, the chain-rule composition
`W2 ∘ ReLU'(z1) ∘ W1` with `z1 = W1 x + b1`.
-/
def Vec.mlpDeriv {inDim hidDim outDim : Nat}
    (l1 : Linear ℝ inDim hidDim)
    (l2 : Linear ℝ hidDim outDim)
    (x : Vec inDim) : Vec inDim →L[ℝ] Vec outDim :=
  let W1 := Rep.matrixEquiv ℝ hidDim inDim l1.weights
  let W2 := Rep.matrixEquiv ℝ outDim hidDim l2.weights
  let b1 : Vec hidDim := getScalarE l1.bias
  let z1 := affine (inDim := inDim) (outDim := hidDim) W1 b1 x
  (Matrix.toEuclideanLin W2).toContinuousLinearMap.comp
    ((reluDerivCLM (n := hidDim) z1).comp (Matrix.toEuclideanLin W1).toContinuousLinearMap)

/--
If every pre-activation coordinate `z1ᵢ` of `W1 x + b1` is nonzero, the 2-layer MLP
(Linear → ReLU → Linear) has Fréchet derivative `Vec.mlpDeriv l1 l2 x` at `x`.
-/
theorem Vec.hasFDerivAt_mlp {inDim hidDim outDim : Nat}
    (l1 : Linear ℝ inDim hidDim)
    (l2 : Linear ℝ hidDim outDim)
    (x : Vec inDim)
    (hx : ∀ i : Fin hidDim,
      let W1 := Rep.matrixEquiv ℝ hidDim inDim l1.weights
      let b1 : Vec hidDim := getScalarE l1.bias
      (affine (inDim := inDim) (outDim := hidDim) W1 b1 x) i ≠ 0) :
    HasFDerivAt (mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2)
      (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 x) x := by
  dsimp [mlp, mlpDeriv]
  let W1 := Rep.matrixEquiv ℝ hidDim inDim l1.weights
  let b1 : Vec hidDim := getScalarE l1.bias
  let W2 := Rep.matrixEquiv ℝ outDim hidDim l2.weights
  let b2 : Vec outDim := getScalarE l2.bias
  let z1 := affine (inDim := inDim) (outDim := hidDim) W1 b1 x

  have hlin1 :
      HasFDerivAt (affine (inDim := inDim) (outDim := hidDim) W1 b1)
        (Matrix.toEuclideanLin W1).toContinuousLinearMap x :=
    hasFDerivAt_affine (W := W1) (b := b1) x

  have hrelu :
      HasFDerivAt (relu (n := hidDim)) (reluDerivCLM (n := hidDim) z1) z1 :=
    hasFDerivAt_relu (x := z1) (n := hidDim) (hx := by
      intro i
      simpa [z1] using hx i)

  have hlin2 :
      HasFDerivAt (affine (inDim := hidDim) (outDim := outDim) W2 b2)
        (Matrix.toEuclideanLin W2).toContinuousLinearMap (relu z1) :=
    hasFDerivAt_affine (W := W2) (b := b2) (x := relu z1)

  have hcomp1 := hrelu.comp x hlin1
  have hcomp2 := hlin2.comp x hcomp1
  change HasFDerivAt
    ((affine (inDim := hidDim) (outDim := outDim) W2 b2) ∘
      (relu (n := hidDim)) ∘ (affine (inDim := inDim) (outDim := hidDim) W1 b1))
    ((Matrix.toEuclideanLin W2).toContinuousLinearMap ∘SL
      reluDerivCLM (n := hidDim) z1 ∘SL (Matrix.toEuclideanLin W1).toContinuousLinearMap) x
  simpa [z1, ContinuousLinearMap.comp_assoc] using hcomp2

-- ---------------------------------------------------------------------------
-- Connect proved-correct `VJPOp.backward` to analytic VJP (adjoint of `fderiv`)
-- ---------------------------------------------------------------------------

/--
The adjointness certificate of the MLP, composed from the linear and ReLU certificates. Its proof
field is the dot-level adjointness `⟪JVP,δ⟫ = ⟪dx,VJP⟫`.
-/
def AdjointOp.mlp {inDim hidDim outDim : Nat}
    (l1 : Linear ℝ inDim hidDim)
    (l2 : Linear ℝ hidDim outDim) :
    Tape.AdjointOp ℝ (.dim inDim .scalar) (.dim outDim .scalar) :=
  Tape.AdjointOp.compose
    (Tape.AdjointOp.linear (inDim := inDim) (outDim := hidDim) l1)
    (Tape.AdjointOp.compose (Tape.AdjointOp.relu (s := .dim hidDim .scalar))
      (Tape.AdjointOp.linear (inDim := hidDim) (outDim := outDim) l2))

/--
After vectorizing tensors to Euclidean vectors, the JVP of `mlp` is the analytic derivative
`mlpDeriv`.
-/
theorem AdjointOp.getScalarE_mlp_jvp {inDim hidDim outDim : Nat}
    (l1 : Linear ℝ inDim hidDim)
    (l2 : Linear ℝ hidDim outDim)
    (x dx : Tensor ℝ [inDim]) :
    getScalarE ((mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).jvp x dx)
      =
    (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 (getScalarE x))
      (getScalarE dx) := by
  classical
  -- Name the intermediate pre-activation and its Euclidean version.
  let z1T : Tensor ℝ [hidDim] := Linear.forward (α := ℝ) l1 x
  let xV : Vec inDim := getScalarE x
  let dxV : Vec inDim := getScalarE dx
  let W1 := Rep.matrixEquiv ℝ hidDim inDim l1.weights
  let W2 := Rep.matrixEquiv ℝ outDim hidDim l2.weights
  let b1 : Vec hidDim := getScalarE l1.bias
  let z1V : Vec hidDim := affine (inDim := inDim) (outDim := hidDim) W1 b1 xV

  have hz1 : getScalarE z1T = z1V := by
    simpa [z1T, z1V, xV, W1, b1] using
      (Linear.getScalarE_forward (inDim := inDim) (outDim := hidDim) (l := l1) (x := x))

  -- Expand the JVP produced by `mlp`.
  have hjvp :
      (mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).jvp x dx
        =
      Tensor.contract l2.weights
        (Tensor.contract l1.weights dx _ _ [hidDim] .matVec *
          (Tensor.map Activation.reluDeriv (Linear.forward (α := ℝ) l1 x))) _ _ [outDim]
            .matVec := by
    simp [mlp, Tape.AdjointOp.compose, Tape.AdjointOp.linear, Tape.AdjointOp.relu]

  -- Translate the inner "ReLU JVP" tensor to a Euclidean vector and match it to `reluDerivCLM`.
  let innerT : Tensor ℝ [hidDim] :=
    Tensor.contract l1.weights dx _ _ [hidDim] .matVec *
      (Tensor.map Activation.reluDeriv (Linear.forward (α := ℝ) l1 x))

  have hInner :
      getScalarE innerT =
        (reluDerivCLM (n := hidDim) z1V)
          ((Matrix.toEuclideanLin W1).toContinuousLinearMap dxV) := by
    ext j
    have hdx1 :
        Tensor.getScalar (Tensor.contract l1.weights dx _ _ [hidDim] .matVec) j =
          ((Matrix.toEuclideanLin W1).toContinuousLinearMap dxV).ofLp j := by
      have h :=
        congrArg (fun v : Vec hidDim => v.ofLp j)
          (getScalarE_contract_matVec (m := hidDim) (n := inDim) (A := l1.weights) (v := dx))
      simpa [W1, dxV, getScalarE_ofLp] using h

    have hz1j : Tensor.getScalar z1T j = z1V.ofLp j := by
      have h := congrArg (fun v : Vec hidDim => v.ofLp j) hz1
      simpa [getScalarE_ofLp] using h

    have hrelu' :
        Tensor.getScalar
            (Tensor.map Activation.reluDeriv (Linear.forward (α := ℝ) l1 x)) j
          =
        Activation.reluDeriv (z1V.ofLp j) := by
      rw [getScalar_map]
      -- Replace `TorchLean.Tensor.getScalar (Linear.forward l1 x) j` by the corresponding
      -- coordinate of `z1V`.
      exact congrArg Activation.reluDeriv hz1j

    have hR :
        ((reluDerivCLM (n := hidDim) z1V)
              ((Matrix.toEuclideanLin W1).toContinuousLinearMap dxV)).ofLp j
          =
        ((Matrix.toEuclideanLin W1).toContinuousLinearMap dxV).ofLp j *
          Activation.reluDeriv (z1V.ofLp j) :=
      -- `reluDerivCLM` is a diagonal `elemwiseDerivCLM`, so coordinate `j` holds by definition.
      rfl

    -- Left side is the elementwise product `dx₁ ⊙ relu'(z₁)` in coordinate form.
    -- Right side is the same coordinate as computed by `reluDerivCLM`.
    rw [hR]
    simp [innerT, getScalarE_ofLp, Tensor.getScalar_mul, hdx1, hrelu']

  -- Finish: translate the outer mat-vec and compare with `mlpDeriv`.
  -- First, rewrite the JVP `Tensor` as a mat-vec against `innerT`.
  have hjvp' :
      (mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).jvp x dx
        =
      Tensor.contract l2.weights innerT _ _ [outDim] .matVec := by
    simpa [innerT] using hjvp

  -- Both sides are `W2` applied to the same hidden vector.
  -- Use `getScalarE_contract_matVec` for the tensor mat-vec, and unfold `mlpDeriv`.
  calc
    getScalarE ((mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2).jvp x dx)
        = (Matrix.toEuclideanLin W2).toContinuousLinearMap (getScalarE innerT) := by
            simpa [hjvp', W2] using
              (getScalarE_contract_matVec (m := outDim) (n := hidDim) (A := l2.weights)
                (v := innerT))
    _ =
        (Matrix.toEuclideanLin W2).toContinuousLinearMap
          ((reluDerivCLM (n := hidDim) z1V)
            ((Matrix.toEuclideanLin W1).toContinuousLinearMap dxV)) := by
            simpa using congrArg
              (fun v : Vec hidDim => (Matrix.toEuclideanLin W2).toContinuousLinearMap v) hInner
    _ = (mlpDeriv (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2 xV) dxV := by
            simp [mlpDeriv, xV, z1V, W1, W2, b1, ContinuousLinearMap.comp_apply]

/--
Analytic soundness of the 2-layer MLP `VJPOp` backward in the input, for fixed weights. If no
pre-activation coordinate is zero, then after vectorization `(AdjointOp.mlp l1 l2).op.backward x δ`
equals the analytic VJP `(fderiv ℝ (Vec.mlp l1 l2) x).adjoint δ`.
-/
theorem VJPOp.mlp_backward_eq_fderiv {inDim hidDim outDim : Nat}
    (l1 : Linear ℝ inDim hidDim)
    (l2 : Linear ℝ hidDim outDim)
    (x : Tensor ℝ [inDim])
    (hx : ∀ i : Fin hidDim,
      let W1 := Rep.matrixEquiv ℝ hidDim inDim l1.weights
      let b1 : Vec hidDim := getScalarE l1.bias
      (affine (inDim := inDim) (outDim := hidDim) W1 b1 (getScalarE x)) i ≠ 0) :
    ∀ δ : Tensor ℝ [outDim],
      getScalarE ((AdjointOp.mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim)
          l1 l2).op.backward x δ)
        =
      (fderiv ℝ (Vec.mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim) l1 l2)
        (getScalarE x)).adjoint (getScalarE δ) := by
  intro δ
  let f := Vec.mlp l1 l2
  let xV := getScalarE x
  have hf : HasFDerivAt f (mlpDeriv l1 l2 xV) xV :=
    hasFDerivAt_mlp l1 l2 xV hx
  change getScalarE ((AdjointOp.mlp l1 l2).op.backward x δ) =
    (fderiv ℝ f xV).adjoint (getScalarE δ)
  rw [hf.fderiv]
  apply ext_inner_left ℝ
  intro dxV
  rw [ContinuousLinearMap.adjoint_inner_right]
  have hdot := (AdjointOp.mlp l1 l2).isAdjointPair x (ofFnE dxV) δ
  simpa only [dot_eq_inner_getScalarE, AdjointOp.getScalarE_mlp_jvp, getScalarE_ofFnE]
    using hdot.symm

end
end TorchLean
