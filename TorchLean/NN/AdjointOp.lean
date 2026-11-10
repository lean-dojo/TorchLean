/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.AdjointOp
public import TorchLean.Tensor.Algebra.Real
public import TorchLean.Tensor.LinearAlgebra.Norm

/-!
# Adjointness certificates over ℝ

Five adjointness certificates over `ℝ`: sigmoid, tanh, exp, square, and the total sum. Each is a
value of `TorchLean.Tape.AdjointOp ℝ` from `TorchLean.NN.Tape.AdjointOp`, which holds the
interface, the chain rule `compose`, and the two certificates that need only commutative semiring
algebra, for ReLU and linear layers.

A certificate proves that the selected tangent map and the backward pass are adjoint for the
pairing `dot`. It does not prove that the tangent map is the derivative of the forward map. With
the `HasFDerivAt` theorems under `TorchLean.NN.FDeriv` and their domain hypotheses, the chain
rule identifies a composed tangent map with the derivative of the composed forward map, and
adjointness identifies the composed backward pass with its adjoint.

Baydin et al. describe forward mode as computing the Jacobian-vector product `J r`
(Section 3.1, Eq. (4)) and reverse mode the transposed product `J⊺ r` (Section 3.2). The
certificates prove the identity that pairs the two, `dot (jvp x r) δ = dot r (op.backward x δ)`,
for the maps selected here. We wrote the proofs by hand: the four pointwise ones close with
`Tensor.dot_elemwise_adjoint`, and `sum`'s goes through `Tensor.dot_scale_left`.

References:
- `torch.autograd` (PyTorch 2.8 docs), https://docs.pytorch.org/docs/2.8/autograd.html, and custom
  rules through `torch.autograd.Function`,
  https://docs.pytorch.org/docs/2.8/autograd.html#torch.autograd.Function
- Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul and Jeffrey Mark Siskind,
  "Automatic Differentiation in Machine Learning: a Survey", JMLR 18(153):1-43, 2018
  (`arXiv:1502.05767`), Sections 3.1 (Eq. (4)) and 3.2
- Griewank and Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic
  Differentiation*, 2nd ed., SIAM 2008, https://doi.org/10.1137/1.9780898717761
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open Tensor
open Rep (dot)

noncomputable section

-- Over `ℝ`, the sigmoid, exponential and tanh backward specs equal their derivatives times the
-- seed.

private theorem Tensor.sigmoidBackward_eq_mul {s : Shape} (x seed : Tensor ℝ s) :
    sigmoidBackward x seed =
      Tensor.map Numeric.sigmoidDeriv x * seed := by
  apply Tensor.Rep.ext
  intro i
  simp [sigmoidBackward,
    Numeric.sigmoidBackward, Numeric.sigmoidDeriv,
    Numeric.applySlope, Numeric.QuotientArithmetic.supported_real, mul_def]

private theorem Tensor.zipWith_expMul_eq_mul {s : Shape} (x seed : Tensor ℝ s) :
    Tensor.zipWith Context.expMul x seed = Tensor.exp x * seed := by
  apply Tensor.Rep.ext
  intro i
  simp [Context.expMul, mul_def, Tensor.exp]

private theorem Tensor.tanhBackward_eq_mul {s : Shape} (x seed : Tensor ℝ s) :
    tanhBackward x seed = Tensor.map Numeric.tanhDeriv x * seed := by
  apply Tensor.Rep.ext
  intro i
  simp [tanhBackward,
    Numeric.tanhBackward, Numeric.tanhDeriv,
    Numeric.applySlope, Numeric.QuotientArithmetic.supported_real, mul_def]

/--
Correctness of sigmoid’s backward rule.

PyTorch analogue: `torch.sigmoid`.
-/
def AdjointOp.sigmoid {s : Shape} :
  Tape.AdjointOp ℝ s s :=
{
  op :=
    { forward := Tensor.map Context.sigmoid
      backward := Tensor.sigmoidBackward }
  jvp := fun x dx => dx * Tensor.map Numeric.sigmoidDeriv x
  isAdjointPair := by
    intro x dx δ
    simpa [sigmoidBackward_eq_mul] using (Tensor.dot_elemwise_adjoint (dx:=dx)
        (df:=Tensor.map Numeric.sigmoidDeriv x) (δ:=δ))
}

/--
Correctness of tanh’s backward rule.

PyTorch analogue: `torch.tanh`.
-/
def AdjointOp.tanh {s : Shape} :
  Tape.AdjointOp ℝ s s :=
{
  op :=
    { forward := Tensor.map MathFunctions.tanh
      backward := Tensor.tanhBackward }
  jvp := fun x dx => dx * Tensor.map Numeric.tanhDeriv x
  isAdjointPair := by
    intro x dx δ
    simpa [tanhBackward_eq_mul] using (Tensor.dot_elemwise_adjoint (dx:=dx)
        (df:=Tensor.map Numeric.tanhDeriv x) (δ:=δ))
}

/--
Correctness of `exp`’s backward rule.

PyTorch analogue: `torch.exp`.
-/
def AdjointOp.exp {s : Shape} :
  Tape.AdjointOp ℝ s s :=
{
  op :=
    { forward := fun x => Tensor.exp x
      backward := fun x g => Tensor.zipWith Context.expMul x g }
  jvp := fun x dx => dx * Tensor.exp (α:=ℝ) (s:=s) x
  isAdjointPair := by
    intro x dx δ
    simpa [zipWith_expMul_eq_mul] using (Tensor.dot_elemwise_adjoint (dx:=dx)
      (df:=Tensor.exp (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of `square`'s backward rule.

PyTorch analogue: `torch.square`. The local derivative is $2x$.
-/
def AdjointOp.square {s : Shape} :
  Tape.AdjointOp ℝ s s :=
{
  op :=
    { forward := fun x => Tensor.square x
      backward := fun x g => (Tensor.full _ 2 * x) * g }
  jvp := fun x dx => dx * (Tensor.full s (2 : ℝ) * x)
  isAdjointPair := by
    intro x dx δ
    simpa  using (Tensor.dot_elemwise_adjoint (dx:=dx)
      (df:=Tensor.full s (2 : ℝ) * x) (δ:=δ))
}

/--
Correctness of `sum` (reduce-all) backward rule.

Informally, $\frac{d}{dx}\sum x=1$, so the VJP replicates the upstream scalar gradient into every
entry.

PyTorch analogue: `torch.sum` (over all elements).
-/
def AdjointOp.sum {s : Shape} : Tape.AdjointOp ℝ s Shape.scalar :=
{
  op :=
    { forward := fun x => Tensor.scalar (Tensor.sum (α:=ℝ) (s:=s) x)
      backward := fun _x dLdy => Tensor.full (α := ℝ) s dLdy.item }
  jvp := fun _x dx => Tensor.scalar (Tensor.sum (α:=ℝ) (s:=s) dx)
  isAdjointPair := by
    intro x dx δ
    let g := δ.item
    rw [← Tensor.scalar_item δ]
      -- The all-`g` tensor is `g • Tensor.full s 1`.
    have hrep :
        Tensor.full (α := ℝ) s g =
          Tensor.scale (α:=ℝ) (s:=s) (Tensor.full s (1 : ℝ)) g := by
      apply Tensor.Rep.ext
      intro coordinate
      simp [Tensor.scale]
    -- Reduce both sides using `dot_scale_left` and the fact that the all-ones tensor is the
    -- multiplicative identity.
    calc
      Rep.dot (Tensor.scalar (Tensor.sum (α:=ℝ) (s:=s) dx)) (Tensor.scalar g)
          = (Tensor.sum (α:=ℝ) (s:=s) dx) * g := by
              simp only [Tensor.dot_scalar, Tensor.item_scalar]
      _ = g * (Tensor.sum (α:=ℝ) (s:=s) dx) := by ring
      _ = g * Rep.dot dx (Tensor.full s (1 : ℝ)) := by
            simp [sum_eq_coord_sum, Rep.dot]
      _ = g * Rep.dot (Tensor.full s (1 : ℝ)) dx := by rw [Tensor.dot_comm]
      _ = Rep.dot (Tensor.scale (α:=ℝ) (s:=s) (Tensor.full s (1 : ℝ)) g) dx := by
            rw [Tensor.dot_scale_left, mul_comm]
      _ = Rep.dot dx (Tensor.scale (α:=ℝ) (s:=s) (Tensor.full s (1 : ℝ)) g) :=
            Tensor.dot_comm _ _
      _ = Rep.dot dx (Tensor.full (α := ℝ) s g) := by rw [hrep]
}

end
end TorchLean
