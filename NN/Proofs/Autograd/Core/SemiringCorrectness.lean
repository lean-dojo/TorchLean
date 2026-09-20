/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Algebra
public import NN.Spec.Autograd.Ops

/-!
# SemiringCorrectness

Semiring-generic autograd correctness layer (backend-generic).

This mirrors `NN/Proofs/Autograd/Core/RealCorrectness.lean`, but avoids analytic assumptions and
works over any commutative semiring. In particular, it applies to exact backends like `ℚ`.

The correctness notion is the standard reverse-mode / forward-mode adjointness law:

$$
\left\langle \operatorname{JVP}(x,dx),\delta\right\rangle
=\left\langle dx,\operatorname{VJP}(x,\delta)\right\rangle.
$$

where $\langle\cdot,\cdot\rangle$ is the tensor dot product from
`NN/Proofs/Tensor/Algebra.lean`.

## Why this is separate from the $\mathbb R$ file

Many ML ops are definable over a commutative semiring (addition/multiplication/linear maps), and
their reverse-mode rules can be proved from algebraic identities alone. This file isolates that
“pure algebra” portion so it can be instantiated for exact backends (e.g. `ℚ`) without pulling in
real-analytic structure.

Ops that require extra structure (e.g. ReLU needs an order/max, MSE needs division by
`Spec.Shape.size`) appear here only under the corresponding extra typeclass assumptions.

If you only care about real-valued training semantics, prefer
`NN.Proofs.Autograd.Core.RealCorrectness`. If you want proofs that can be instantiated for exact
backends (`ℚ`, etc.), prefer this file.

## PyTorch correspondence / citations
This is the proof-level analogue of the “VJP correctness” property implicitly relied upon by
PyTorch Autograd: each primitive op must supply a correct local backward/VJP rule.
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Algebra

open Spec TorchLean
open TorchLean TorchLean.Tensor
open TensorAlgebra

noncomputable section

/-- VJP/JVP adjointness for a unary op `σ → τ`. -/
def VJPCorrect {α : Type} [TorchLean.Storage α] [CommSemiring α] {σ τ : Shape}
  (_forward : Tensor α σ → Tensor α τ)
  (jvp     : Tensor α σ → Tensor α σ → Tensor α τ)
  (vjp     : Tensor α σ → Tensor α τ → Tensor α σ) : Prop :=
  ∀ x dx δ, dot (α := α) (jvp x dx) δ = dot (α := α) dx (vjp x δ)

/--
An `OpSpec` together with a matching JVP and a proof of VJP/JVP adjointness.

This is the backend-generic analogue of `Proofs.Autograd.OpSpecCorrect` from
`NN.Proofs.Autograd.Core.RealCorrectness`.
-/
structure OpSpecCorrect (α : Type) [TorchLean.Storage α] [CommSemiring α] (σ τ : Shape) where
  /-- The operation being certified, forward and backward together. -/
  op : Spec.OpSpec α σ τ
  /-- Forward-mode derivative at a basepoint, applied to a tangent. -/
  jvp : Tensor α σ → Tensor α σ → Tensor α τ
  /-- The adjointness proof. Bundling it with the operation is what makes a value of this type a
  certificate: you cannot obtain one without having shown the backward pass is the transpose. -/
  correct : VJPCorrect (α := α) op.forward jvp op.backward

namespace OpSpecCorrect

/--
Composition preserves VJP/JVP correctness (reverse-mode chain rule).

Informally, if $f$ and $g$ satisfy
$\langle\operatorname{JVP},\cdot\rangle
=\langle\cdot,\operatorname{VJP}\rangle$,
then so does $g\circ f$, with the obvious composed JVP and VJP.
-/
def compose {α : Type} [TorchLean.Storage α] [CommSemiring α] {σ τ υ : Shape}
  (f : OpSpecCorrect (α := α) σ τ) (g : OpSpecCorrect (α := α) τ υ) :
  OpSpecCorrect (α := α) σ υ :=
{
  op := Spec.OpSpec.compose (α := α) f.op g.op
  jvp := fun x dx => g.jvp (f.op.forward x) (f.jvp x dx)
  correct := by
    intro x dx δ
    have hg := g.correct (f.op.forward x) (f.jvp x dx) δ
    have hf := f.correct x dx (g.op.backward (f.op.forward x) δ)
    simpa [Spec.OpSpec.compose, VJPCorrect] using hg.trans hf
}

end OpSpecCorrect

/--
Elementwise multiplication is self-adjoint with respect to the algebraic tensor dot-product.

Informally,
$\langle dx\odot df,\delta\rangle=\langle dx,df\odot\delta\rangle$.
This is the main identity used to justify elementwise backward rules in a backend-generic way.
-/
private theorem dot_elemwise_adjoint {α : Type} [TorchLean.Storage α] [CommSemiring α] {s : Shape}
  (dx df δ : Tensor α s) :
  dot (α := α) (mulSpec dx df) δ = dot (α := α) dx (mulSpec df δ) := by
  induction s with
  | scalar =>
      simpa [TensorAlgebra.dot, mulSpec, map2Spec, Tensor.item] using
        (mul_assoc dx.item df.item δ.item)
  | dim n s ih =>
      have hterm :
          ∀ i : Fin n,
            dot (α := α) (mulSpec (dx.unstack i) (df.unstack i)) (δ.unstack i) =
              dot (α := α) (dx.unstack i) (mulSpec (df.unstack i) (δ.unstack i)) := by
        intro i
        exact ih (dx := dx.unstack i) (df := df.unstack i) (δ := δ.unstack i)
      have hfold :=
        List.foldl_add_congr (l := List.finRange n)
          (f := fun i => dot (α := α) (mulSpec (dx.unstack i) (df.unstack i)) (δ.unstack i))
          (g := fun i => dot (α := α) (dx.unstack i) (mulSpec (df.unstack i) (δ.unstack i)))
          (a := (0 : α)) hterm
      have hdxdf : ∀ i : Fin n,
          (mulSpec dx df).unstack i = mulSpec (dx.unstack i) (df.unstack i) := by
        intro i
        exact (TorchLean.Tensor.Internal.Rep.zipWith_unstack (· * ·) dx df i).symm
      have hdfδ : ∀ i : Fin n,
          (mulSpec df δ).unstack i = mulSpec (df.unstack i) (δ.unstack i) := by
        intro i
        exact (TorchLean.Tensor.Internal.Rep.zipWith_unstack (· * ·) df δ i).symm
      simp only [TensorAlgebra.dot, hdxdf, hdfδ]
      exact hfold

/--
Correctness of ReLU’s backward rule, stated generically over `α`.

We assume the extra structure needed to define ReLU and its derivative: maximum, scalar equality,
order, and decidable comparison.
PyTorch analogue: `torch.relu` / `torch.nn.functional.relu`.
-/
def reluCorrect {α : Type} [TorchLean.Storage α] [CommSemiring α]
  [Max α] [BEq α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape} :
  OpSpecCorrect (α := α) s s :=
{
  op := Spec.reluOp (α := α) (s := s)
  jvp := fun x dx => mulSpec dx (Activation.reluDerivSpec (α := α) (s := s) x)
  correct := by
    intro x dx δ
    simpa [Spec.reluOp, Spec.liftElementwiseBackward, Spec.liftElementwise,
      Activation.reluDerivSpec] using
        (dot_elemwise_adjoint (α := α) (s := s) (dx := dx)
          (df := Activation.reluDerivSpec (α := α) (s := s) x) (δ := δ))
}

/--
Correctness of a linear layer’s backward rule (matrix–vector multiply), stated generically over `α`.

This is purely algebraic: it relies only on semiring laws and the adjointness lemma for matrix
multiplication in `TensorAlgebra`.
PyTorch analogue: the affine map implemented by `torch.nn.Linear`.
-/
def linearCorrect {α : Type} [TorchLean.Storage α] [CommSemiring α]
  {inDim outDim : Nat} (m : Spec.LinearSpec α inDim outDim) :
  OpSpecCorrect (α := α) (.dim inDim .scalar) (.dim outDim .scalar) :=
{
  op := Spec.linearOp (α := α) (inDim := inDim) (outDim := outDim) m
  jvp := fun _x dx => matVecMulSpec m.weights dx
  correct := by
    intro x dx δ
    classical
    have hadj :=
      TensorAlgebra.dot_mat_linear_adjoint (α := α) (W := m.weights) (dLdy := δ) (dx := dx)
    calc
      dot (α := α) (matVecMulSpec m.weights dx) δ
          = dot (α := α) δ (matVecMulSpec m.weights dx) := by
              simpa using (TensorAlgebra.dot_comm (α := α) (a := matVecMulSpec m.weights dx) (b
                := δ))
      _ = dot (α := α) (vecMatMulSpec δ m.weights) dx := hadj
      _ = dot (α := α) dx (vecMatMulSpec δ m.weights) := by
              simpa using (TensorAlgebra.dot_comm (α := α) (a := vecMatMulSpec δ m.weights) (b :=
                dx))
}

/--
Correctness of scaling by a constant: forward and backward are both $x\mapsto cx$.

PyTorch analogue: $cx$ (with broadcasting aligned to shape).
-/
def scaleCorrect {α : Type} [TorchLean.Storage α] [CommSemiring α] {s : Shape} (c : α) :
  OpSpecCorrect (α := α) s s :=
{
  op :=
    { forward := fun x => scaleSpec (α := α) (s := s) x c
      backward := fun _x dLdy => scaleSpec (α := α) (s := s) dLdy c }
  jvp := fun _x dx => scaleSpec (α := α) (s := s) dx c
  correct := by
    intro x dx δ
    have hL := TensorAlgebra.dot_scale_left (α := α) (s := s) (a := dx) (b := δ) (k := c)
    have hR := TensorAlgebra.dot_scale_right (α := α) (s := s) (a := dx) (b := δ) (k := c)
    -- Both sides reduce to `dot dx δ * c`.
    simpa [VJPCorrect] using hL.trans hR.symm
}

/--
Correctness of pointwise multiplication by a fixed tensor `rhs`.

PyTorch analogue: $x\odot\operatorname{rhs}$ (elementwise).
-/
def mulCorrect {α : Type} [TorchLean.Storage α] [CommSemiring α] {s : Shape} (rhs : Tensor α s) :
  OpSpecCorrect (α := α) s s :=
{
  op :=
    { forward := fun x => mulSpec (α := α) (s := s) x rhs
      backward := fun _x dLdy => mulSpec (α := α) (s := s) rhs dLdy }
  jvp := fun _x dx => mulSpec (α := α) (s := s) dx rhs
  correct := by
    intro x dx δ
    simpa [VJPCorrect] using
      (dot_elemwise_adjoint (α := α) (s := s) (dx := dx) (df := rhs) (δ := δ))
}

section

variable {α : Type} [TorchLean.Storage α] [CommSemiring α] [Sub α] [Div α]

/--
Correctness of mean-squared error loss (MSE) as an `OpSpecCorrect`.

The MSE correctness declaration assumes extra operations (`Sub`, `Div`, and coercions from
naturals) because the MSE definition uses subtraction and division by the totalized element count
`TorchLean.Tensor.meanDenominator`.
PyTorch analogue: `torch.nn.functional.mse_loss(reduction="mean")` (up to normalization
  conventions).
-/
def mseLossCorrect {s : Shape} (target : Tensor α s) :
  OpSpecCorrect (α := α) s Shape.scalar :=
{
  op :=
    { forward := fun yhat => Tensor.scalar (Spec.mseSpec (α := α) yhat target)
      backward := fun yhat dLdy =>
        let g := Tensor.item dLdy
        scaleSpec (α := α) (s := s) (Spec.mseDerivSpec (α := α) yhat target) g
    }
  jvp := fun yhat dyhat =>
    let grad := Spec.mseDerivSpec (α := α) yhat target
    Tensor.scalar (dot (α := α) (shape := s) dyhat grad)
  correct := by
    intro yhat dyhat δ
    change
      dot (α := α)
          (Tensor.scalar (dot (α := α) (shape := s) dyhat
            (Spec.mseDerivSpec (α := α) yhat target))) δ =
        dot (α := α) (shape := s) dyhat
          (scaleSpec (α := α) (s := s) (Spec.mseDerivSpec (α := α) yhat target) δ.item)
    set g := δ.item
    set grad := Spec.mseDerivSpec (α := α) yhat target
    have hscale :=
      TensorAlgebra.dot_scale_right (α := α) (s := s) (a := dyhat) (b := grad) (k := g)
    -- LHS: ⟪⟪dyhat, grad⟫, g⟫ = (⟪dyhat, grad⟫) * g
    -- RHS: ⟪dyhat, g • grad⟫ = (⟪dyhat, grad⟫) * g
    calc
      dot (α := α) (Tensor.scalar (dot (α := α) (shape := s) dyhat grad)) δ
          = dot (α := α) (shape := s) dyhat grad * g := by
              simp [TensorAlgebra.dot, g]
      _ = dot (α := α) (shape := s) dyhat (scaleSpec (α := α) (s := s) grad g) := by
              exact hscale.symm
}

end

end
end Algebra
end Autograd
end Proofs
