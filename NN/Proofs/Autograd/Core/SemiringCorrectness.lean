/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Algebra
public import NN.Spec.Autograd.Ops
public import NN.Spec.Core.TensorReductionShape

/-!
# SemiringCorrectness

Semiring-generic autograd correctness layer (backend-generic).

This mirrors `NN/Proofs/Autograd/Core/RealCorrectness.lean`, but avoids analytic assumptions and
works over any commutative semiring. In particular, it applies to exact backends like `ℚ`.

The correctness notion is the standard reverse-mode / forward-mode adjointness law:

  ⟪ JVP(x, dx), δ ⟫ = ⟪ dx, VJP(x, δ) ⟫

where `⟪·,·⟫` is the tensor dot-product from `NN/Proofs/Tensor/Algebra.lean`.

## Why this is separate from the ℝ file

Many ML ops are definable over a commutative semiring (addition/multiplication/linear maps), and
their reverse-mode rules can be proved from algebraic identities alone. This file isolates that
“pure algebra” portion so it can be instantiated for exact backends (e.g. `ℚ`) without pulling in
real-analytic structure.

Ops that require extra structure (e.g. ReLU needs an order/max, MSE needs division by `Shape.size`)
appear here only under the corresponding extra typeclass assumptions.

If you only care about real-valued training semantics, prefer `NN.Proofs.Autograd.Core.RealCorrectness`. If you
want proofs that can be instantiated for exact backends (`ℚ`, etc.), prefer this file.

## PyTorch correspondence / citations
This is the proof-level analogue of the “VJP correctness” property implicitly relied upon by
PyTorch Autograd: each primitive op must supply a correct local backward/VJP rule.
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor
open TensorAlgebra

noncomputable section

/-- VJP/JVP adjointness for a unary op `σ → τ`. -/
def VJPCorrect {α : Type} [CommSemiring α] {σ τ : Shape}
  (_forward : Tensor α σ → Tensor α τ)
  (jvp     : Tensor α σ → Tensor α σ → Tensor α τ)
  (vjp     : Tensor α σ → Tensor α τ → Tensor α σ) : Prop :=
  ∀ x dx δ, dot (α := α) (jvp x dx) δ = dot (α := α) dx (vjp x δ)

/--
An `OpSpec` together with a matching JVP and a proof of VJP/JVP adjointness.

This is the backend-generic analogue of `Proofs.Autograd.OpSpecCorrect` from
`NN.Proofs.Autograd.Core.RealCorrectness`.
-/
structure OpSpecCorrect (α : Type) [CommSemiring α] (σ τ : Shape) where
  /-- op. -/
  op : Spec.OpSpec α σ τ
  /-- jvp. -/
  jvp : Tensor α σ → Tensor α σ → Tensor α τ
  /-- correct. -/
  correct : VJPCorrect (α := α) op.forward jvp op.backward

namespace OpSpecCorrect

/--
Composition preserves VJP/JVP correctness (reverse-mode chain rule).

Informally: if `f` and `g` satisfy `⟪JVP,·⟫ = ⟪·,VJP⟫`, then so does `g ∘ f`, with the obvious
composed JVP and VJP.
-/
def compose {α : Type} [CommSemiring α] {σ τ υ : Shape}
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

/-!
## Small list/finite-sum bookkeeping

The semiring-generic dot product is defined by folding over coordinates, so we collect a small
congruence lemma to rewrite the per-coordinate term.
-/

/--
If two functions `f` and `g` are pointwise equal, then folding `(+ f x)` over a list is the same as
folding `(+ g x)`.
-/
private lemma foldl_add_congr {α β : Type} [Add α] (l : List β) (f g : β → α) (a : α)
    (h : ∀ x, f x = g x) :
    l.foldl (fun s x => s + f x) a = l.foldl (fun s x => s + g x) a := by
  induction l generalizing a with
  | nil => simp
  | cons hd tl ih =>
    simp [List.foldl, h hd, ih]

/--
Elementwise multiplication is self-adjoint with respect to the algebraic tensor dot-product.

Informally: `⟪dx ⊙ df, δ⟫ = ⟪dx, df ⊙ δ⟫`.
This is the main identity used to justify elementwise backward rules in a backend-generic way.
-/
private theorem dot_elemwise_adjoint {α : Type} [CommSemiring α] {s : Shape}
  (dx df δ : Tensor α s) :
  dot (α := α) (mulSpec dx df) δ = dot (α := α) dx (mulSpec df δ) := by
  induction s with
  | scalar =>
    cases dx; cases df; cases δ
    simp [mulSpec, map2Spec, mul_assoc]
  | dim n s ih =>
    cases dx with
    | dim fdx =>
      cases df with
      | dim fdf =>
        cases δ with
        | dim fδ =>
          have hterm :
              ∀ i : Fin n,
                dot (α := α) (mulSpec (fdx i) (fdf i)) (fδ i) =
                  dot (α := α) (fdx i) (mulSpec (fdf i) (fδ i)) := by
            intro i
            simpa using (ih (dx := fdx i) (df := fdf i) (δ := fδ i))
          have hfold :=
            foldl_add_congr (l := List.finRange n)
              (f := fun i => dot (α := α) (mulSpec (fdx i) (fdf i)) (fδ i))
              (g := fun i => dot (α := α) (fdx i) (mulSpec (fdf i) (fδ i)))
              (a := (0 : α)) hterm
          simpa [TensorAlgebra.dot, mulSpec, map2Spec] using hfold

/--
Correctness of ReLU’s backward rule, stated generically over `α`.

We assume the extra structure needed to *define* ReLU and its derivative (`Max`, order, and
decidable comparison).
PyTorch analogue: `torch.relu` / `torch.nn.functional.relu`.
-/
def reluCorrect {α : Type} [CommSemiring α]
  [Max α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape} :
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
PyTorch analogue: `torch.nn.Linear`’s linear map.
-/
def linearCorrect {α : Type} [CommSemiring α]
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
Correctness of scaling by a constant: forward and backward are both `x ↦ c • x`.

PyTorch analogue: `c * x` (with broadcasting aligned to shape).
-/
def scaleCorrect {α : Type} [CommSemiring α] {s : Shape} (c : α) :
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

PyTorch analogue: `x * rhs` (elementwise).
-/
def mulCorrect {α : Type} [CommSemiring α] {s : Shape} (rhs : Tensor α s) :
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

variable {α : Type} [CommSemiring α] [Sub α] [Div α] [Coe Nat α]

/--
Basic mean-squared error (MSE) scalar value:

`mse(predicted, target) = (∑ (predicted - target)^2) / size`.

This local definition is used only to define the loss `OpSpec`.
-/
def mseSpecBasic {s : Shape} (predicted target : Tensor α s) : α :=
  let diff := subSpec (α := α) (s := s) predicted target
  let squared := mulSpec (α := α) (s := s) diff diff
  let sum := sumSpec (α := α) (s := s) squared
  sum / (Shape.size s : α)

/--
Gradient of `mse_spec_basic` with respect to `predicted`, as a tensor of the same shape.

Up to conventions, this is `2*(predicted-target)/size`.
-/
def mseDerivSpecBasic {s : Shape} (predicted target : Tensor α s) : Tensor α s :=
  let diff := subSpec (α := α) (s := s) predicted target
  let two : α := (1 : α) + 1
  scaleSpec (α := α) (s := s) diff (two / (Shape.size s : α))

/--
Correctness of mean-squared error loss (MSE) as an `OpSpecCorrect`.

This section assumes extra operations (`Sub`, `Div`, and coercions from naturals) because the MSE
definition uses subtraction and division by `Shape.size`.
PyTorch analogue: `torch.nn.functional.mse_loss(reduction="mean")` (up to normalization
  conventions).
-/
def mseLossCorrect {s : Shape} (target : Tensor α s) :
  OpSpecCorrect (α := α) s Shape.scalar :=
{
  op :=
    { forward := fun yhat => Tensor.scalar (mseSpecBasic (α := α) (s := s) yhat target)
      backward := fun yhat dLdy =>
        let g := Tensor.toScalar dLdy
        scaleSpec (α := α) (s := s) (mseDerivSpecBasic (α := α) (s := s) yhat target) g
    }
  jvp := fun yhat dyhat =>
    let grad := mseDerivSpecBasic (α := α) (s := s) yhat target
    Tensor.scalar (dot (α := α) (s := s) dyhat grad)
  correct := by
    intro yhat dyhat δ
    cases δ with
    | scalar g =>
      set grad := mseDerivSpecBasic (α := α) (s := s) yhat target
      have hscale :=
        TensorAlgebra.dot_scale_right (α := α) (s := s) (a := dyhat) (b := grad) (k := g)
      -- LHS: ⟪⟪dyhat, grad⟫, g⟫ = (⟪dyhat, grad⟫) * g
      -- RHS: ⟪dyhat, g • grad⟫ = (⟪dyhat, grad⟫) * g
      calc
        dot (α := α) (Tensor.scalar (dot (α := α) (s := s) dyhat grad)) (Tensor.scalar g)
            = dot (α := α) (s := s) dyhat grad * g := by
                simp [TensorAlgebra.dot]
        _ = dot (α := α) (s := s) dyhat (scaleSpec (α := α) (s := s) grad g) := by
                simpa using hscale.symm
}

end

end
end Algebra
end Autograd
end Proofs
