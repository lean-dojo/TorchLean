/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic
public import NN.Spec.Autograd.Ops

/-!
# RealCorrectness

Real-valued autograd correctness layer (proof-only).

This file does **not** talk about calculus (`HasFDerivAt`) yet. Instead it proves the standard
reverse-mode / forward-mode *adjointness* law (aka VJP/JVP duality) for a core set of ops:

  ⟪ JVP(x, dx), δ ⟫ = ⟪ dx, VJP(x, δ) ⟫

where `⟪·,·⟫` is the tensor dot-product (sum of elementwise products).

This is strong enough to justify the reverse-mode chain rule and to build a proved-correct layer
on top of `Spec.OpSpec.compose`.

## Why this file exists (and why there is a second “algebraic” file)

We keep two correctness developments:

- `real_correctness.lean` (this file) specializes to `ℝ` and is the home for rules whose
  definitions/proofs genuinely depend on real-analytic structure (e.g. smooth activations and
  `exp/log`-style ops).
- `semiring_correctness.lean` is backend-generic over a type `α` with `[CommSemiring α]`. It is
  meant to instantiate to *exact* backends like `ℚ`, so it avoids assuming division, order, or
  transcendental functions unless an op explicitly requires them.

Keeping them separate prevents importing analysis-heavy assumptions into the semiring-generic proofs
and keeps compilation dependencies smaller.

## Technical difference

- This file uses the `Spec.dot`/`Tensor` theory from `NN/Proofs/Tensor/Basic.lean` (specialized to
  `ℝ`).
- The semiring-generic file uses `TensorAlgebra.dot` from `NN/Proofs/Tensor/Algebra.lean` and keeps
  all statements polymorphic in `α` with `[CommSemiring α]`.

## Runtime note
- The **runtime engine** in `NN.Runtime.Autograd.Engine` remains generic over `α` and works
  whenever the needed ops exist. Relating a concrete backend to these ℝ-proofs may require a
  separate semantic model (e.g. mapping to `ℝ` with rounding error bounds for NeuralFloat).

## PyTorch correspondence / citations
- PyTorch AD background and conventions (VJP in reverse-mode):
  https://pytorch.org/docs/stable/autograd.html
- Custom VJP rules are analogous to implementing `torch.autograd.Function`:
  https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function

References (background):
- Baydin et al., “Automatic Differentiation in Machine Learning: a Survey”, JMLR 2018
  (originally circulated as `arXiv:1502.05767`).
- Griewank & Walther, *Evaluating Derivatives* (2nd ed.), SIAM 2008 (reverse-mode AD foundations).
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

/-- VJP/JVP adjointness for a unary op `σ → τ`. -/
def VJPCorrect {σ τ : Shape}
  (_forward : Tensor ℝ σ → Tensor ℝ τ)
  (jvp     : Tensor ℝ σ → Tensor ℝ σ → Tensor ℝ τ)
  (vjp     : Tensor ℝ σ → Tensor ℝ τ → Tensor ℝ σ) : Prop :=
  ∀ x dx δ, dot (jvp x dx) δ = dot dx (vjp x δ)

/--
An `OpSpec` together with a matching JVP and a proof of VJP/JVP adjointness.

This is the “proved-correct local op” interface needed to build a sound reverse-mode tape.
-/
structure OpSpecCorrect (σ τ : Shape) where
  /-- op. -/
  op : Spec.OpSpec ℝ σ τ
  /-- jvp. -/
  jvp : Tensor ℝ σ → Tensor ℝ σ → Tensor ℝ τ
  /-- correct. -/
  correct : VJPCorrect op.forward jvp op.backward

namespace OpSpecCorrect

/--
Composition preserves VJP/JVP correctness (reverse-mode chain rule).

Informally: if `f` and `g` each satisfy the adjointness law, then `g ∘ f` does as well, with the
composed JVP and the composed VJP.
-/
def compose {σ τ υ : Shape}
  (f : OpSpecCorrect σ τ) (g : OpSpecCorrect τ υ) : OpSpecCorrect σ υ :=
{
  op := Spec.OpSpec.compose (α:=ℝ) f.op g.op
  jvp := fun x dx => g.jvp (f.op.forward x) (f.jvp x dx)
  correct := by
    intro x dx δ
    -- Use g's correctness at y = f x, dy = JVP_f x dx.
    have hg := g.correct (f.op.forward x) (f.jvp x dx) δ
    -- Use f's correctness with δ := VJP_g (f x) δ.
    have hf := f.correct x dx (g.op.backward (f.op.forward x) δ)
    -- Combine.
    simpa [Spec.OpSpec.compose, VJPCorrect] using hg.trans hf
}

end OpSpecCorrect

/-!
## A reusable adjointness identity

Most elementwise ops have JVP of the form `dx ⊙ f'(x)` and VJP of the form `f'(x) ⊙ δ`.
The following lemma is the “commute elementwise factors under dot” fact that makes those proofs
one-liners.
-/

/--
Elementwise multiplication is self-adjoint with respect to the tensor dot-product.

Informally: `⟪dx ⊙ df, δ⟫ = ⟪dx, df ⊙ δ⟫`.
-/
private theorem dot_elemwise_adjoint {s : Shape}
  (dx df δ : Tensor ℝ s) :
  dot (mulSpec dx df) δ = dot dx (mulSpec df δ) := by
  unfold dot
  -- Reduce to associativity of elementwise multiplication.
  -- `mul_spec_assoc` is proved in `NN/Proofs/tensor.lean`.
  simp [mul_spec_assoc]

-- ---------------------------------------------------------------------------
-- Primitive ops
-- ---------------------------------------------------------------------------

/--
Correctness of ReLU’s backward rule.

PyTorch analogue: `torch.nn.functional.relu` / `torch.relu` with its standard VJP.
-/
def reluCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.reluOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (Activation.reluDerivSpec (α:=ℝ) (s:=s) x)
  correct := by
    intro x dx δ
    -- VJP is `relu'(x) ⊙ δ`; JVP is `dx ⊙ relu'(x)`.
    simpa [Spec.reluOp, Spec.liftElementwiseBackward, Spec.liftElementwise,
      Activation.reluDerivSpec] using (dot_elemwise_adjoint (dx:=dx)
        (df:=Activation.reluDerivSpec (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of sigmoid’s backward rule.

PyTorch analogue: `torch.sigmoid`.
-/
def sigmoidCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.sigmoidOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (Activation.sigmoidDerivSpec (α:=ℝ) (s:=s) x)
  correct := by
    intro x dx δ
    simpa [Spec.sigmoidOp, Spec.liftElementwiseBackward, Spec.liftElementwise,
      Activation.sigmoidDerivSpec] using (dot_elemwise_adjoint (dx:=dx)
        (df:=Activation.sigmoidDerivSpec (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of tanh’s backward rule.

PyTorch analogue: `torch.tanh`.
-/
def tanhCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.tanhOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (Activation.tanhDerivSpec (α:=ℝ) (s:=s) x)
  correct := by
    intro x dx δ
    simpa [Spec.tanhOp, Spec.liftElementwiseBackward, Spec.liftElementwise,
      Activation.tanhDerivSpec] using (dot_elemwise_adjoint (dx:=dx)
        (df:=Activation.tanhDerivSpec (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of softplus’s backward rule.

PyTorch analogue: `torch.nn.functional.softplus`.
-/
def softplusCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.softplusOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (Activation.softplusDerivSpec (α:=ℝ) (s:=s) x)
  correct := by
    intro x dx δ
    simpa [Spec.softplusOp, Spec.liftElementwiseBackward, Spec.liftElementwise,
      Activation.softplusDerivSpec] using (dot_elemwise_adjoint (dx:=dx)
        (df:=Activation.softplusDerivSpec (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of SiLU’s backward rule.

PyTorch analogue: `torch.nn.functional.silu`, equivalently `x * sigmoid(x)`.
-/
def siluCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.swishOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (Activation.swishDerivSpec (α:=ℝ) (s:=s) x)
  correct := by
    intro x dx δ
    simpa [Spec.swishOp, Activation.swishDerivSpec] using (dot_elemwise_adjoint (dx:=dx)
      (df:=Activation.swishDerivSpec (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of tanh-approximate GELU's VJP/JVP adjointness rule.

This proves the linear-algebraic part of the `gelu` backward rule used by Transformer-style
feed-forward blocks: multiplying the upstream cotangent by the local derivative mask is adjoint to
multiplying the tangent by the same mask. The scalar calculus theorem for the full tanh
approximation is intentionally separate because it depends on a longer chain-rule proof through
`tanh`, `sqrt`, and the cubic inner polynomial.

PyTorch analogue: `torch.nn.functional.gelu(..., approximate="tanh")`.
-/
def geluCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.geluOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (Activation.geluDerivSpec (α:=ℝ) (s:=s) x)
  correct := by
    intro x dx δ
    simpa [Spec.geluOp] using (dot_elemwise_adjoint (dx:=dx)
      (df:=Activation.geluDerivSpec (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of `safe_log`’s backward rule (a log with an `ε` safeguard).

PyTorch analogue: typically implemented as `torch.log(torch.clamp(x, min=ε))` (or similar).
-/
def safeLogCorrect {s : Shape} (ε : ℝ := Numbers.epsilon) :
  OpSpecCorrect s s :=
{
  op := Spec.safeLogOp (α:=ℝ) (s:=s) (ε := ε)
  jvp := fun x dx => mulSpec dx (Activation.safeLogDerivSpec (α:=ℝ) (s:=s) x ε)
  correct := by
    intro x dx δ
    simpa [Spec.safeLogOp, Spec.liftElementwiseBackward, Spec.liftElementwise,
      Activation.safeLogDerivSpec] using (dot_elemwise_adjoint (dx:=dx)
        (df:=Activation.safeLogDerivSpec (α:=ℝ) (s:=s) x ε) (δ:=δ))
}

/--
Correctness of a smooth absolute value’s backward rule (a differentiable approximation to `|x|`).

PyTorch analogue: a custom smooth `abs` implemented via `sqrt(x^2 + ε^2)` or similar.
-/
def smoothAbsCorrect {s : Shape} (ε : ℝ := Numbers.epsilon) :
  OpSpecCorrect s s :=
{
  op := Spec.smoothAbsOp (α:=ℝ) (s:=s) (ε := ε)
  jvp := fun x dx => mulSpec dx (Activation.smoothAbsDerivSpec (α:=ℝ) (s:=s) x ε)
  correct := by
    intro x dx δ
    simpa [Spec.smoothAbsOp, Spec.liftElementwiseBackward, Spec.liftElementwise,
      Activation.smoothAbsDerivSpec] using (dot_elemwise_adjoint (dx:=dx)
        (df:=Activation.smoothAbsDerivSpec (α:=ℝ) (s:=s) x ε) (δ:=δ))
}

/--
Correctness of `exp`’s backward rule.

PyTorch analogue: `torch.exp`.
-/
def expCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.expOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (expSpec (α:=ℝ) (s:=s) x)
  correct := by
    intro x dx δ
    simpa [Spec.expOp] using (dot_elemwise_adjoint (dx:=dx)
      (df:=expSpec (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of `square`'s backward rule.

PyTorch analogue: `torch.square`; the local derivative is `2 * x`.
-/
def squareCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.squareOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (mulSpec (fill (Numbers.two : ℝ) s) x)
  correct := by
    intro x dx δ
    simpa [Spec.squareOp] using (dot_elemwise_adjoint (dx:=dx)
      (df:=mulSpec (fill (Numbers.two : ℝ) s) x) (δ:=δ))
}

/--
Correctness of ELU's VJP/JVP adjointness rule.

This is the algebraic half of the argument: once a local derivative mask is chosen, the VJP
`elu'(x) ⊙ δ` is adjoint to the JVP `dx ⊙ elu'(x)`. The analytic differentiability theorem lives in
`Proofs.elu_deriv_correct`, which correctly excludes the kink at `0` for arbitrary `alpha`.

PyTorch analogue: `torch.nn.functional.elu`.
-/
def eluCorrect {s : Shape} (eluAlpha : ℝ) :
  OpSpecCorrect s s :=
{
  op := Spec.eluOp (α:=ℝ) (s:=s) eluAlpha
  jvp := fun x dx => mulSpec dx (Activation.eluDerivSpec (α:=ℝ) (s:=s) x eluAlpha)
  correct := by
    intro x dx δ
    simpa [Spec.eluOp] using (dot_elemwise_adjoint (dx:=dx)
      (df:=Activation.eluDerivSpec (α:=ℝ) (s:=s) x eluAlpha) (δ:=δ))
}

/--
Correctness of `sinh`'s backward rule.

PyTorch analogue: `torch.sinh`; the local derivative is `cosh`.
-/
def sinhCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.sinhOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (coshSpec (α:=ℝ) (s:=s) x)
  correct := by
    intro x dx δ
    simpa [Spec.sinhOp, Spec.liftElementwiseBackward] using (dot_elemwise_adjoint (dx:=dx)
      (df:=coshSpec (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of `cosh`'s backward rule.

PyTorch analogue: `torch.cosh`; the local derivative is `sinh`.
-/
def coshCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.coshOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (sinhSpec (α:=ℝ) (s:=s) x)
  correct := by
    intro x dx δ
    simpa [Spec.coshOp, Spec.liftElementwiseBackward] using (dot_elemwise_adjoint (dx:=dx)
      (df:=sinhSpec (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of `log`'s backward rule.

PyTorch analogue: `torch.log`.
-/
def logCorrect {s : Shape} :
  OpSpecCorrect s s :=
{
  op := Spec.logOp (α:=ℝ) (s:=s)
  jvp := fun x dx => mulSpec dx (invSpec (α:=ℝ) (s:=s) x)
  correct := by
    intro x dx δ
    simpa [Spec.logOp] using (dot_elemwise_adjoint (dx:=dx)
      (df:=invSpec (α:=ℝ) (s:=s) x) (δ:=δ))
}

/--
Correctness of a linear layer’s backward rule (matrix–vector multiply).

PyTorch analogue: `torch.nn.Linear` (restricted here to the “weights only” linear map).
-/
def linearCorrect {inDim outDim : Nat}
  (m : Spec.LinearSpec ℝ inDim outDim) :
  OpSpecCorrect (.dim inDim .scalar) (.dim outDim .scalar) :=
{
  op := Spec.linearOp (α:=ℝ) (inDim:=inDim) (outDim:=outDim) m
  jvp := fun _x dx => matVecMulSpec m.weights dx
  correct := by
    intro x dx δ
    -- `linear_op.backward` ignores the input `x`, so the proof is purely linear-algebraic.
    -- Use the adjoint lemma from `NN/Proofs/tensor.lean`.
    -- Our goal has dot with the arguments flipped compared to the lemma; use `dot_comm`.
    classical
    have hadj :=
      dot_mat_linear_adjoint (W := m.weights) (dLdy := δ) (dx := dx)
    calc
      dot (matVecMulSpec m.weights dx) δ
          = dot δ (matVecMulSpec m.weights dx) := by
              simpa using (dot_comm (a := matVecMulSpec m.weights dx) (b := δ))
      _ = dot (vecMatMulSpec δ m.weights) dx := hadj
      _ = dot dx (vecMatMulSpec δ m.weights) := by
              simpa using (dot_comm (a := vecMatMulSpec δ m.weights) (b := dx))
}

/--
Correctness of `sum` (reduce-all) backward rule.

Informally: `d/dx (sum x) = 1`, so the VJP replicates the upstream scalar gradient into every entry.

PyTorch analogue: `torch.sum` (over all elements).
-/
def sumCorrect {s : Shape} : OpSpecCorrect s Shape.scalar :=
{
  op :=
    { forward := fun x => Tensor.scalar (sumSpec (α:=ℝ) (s:=s) x)
      backward := fun _x dLdy => replicate (α:=ℝ) (s:=s) dLdy }
  jvp := fun _x dx => Tensor.scalar (sumSpec (α:=ℝ) (s:=s) dx)
  correct := by
    intro x dx δ
    cases δ with
    | scalar g =>
      -- `replicate (scalar g)` is the "all-`g`" tensor, i.e. `g • fill 1`.
      have hrep :
          replicate (α:=ℝ) (s:=s) (Tensor.scalar g) =
            scaleSpec (α:=ℝ) (s:=s) (fill (1 : ℝ) s) g := by
        induction s with
        | scalar =>
          simp [replicate, fill, scaleSpec, mapSpec]
        | dim n s ih =>
          simp [replicate, fill, scaleSpec, mapSpec]
          funext i
          simpa [scaleSpec] using ih
      -- Reduce both sides using `dot_scale_left` and the fact that `fill 1` is multiplicative
      -- identity.
      calc
        dot (Tensor.scalar (sumSpec (α:=ℝ) (s:=s) dx)) (Tensor.scalar g)
            = (sumSpec (α:=ℝ) (s:=s) dx) * g := by
                simp [dot, sumSpec, mulSpec, tensorFoldlSpec, map2Spec]
        _ = g * (sumSpec (α:=ℝ) (s:=s) dx) := by
              ring
        _ = g * dot dx (fill (1 : ℝ) s) := by
              simp [dot, mul_spec_one_right]
        _ = g * dot (fill (1 : ℝ) s) dx := by
              simp [dot_comm]
        _ = dot (scaleSpec (α:=ℝ) (s:=s) (fill (1 : ℝ) s) g) dx := by
              symm
              simpa using (dot_scale_left (a := fill (1 : ℝ) s) (b := dx) (k := g))
        _ = dot dx (scaleSpec (α:=ℝ) (s:=s) (fill (1 : ℝ) s) g) := by
              simpa using (dot_comm (a := scaleSpec (α:=ℝ) (s:=s) (fill (1 : ℝ) s) g) (b := dx))
        _ = dot dx (replicate (α:=ℝ) (s:=s) (Tensor.scalar g)) := by
              simp [hrep]
}

end
end Autograd
end Proofs
