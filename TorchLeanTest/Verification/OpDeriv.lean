/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Op.Attention
public import TorchLean.NN.Op.Elementwise
public import TorchLean.NN.Op.Loss
public import TorchLean.NN.Op.Matmul
public import TorchLean.NN.Op.Order
public import TorchLean.NN.Tactic.Autograd

/-!
# `op_deriv` on programs outside the operator table

No operator certificate covers the programs below. Each one is a forward map that a user writes
from functions of the library, with the backward pass written by hand, often with the library's
own backward maps. `op_deriv` proves that the backward pass is the adjoint of the Fréchet
derivative of the forward map, by composing the pullback rules of the functions in it.

- Scalar functions at every entry: softplus, log-cosh, GELU written out with its two constants,
  the square root, `relu`, the absolute value and `max a 0` off zero, `min a 6` and `min 6 a` off
  `6`, the sigmoid with its slope written out, `a / (a a + 1)` with no hypothesis, the real powers
  `a ^ (3 / 2)` with no hypothesis and `a ^ (1 / 2)` and `a ^ a` at positive entries, and two
  cubics of our own, one with a derivative registered with `@[autograd]` and one with its
  derivative as a hypothesis.
- The goal forms: `DifferentiableAt` alone, `HasFDerivAt` with a given derivative, also one
  rearranged by linearity, the certificate with a given derivative, a composition written with
  `∘`, and a backward pass and an operator's table entry bound by a `let` introduced with `intro`.
- Several inputs and outputs: `lerp` then `sin` and the sum, jointly in three inputs, five inputs
  in a nested L2 product, and a pair of outputs.
- Real arithmetic: `sin` and the square on `ℝ`, products and quotients of real-valued maps of
  tensors, a real constant times or dividing a squared error, a quotient by `y y + 1` with no
  hypothesis, a real variable times a constant tensor, a real-valued map times a tensor map,
  `exp` of a sum and a real power of a sum.
- Functions of the library: unmasked softmax, `Tensor.exp`, the value of a scalar tensor and an
  entry, `Tensor.scale` by a fixed or a varying real, the batched matrix product, a contraction,
  and `zipWith` of a binary function with one or two varying operands.
- Layers: a linear layer with `tanh`, in the weights and the bias and in all three of weights, bias
  and input, a residual block, and a two-layer network in its four parameters, with the factors of
  the products in the backward pass in either order.
- Masked single-head attention `softmax(Q Kᵀ) V`, also with the scores scaled by a real, jointly
  in `Q`, `Kᵀ` and `V`.
- Losses: cross-entropy through `logSoftmax`, a layer normalization and an RMS normalization
  written by hand, a `Finset` sum of scaled maps, and the sum of the squared entries over all
  coordinates.
- Order statistics: an ordered weighted sum of the sorted entries at a point with distinct
  entries, and the sorted entries plus the running maximum.

The lemmas that move sums and scalars of the cotangent out of the backward maps of the library,
which `op_deriv` uses to compare a backward pass with a composed pullback, are checked on their
own. The last examples check that `op_deriv` rejects a wrong backward pass, the square root,
`relu` and `a ^ (1 / 2)` without the hypothesis that no entry is zero, and a definition that is not
given in brackets.
-/

@[expose] public section

namespace TorchLean.Test.OpDeriv

open TorchLean Tensor

variable {s : Shape}

section Scalar

/-- Softplus at every entry, `log (1 + exp x)`, has the backward pass `δ * exp x / (1 + exp x)`. -/
theorem softplus_backward (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => Real.log (1 + Real.exp a)) y)
        x ∧
      δ * Tensor.map (fun a => Real.exp a / (1 + Real.exp a)) x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => Real.log (1 + Real.exp a)) y)
          x).adjoint δ := by
  op_deriv

/-- The log-cosh loss `∑ log (cosh (p - t))` of a prediction `p` against a target `t` has the
backward pass `δ * tanh (p - t)` in the prediction. -/
theorem logCosh_backward (p t : Tensor ℝ s) (δ : ℝ) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s =>
        Tensor.sum (Tensor.map (fun a => Real.log (Real.cosh a)) (y - t))) p ∧
      Tensor.full s δ * Tensor.map Real.tanh (p - t) =
        (fderiv ℝ (fun y : Tensor ℝ s =>
          Tensor.sum (Tensor.map (fun a => Real.log (Real.cosh a)) (y - t))) p).adjoint δ := by
  have _ : ∀ c, Real.cosh ((p - t) c) ≠ 0 := fun c => (Real.cosh_pos _).ne'
  op_deriv [Real.tanh_eq_sinh_div_cosh]

/-- GELU written out as `a (1 + tanh (c₁ (a + c₂ a³)))` at every entry, for any constants `c₁` and
`c₂`, has the backward pass `δ` times its derivative by the product and chain rules. -/
theorem gelu_backward (c₁ c₂ : ℝ) (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s =>
        Tensor.map (fun a => a * (1 + Real.tanh (c₁ * (a + c₂ * a ^ 3)))) y) x ∧
      δ * Tensor.map (fun a => (1 + Real.tanh (c₁ * (a + c₂ * a ^ 3))) +
          a * ((1 - Real.tanh (c₁ * (a + c₂ * a ^ 3)) ^ 2) * (c₁ * (1 + c₂ * (3 * a ^ 2))))) x =
        (fderiv ℝ (fun y : Tensor ℝ s =>
          Tensor.map (fun a => a * (1 + Real.tanh (c₁ * (a + c₂ * a ^ 3)))) y) x).adjoint δ := by
  op_deriv

/-- Where no entry is zero, the square root at every entry has the backward pass
`δ / (2 √x)`. -/
theorem sqrt_backward (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    δ * Tensor.map (fun a => 1 / (2 * Real.sqrt a)) x =
      (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map Real.sqrt y) x).adjoint δ := by
  op_deriv

/-- Where no entry is zero, `relu` at every entry has the backward pass `δ * reluDeriv x`. -/
theorem relu_backward (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map Activation.relu y) x ∧
      δ * Tensor.map Activation.reluDeriv x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map Activation.relu y) x).adjoint δ := by
  op_deriv

/-- Where no entry is zero, the absolute value at every entry has the backward pass
`δ * sign x`. -/
theorem abs_backward (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => |a|) y) x ∧
      δ * Tensor.map (fun a => (SignType.sign a : ℝ)) x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => |a|) y) x).adjoint δ := by
  op_deriv

/-- Where no entry is zero, `max a 0` at every entry, `relu` with the constant on the right, has
the backward pass `δ` at a positive entry and zero at a negative one. -/
theorem maxZero_backward (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => max a 0) y) x ∧
      δ * Tensor.map (fun a => if 0 < a then 1 else 0) x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => max a 0) y) x).adjoint δ := by
  op_deriv

/-- Where no entry is `6`, `min a 6` and `min 6 a` at every entry are differentiable. -/
theorem minSix_differentiableAt (x : Tensor ℝ s) (hx : ∀ c, x c ≠ 6) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => min a 6) y) x ∧
      DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => min 6 a) y) x :=
  ⟨by op_deriv, by op_deriv⟩

/-- The sigmoid at every entry has the backward pass `δ σ(x) (1 - σ(x))`. -/
theorem sigmoid_backward (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map Context.sigmoid y) x ∧
      δ * Tensor.map (fun a => Context.sigmoid a * (1 - Context.sigmoid a)) x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map Context.sigmoid y) x).adjoint δ := by
  op_deriv

/-- `a / (a a + 1)` at every entry is differentiable at every point, since its denominator is
positive. -/
theorem map_div_mul_self_add_one_differentiableAt (x : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => a / (a * a + 1)) y) x := by
  op_deriv

/-- The real power `a ^ (3 / 2)` at every entry is differentiable at every point, since its
exponent is at least one, and has the backward pass `δ (3 / 2) a ^ (1 / 2)`. -/
theorem rpow_backward (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => a ^ (3 / 2 : ℝ)) y) x ∧
      δ * Tensor.map (fun a => 3 / 2 * a ^ (1 / 2 : ℝ)) x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => a ^ (3 / 2 : ℝ)) y)
          x).adjoint δ := by
  op_deriv

/-- Where every entry is positive, the real power `a ^ (1 / 2)` at every entry is
differentiable. -/
theorem rpow_half_differentiableAt (x : Tensor ℝ s) (hx : ∀ c, 0 < x c) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => a ^ (1 / 2 : ℝ)) y) x := by
  op_deriv

/-- Where every entry is positive, `a ^ a` at every entry is differentiable. -/
theorem rpow_self_differentiableAt (x : Tensor ℝ s) (hx : ∀ c, 0 < x c) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => a ^ a) y) x := by
  op_deriv

/-- The cubic `a ^ 3 + a`, a scalar function of our own. -/
noncomputable def cubic (a : ℝ) : ℝ := a ^ 3 + a

/-- The derivative of `cubic`, registered for `autograd` and so for `op_deriv`. -/
@[autograd] theorem hasDerivAt_cubic (a : ℝ) : HasDerivAt cubic (3 * a ^ 2 + 1) a :=
  ((hasDerivAt_pow 3 a).add (hasDerivAt_id' a)).congr_deriv (by norm_num)

/-- `cubic` at every entry has the backward pass `δ (3 x² + 1)`, from its registered derivative. -/
theorem cubic_backward (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map cubic y) x ∧
      δ * Tensor.map (fun a => 3 * a ^ 2 + 1) x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map cubic y) x).adjoint δ := by
  op_deriv

/-- The cubic `a ^ 3 - a`, whose derivative we do not register. -/
noncomputable def cubicSub (a : ℝ) : ℝ := a ^ 3 - a

/-- Given the derivative of `cubicSub` at every entry as a hypothesis, `cubicSub` at every entry
has the backward pass `δ (3 x² - 1)`. -/
theorem cubicSub_backward (x δ : Tensor ℝ s)
    (hd : ∀ c, HasDerivAt cubicSub (3 * x c ^ 2 - 1) (x c)) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map cubicSub y) x ∧
      δ * Tensor.map (fun a => 3 * a ^ 2 - 1) x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map cubicSub y) x).adjoint δ := by
  op_deriv

end Scalar

section Forms

/-- `exp` at every entry is differentiable. -/
theorem exp_differentiableAt (x : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map Real.exp y) x := by
  op_deriv

/-- The sum of the squares of the entries has the derivative `v ↦ ∑ (x v + x v)` at `x`. -/
theorem sum_mul_self_hasFDerivAt (x : Tensor ℝ s) :
    HasFDerivAt (fun y : Tensor ℝ s => Tensor.sum (y * y))
      ((Tensor.sumCLM s).comp (Op.slopeCLM (fun c => x c) + Op.slopeCLM (fun c => x c))) x := by
  op_deriv

/-- The sum of the squares of the entries has the derivative `2 • (v ↦ ∑ x v)` at `x`, which is
the composed derivative rearranged by linearity. -/
theorem sum_mul_self_hasFDerivAt_smul (x : Tensor ℝ s) :
    HasFDerivAt (fun y : Tensor ℝ s => Tensor.sum (y * y))
      ((2 : ℝ) • (Tensor.sumCLM s).comp (Op.slopeCLM fun c => x c)) x := by
  op_deriv

/-- The sum has the derivative `Tensor.sumCLM s` and the backward pass `full s δ`. -/
theorem sum_cert (x : Tensor ℝ s) (δ : ℝ) :
    HasFDerivAt (fun y : Tensor ℝ s => Tensor.sum y) (Tensor.sumCLM s) x ∧
      Tensor.full s δ = (fderiv ℝ (fun y : Tensor ℝ s => Tensor.sum y) x).adjoint δ := by
  op_deriv

/-- `sin` then `exp` at every entry, written with `∘`, has a derivative and a pullback. -/
theorem exp_comp_sin (x : Tensor ℝ s) :
    ∃ D b, HasPullbackAt (Tensor.map Real.exp ∘ Tensor.map Real.sin) D b x :=
  ⟨_, _, by op_deriv⟩

/-- With the image `g` of `exp` bound by a `let` and introduced with `intro`, the backward pass
`δ * g` of `exp` at every entry is the adjoint of its derivative. -/
theorem let_exp_backward (x δ : Tensor ℝ s) :
    let g := Tensor.map Real.exp x
    δ * g = (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map Real.exp y) x).adjoint δ := by
  intro g
  op_deriv

/-- With the table entry of `Op.hypot` bound by a `let` and introduced with `intro`, its stored
backward pass is the adjoint of the derivative of its forward map where no entry of
`a * a + b * b` is zero. -/
theorem let_hypot_backward (a b δ : Tensor ℝ s) (h : ∀ c, (a * a + b * b) c ≠ 0) :
    let r := (Op.hypot (s := s)).rule? (α := ℝ) |>.get rfl
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (r.forward (p.fst, p.snd, ())).1) (WithLp.toLp 2 (a, b)) ∧
      WithLp.toLp 2 ((r.backward (a, b, ()) (δ, ())).1, (r.backward (a, b, ()) (δ, ())).2.1) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (r.forward (p.fst, p.snd, ())).1) (WithLp.toLp 2 (a, b))).adjoint δ := by
  intro r
  op_deriv

end Forms

section Inputs

/-- `lerp a b w`, then `sin` at every entry, then the sum, jointly in `a`, `b` and `w`. -/
theorem lerpSin_backward (a b w : Tensor ℝ s) (δ : ℝ) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s × Tensor ℝ s)) =>
        Tensor.sum (Tensor.map Real.sin (Tensor.lerp p.fst p.snd.fst p.snd.snd)))
        (WithLp.toLp 2 (a, WithLp.toLp 2 (b, w))) ∧
      WithLp.toLp 2 (Tensor.full s δ * Tensor.map Real.cos (Tensor.lerp a b w) *
          Tensor.map (fun t => 1 - t) w,
          WithLp.toLp 2 (Tensor.full s δ * Tensor.map Real.cos (Tensor.lerp a b w) * w,
            Tensor.full s δ * Tensor.map Real.cos (Tensor.lerp a b w) * (b - a))) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s × Tensor ℝ s)) =>
          Tensor.sum (Tensor.map Real.sin (Tensor.lerp p.fst p.snd.fst p.snd.snd)))
          (WithLp.toLp 2 (a, WithLp.toLp 2 (b, w)))).adjoint δ := by
  op_deriv

/-- `a b + c d - e` in five inputs has the backward pass `(δ b, δ a, δ d, δ c, -δ)`. -/
theorem five_backward (a b c d e δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s ×
          WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s × Tensor ℝ s)))) =>
        p.fst * p.snd.fst + p.snd.snd.fst * p.snd.snd.snd.fst - p.snd.snd.snd.snd)
        (WithLp.toLp 2 (a, WithLp.toLp 2 (b, WithLp.toLp 2 (c, WithLp.toLp 2 (d, e))))) ∧
      WithLp.toLp 2 (δ * b, WithLp.toLp 2 (δ * a, WithLp.toLp 2 (δ * d,
          WithLp.toLp 2 (δ * c, -δ)))) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s ×
            WithLp 2 (Tensor ℝ s × WithLp 2 (Tensor ℝ s × Tensor ℝ s)))) =>
          p.fst * p.snd.fst + p.snd.snd.fst * p.snd.snd.snd.fst - p.snd.snd.snd.snd)
          (WithLp.toLp 2 (a, WithLp.toLp 2 (b, WithLp.toLp 2 (c,
            WithLp.toLp 2 (d, e)))))).adjoint δ := by
  op_deriv

/-- The pair of `exp` at every entry and the sum has the backward pass
`δ.fst * exp x + full s δ.snd`. -/
theorem expSum_backward (x : Tensor ℝ s) (δ : WithLp 2 (Tensor ℝ s × ℝ)) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s =>
        WithLp.toLp 2 (Tensor.map Real.exp y, Tensor.sum y)) x ∧
      δ.fst * Tensor.map Real.exp x + Tensor.full s δ.snd =
        (fderiv ℝ (fun y : Tensor ℝ s =>
          WithLp.toLp 2 (Tensor.map Real.exp y, Tensor.sum y)) x).adjoint δ := by
  op_deriv

end Inputs

section Real

/-- `sin` on `ℝ` has the backward pass `cos a * δ`. -/
theorem sin_backward (a δ : ℝ) :
    DifferentiableAt ℝ (fun y : ℝ => Real.sin y) a ∧
      Real.cos a * δ = (fderiv ℝ (fun y : ℝ => Real.sin y) a).adjoint δ := by
  op_deriv

/-- The real square `y * y` has the backward pass `2 a δ`. -/
theorem mul_self_backward (a δ : ℝ) :
    DifferentiableAt ℝ (fun y : ℝ => y * y) a ∧
      2 * a * δ = (fderiv ℝ (fun y : ℝ => y * y) a).adjoint δ := by
  op_deriv

/-- The sum of the entries times the sum of their squares has the backward pass
`full (δ ∑ x²) + full (δ ∑ x) (x + x)`. -/
theorem sum_mul_sum_backward (x : Tensor ℝ s) (δ : ℝ) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.sum y * Tensor.sum (y * y)) x ∧
      Tensor.full s (δ * Tensor.sum (x * x)) + Tensor.full s (δ * Tensor.sum x) * (x + x) =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.sum y * Tensor.sum (y * y)) x).adjoint δ := by
  op_deriv

/-- The squared error `∑ (y - t)²` divided by a real `m` has the backward pass
`full (δ / m) ((x - t) + (x - t))`, also at `m = 0`. -/
theorem squaredError_div_backward (x t : Tensor ℝ s) (δ m : ℝ) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.sum ((y - t) * (y - t)) / m) x ∧
      Tensor.full s (δ / m) * ((x - t) + (x - t)) =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.sum ((y - t) * (y - t)) / m) x).adjoint δ := by
  op_deriv

/-- A real `m` times the squared error `∑ (y - t)²` has the backward pass
`full (m δ) ((x - t) + (x - t))`. -/
theorem mul_squaredError_backward (x t : Tensor ℝ s) (δ m : ℝ) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => m * Tensor.sum ((y - t) * (y - t))) x ∧
      Tensor.full s (m * δ) * ((x - t) + (x - t)) =
        (fderiv ℝ (fun y : Tensor ℝ s => m * Tensor.sum ((y - t) * (y - t))) x).adjoint δ := by
  op_deriv

/-- The sum of the entries divided by the sum of their squares has a derivative and a pullback
where the denominator is not zero. -/
theorem sum_div_hasPullbackAt (x : Tensor ℝ s) (hx : Tensor.sum (x * x) ≠ 0) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s => Tensor.sum y / Tensor.sum (y * y)) D b x :=
  ⟨_, _, by op_deriv⟩

/-- `y / (y y + 1)` at every entry has a derivative and a pullback at every point, since its
denominator is positive. -/
theorem div_mul_self_add_one_hasPullbackAt (x : Tensor ℝ s) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s => y / (y * y + Tensor.full s 1)) D b x :=
  ⟨_, _, by op_deriv⟩

/-- A real variable times a constant tensor `x` has the backward pass `∑ δ x`. -/
theorem smul_const_backward (x : Tensor ℝ s) (a : ℝ) (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : ℝ => y • x) a ∧
      Tensor.sum (δ * x) = (fderiv ℝ (fun y : ℝ => y • x) a).adjoint δ := by
  op_deriv [Tensor.sum_mul_eq_inner, real_inner_comm]

/-- The sum of the entries times the entrywise square has a derivative and a pullback. -/
theorem sum_smul_hasPullbackAt (x : Tensor ℝ s) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s => Tensor.sum y • (y * y)) D b x :=
  ⟨_, _, by op_deriv⟩

/-- `exp` of the sum of the squares of the entries has a derivative and a pullback. -/
theorem exp_sum_hasPullbackAt (x : Tensor ℝ s) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s => Real.exp (Tensor.sum (y * y))) D b x :=
  ⟨_, _, by op_deriv⟩

/-- The sum of the squares of the entries raised to the real power `3 / 2` is differentiable at
every point. -/
theorem sum_mul_self_rpow_differentiableAt (x : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.sum (y * y) ^ (3 / 2 : ℝ)) x := by
  op_deriv

end Real

section Functions

variable {n d : Nat}

/-- Softmax along `dim` with no mask has a derivative and a pullback. -/
theorem softmax_hasPullbackAt (x : Tensor ℝ s) (dim : Fin s.rank) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s => Tensor.softmax y dim) D b x :=
  ⟨_, _, by op_deriv⟩

/-- Softmax along `dim` of the entrywise square has the backward pass
`(x + x) * softmaxBackward (x * x) δ`. -/
theorem softmax_mul_self_backward (x δ : Tensor ℝ s) (dim : Fin s.rank) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.softmax (y * y) dim) x ∧
      (x + x) * Tensor.softmaxBackward (x * x) δ dim =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.softmax (y * y) dim) x).adjoint δ := by
  op_deriv

/-- `Tensor.exp` has the backward pass `δ * map exp x`. -/
theorem exp_backward (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.exp y) x ∧
      δ * Tensor.map Real.exp x = (fderiv ℝ (fun y : Tensor ℝ s => Tensor.exp y) x).adjoint δ := by
  op_deriv

/-- The value of a scalar tensor and an entry of a tensor have derivatives and pullbacks. -/
theorem item_entry_hasPullbackAt (x : Tensor ℝ []) (y : Tensor ℝ s) (c : s.Coord) :
    (∃ D b, HasPullbackAt (fun z : Tensor ℝ [] => Tensor.item z) D b x) ∧
      ∃ D b, HasPullbackAt (fun z : Tensor ℝ s => z c) D b y :=
  ⟨⟨_, _, by op_deriv⟩, ⟨_, _, by op_deriv⟩⟩

/-- `Tensor.scale` of the entrywise square has a derivative and a pullback. -/
theorem scale_hasPullbackAt (x : Tensor ℝ s) (c : ℝ) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s => Tensor.scale (y * y) c) D b x :=
  ⟨_, _, by op_deriv⟩

/-- `Tensor.scale` of a tensor by a real, both varying, has the backward pass
`(Tensor.scale δ c, ⟪x, δ⟫)`. -/
theorem scale_pair_backward (x δ : Tensor ℝ s) (c : ℝ) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × ℝ) => Tensor.scale p.fst p.snd)
        (WithLp.toLp 2 (x, c)) ∧
      WithLp.toLp 2 (Tensor.scale δ c, inner ℝ x δ) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × ℝ) => Tensor.scale p.fst p.snd)
          (WithLp.toLp 2 (x, c))).adjoint δ := by
  op_deriv

/-- The entries scaled by their sum have a derivative and a pullback. -/
theorem scale_sum_hasPullbackAt (x : Tensor ℝ s) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s => Tensor.scale y (Tensor.sum y)) D b x :=
  ⟨_, _, by op_deriv⟩

/-- The common-batch matrix product of `y` and `y + y` has a derivative and a pullback. -/
theorem matmulCommonBatch_hasPullbackAt {batch : Shape} (x : Tensor ℝ (batch ++ [n, n])) :
    ∃ D b, HasPullbackAt (fun y => Tensor.matmulCommonBatch y (y + y)) D b x :=
  ⟨_, _, by op_deriv⟩

/-- The contraction of `y` with `y * y` has a derivative and a pullback. -/
theorem contract_hasPullbackAt (dims batchDims : List (Fin s.rank × Fin s.rank))
    (h : Shape.ContractValid s s dims batchDims) (x : Tensor ℝ s) :
    ∃ D b, HasPullbackAt (fun y => Tensor.contract y (y * y) dims batchDims _ h rfl) D b x :=
  ⟨_, _, by op_deriv⟩

/-- `zipWith φ y y` has a derivative and a pullback where `φ` has the derivative
`(u, v) ↦ φ₀ c u + φ₁ c v` at `(x c, x c)`. -/
theorem zipWith_self_hasPullbackAt {φ : ℝ → ℝ → ℝ} {φ₀ φ₁ : s.Coord → ℝ} (x : Tensor ℝ s)
    (hφ : ∀ c, HasFDerivAt (fun q : ℝ × ℝ => φ q.1 q.2)
      (φ₀ c • ContinuousLinearMap.fst ℝ ℝ ℝ + φ₁ c • ContinuousLinearMap.snd ℝ ℝ ℝ) (x c, x c)) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s => Tensor.zipWith φ y y) D b x :=
  ⟨_, _, by op_deriv⟩

/-- `zipWith φ` of the two components of a pair has the backward pass
`(slopeCLM φ₀ δ, slopeCLM φ₁ δ)` where `φ` has the derivative `(u, v) ↦ φ₀ c u + φ₁ c v` at
`(a c, b c)`, as for `fmod` and `remainder`. -/
theorem zipWith_pair_backward {φ : ℝ → ℝ → ℝ} {φ₀ φ₁ : s.Coord → ℝ} (a b δ : Tensor ℝ s)
    (hφ : ∀ c, HasFDerivAt (fun q : ℝ × ℝ => φ q.1 q.2)
      (φ₀ c • ContinuousLinearMap.fst ℝ ℝ ℝ + φ₁ c • ContinuousLinearMap.snd ℝ ℝ ℝ) (a c, b c)) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        Tensor.zipWith φ p.fst p.snd) (WithLp.toLp 2 (a, b)) ∧
      WithLp.toLp 2 (Op.slopeCLM φ₀ δ, Op.slopeCLM φ₁ δ) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) => Tensor.zipWith φ p.fst p.snd)
          (WithLp.toLp 2 (a, b))).adjoint δ := by
  op_deriv

end Functions

section Dense

variable {n h k : Nat}

/-- The sum of `tanh` of a linear layer with weights `W` and bias `b` at the input `v`. -/
noncomputable def dense (W : Tensor ℝ [h, n]) (b : Tensor ℝ [h]) (v : Tensor ℝ [n]) : ℝ :=
  Tensor.sum (Tensor.map Real.tanh (Linear.forward { weights := W, bias := b } v))

/-- The gradients of `dense` in the weights and the bias at the output cotangent `δ`: the cotangent
`δ (1 - tanh² z)` of the layer's output `z`, then `Linear.backward`. -/
noncomputable def denseBackward (W : Tensor ℝ [h, n]) (b : Tensor ℝ [h]) (v : Tensor ℝ [n])
    (δ : ℝ) : WithLp 2 (Tensor ℝ [h, n] × Tensor ℝ [h]) :=
  let z := Linear.forward { weights := W, bias := b } v
  let g := Tensor.full [h] δ * Tensor.map (fun t => 1 - Real.tanh t ^ 2) z
  let grads := Linear.backward { weights := W, bias := b } v g
  WithLp.toLp 2 (grads.parameters.weights, grads.parameters.bias)

/-- `denseBackward` is the adjoint of the derivative of `dense` in the weights and the bias. -/
theorem dense_backward (W : Tensor ℝ [h, n]) (b : Tensor ℝ [h]) (v : Tensor ℝ [n]) (δ : ℝ) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ [h, n] × Tensor ℝ [h]) => dense p.fst p.snd v)
        (WithLp.toLp 2 (W, b)) ∧
      denseBackward W b v δ =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ [h, n] × Tensor ℝ [h]) => dense p.fst p.snd v)
          (WithLp.toLp 2 (W, b))).adjoint δ := by
  op_deriv [dense, denseBackward, Linear.backward]

/-- `dense` has the backward pass `(weightsDeriv v g, g, inputDeriv W g)` jointly in the weights,
the bias and the input, for `g = δ (1 - tanh² z)` at the layer's output `z`. -/
theorem dense_backward_input (W : Tensor ℝ [h, n]) (b : Tensor ℝ [h]) (v : Tensor ℝ [n])
    (δ : ℝ) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ [h, n] ×
          WithLp 2 (Tensor ℝ [h] × Tensor ℝ [n])) => dense p.fst p.snd.fst p.snd.snd)
        (WithLp.toLp 2 (W, WithLp.toLp 2 (b, v))) ∧
      WithLp.toLp 2 (Linear.weightsDeriv v (Tensor.full [h] δ *
          Tensor.map (fun t => 1 - Real.tanh t ^ 2) (Linear.forward { weights := W, bias := b } v)),
        WithLp.toLp 2 (Tensor.full [h] δ * Tensor.map (fun t => 1 - Real.tanh t ^ 2)
            (Linear.forward { weights := W, bias := b } v),
          Linear.inputDeriv W (Tensor.full [h] δ * Tensor.map (fun t => 1 - Real.tanh t ^ 2)
            (Linear.forward { weights := W, bias := b } v)))) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ [h, n] ×
            WithLp 2 (Tensor ℝ [h] × Tensor ℝ [n])) => dense p.fst p.snd.fst p.snd.snd)
          (WithLp.toLp 2 (W, WithLp.toLp 2 (b, v)))).adjoint δ := by
  op_deriv [dense]

/-- The residual block `y + tanh (W y + b)` has the backward pass
`δ + inputDeriv W (δ (1 - tanh² (W v + b)))` in its input. -/
theorem residual_backward (W : Tensor ℝ [n, n]) (b : Tensor ℝ [n]) (v δ : Tensor ℝ [n]) :
    DifferentiableAt ℝ (fun y : Tensor ℝ [n] =>
        y + Tensor.map Real.tanh (Linear.forward { weights := W, bias := b } y)) v ∧
      δ + Linear.inputDeriv W (δ * Tensor.map (fun t => 1 - Real.tanh t ^ 2)
          (Linear.forward { weights := W, bias := b } v)) =
        (fderiv ℝ (fun y : Tensor ℝ [n] =>
          y + Tensor.map Real.tanh (Linear.forward { weights := W, bias := b } y))
          v).adjoint δ := by
  op_deriv

/-- The sum of a two-layer network with `tanh` after each linear layer. -/
noncomputable def mlp (W₁ : Tensor ℝ [h, n]) (b₁ : Tensor ℝ [h]) (W₂ : Tensor ℝ [k, h])
    (b₂ : Tensor ℝ [k]) (v : Tensor ℝ [n]) : ℝ :=
  Tensor.sum (Tensor.map Real.tanh (Linear.forward { weights := W₂, bias := b₂ }
    (Tensor.map Real.tanh (Linear.forward { weights := W₁, bias := b₁ } v))))

/-- The gradients of `mlp` in its four parameters at the output cotangent `δ`, with the incoming
cotangent on the left of each product, as PyTorch writes them. -/
noncomputable def mlpBackward (W₁ : Tensor ℝ [h, n]) (b₁ : Tensor ℝ [h]) (W₂ : Tensor ℝ [k, h])
    (b₂ : Tensor ℝ [k]) (v : Tensor ℝ [n]) (δ : ℝ) :
    WithLp 2 (Tensor ℝ [h, n] × WithLp 2 (Tensor ℝ [h] ×
      WithLp 2 (Tensor ℝ [k, h] × Tensor ℝ [k]))) :=
  let z₁ := Linear.forward { weights := W₁, bias := b₁ } v
  let a₁ := Tensor.map Real.tanh z₁
  let z₂ := Linear.forward { weights := W₂, bias := b₂ } a₁
  let g₂ := Tensor.full [k] δ * Tensor.map (fun t => 1 - Real.tanh t ^ 2) z₂
  let g₁ := Linear.inputDeriv W₂ g₂ * Tensor.map (fun t => 1 - Real.tanh t ^ 2) z₁
  WithLp.toLp 2 (Linear.weightsDeriv v g₁, WithLp.toLp 2 (g₁,
    WithLp.toLp 2 (Linear.weightsDeriv a₁ g₂, g₂)))

/-- The gradients of `mlp` as in `mlpBackward`, with the slope on the left of each product. -/
noncomputable def mlpBackwardSlopeFirst (W₁ : Tensor ℝ [h, n]) (b₁ : Tensor ℝ [h])
    (W₂ : Tensor ℝ [k, h]) (b₂ : Tensor ℝ [k]) (v : Tensor ℝ [n]) (δ : ℝ) :
    WithLp 2 (Tensor ℝ [h, n] × WithLp 2 (Tensor ℝ [h] ×
      WithLp 2 (Tensor ℝ [k, h] × Tensor ℝ [k]))) :=
  let z₁ := Linear.forward { weights := W₁, bias := b₁ } v
  let a₁ := Tensor.map Real.tanh z₁
  let z₂ := Linear.forward { weights := W₂, bias := b₂ } a₁
  let g₂ := Tensor.map (fun t => 1 - Real.tanh t ^ 2) z₂ * Tensor.full [k] δ
  let g₁ := Tensor.map (fun t => 1 - Real.tanh t ^ 2) z₁ * Linear.inputDeriv W₂ g₂
  WithLp.toLp 2 (Linear.weightsDeriv v g₁, WithLp.toLp 2 (g₁,
    WithLp.toLp 2 (Linear.weightsDeriv a₁ g₂, g₂)))

/-- `mlpBackward` is the adjoint of the derivative of `mlp` in its four parameters. -/
theorem mlp_backward (W₁ : Tensor ℝ [h, n]) (b₁ : Tensor ℝ [h]) (W₂ : Tensor ℝ [k, h])
    (b₂ : Tensor ℝ [k]) (v : Tensor ℝ [n]) (δ : ℝ) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ [h, n] × WithLp 2 (Tensor ℝ [h] ×
          WithLp 2 (Tensor ℝ [k, h] × Tensor ℝ [k]))) =>
        mlp p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd v)
        (WithLp.toLp 2 (W₁, WithLp.toLp 2 (b₁, WithLp.toLp 2 (W₂, b₂)))) ∧
      mlpBackward W₁ b₁ W₂ b₂ v δ =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ [h, n] × WithLp 2 (Tensor ℝ [h] ×
            WithLp 2 (Tensor ℝ [k, h] × Tensor ℝ [k]))) =>
          mlp p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd v)
          (WithLp.toLp 2 (W₁, WithLp.toLp 2 (b₁, WithLp.toLp 2 (W₂, b₂))))).adjoint δ := by
  op_deriv [mlp, mlpBackward]

/-- `mlpBackwardSlopeFirst` is the adjoint of the derivative of `mlp` in its four parameters. -/
theorem mlp_backward_slopeFirst (W₁ : Tensor ℝ [h, n]) (b₁ : Tensor ℝ [h])
    (W₂ : Tensor ℝ [k, h]) (b₂ : Tensor ℝ [k]) (v : Tensor ℝ [n]) (δ : ℝ) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ [h, n] × WithLp 2 (Tensor ℝ [h] ×
          WithLp 2 (Tensor ℝ [k, h] × Tensor ℝ [k]))) =>
        mlp p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd v)
        (WithLp.toLp 2 (W₁, WithLp.toLp 2 (b₁, WithLp.toLp 2 (W₂, b₂)))) ∧
      mlpBackwardSlopeFirst W₁ b₁ W₂ b₂ v δ =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ [h, n] × WithLp 2 (Tensor ℝ [h] ×
            WithLp 2 (Tensor ℝ [k, h] × Tensor ℝ [k]))) =>
          mlp p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd v)
          (WithLp.toLp 2 (W₁, WithLp.toLp 2 (b₁, WithLp.toLp 2 (W₂, b₂))))).adjoint δ := by
  op_deriv [mlp, mlpBackwardSlopeFirst]

end Dense

section Attention

variable {n m d e : Nat}

/-- Single-head attention `softmax(Q Kᵀ) V` with the softmax along the keys and blocked entries
where `mask` is false, from the queries `q`, the transposed keys `kt` and the values `v`. -/
noncomputable def attend (mask : Tensor Bool [n, m]) (q : Tensor ℝ [n, d]) (kt : Tensor ℝ [d, m])
    (v : Tensor ℝ [m, e]) : Tensor ℝ [n, e] :=
  Tensor.matmul (Tensor.softmax (Tensor.matmul q kt) ⟨1, Nat.one_lt_two⟩ (some mask)) v

/-- The backward pass of `attend` at the output cotangent `δ`, with the library's backward maps of
the matrix product and of the masked softmax. -/
noncomputable def attendBackward (mask : Tensor Bool [n, m]) (q : Tensor ℝ [n, d])
    (kt : Tensor ℝ [d, m]) (v : Tensor ℝ [m, e]) (δ : Tensor ℝ [n, e]) :
    WithLp 2 (Tensor ℝ [n, d] × WithLp 2 (Tensor ℝ [d, m] × Tensor ℝ [m, e])) :=
  let scores := Tensor.matmul q kt
  let weights := Tensor.softmax scores ⟨1, Nat.one_lt_two⟩ (some mask)
  let dOut := Tensor.matmulBackward (Shape.CanBroadcastTo.refl .scalar)
    (Shape.CanBroadcastTo.refl .scalar) weights v δ
  let dScores := Tensor.maskedSoftmaxBackward scores mask dOut.1 ⟨1, Nat.one_lt_two⟩
  let dIn := Tensor.matmulBackward (Shape.CanBroadcastTo.refl .scalar)
    (Shape.CanBroadcastTo.refl .scalar) q kt dScores
  WithLp.toLp 2 (dIn.1, WithLp.toLp 2 (dIn.2, dOut.2))

/-- `attendBackward` is the adjoint of the derivative of `attend` jointly in `Q`, `Kᵀ` and `V`. -/
theorem attend_backward (mask : Tensor Bool [n, m]) (q : Tensor ℝ [n, d]) (kt : Tensor ℝ [d, m])
    (v : Tensor ℝ [m, e]) (δ : Tensor ℝ [n, e]) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ [n, d] ×
          WithLp 2 (Tensor ℝ [d, m] × Tensor ℝ [m, e])) =>
        attend mask p.fst p.snd.fst p.snd.snd) (WithLp.toLp 2 (q, WithLp.toLp 2 (kt, v))) ∧
      attendBackward mask q kt v δ =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ [n, d] ×
            WithLp 2 (Tensor ℝ [d, m] × Tensor ℝ [m, e])) =>
          attend mask p.fst p.snd.fst p.snd.snd)
          (WithLp.toLp 2 (q, WithLp.toLp 2 (kt, v)))).adjoint δ := by
  op_deriv [attend, attendBackward]

/-- `attend` with the scores scaled by a real `c`. -/
noncomputable def scaledAttend (c : ℝ) (mask : Tensor Bool [n, m]) (q : Tensor ℝ [n, d])
    (kt : Tensor ℝ [d, m]) (v : Tensor ℝ [m, e]) : Tensor ℝ [n, e] :=
  Tensor.matmul (Tensor.softmax (c • Tensor.matmul q kt) ⟨1, Nat.one_lt_two⟩ (some mask)) v

/-- The backward pass of `scaledAttend` at the output cotangent `δ`, with `c` applied to the
cotangent of the scores after the masked softmax. -/
noncomputable def scaledAttendBackward (c : ℝ) (mask : Tensor Bool [n, m]) (q : Tensor ℝ [n, d])
    (kt : Tensor ℝ [d, m]) (v : Tensor ℝ [m, e]) (δ : Tensor ℝ [n, e]) :
    WithLp 2 (Tensor ℝ [n, d] × WithLp 2 (Tensor ℝ [d, m] × Tensor ℝ [m, e])) :=
  let scores := c • Tensor.matmul q kt
  let weights := Tensor.softmax scores ⟨1, Nat.one_lt_two⟩ (some mask)
  let dOut := Tensor.matmulBackward (Shape.CanBroadcastTo.refl .scalar)
    (Shape.CanBroadcastTo.refl .scalar) weights v δ
  let dScores := c • Tensor.maskedSoftmaxBackward scores mask dOut.1 ⟨1, Nat.one_lt_two⟩
  let dIn := Tensor.matmulBackward (Shape.CanBroadcastTo.refl .scalar)
    (Shape.CanBroadcastTo.refl .scalar) q kt dScores
  WithLp.toLp 2 (dIn.1, WithLp.toLp 2 (dIn.2, dOut.2))

/-- `scaledAttendBackward` is the adjoint of the derivative of `scaledAttend` jointly in `Q`, `Kᵀ`
and `V`. -/
theorem scaledAttend_backward (c : ℝ) (mask : Tensor Bool [n, m]) (q : Tensor ℝ [n, d])
    (kt : Tensor ℝ [d, m]) (v : Tensor ℝ [m, e]) (δ : Tensor ℝ [n, e]) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ [n, d] ×
          WithLp 2 (Tensor ℝ [d, m] × Tensor ℝ [m, e])) =>
        scaledAttend c mask p.fst p.snd.fst p.snd.snd)
        (WithLp.toLp 2 (q, WithLp.toLp 2 (kt, v))) ∧
      scaledAttendBackward c mask q kt v δ =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ [n, d] ×
            WithLp 2 (Tensor ℝ [d, m] × Tensor ℝ [m, e])) =>
          scaledAttend c mask p.fst p.snd.fst p.snd.snd)
          (WithLp.toLp 2 (q, WithLp.toLp 2 (kt, v)))).adjoint δ := by
  op_deriv [scaledAttend, scaledAttendBackward]

end Attention

section Loss

/-- The cross-entropy `-∑ t log softmax x` of the logits `x` against a target `t` has the backward
pass `-logSoftmaxBackward (logSoftmax x) (δ t)`. -/
theorem crossEntropy_backward (x t : Tensor ℝ s) (dim : Fin s.rank) (δ : ℝ) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => -Tensor.sum (t * Tensor.logSoftmax y dim)) x ∧
      -Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) (Tensor.full s δ * t) dim =
        (fderiv ℝ (fun y : Tensor ℝ s =>
          -Tensor.sum (t * Tensor.logSoftmax y dim)) x).adjoint δ := by
  op_deriv

/-- A layer normalization `(y - μ) / √(σ² + ε)` written by hand, with the mean `μ` and the variance
`σ²` taken as `m⁻¹` times sums, has a derivative and a pullback where the argument of the square
root and the denominator have no zero entry at `x`. -/
theorem layerNorm_hasPullbackAt (x : Tensor ℝ s) (ε m : ℝ)
    (hsq : ∀ c, (Tensor.full s (m⁻¹ • Tensor.sum ((x - Tensor.full s (m⁻¹ • Tensor.sum x)) *
      (x - Tensor.full s (m⁻¹ • Tensor.sum x)))) + Tensor.full s ε) c ≠ 0)
    (hden : ∀ c, Tensor.map Real.sqrt (Tensor.full s (m⁻¹ • Tensor.sum
      ((x - Tensor.full s (m⁻¹ • Tensor.sum x)) * (x - Tensor.full s (m⁻¹ • Tensor.sum x)))) +
      Tensor.full s ε) c ≠ 0) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s =>
      (y - Tensor.full s (m⁻¹ • Tensor.sum y)) /
        Tensor.map Real.sqrt (Tensor.full s (m⁻¹ • Tensor.sum
          ((y - Tensor.full s (m⁻¹ • Tensor.sum y)) * (y - Tensor.full s (m⁻¹ • Tensor.sum y)))) +
          Tensor.full s ε)) D b x :=
  ⟨_, _, by op_deriv⟩

/-- The RMS normalization `y / √(m⁻¹ ∑ y² + ε)` has a derivative and a pullback for positive `m`
and `ε`. -/
theorem rmsNorm_hasPullbackAt (x : Tensor ℝ s) (ε m : ℝ) (hm : 0 < m) (hε : 0 < ε) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s =>
      y / Tensor.full s (Real.sqrt (m⁻¹ * Tensor.sum (y * y) + ε))) D b x :=
  ⟨_, _, by op_deriv⟩

/-- The `Finset` sum over `i < 3` of the entries scaled by `i` has the backward pass
`∑ i < 3, δ i`, with `i` as a constant tensor. -/
theorem finsetSum_backward (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s =>
        ∑ i ∈ Finset.range 3, Tensor.map (fun a => a * (i : ℝ)) y) x ∧
      ∑ i ∈ Finset.range 3, δ * Tensor.full s (i : ℝ) =
        (fderiv ℝ (fun y : Tensor ℝ s =>
          ∑ i ∈ Finset.range 3, Tensor.map (fun a => a * (i : ℝ)) y) x).adjoint δ := by
  op_deriv

/-- The sum over all coordinates of the squared entries has the backward pass `2 δ x`. -/
theorem coordSum_sq_backward (x : Tensor ℝ s) (δ : ℝ) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => ∑ c : s.Coord, y c ^ 2) x ∧
      Tensor.full s (2 * δ) * x =
        (fderiv ℝ (fun y : Tensor ℝ s => ∑ c : s.Coord, y c ^ 2) x).adjoint δ := by
  op_deriv

end Loss

section Order

/-- The weights `w` applied to the entries of `x` sorted in increasing order along `dim`, and
summed. -/
noncomputable def orderedSum (w : Tensor ℝ s) (dim : Fin s.rank) (x : Tensor ℝ s) : ℝ :=
  Tensor.sum (w * (Tensor.sort x dim false).1)

/-- At a point whose fibers along `dim` have distinct entries, the ordered weighted sum has the
backward pass that sends the weights back to the positions of the sorted entries. -/
theorem orderedSum_backward (w : Tensor ℝ s) (dim : Fin s.rank) (x : Tensor ℝ s) (δ : ℝ)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) ≠ x (Shape.Coord.update c dim j)) :
    DifferentiableAt ℝ (fun y => orderedSum w dim y) x ∧
      Tensor.sortBackward x (Tensor.full s δ * w) dim false =
        (fderiv ℝ (fun y => orderedSum w dim y) x).adjoint δ := by
  op_deriv [orderedSum]

/-- The sorted entries plus the running maximum along `dim` are differentiable at a point where
the fibers along `dim` have distinct entries. -/
theorem sortCummax_differentiableAt (dim : Fin s.rank) (x : Tensor ℝ s)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) ≠ x (Shape.Coord.update c dim j)) :
    ∃ D b, HasPullbackAt (fun y : Tensor ℝ s =>
      (Tensor.sort y dim false).1 + (Tensor.cummax y dim).1) D b x := by
  have _ : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i < j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        ∃ l < j, x (Shape.Coord.update c dim j) < x (Shape.Coord.update c dim l) :=
    fun c i j hij heq => absurd heq (hx c i j hij.ne)
  exact ⟨_, _, by op_deriv⟩

end Order

section Linearity

variable {n m d : Nat}

/-- The `op_deriv_simps` lemmas move a sum and a scalar of the cotangent out of the backward maps
of the masked softmax and of the matrix product. -/
theorem matmulBackward_maskedSoftmaxBackward_linear (k : ℝ) (mask : Tensor Bool [n, m])
    (q : Tensor ℝ [n, d]) (kt : Tensor ℝ [d, m]) (z δ δ' : Tensor ℝ [n, m]) :
    (matmulBackward (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar) q kt
        (k • Tensor.maskedSoftmaxBackward z mask (δ + δ') ⟨1, Nat.one_lt_two⟩)).2 =
      k • ((matmulBackward (Shape.CanBroadcastTo.refl .scalar)
          (Shape.CanBroadcastTo.refl .scalar) q kt
          (Tensor.maskedSoftmaxBackward z mask δ ⟨1, Nat.one_lt_two⟩)).2 +
        (matmulBackward (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
          q kt (Tensor.maskedSoftmaxBackward z mask δ' ⟨1, Nat.one_lt_two⟩)).2) := by
  simp only [Op.maskedSoftmaxBackward_add, Tensor.matmulBackward_refl_snd_smul,
    Tensor.matmulBackward_refl_snd_add]

/-- The `op_deriv_simps` lemmas move a scalar and a negation of the cotangent out of the backward
map of a batched matrix product. -/
theorem matmulBackward_batch_linear {batch : Shape} (a : Tensor ℝ (batch ++ [m, n]))
    (b : Tensor ℝ (batch ++ [n, d])) (k : ℝ) (δ δ' : Tensor ℝ (batch ++ [m, d])) :
    (matmulBackward (Shape.CanBroadcastTo.refl batch) (Shape.CanBroadcastTo.refl batch) a b
        (k • δ + -δ')).1 =
      k • (matmulBackward (Shape.CanBroadcastTo.refl batch) (Shape.CanBroadcastTo.refl batch)
          a b δ).1 +
        -(matmulBackward (Shape.CanBroadcastTo.refl batch) (Shape.CanBroadcastTo.refl batch)
          a b δ').1 := by
  simp only [Tensor.matmulBackward_fst_add, Tensor.matmulBackward_fst_smul,
    Tensor.matmulBackward_fst_neg]

/-- The `op_deriv_simps` lemmas move a scalar and a negation of the cotangent out of the input and
weight gradients of a linear layer and out of the log-softmax pullback. -/
theorem linear_logSoftmaxBackward_linear (W : Tensor ℝ [m, n]) (v : Tensor ℝ [n]) (k : ℝ)
    (δ δ' : Tensor ℝ [m]) (x ε ε' : Tensor ℝ s) (dim : Fin s.rank) :
    Linear.inputDeriv W (k • δ + -δ') = k • Linear.inputDeriv W δ + -Linear.inputDeriv W δ' ∧
      Linear.weightsDeriv v (k • δ + -δ') =
        k • Linear.weightsDeriv v δ + -Linear.weightsDeriv v δ' ∧
      Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) (k • ε + -ε') dim =
        k • Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) ε dim +
          -Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) ε' dim := by
  simp only [Linear.inputDeriv_add, Linear.inputDeriv_smul, Linear.inputDeriv_neg,
    Linear.weightsDeriv_add, Linear.weightsDeriv_smul, Linear.weightsDeriv_neg,
    Op.logSoftmaxBackward_add, Op.logSoftmaxBackward_smul, Op.logSoftmaxBackward_neg, and_self]

/-- The `op_deriv_simps` lemma `Tensor.Rep.sum_apply` takes an entry of a finite sum of
tensors. -/
theorem finsetSum_apply (f : ℕ → Tensor ℝ s) (c : s.Coord) :
    (∑ i ∈ Finset.range 3, f i) c = ∑ i ∈ Finset.range 3, f i c := by
  simp only [Tensor.Rep.sum_apply]

end Linearity

section Reject

/-- `op_deriv` proves the backward pass `δ * exp x` of `exp` and rejects `δ * x`. -/
example (x δ : Tensor ℝ s) :
    δ * Tensor.map Real.exp x =
      (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map Real.exp y) x).adjoint δ := by
  fail_if_success have : δ * x =
      (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map Real.exp y) x).adjoint δ := by op_deriv
  op_deriv

/-- Without the hypothesis that no entry is zero, `op_deriv` proves neither the backward passes
of the square root and of `relu` nor that `a ^ (1 / 2)` at every entry is differentiable, and none
of the three has a derivative at zero. -/
example : True := by
  fail_if_success
    have : ∀ (s : Shape) (x δ : Tensor ℝ s),
        δ * Tensor.map (fun a => 1 / (2 * Real.sqrt a)) x =
          (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map Real.sqrt y) x).adjoint δ := by
      intro s x δ
      op_deriv
  fail_if_success
    have : ∀ (s : Shape) (x δ : Tensor ℝ s),
        δ * Tensor.map Activation.reluDeriv x =
          (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map Activation.relu y) x).adjoint δ := by
      intro s x δ
      op_deriv
  fail_if_success
    have : ∀ (s : Shape) (x : Tensor ℝ s),
        DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => a ^ (1 / 2 : ℝ)) y) x := by
      intro s x
      op_deriv
  trivial

/-- `op_deriv` does not unfold `dense` when it is not given in brackets, since no rule names it. -/
example : True := by
  fail_if_success
    have : ∀ (n h : Nat) (W : Tensor ℝ [h, n]) (b : Tensor ℝ [h]) (v : Tensor ℝ [n]),
        ∃ D g, HasPullbackAt (fun p : WithLp 2 (Tensor ℝ [h, n] × Tensor ℝ [h]) =>
          dense p.fst p.snd v) D g (WithLp.toLp 2 (W, b)) := by
      intro n h W b v
      exact ⟨_, _, by op_deriv⟩
  trivial

end Reject

end TorchLean.Test.OpDeriv
