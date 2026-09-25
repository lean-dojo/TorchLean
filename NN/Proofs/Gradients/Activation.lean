/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.Sigmoid
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
public import NN.Proofs.Utils.MathFunctions
public import NN.Spec.Layers.Activation
public import NN.Spec.Core.Context.Real

/-!
# `NN.Proofs.Gradients.Activation`

Real calculus lemmas (`HasDerivAt`, etc.) for scalar activation functions, used as building blocks
for TorchLean autograd correctness proofs.
-/

@[expose] public section

open Complex
open Real
open Activation
open Spec TorchLean
open TorchLean TorchLean.Tensor
open scoped Topology
open Filter

namespace Proofs

/-!
# Calculus lemmas for activation functions (real-valued)

This file proves `HasDerivAt` facts for common scalar activations (ReLU, leaky ReLU, sigmoid, …)
and connects them to the derivative “spec” functions used elsewhere in TorchLean.

## Why this is here
TorchLean’s autograd correctness theorems often come in two layers:
1) algebraic adjointness theorems (VJP/JVP duality) for tensor programs, and
2) calculus facts that the chosen scalar primitives really have the stated derivatives.

This file contributes to (2) in the simplest setting: scalar functions `ℝ → ℝ`.

## PyTorch correspondence / citations

These theorems are best read as “the scalar formulas behind PyTorch agree with the spec”:

- ReLU: `torch.relu` / `torch.nn.functional.relu`
  https://pytorch.org/docs/stable/generated/torch.relu.html
  https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
- Leaky ReLU: `torch.nn.functional.leaky_relu`
  https://pytorch.org/docs/stable/generated/torch.nn.functional.leaky_relu.html
- Sigmoid (logistic): `torch.sigmoid`
  https://pytorch.org/docs/stable/generated/torch.sigmoid.html
- Tanh: `torch.tanh`
  https://pytorch.org/docs/stable/generated/torch.tanh.html
- Softplus: `torch.nn.functional.softplus`
  https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html
- SiLU / Swish: `torch.nn.functional.silu`
  https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html
- ELU: `torch.nn.functional.elu`
  https://pytorch.org/docs/stable/generated/torch.nn.functional.elu.html
- GELU: `torch.nn.functional.gelu(..., approximate="tanh")`
  https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html
- Hyperbolic sine/cosine: `torch.sinh`, `torch.cosh`
  https://pytorch.org/docs/stable/generated/torch.sinh.html
  https://pytorch.org/docs/stable/generated/torch.cosh.html

Important caveat (matches PyTorch practice): `relu` (and `leakyRelu`) are not differentiable at
`0`. ELU is differentiable at `0` only for the special case `alpha = 1`; the reusable theorem below
therefore also takes `x ≠ 0`.

## References
- Mathlib’s calculus library is the main dependency:
  `Mathlib.Analysis.Calculus.Deriv.*` and `Mathlib.Analysis.SpecialFunctions.ExpDeriv`.
- The derivatives themselves are standard and can be found in any calculus textbook; the value here
  is turning them into reusable Lean lemmas.
-/

/--
Correctness of the ReLU derivative spec away from the kink at `0`.

PyTorch note: `torch.relu` uses a subgradient convention at `0`; in this file we avoid that
subtlety by assuming `x ≠ 0`.
-/
theorem relu_deriv_correct (x : ℝ) (h : x ≠ 0) :
    HasDerivAt Activation.Math.reluSpec (Activation.Math.reluDerivSpec x) x := by
  unfold Activation.Math.reluDerivSpec
  by_cases hx : 0 < x
  · -- Case: x > 0
    simp only [ite_eq_left hx]
    apply (hasDerivAt_id' x).congr_of_eventuallyEq
    filter_upwards [Ioi_mem_nhds hx] with y hy
    simpa only [Activation.Math.reluSpec_eq_max] using max_eq_left (le_of_lt hy)
  · -- Case: x ≤ 0 but x ≠ 0 ⇒ x < 0
    push Not at hx
    have hx' : x < 0 := lt_of_le_of_ne hx h
    simp only [ite_eq_right (not_lt.mpr hx)]
    apply (hasDerivAt_const x 0).congr_of_eventuallyEq
    filter_upwards [Iio_mem_nhds hx'] with y hy
    simpa only [Activation.Math.reluSpec_eq_max] using max_eq_right (le_of_lt hy)

/--
Correctness of the leaky-ReLU derivative spec away from the kink at `0`.

PyTorch correspondence: `torch.nn.functional.leaky_relu(x, negative_slope = αₗ)` with `αₗ > 0`.
-/
theorem leaky_relu_deriv_correct (x : ℝ) (h : x ≠ 0) (αₗ : ℝ) (_ : αₗ > 0) :
  HasDerivAt (fun x ↦ Activation.Math.leakyReluSpec x αₗ) (Activation.Math.leakyReluDerivSpec x
    αₗ) x := by
  unfold Activation.Math.leakyReluSpec Activation.Math.leakyReluDerivSpec
  by_cases hx : 0 < x
  · simp only [ite_eq_left hx]
    apply (hasDerivAt_id' x).congr_of_eventuallyEq
    filter_upwards [Ioi_mem_nhds hx] with y hy
    show Activation.Math.leakyReluSpec y αₗ = y
    dsimp [Activation.Math.leakyReluSpec]
    split_ifs with h'
    · rfl
    · contradiction
  · push Not at hx
    have hx' : x < 0 := lt_of_le_of_ne hx h
    simp only [ite_eq_right (not_lt.mpr hx)]
    -- derivative is αₗ * id derivative = αₗ * 1 = αₗ
    apply (hasDerivAt_mul_const αₗ).congr_of_eventuallyEq
    filter_upwards [Iio_mem_nhds hx'] with y hy
    show Activation.Math.leakyReluSpec y αₗ = y * αₗ
    dsimp [Activation.Math.leakyReluSpec]
    split_ifs with h'
    · exact (lt_irrefl _ (lt_trans hy h')).elim
    · rw [mul_comm]

/-- Correctness of the square derivative spec: `d/dx x^2 = 2x`. -/
theorem square_deriv_correct (x : ℝ) :
    HasDerivAt (fun y : ℝ => y * y) ((2 : ℝ) * x) x := by
  have hid : HasDerivAt (fun y : ℝ => y) (1 : ℝ) x := hasDerivAt_id' x
  have hmul := hid.mul hid
  have hderiv : (1 : ℝ) * x + x * (1 : ℝ) = (2 : ℝ) * x := by
    norm_num
    ring
  exact hmul.congr_deriv hderiv

/-- Correctness of the hyperbolic-sine derivative spec: `sinh' = cosh`. -/
theorem sinh_deriv_correct (x : ℝ) :
    HasDerivAt Activation.Math.sinhSpec (Activation.Math.sinhDerivSpec x) x := by
  change HasDerivAt (fun y : ℝ => Real.sinh y) (Real.cosh x) x
  exact Real.hasDerivAt_sinh x

/-- Correctness of the hyperbolic-cosine derivative spec: `cosh' = sinh`. -/
theorem cosh_deriv_correct (x : ℝ) :
    HasDerivAt Activation.Math.coshSpec (Activation.Math.coshDerivSpec x) x := by
  change HasDerivAt (fun y : ℝ => Real.cosh y) (Real.sinh x) x
  exact Real.hasDerivAt_cosh x

/--
Correctness of the ELU derivative spec away from the kink at `0`.

PyTorch correspondence: `torch.nn.functional.elu(x, alpha = α)`. For arbitrary `α`, the left
derivative at `0` is `α` and the right derivative is `1`, so the clean reusable statement is the
pointwise theorem away from `0`.
-/
theorem elu_deriv_correct (x α : ℝ) (h : x ≠ 0) :
    HasDerivAt (fun y : ℝ => Activation.Math.eluSpec y α)
      (Activation.Math.eluDerivSpec x α) x := by
  unfold Activation.Math.eluSpec Activation.Math.eluDerivSpec
  by_cases hx : 0 < x
  · simp only [ite_eq_left hx]
    apply (hasDerivAt_id' x).congr_of_eventuallyEq
    filter_upwards [Ioi_mem_nhds hx] with y hy
    have hy' : 0 < y := hy
    simp [hy']
  · push Not at hx
    have hx' : x < 0 := lt_of_le_of_ne hx h
    simp only [ite_eq_right (not_lt.mpr hx)]
    have hbase : HasDerivAt (fun y : ℝ => Real.exp y - 1) (Real.exp x) x :=
      (Real.hasDerivAt_exp x).sub_const 1
    have hscaled : HasDerivAt (fun y : ℝ => α * (Real.exp y - 1)) (α * Real.exp x) x :=
      hbase.const_mul α
    have hscaled' :
        HasDerivAt (fun y : ℝ => α * (MathFunctions.exp y - 1))
          (α * MathFunctions.exp x) x := by
      simpa [mathfunc_exp_eq_rexp] using hscaled
    apply hscaled'.congr_of_eventuallyEq
    filter_upwards [Iio_mem_nhds hx'] with y hy
    have hy' : y < 0 := hy
    simp [not_lt.mpr (le_of_lt hy')]

/-- Both evaluation branches of sigmoid define `(1 + exp (-x))⁻¹` over the reals.

The negative branch multiplies numerator and denominator by `exp x`. This identity lets the
calculus proofs use one smooth expression, including at the branch boundary `x = 0`. -/
theorem sigmoid_eq_inv_exp (x : ℝ) : Activation.Math.sigmoidSpec x = (1 + Real.exp (-x))⁻¹ := by
  simp only [Activation.Math.sigmoidSpec, mathfunc_exp_eq_rexp]
  split_ifs
  · exact one_div _
  · rw [Real.exp_neg]
    have hexp : Real.exp x ≠ 0 := Real.exp_ne_zero x
    have hden : (1 : ℝ) + Real.exp x ≠ 0 := by positivity
    field_simp [hexp, hden]
    ring

/-- Sigmoid also equals `exp x / (1 + exp x)` over the reals.

This form is the negative-input evaluation branch and the derivative obtained by differentiating
`log (1 + exp x)`. -/
theorem sigmoid_eq_exp_div (x : ℝ) :
    Activation.Math.sigmoidSpec x = Real.exp x / (1 + Real.exp x) := by
  rw [sigmoid_eq_inv_exp, Real.exp_neg]
  have hexp : Real.exp x ≠ 0 := Real.exp_ne_zero x
  have hden : (1 : ℝ) + Real.exp x ≠ 0 := by positivity
  field_simp [hexp, hden]
  ring

/--
Correctness of the sigmoid derivative spec.

PyTorch correspondence: `torch.sigmoid`.
-/
theorem sigmoid_deriv_correct (x : ℝ) :
  HasDerivAt Activation.Math.sigmoidSpec (Activation.Math.sigmoidDerivSpec x) x := by
  have hs : (Activation.Math.sigmoidSpec : ℝ → ℝ) = Real.sigmoid :=
    funext fun y => by rw [sigmoid_eq_inv_exp, Real.sigmoid_def]
  simpa only [Activation.Math.sigmoidDerivSpec, hs] using Real.hasDerivAt_sigmoid x

/-- Differentiating `σ(x)(1 - σ(x))` gives `σ'(x)(1 - 2σ(x))`.

This is the scalar second derivative used when a sigmoid VJP is evaluated over dual numbers.
The real identity holds at zero as well as on either evaluation branch. -/
theorem sigmoid_deriv_spec_deriv_correct (x : ℝ) :
    HasDerivAt Activation.Math.sigmoidDerivSpec
      (Activation.Math.sigmoidDerivSpec x * (1 - 2 * Activation.Math.sigmoidSpec x)) x := by
  have hsig : HasDerivAt Activation.Math.sigmoidSpec (Activation.Math.sigmoidDerivSpec x) x :=
    sigmoid_deriv_correct x
  have hproduct :
      HasDerivAt (fun y : ℝ => Activation.Math.sigmoidSpec y * (1 - Activation.Math.sigmoidSpec y))
        (Activation.Math.sigmoidDerivSpec x * (1 - Activation.Math.sigmoidSpec x) +
          Activation.Math.sigmoidSpec x * -Activation.Math.sigmoidDerivSpec x) x := by
    change HasDerivAt (Activation.Math.sigmoidSpec * fun y : ℝ => 1 - Activation.Math.sigmoidSpec y)
      (Activation.Math.sigmoidDerivSpec x * (1 - Activation.Math.sigmoidSpec x) +
        Activation.Math.sigmoidSpec x * -Activation.Math.sigmoidDerivSpec x) x
    exact hsig.mul (hsig.const_sub 1)
  have hderiv :
      Activation.Math.sigmoidDerivSpec x * (1 - Activation.Math.sigmoidSpec x) +
          Activation.Math.sigmoidSpec x * -Activation.Math.sigmoidDerivSpec x =
        Activation.Math.sigmoidDerivSpec x * (1 - 2 * Activation.Math.sigmoidSpec x) := by
    ring
  exact hproduct.congr_deriv hderiv

/--
Correctness of the derivative spec for `Activation.Math.logisticSpec`.

This is the scalar logistic formula `exp x / (exp x + 1)`. It is not named
`softmax`: a one-entry softmax is always `1`, while TorchLean's actual axis-normalizing softmax is
the tensor-level `Activation.softmaxSpec` in `NN/Spec/Layers/Activation.lean`.
-/
theorem logistic_deriv_correct (x : ℝ) :
  HasDerivAt Activation.Math.logisticSpec (Activation.Math.logisticDerivSpec x) x := by
  -- Scalar logistic: `exp x / (exp x + 1)`.
  have hdenom : MathFunctions.exp x + 1 ≠ 0 := by
    -- reduce to `Real.exp_pos`
    simpa [mathfunc_exp_eq_rexp] using (by linarith [Real.exp_pos x] : Real.exp x + 1 ≠ 0)
  have hu : HasDerivAt (fun y : ℝ => MathFunctions.exp y) (MathFunctions.exp x) x := by
    simpa [mathfunc_exp_eq_rexp] using (Real.hasDerivAt_exp x)
  have hv : HasDerivAt (fun y : ℝ => MathFunctions.exp y + 1) (MathFunctions.exp x) x := by
    simpa [mathfunc_exp_eq_rexp] using (Real.hasDerivAt_exp x).add_const 1
  have hdiv := hu.div hv hdenom
  have hdiv' :
      HasDerivAt (fun y : ℝ => MathFunctions.exp y / (MathFunctions.exp y + 1))
        ((MathFunctions.exp x * (MathFunctions.exp x + 1) - MathFunctions.exp x * MathFunctions.exp
          x) /
          ((MathFunctions.exp x + 1) * (MathFunctions.exp x + 1)))
        x := by
    change HasDerivAt
      ((fun y : ℝ => MathFunctions.exp y) / fun y : ℝ => MathFunctions.exp y + 1)
      ((MathFunctions.exp x * (MathFunctions.exp x + 1) - MathFunctions.exp x * MathFunctions.exp
        x) /
        ((MathFunctions.exp x + 1) * (MathFunctions.exp x + 1)))
      x
    simpa [pow_two] using hdiv
  have hsimp :
      (MathFunctions.exp x * (MathFunctions.exp x + 1) - MathFunctions.exp x * MathFunctions.exp x)
        /
          ((MathFunctions.exp x + 1) * (MathFunctions.exp x + 1))
        =
      (MathFunctions.exp x) / ((MathFunctions.exp x + 1) * (MathFunctions.exp x + 1)) := by
    ring_nf
  have hderiv :
      Activation.Math.logisticDerivSpec x =
        (MathFunctions.exp x) / ((MathFunctions.exp x + 1) * (MathFunctions.exp x + 1)) := by
    unfold Activation.Math.logisticDerivSpec Activation.Math.logisticSpec
    field_simp [hdenom]
    ring
  -- Replace the quotient-rule derivative by `logisticDerivSpec x`.
  have hdiv'' :
      HasDerivAt (fun y : ℝ => MathFunctions.exp y / (MathFunctions.exp y + 1))
        (Activation.Math.logisticDerivSpec x) x := by
    simpa [hsimp, hderiv] using hdiv'
  -- Rewrite the function to `Activation.Math.logisticSpec`.
  change HasDerivAt (fun y : ℝ => MathFunctions.exp y / (MathFunctions.exp y + 1))
    (Activation.Math.logisticDerivSpec x) x
  exact hdiv''

/-- Real tanh is smooth at every order because its cosh denominator never vanishes. -/
@[fun_prop] theorem contDiff_tanh {n : WithTop ℕ∞} : ContDiff ℝ n Real.tanh := by
  change ContDiff ℝ n (fun x => Real.tanh x)
  simp only [Real.tanh_eq_sinh_div_cosh]
  exact Real.contDiff_sinh.fun_div Real.contDiff_cosh (fun x => (Real.cosh_pos x).ne')

/-- `tanh` in terms of `exp`, which is the form every derivative computation below wants.

Mathlib defines `Real.tanh` through the complex hyperbolic functions, so getting to this elementary
identity means descending to `Complex.sinh`/`Complex.cosh` and taking real parts. The long `calc`
below does exactly that, and it is worth doing once here rather than inside each proof. -/
theorem tanh_exp_eq (x : ℝ) :
  Real.tanh x = (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x)) := by
  calc
    Real.tanh x = (Complex.sinh ↑x).re / (Complex.cosh ↑x).re := by
      rw [Real.tanh_eq_sinh_div_cosh, Real.sinh, Real.cosh]
    _ = ((cexp ↑x - cexp (-↑x)) / 2).re / ((cexp ↑x + cexp (-↑x)) / 2).re := by
      dsimp only [Complex.sinh, Complex.cosh]
    _ = ((rexp x * Real.cos 0 - rexp (-x) * Real.cos 0) * re 2 / normSq 2 +
         (rexp x * Real.sin 0 - rexp (-x) * Real.sin 0) * im 2 / normSq 2) /
        ((rexp x * Real.cos 0 + rexp (-x) * Real.cos 0) * re 2 / normSq 2 +
         (rexp x * Real.sin 0 + rexp (-x) * Real.sin 0) * im 2 / normSq 2) := by
      rw [Complex.div_re, Complex.sub_re, Complex.div_re, Complex.add_re, Complex.add_im,
        Complex.sub_im]
      rw [Complex.exp_im, Complex.exp_re]
      rw [Complex.exp_im, Complex.exp_re]
      rw [neg_re, neg_im]
      rw [ofReal_re, ofReal_im, neg_zero]
    _ = ((rexp x - rexp (-x)) * re 2 / normSq 2) /
        ((rexp x + rexp (-x)) * re 2 / normSq 2) := by
      simp only [Real.cos_zero, Real.sin_zero, mul_one, mul_zero, zero_mul, add_zero]
      simp only [sub_zero, zero_mul, zero_div, add_zero]
    _ = ((rexp x - rexp (-x)) * 2 / 4) /
        ((rexp x + rexp (-x)) * 2 / 4) := by norm_num
    _ = ((2 / 4) * ((rexp x - rexp (-x))) /
        ((2 / 4) * (rexp x + rexp (-x)))) := by
          rw [mul_div_assoc]
          rw [mul_comm]
          rw [mul_div_assoc]
          rw [mul_div_assoc]
          conv =>
            pattern (rexp x + rexp (-x)) * (2 / 4)
            rw [mul_comm]
          rw [mul_div_assoc']
    _ = (2 / 4) / (2 / 4) * ((rexp x - rexp (-x)) /
        (rexp x + rexp (-x))) := by
          rw [mul_div_mul_comm]
    _ = 1 * ((rexp x - rexp (-x)) /
        (rexp x + rexp (-x))) := by norm_num
    _ = (rexp x - rexp (-x)) / (rexp x + rexp (-x)) := by simp

/--
Correctness of the tanh derivative spec.

PyTorch correspondence: `torch.tanh`.
-/
theorem tanh_deriv_correct (x : ℝ) :
    HasDerivAt Activation.Math.tanhSpec (Activation.Math.tanhDerivSpec x) x := by
  -- Unfold definitions
  unfold Activation.Math.tanhSpec Activation.Math.tanhDerivSpec

  -- Define numerator and denominator of tanh(x) = f(x)/g(x)
  let f : ℝ → ℝ := fun t => Real.exp t - Real.exp (-t)
  let g : ℝ → ℝ := fun t => Real.exp t + Real.exp (-t)

  -- Derivative of f and g using chain rule and derivative rules
  have h_exp_neg : HasDerivAt (fun t => Real.exp (-t)) (-Real.exp (-x)) x := by
    have h_neg : HasDerivAt (fun t ↦ -t) (-1) x := hasDerivAt_neg x
    have h_comp := (Real.hasDerivAt_exp (-x)).comp x h_neg
    simpa [Function.comp_def] using h_comp

  have hf : HasDerivAt f (Real.exp x + Real.exp (-x)) x := by
    rw [← sub_neg_eq_add]
    exact HasDerivAt.sub (hasDerivAt_exp x) h_exp_neg

  have hg : HasDerivAt g (Real.exp x - Real.exp (-x)) x :=
    HasDerivAt.add (hasDerivAt_exp x) h_exp_neg

  -- Denominator is nonzero for all x
  have h_denom_ne_zero : g x ≠ 0 :=
    ne_of_gt (add_pos (exp_pos x) (exp_pos (-x)))

  -- Show that tanhAct equals f/g in a neighborhood
  have h_func_eq : ∀ᶠ y in 𝓝 x, Activation.Math.tanhSpec y = f y / g y := by
    apply Filter.Eventually.of_forall
    intro y
    unfold Activation.Math.tanhSpec
    rw [mathfunc_tanh_eq_rtanh, tanh_exp_eq]

  -- Show that the derivative equals 1 - tanh²
  have h_derivative_eq : ((Real.exp x + Real.exp (-x)) * g x - f x * (Real.exp x - Real.exp (-x))) /
    g x ^ 2 =
    1 - MathFunctions.tanh x * MathFunctions.tanh x := by
    -- First establish the numerator identity
    have h_num : (Real.exp x + Real.exp (-x)) * g x - f x * (Real.exp x - Real.exp (-x)) = 4 := by
      simp only [f, g]
      ring_nf
      rw [← Real.exp_add x (-x)]
      simp [Real.exp_zero]

    -- Then show 1 - tanh²(x) = 4/(g x)²
    have h_tanh_identity :
      1 - MathFunctions.tanh x ^ 2 = 4 / (g x) ^ 2 := by
      rw [mathfunc_tanh_eq_rtanh, tanh_exp_eq]
      -- now tanh x = (exp x - exp (-x)) / (exp x + exp (-x))
      simp only [g]
      -- rewrite tanh squared explicitly
      field_simp [add_pos (exp_pos x) (exp_pos (-x))] -- denominator nonzero
      ring_nf
      rw [← Real.exp_add x (-x)]
      simp [Real.exp_zero]

    rw [h_num, ← h_tanh_identity]
    rw [pow_two]

  -- Apply quotient rule and use the established equalities
  have h_deriv := HasDerivAt.div hf hg h_denom_ne_zero

  -- Use congr_of_eventuallyEq with the correct derivative
  exact HasDerivAt.congr_of_eventuallyEq
    (h_deriv.congr_deriv h_derivative_eq)
    h_func_eq

/-- The real tanh chain rule, using the same derivative expression as the runtime. -/
theorem _root_.HasDerivAt.tanh {f : ℝ → ℝ} {f' x : ℝ} (hf : HasDerivAt f f' x) :
    HasDerivAt (fun y => Real.tanh (f y))
      ((1 - Real.tanh (f x) * Real.tanh (f x)) * f') x := by
  simpa only [Function.comp_def, Activation.Math.tanhSpec, Activation.Math.tanhDerivSpec,
    Proofs.mathfunc_tanh_eq_rtanh] using (Proofs.tanh_deriv_correct (f x)).comp x hf

/--
Correctness of the tanh-approximate GELU derivative spec.

PyTorch correspondence: `torch.nn.functional.gelu(x, approximate = "tanh")`. The proof follows
the product rule for
`x * (1 + tanh(c * (x + k*x^3))) / 2`, plus the chain rule through the tanh inner polynomial.
-/
theorem gelu_deriv_correct (x : ℝ) :
    HasDerivAt Activation.Math.geluSpec (Activation.Math.geluDerivSpec x) x := by
  unfold Activation.Math.geluSpec Activation.Math.geluDerivSpec
  let two : ℝ := (1 : ℝ) + (1 : ℝ)
  let three : ℝ := (1 : ℝ) + (1 : ℝ) + (1 : ℝ)
  let pi : ℝ := MathFunctions.pi
  let sqrt_two_over_pi : ℝ := MathFunctions.sqrt (two / pi)
  let coeff : ℝ := Activation.Math.geluTanhCoeff
  let u : ℝ → ℝ := fun y => sqrt_two_over_pi * (y + coeff * y * y * y)
  let tanh_term : ℝ := MathFunctions.tanh (u x)
  let sech_term : ℝ := (1 : ℝ) - tanh_term * tanh_term
  let inner_deriv : ℝ := sqrt_two_over_pi * ((1 : ℝ) + three * coeff * x * x)
  have hid : HasDerivAt (fun y : ℝ => y) (1 : ℝ) x := hasDerivAt_id' x

  -- Inner cubic: `y ↦ coeff * y^3`, written in the same multiplication shape as the spec.
  have hcoeff_y3 : HasDerivAt (fun y : ℝ => coeff * y * y * y)
      (three * coeff * x * x) x := by
    have hpow : HasDerivAt (fun y : ℝ => y ^ 3) ((3 : ℝ) * x ^ (3 - 1)) x :=
      hasDerivAt_pow 3 x
    have hscaled : HasDerivAt (fun y : ℝ => coeff * (y ^ 3))
        (coeff * ((3 : ℝ) * x ^ (3 - 1))) x :=
      hpow.const_mul coeff
    have hderiv : coeff * ((3 : ℝ) * x ^ (3 - 1)) = three * coeff * x * x := by
      norm_num [three]
      ring
    have hscaled' := hscaled.congr_deriv hderiv
    apply hscaled'.congr_of_eventuallyEq
    exact Filter.Eventually.of_forall (fun y => by ring)

  have hpoly : HasDerivAt (fun y : ℝ => y + coeff * y * y * y)
      ((1 : ℝ) + three * coeff * x * x) x := by
    change HasDerivAt ((fun y : ℝ => y) + fun y : ℝ => coeff * y * y * y)
      ((1 : ℝ) + three * coeff * x * x) x
    exact hid.add hcoeff_y3
  have hu : HasDerivAt u inner_deriv x := by
    have h := hpoly.const_mul sqrt_two_over_pi
    simpa [u, inner_deriv, mul_assoc] using h
  have htanh0 :
      HasDerivAt Activation.Math.tanhSpec (Activation.Math.tanhDerivSpec (u x)) (u x) :=
    tanh_deriv_correct (u x)
  have htanh : HasDerivAt (fun y : ℝ => Activation.Math.tanhSpec (u y))
      (sech_term * inner_deriv) x := by
    have h := htanh0.comp x hu
    simpa [Function.comp_def, Activation.Math.tanhDerivSpec, sech_term, tanh_term] using h
  have hA : HasDerivAt (fun y : ℝ => (1 : ℝ) + Activation.Math.tanhSpec (u y))
      (sech_term * inner_deriv) x := by
    simpa using htanh.const_add (1 : ℝ)
  have hprod : HasDerivAt
      (fun y : ℝ => y * ((1 : ℝ) + Activation.Math.tanhSpec (u y)))
      ((1 : ℝ) * ((1 : ℝ) + Activation.Math.tanhSpec (u x)) +
        x * (sech_term * inner_deriv)) x := by
    change HasDerivAt
      ((fun y : ℝ => y) * fun y : ℝ => (1 : ℝ) + Activation.Math.tanhSpec (u y))
      ((1 : ℝ) * ((1 : ℝ) + Activation.Math.tanhSpec (u x)) +
        x * (sech_term * inner_deriv)) x
    exact hid.mul hA
  have hdiv : HasDerivAt
      (fun y : ℝ => y * ((1 : ℝ) + Activation.Math.tanhSpec (u y)) / two)
      (((1 : ℝ) * ((1 : ℝ) + Activation.Math.tanhSpec (u x)) +
        x * (sech_term * inner_deriv)) / two) x :=
    hprod.div_const two
  have hderiv :
      (((1 : ℝ) * ((1 : ℝ) + Activation.Math.tanhSpec (u x)) +
          x * (sech_term * inner_deriv)) / two)
        =
      ((1 : ℝ) + tanh_term + x * sech_term * inner_deriv) / two := by
    simp [tanh_term, sech_term, Activation.Math.tanhSpec, mul_assoc]
  simpa [u, two, three, pi, sqrt_two_over_pi, coeff, tanh_term, sech_term, inner_deriv,
    Activation.Math.tanhSpec] using hdiv.congr_deriv hderiv

/-- The branch-stable real softplus specification is extensionally `log (1 + exp x)`. -/
theorem softplus_spec_eq_log_one_add_exp (x : ℝ) :
    Activation.Math.softplusSpec (α := ℝ) x = Real.log (1 + Real.exp x) := by
  simp only [Activation.Math.softplusSpec, MathFunctions.log, MathFunctions.exp]
  split_ifs with hx
  · have hexp : Real.exp x ≠ 0 := ne_of_gt (Real.exp_pos x)
    have hinner : (1 : ℝ) + Real.exp (-x) ≠ 0 := by positivity
    calc
      x + Real.log (1 + Real.exp (-x)) =
          Real.log (Real.exp x) + Real.log (1 + Real.exp (-x)) := by rw [Real.log_exp]
      _ = Real.log (Real.exp x * (1 + Real.exp (-x))) :=
        (Real.log_mul hexp hinner).symm
      _ = Real.log (1 + Real.exp x) := by
        congr 1
        rw [mul_add, mul_one, ← Real.exp_add]
        simp [add_comm]
  · rfl

/--
Correctness of the softplus derivative spec.

PyTorch correspondence: `torch.nn.functional.softplus`.
-/
theorem softplus_deriv_correct (x : ℝ) :
    HasDerivAt Activation.Math.softplusSpec (Activation.Math.softplusDerivSpec x) x := by
  have softplus_eq_log_exp : Activation.Math.softplusSpec (α := ℝ) =
      fun y : ℝ => Real.log (1 + Real.exp y) := by
    funext y
    exact softplus_spec_eq_log_one_add_exp y
  rw [softplus_eq_log_exp]
  unfold Activation.Math.softplusDerivSpec
  -- derivative of `log (1 + exp x)`
  have hx0 : (1 : ℝ) + Real.exp x ≠ 0 := by
    have : 0 < (1 : ℝ) + Real.exp x := by linarith [Real.exp_pos x]
    exact ne_of_gt this
  have h_inner : HasDerivAt (fun y : ℝ => (1 : ℝ) + Real.exp y) (Real.exp x) x := by
    simpa using (Real.hasDerivAt_exp x).const_add (1 : ℝ)
  have h_comp' := (Real.hasDerivAt_log hx0).comp x h_inner
  have h_comp :
      HasDerivAt (fun y : ℝ => Real.log ((1 : ℝ) + Real.exp y)) ((1 + Real.exp x)⁻¹ * Real.exp x) x
        := by
    simpa [Function.comp_def, mul_assoc, mul_left_comm, mul_comm, add_comm, add_left_comm,
      add_assoc] using h_comp'
  -- The exponential ratio is the derivative of the logarithm in this expression.
  have hsig : (Activation.Math.sigmoidSpec (α := ℝ) x) = (Real.exp x) * (1 + Real.exp x)⁻¹ := by
    rw [sigmoid_eq_exp_div, div_eq_mul_inv]
  simpa [MathFunctions.log, MathFunctions.exp, one_div, div_eq_mul_inv, hsig,
    mul_comm, mul_left_comm, mul_assoc, add_comm, add_left_comm, add_assoc] using h_comp

/-- The softplus derivative is sigmoid, so its derivative is `σ(x)(1 - σ(x))` at every real
input. This supplies the second-derivative identity without differentiating the branch test. -/
theorem softplus_deriv_spec_deriv_correct (x : ℝ) :
    HasDerivAt Activation.Math.softplusDerivSpec (Activation.Math.sigmoidDerivSpec x) x := by
  exact sigmoid_deriv_correct x

/--
Correctness of the SiLU derivative spec.

`silu(x) = x * sigmoid(x)`, so the proof is just the product rule plus the already-proved
sigmoid derivative. This is the scalar calculus fact used by the tensor-level VJP proof bridge.
-/
theorem silu_deriv_correct (x : ℝ) :
    HasDerivAt Activation.Math.swishSpec (Activation.Math.swishDerivSpec x) x := by
  unfold Activation.Math.swishSpec Activation.Math.swishDerivSpec
  have hid : HasDerivAt (fun y : ℝ => y) (1 : ℝ) x := hasDerivAt_id' x
  have hsig : HasDerivAt Activation.Math.sigmoidSpec
      (Activation.Math.sigmoidDerivSpec x) x :=
    sigmoid_deriv_correct x
  have hprod := hid.mul hsig
  have hprod' :
      HasDerivAt (fun y : ℝ => y * Activation.Math.sigmoidSpec y)
        (1 * Activation.Math.sigmoidSpec x + x * Activation.Math.sigmoidDerivSpec x) x := by
    change HasDerivAt ((fun y : ℝ => y) * Activation.Math.sigmoidSpec)
      (1 * Activation.Math.sigmoidSpec x + x * Activation.Math.sigmoidDerivSpec x) x
    exact hprod
  have hderiv :
      1 * Activation.Math.sigmoidSpec x + x * Activation.Math.sigmoidDerivSpec x =
        (let s := Activation.Math.sigmoidSpec x; s + x * s * (1 - s)) := by
    simp [Activation.Math.sigmoidDerivSpec]
    ring
  exact hprod'.congr_deriv hderiv

/--
Correctness of the `safeLog` derivative spec (a smooth log surrogate).

`safeLog` is not a standard PyTorch primitive; conceptually it is “log-like but always defined”
using `softplus` to avoid a strict-positivity side condition.

Related PyTorch primitives:
https://pytorch.org/docs/stable/generated/torch.log.html
-/
theorem safe_log_deriv_correct (x ε : ℝ) (hε : 0 < ε) :
    HasDerivAt (fun y => Activation.Math.safeLogSpec y ε) (Activation.Math.safeLogDerivSpec x
      ε) x := by
  unfold Activation.Math.safeLogSpec Activation.Math.safeLogDerivSpec
  -- `safe_log(x) = log(softplus(x) + ε)`
  have hsoft : HasDerivAt Activation.Math.softplusSpec (Activation.Math.sigmoidSpec x) x := by
    simpa [Activation.Math.softplusDerivSpec] using softplus_deriv_correct x
  have h_inner : HasDerivAt (fun y : ℝ => Activation.Math.softplusSpec y + ε)
    (Activation.Math.sigmoidSpec x) x := by
    simpa using hsoft.const_add ε
  have hx0 : Activation.Math.softplusSpec x + ε ≠ 0 := by
    have : 0 < Activation.Math.softplusSpec x + ε := by
      -- `softplus(x) = log(1+exp x) ≥ 0` and `ε > 0`
      have hpos : 0 ≤ Activation.Math.softplusSpec x := by
        -- `1 ≤ 1 + exp x`, and `log` is monotone on `(0,∞)`
        have h1 : (1 : ℝ) ≤ 1 + Real.exp x := by linarith [Real.exp_pos x]
        rw [softplus_spec_eq_log_one_add_exp]
        exact Real.log_nonneg h1
      linarith
    exact ne_of_gt this
  have h_comp := (Real.hasDerivAt_log hx0).comp x h_inner
  -- `log' u = 1/u`
  simpa [Function.comp_def, MathFunctions.log, Activation.Math.softplusDerivSpec, one_div,
    div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using h_comp

/--
Correctness of the `smoothAbs` derivative spec (a smooth absolute-value surrogate).

`smooth_abs(x; ε) = sqrt(x^2 + ε)` is a differentiable replacement for `|x|` near `0`.

Related PyTorch primitives:
https://pytorch.org/docs/stable/generated/torch.abs.html
-/
theorem smooth_abs_deriv_correct (x ε : ℝ) (hε : 0 < ε) :
    HasDerivAt (fun y => Activation.Math.smoothAbsSpec y ε) (Activation.Math.smoothAbsDerivSpec
      x ε) x := by
  unfold Activation.Math.smoothAbsSpec Activation.Math.smoothAbsDerivSpec
  -- `smooth_abs(x) = sqrt(x^2 + ε)`
  have hx0 : x * x + ε ≠ 0 := by
    have : 0 < x * x + ε := by
      have hx2 : 0 ≤ x * x := by nlinarith
      linarith
    exact ne_of_gt this
  have h_inner : HasDerivAt (fun y : ℝ => y * y + ε) (2 * x) x := by
    have hid : HasDerivAt (fun y : ℝ => y) (1 : ℝ) x := hasDerivAt_id' x
    have hmul : HasDerivAt (fun y : ℝ => y * y) (x + x) x := by
      -- product rule on `y ↦ y * y` gives derivative `x + x`
      change HasDerivAt ((fun y : ℝ => y) * fun y : ℝ => y) (x + x) x
      exact (hid.mul hid).congr_deriv (by ring)
    have htwo : x + x = 2 * x := by ring
    have hsq : HasDerivAt (fun y : ℝ => y * y) (2 * x) x :=
      hmul.congr_deriv htwo
    simpa using (hsq.const_add ε)
  have h_comp := (hasDerivAt_sqrt hx0).comp x h_inner
  -- simplify `(1 / (2 * sqrt u)) * (2 * x)` to `x / sqrt u`
  have : (1 / (2 * Real.sqrt (x * x + ε))) * (2 * x) = x / Real.sqrt (x * x + ε) := by
    field_simp
  simpa [Function.comp_def, Activation.Math.smoothAbsSpec, MathFunctions.sqrt, this, div_eq_mul_inv,
    mul_assoc, mul_left_comm, mul_comm, add_assoc, add_left_comm, add_comm] using h_comp

end Proofs
