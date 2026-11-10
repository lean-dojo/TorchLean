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
public import TorchLean.Tensor.Numeric.Real
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Elementwise.Real
public import TorchLean.Tensor.Layers.Activation

/-!
# Scalar Activation Derivatives

Real calculus lemmas for the scalar activation specifications: their derivatives (`HasDerivAt`),
the closed forms their evaluation branches take over `ℝ`, and a few bounds. The TorchLean autograd
correctness proofs build on them.
-/

@[expose] public section

open Real
open TorchLean.Tensor
open scoped Topology
open Filter
open FloatLib.Numerics (MathFunctions)

namespace TorchLean

/-!
## Related PyTorch operators

Each activation below is paired with the PyTorch operator whose formula it follows. The theorems
hold over the reals, so native evaluation, rounding and implementation-specific branches lie
outside them.

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

The ReLU, leaky-ReLU and ELU theorems assume `x ≠ 0`. ReLU has a kink at `0`. Leaky ReLU and ELU
are differentiable there when their parameter (`slope` or `α`) is `1`, a case the theorems leave
out.
-/

/--
Away from the kink at `0`, ReLU has derivative `Activation.reluDeriv x`.

`torch.relu` picks a subgradient at `0`, and the hypothesis `x ≠ 0` keeps the theorem clear of that
choice.
-/
theorem Activation.hasDerivAt_relu (x : ℝ) (h : x ≠ 0) :
    HasDerivAt Activation.relu (Activation.reluDeriv x) x := by
  unfold Activation.reluDeriv
  by_cases hx : 0 < x
  ·
    simp only [ite_eq_left hx]
    apply (hasDerivAt_id' x).congr_of_eventuallyEq
    filter_upwards [Ioi_mem_nhds hx] with y hy
    simpa only [Activation.relu_eq_max] using max_eq_left (le_of_lt hy)
  ·
    push Not at hx
    have hx' : x < 0 := lt_of_le_of_ne hx h
    simp only [ite_eq_right (not_lt.mpr hx)]
    apply (hasDerivAt_const x 0).congr_of_eventuallyEq
    filter_upwards [Iio_mem_nhds hx'] with y hy
    simpa only [Activation.relu_eq_max] using max_eq_right (le_of_lt hy)

/-- Away from zero, leaky ReLU with any coefficient `slope` has derivative
`leakyReLUDeriv x slope`. -/
theorem Activation.hasDerivAt_leakyReLU (slope : ℝ) {x : ℝ} (hx : x ≠ 0) :
    HasDerivAt (fun y => Activation.leakyReLU y slope)
      (Activation.leakyReLUDeriv x slope) x := by
  unfold Activation.leakyReLU Activation.leakyReLUDeriv
  rcases hx.lt_or_gt with h | h
  · rw [ite_eq_right (not_lt.mpr h.le)]
    have hd : HasDerivAt (fun y => slope * y) slope x :=
      ((hasDerivAt_id' x).const_mul slope).congr_deriv (mul_one slope)
    refine hd.congr_of_eventuallyEq ?_
    filter_upwards [Iio_mem_nhds h] with y hy
    exact ite_eq_right (not_lt.mpr (le_of_lt hy))
  · rw [ite_eq_left h]
    refine (hasDerivAt_id' x).congr_of_eventuallyEq ?_
    filter_upwards [Ioi_mem_nhds h] with y hy
    exact ite_eq_left hy

/-- Leaky ReLU with coefficient one is the identity, so it has derivative
`leakyReLUDeriv x 1 = 1` at every `x`. -/
theorem Activation.hasDerivAt_leakyReLU_one (x : ℝ) :
    HasDerivAt (fun y => Activation.leakyReLU y 1) (Activation.leakyReLUDeriv x 1) x := by
  have h : (fun y : ℝ => Activation.leakyReLU y 1) = id := funext fun y => by
    unfold Activation.leakyReLU
    split_ifs <;> simp
  rw [h]
  unfold Activation.leakyReLUDeriv
  split_ifs <;> exact hasDerivAt_id x

/-- Correctness of the square derivative spec: `d/dx x^2 = 2x`. -/
theorem Activation.hasDerivAt_mul_self (x : ℝ) :
    HasDerivAt (fun y : ℝ => y * y) ((2 : ℝ) * x) x := by
  simpa [pow_two] using hasDerivAt_pow 2 x

/-- The compensated exponential difference agrees with `exp x - 1` over the reals. -/
theorem Context.expm1_eq_exp_sub_one (x : ℝ) :
    Context.expm1 x = Real.exp x - 1 := by
  simp only [Context.expm1, Numeric.expm1,
    MathFunctions.exp, MathFunctions.log]
  split_ifs with hnear h₁ h₂
  · have hx : x = 0 := by
      simpa only [Real.log_exp, Real.log_one] using congrArg Real.log (beq_iff_eq.mp h₁)
    simp [hx]
  · exact (Real.exp_ne_zero x (beq_iff_eq.mp h₂)).elim
  · have hx : x ≠ 0 := by intro h; simp [h] at h₁
    simp [Real.log_exp, hx]
  · rfl

/--
For `x ≠ 0`, ELU with parameter `α` has derivative `Activation.eluDeriv x α`.

PyTorch: `torch.nn.functional.elu(x, alpha = α)`. For general `α` the one-sided derivatives at `0`
are `α` on the left and `1` on the right, so the theorem is stated away from `0`.
-/
theorem Activation.hasDerivAt_elu (x α : ℝ) (h : x ≠ 0) :
    HasDerivAt (fun y : ℝ => Activation.elu y α)
      (Activation.eluDeriv x α) x := by
  unfold Activation.elu Activation.eluDeriv
  have hexpm1 : (Context.expm1 : ℝ → ℝ) = fun y => Real.exp y - 1 :=
    funext Context.expm1_eq_exp_sub_one
  simp only [hexpm1]
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
      simpa [MathFunctions.exp_real] using hscaled
    apply hscaled'.congr_of_eventuallyEq
    filter_upwards [Iio_mem_nhds hx'] with y hy
    have hy' : y < 0 := hy
    simp [not_lt.mpr (le_of_lt hy'), MathFunctions.exp_real]

/-- ELU with parameter one has derivative `eluDeriv x 1` at every `x`, since at `0` both one-sided
derivatives are one. -/
theorem Activation.hasDerivAt_elu_one (x : ℝ) :
    HasDerivAt (fun y : ℝ => Activation.elu y 1) (Activation.eluDeriv x 1) x := by
  rcases ne_or_eq x 0 with hx | rfl
  · exact hasDerivAt_elu x 1 hx
  have hexpm1 : (Context.expm1 : ℝ → ℝ) = fun y => Real.exp y - 1 :=
    funext Context.expm1_eq_exp_sub_one
  have h0 : Activation.elu (0 : ℝ) 1 = 0 := by
    simp [Activation.elu, hexpm1]
  have hd : Activation.eluDeriv (0 : ℝ) 1 = 1 := by
    simp [Activation.eluDeriv, MathFunctions.exp_real]
  rw [hd, ← hasDerivWithinAt_univ, ← Set.Iic_union_Ici (a := (0 : ℝ))]
  refine HasDerivWithinAt.union ?_ ?_
  · have h := ((Real.hasDerivAt_exp 0).sub_const 1).hasDerivWithinAt (s := Set.Iic 0)
    rw [Real.exp_zero] at h
    refine h.congr_of_mem (fun y hy => ?_) Set.self_mem_Iic
    simp [Activation.elu, hexpm1, not_lt.mpr (Set.mem_Iic.mp hy)]
  · refine (hasDerivWithinAt_id (0 : ℝ) (Set.Ici 0)).congr_of_mem (fun y hy => ?_)
      Set.self_mem_Ici
    rcases (Set.mem_Ici.mp hy).lt_or_eq with h | h
    · simp [Activation.elu, h]
    · rw [← h, h0, id]

/-- Both evaluation branches of sigmoid equal `(1 + exp (-x))⁻¹` over the reals.

The branch for `x ≤ 0` is that expression with numerator and denominator multiplied by `exp x`.
The calculus proofs can then work with one smooth expression, at the branch boundary `x = 0` as
well. -/
theorem Context.sigmoid_eq_inv_exp (x : ℝ) : Context.sigmoid x = (1 + Real.exp (-x))⁻¹ := by
  simp only [Context.sigmoid, Numeric.sigmoid,
    Numeric.QuotientArithmetic.supported_real, Bool.not_false, ↓reduceIte,
    MathFunctions.exp_real]
  split_ifs
  · exact one_div _
  · rw [Real.exp_neg]
    have hexp : Real.exp x ≠ 0 := Real.exp_ne_zero x
    have hden : (1 : ℝ) + Real.exp x ≠ 0 := by positivity
    field_simp [hexp, hden]
    ring

/-- The real sigmoid is nonnegative. -/
theorem Context.sigmoid_nonneg (x : ℝ) :
    0 ≤ Context.sigmoid (α := ℝ) x := by
  simpa only [sigmoid_eq_inv_exp, Real.sigmoid_def] using Real.sigmoid_nonneg x

/-- The real sigmoid is at most `1`. -/
theorem Context.sigmoid_le_one (x : ℝ) :
    Context.sigmoid (α := ℝ) x ≤ 1 := by
  simpa only [sigmoid_eq_inv_exp, Real.sigmoid_def] using Real.sigmoid_le_one x

/-- Over the reals sigmoid also equals `exp x / (1 + exp x)`, its evaluation branch for `x ≤ 0`
and the derivative of `log (1 + exp x)`. -/
theorem Context.sigmoid_eq_exp_div (x : ℝ) :
    Context.sigmoid x = Real.exp x / (1 + Real.exp x) := by
  rw [sigmoid_eq_inv_exp, Real.exp_neg]
  have hexp : Real.exp x ≠ 0 := Real.exp_ne_zero x
  have hden : (1 : ℝ) + Real.exp x ≠ 0 := by positivity
  field_simp [hexp, hden]
  ring

/-- Over the reals the sigmoid slope `z / (1 + z)²` with `z = exp (-|x|)` is the usual derivative
`σ(x)(1 - σ(x))`. -/
theorem Numeric.sigmoidDeriv_eq (x : ℝ) : Numeric.sigmoidDeriv x =
    Context.sigmoid x * (1 - Context.sigmoid x) := by
  unfold Numeric.sigmoidDeriv
  simp only [Numeric.applySlope,
    Numeric.QuotientArithmetic.supported_real, Bool.false_and, Bool.not_false,
    Bool.false_eq_true, ↓reduceIte]
  by_cases hx : x > 0
  · rw [ite_eq_left hx, Context.sigmoid_eq_inv_exp]
    simp only [MathFunctions.exp]
    have hden : (1 : ℝ) + Real.exp (-x) ≠ 0 := by positivity
    field_simp
    ring
  · rw [ite_eq_right hx, Context.sigmoid_eq_exp_div]
    simp only [MathFunctions.exp]
    have hden : (1 : ℝ) + Real.exp x ≠ 0 := by positivity
    field_simp
    ring

/-- The real sigmoid has derivative `TorchLean.Numeric.sigmoidDeriv x` at every `x` (PyTorch:
`torch.sigmoid`). -/
theorem Context.hasDerivAt_sigmoid (x : ℝ) :
  HasDerivAt Context.sigmoid (Numeric.sigmoidDeriv x) x := by
  have hs : (Context.sigmoid : ℝ → ℝ) = Real.sigmoid :=
    funext fun y => by rw [sigmoid_eq_inv_exp, Real.sigmoid_def]
  simpa only [Numeric.sigmoidDeriv_eq, hs] using Real.hasDerivAt_sigmoid x

/-- Real tanh is smooth at every order because its cosh denominator never vanishes. -/
@[fun_prop] theorem Activation.contDiff_tanh {n : WithTop ℕ∞} : ContDiff ℝ n Real.tanh := by
  change ContDiff ℝ n (fun x => Real.tanh x)
  simp only [Real.tanh_eq_sinh_div_cosh]
  exact Real.contDiff_sinh.fun_div Real.contDiff_cosh (fun x => (Real.cosh_pos x).ne')

/-- Over the reals the tanh slope `(2z / (1 + z²))²` with `z = exp (-|x|)`, the square of
`1 / cosh x`, equals `1 - tanh² x`. -/
theorem Numeric.tanhDeriv_eq (x : ℝ) :
    Numeric.tanhDeriv x = 1 - Real.tanh x * Real.tanh x := by
  unfold Numeric.tanhDeriv
  simp only [Numeric.applySlope,
    Numeric.QuotientArithmetic.supported_real, Bool.false_and, Bool.not_false,
    Bool.false_eq_true, ↓reduceIte]
  have he : Real.exp x ≠ 0 := Real.exp_ne_zero x
  have hn : Real.exp (-x) ≠ 0 := Real.exp_ne_zero (-x)
  have hd : Real.exp x + Real.exp (-x) ≠ 0 := by positivity
  have hdn : (1 : ℝ) + Real.exp (-x) * Real.exp (-x) ≠ 0 := by positivity
  have hdp : (1 : ℝ) + Real.exp x * Real.exp x ≠ 0 := by positivity
  rw [Real.tanh_eq]
  split_ifs
  · simp only [MathFunctions.exp]
    rw [Real.exp_neg]
    field_simp
    ring
  · simp only [MathFunctions.exp]
    rw [Real.exp_neg]
    field_simp
    ring

/-- The real tanh has derivative `TorchLean.Numeric.tanhDeriv x` at every `x` (PyTorch:
`torch.tanh`). -/
theorem Activation.hasDerivAt_tanh (x : ℝ) :
    HasDerivAt MathFunctions.tanh (Numeric.tanhDeriv x) x := by
  have hcosh : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  have hfun : (MathFunctions.tanh : ℝ → ℝ) = fun y => Real.sinh y / Real.cosh y :=
    funext fun y => Real.tanh_eq_sinh_div_cosh y
  -- The quotient rule for `sinh / cosh` gives `(cosh² - sinh²) / cosh²`. That equals `1 - tanh²`
  -- as a rational identity in `sinh` and `cosh`, so `cosh² - sinh² = 1` is never used.
  have hderiv :
      (Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) / Real.cosh x ^ 2 =
        Numeric.tanhDeriv x := by
    rw [Numeric.tanhDeriv_eq, Real.tanh_eq_sinh_div_cosh]
    field_simp
  rw [hfun, ← hderiv]
  exact (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hcosh

/-- The chain rule for real `tanh`: if `f` has derivative `f'` at `x`, then `tanh ∘ f` has
derivative `(1 - tanh² (f x)) f'` there. -/
theorem _root_.HasDerivAt.tanh {f : ℝ → ℝ} {f' x : ℝ} (hf : HasDerivAt f f' x) :
    HasDerivAt (fun y => Real.tanh (f y))
      ((1 - Real.tanh (f x) * Real.tanh (f x)) * f') x := by
  simpa only [Function.comp_def, Numeric.tanhDeriv_eq,
    MathFunctions.tanh_real] using (Activation.hasDerivAt_tanh (f x)).comp x hf

/--
The tanh-approximate GELU, `torch.nn.functional.gelu(x, approximate = "tanh")`, has derivative
`Activation.geluTanhDeriv x` at every real `x`.

The proof rewrites the gate `2 sigmoid (2 u)` as `1 + tanh u` and applies the product rule to
`x (1 + tanh (β (x + κ x³))) / 2`, with the chain rule through the inner polynomial. Where
`TorchLean.Numeric.tanhDeriv u` is zero, the second term that the spec skips is zero as well.
-/
theorem Activation.hasDerivAt_geluTanh (x : ℝ) :
    HasDerivAt Activation.geluTanh (Activation.geluTanhDeriv x) x := by
  unfold Activation.geluTanh Activation.geluTanhDeriv
  simp only [Context.isInf_real, Bool.false_eq_true, ↓reduceIte]
  have hgate (a : ℝ) : Context.sigmoid (a + a) =
      (1 + MathFunctions.tanh a) / ((1 : ℝ) + 1) := by
    rw [Context.sigmoid_eq_exp_div, Real.exp_add, MathFunctions.tanh_real,
      Real.tanh_eq, Real.exp_neg]
    have he := Real.exp_ne_zero a
    have hd : (1 : ℝ) + Real.exp a * Real.exp a ≠ 0 := by positivity
    field_simp
    ring
  have hscale (a : ℝ) : ((1 : ℝ) + 1) * ((1 + a) / ((1 : ℝ) + 1)) = 1 + a := by
    ring
  simp_rw [hgate, hscale]
  let beta : ℝ := Activation.geluBeta
  let kappa : ℝ := Activation.geluKappa
  let half : ℝ := Rat.cast (1 / 2)
  let u : ℝ → ℝ := fun y => beta * (y + kappa * (y * y * y))
  let sech_term : ℝ := Numeric.tanhDeriv (u x)
  let inner_deriv : ℝ := beta * ((1 : ℝ) + ((1 : ℝ) + 1 + 1) * kappa * x * x)
  have hid : HasDerivAt (fun y : ℝ => y) (1 : ℝ) x := hasDerivAt_id' x
  have hpoly : HasDerivAt (fun y : ℝ => y + kappa * (y * y * y))
      ((1 : ℝ) + ((1 : ℝ) + 1 + 1) * kappa * x * x) x := by
    have h := hid.add (((hid.mul hid).mul hid).const_mul kappa)
    convert h using 1
    simp only [Pi.mul_apply]
    ring
  have hu : HasDerivAt u inner_deriv x := hpoly.const_mul beta
  have htanh : HasDerivAt (fun y : ℝ => MathFunctions.tanh (u y))
      (sech_term * inner_deriv) x := (hasDerivAt_tanh (u x)).comp x hu
  have hprod : HasDerivAt (fun y : ℝ => half * y * ((1 : ℝ) + MathFunctions.tanh (u y)))
      (half * 1 * ((1 : ℝ) + MathFunctions.tanh (u x)) +
        half * x * (sech_term * inner_deriv)) x :=
    (hid.const_mul half).mul (htanh.const_add 1)
  have hhalf : half = 1 / 2 := by norm_num [half]
  have hCorrection : (if sech_term == 0 then 0 else x * sech_term * inner_deriv) =
      x * sech_term * inner_deriv := by
    split_ifs with h
    · simp [beq_iff_eq.mp h]
    · rfl
  change HasDerivAt (fun y : ℝ => half * y * ((1 : ℝ) + MathFunctions.tanh (u y)))
    (((1 : ℝ) + MathFunctions.tanh (u x) +
      (if sech_term == 0 then 0 else x * sech_term * inner_deriv)) / ((1 : ℝ) + 1)) x
  rw [hCorrection]
  refine hprod.congr_deriv ?_
  rw [hhalf]
  ring

/-- The compensated formula agrees with `log(1 + x)` over the reals. -/
theorem Context.log1p_eq_log_one_add (x : ℝ) :
    Context.log1p x = Real.log (1 + x) := by
  simp only [Context.log1p, Numeric.log1p,
    MathFunctions.log, sub_self, beq_self_eq_true, ↓reduceIte]
  split_ifs with h₁ h₂
  · have hzero : x = 0 := by have := beq_iff_eq.mp h₁; linarith
    simp [hzero]
  · have := beq_iff_eq.mp h₂
    linarith
  · have hzero : x ≠ 0 := by intro h; simp [h] at h₁
    have hshift : (1 : ℝ) + x - 1 = x := by ring
    simp [hshift, hzero]

/-- Over the reals both branches of softplus give `log (1 + exp x)`. -/
theorem Activation.softplus_eq_log_one_add_exp (x : ℝ) :
    Activation.softplus (α := ℝ) x = Real.log (1 + Real.exp x) := by
  simp only [Activation.softplus, Context.log1p_eq_log_one_add, MathFunctions.exp]
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

/-- Softplus has derivative `Context.sigmoid x` at every real `x` (PyTorch:
`torch.nn.functional.softplus`). -/
theorem Activation.hasDerivAt_softplus (x : ℝ) :
    HasDerivAt Activation.softplus (Context.sigmoid x) x := by
  have softplus_eq_log_exp : Activation.softplus (α := ℝ) =
      fun y : ℝ => Real.log (1 + Real.exp y) := by
    funext y
    exact softplus_eq_log_one_add_exp y
  rw [softplus_eq_log_exp]
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
  -- The derivative of `log (1 + exp x)` is the ratio `exp x / (1 + exp x)`, which is sigmoid.
  have hsig : (Context.sigmoid (α := ℝ) x) = (Real.exp x) * (1 + Real.exp x)⁻¹ := by
    rw [Context.sigmoid_eq_exp_div, div_eq_mul_inv]
  simpa [MathFunctions.log, MathFunctions.exp, one_div, div_eq_mul_inv, hsig,
    mul_comm, mul_left_comm, mul_assoc, add_comm, add_left_comm, add_assoc] using h_comp

/-- Over the reals SiLU is `x * sigmoid x`: splitting the exponential on the branch for `x ≤ 0`
leaves the value unchanged. -/
theorem Activation.silu_eq_mul_sigmoid (x : ℝ) :
    Activation.silu x = x * Context.sigmoid x := by
  unfold Activation.silu
  simp only [Context.isInf_real, Bool.false_eq_true, ↓reduceIte]
  split_ifs
  · rfl
  · have hr : Real.exp (x / 2) * Real.exp (x / 2) = Real.exp x := by
      rw [← Real.exp_add]
      congr 1
      ring
    simp only [MathFunctions.exp]
    rw [mul_assoc x, hr, Context.sigmoid_eq_exp_div]
    ring

/-- Over the reals the split-exponential slope of SiLU is the product-rule derivative
`σ(x) + x σ'(x)`. -/
theorem Activation.siluDeriv_eq (x : ℝ) :
    Activation.siluDeriv x =
      Context.sigmoid x + x * Numeric.sigmoidDeriv x := by
  unfold Activation.siluDeriv
  simp only [Context.isInf_real, Bool.false_eq_true, ↓reduceIte]
  split_ifs
  · rfl
  · have hr : Real.exp (x / 2) * Real.exp (x / 2) = Real.exp x := by
      rw [← Real.exp_add]
      congr 1
      ring
    simp only [MathFunctions.exp]
    rw [mul_assoc (x + _), hr, Numeric.sigmoidDeriv_eq, Context.sigmoid_eq_exp_div]
    have hd : (1 : ℝ) + Real.exp x ≠ 0 := by positivity
    field_simp
    ring

/-- SiLU has derivative `Activation.siluDeriv x` at every real `x`.

Since `silu(x) = x * sigmoid(x)`, the proof is the product rule with `Context.hasDerivAt_sigmoid`.
The tensor-level VJP proofs build on this scalar fact.
-/
theorem Activation.hasDerivAt_silu (x : ℝ) :
    HasDerivAt Activation.silu (Activation.siluDeriv x) x := by
  rw [show Activation.silu = (fun y : ℝ => y * Context.sigmoid y)
    from funext silu_eq_mul_sigmoid, siluDeriv_eq]
  have hid : HasDerivAt (fun y : ℝ => y) (1 : ℝ) x := hasDerivAt_id' x
  have hprod : HasDerivAt (fun y : ℝ => y * Context.sigmoid y)
      (1 * Context.sigmoid x + x * Numeric.sigmoidDeriv x) x :=
    hid.mul (Context.hasDerivAt_sigmoid x)
  simpa only [one_mul] using hprod

/-- Where `softplus x + ε` is nonzero, the smooth log surrogate
`safeLog x ε = log (softplus x + ε)` has derivative `safeLogDeriv x ε`. -/
theorem Activation.hasDerivAt_safeLog_of_ne_zero (x ε : ℝ) (h : Activation.softplus x + ε ≠ 0) :
    HasDerivAt (fun y => Activation.safeLog y ε) (Activation.safeLogDeriv x ε) x := by
  unfold Activation.safeLog Activation.safeLogDeriv
  have h_inner : HasDerivAt (fun y : ℝ => Activation.softplus y + ε)
    (Context.sigmoid x) x := by
    simpa using (hasDerivAt_softplus x).const_add ε
  have h_comp := (Real.hasDerivAt_log h).comp x h_inner
  simpa [Function.comp_def, MathFunctions.log, one_div,
    div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using h_comp

/--
For `0 < ε`, the smooth log surrogate `Activation.safeLog x ε = log (softplus x + ε)` has derivative
`Activation.safeLogDeriv x ε`.

PyTorch has no such primitive. With `softplus` nonnegative and `0 < ε`, the logarithm's argument
is positive at every real `x`, so the surrogate needs no strict-positivity side condition on `x`.

Related PyTorch primitives:
https://pytorch.org/docs/stable/generated/torch.log.html
-/
theorem Activation.hasDerivAt_safeLog (x ε : ℝ) (hε : 0 < ε) :
    HasDerivAt (fun y => Activation.safeLog y ε) (Activation.safeLogDeriv x
      ε) x := by
  refine hasDerivAt_safeLog_of_ne_zero x ε (ne_of_gt ?_)
  -- `softplus(x) = log(1+exp x) ≥ 0` and `ε > 0`
  have h1 : (1 : ℝ) ≤ 1 + Real.exp x := by linarith [Real.exp_pos x]
  rw [softplus_eq_log_one_add_exp]
  linarith [Real.log_nonneg h1]

/-- The scaled evaluation of the smooth absolute value equals `sqrt (x * x + ε)` over the reals,
for every `ε`. -/
theorem Activation.smoothAbs_eq_sqrt (x ε : ℝ) :
    Activation.smoothAbs x ε = Real.sqrt (x * x + ε) := by
  let scale := max |x| (Real.sqrt |ε|)
  simp only [Activation.smoothAbs, MathFunctions.abs, MathFunctions.sqrt]
  change (if scale == 0 || !(scale / scale == scale / scale) then Real.sqrt (x * x + ε)
    else scale * Real.sqrt ((x / scale) * (x / scale) + (ε / scale) / scale)) = _
  by_cases hne : scale = 0
  · simp [hne]
  have hscale : 0 < scale := lt_of_le_of_ne (le_max_of_le_left (abs_nonneg x)) (Ne.symm hne)
  have hinner : (x / scale) * (x / scale) + (ε / scale) / scale =
      (x * x + ε) / (scale * scale) := by
    field_simp
  simp only [show (scale == 0) = false by simp [hne], div_self hne, beq_self_eq_true,
    Bool.not_true, Bool.false_or, Bool.false_eq_true, ↓reduceIte]
  rw [hinner, Real.sqrt_div' _ (mul_self_nonneg scale), Real.sqrt_mul_self hscale.le]
  field_simp [hne]

/-- Where `x * x + ε` is nonzero, the smooth absolute value `√(x * x + ε)` has derivative
`smoothAbsDeriv x ε = x / √(x * x + ε)`, which is zero where `x * x + ε` is negative. -/
theorem Activation.hasDerivAt_smoothAbs_of_ne_zero (x ε : ℝ) (h : x * x + ε ≠ 0) :
    HasDerivAt (fun y => Activation.smoothAbs y ε) (Activation.smoothAbsDeriv x ε) x := by
  simp_rw [Activation.smoothAbsDeriv, smoothAbs_eq_sqrt _ ε]
  have h_inner : HasDerivAt (fun y : ℝ => y * y + ε) (2 * x) x := by
    simpa using (hasDerivAt_mul_self x).const_add ε
  exact (h_inner.sqrt h).congr_deriv (mul_div_mul_left x _ two_ne_zero)

/-- For positive `ε`, `sqrt(x² + ε)` is differentiable with derivative `x / sqrt(x² + ε)`. -/
theorem Activation.hasDerivAt_smoothAbs (x ε : ℝ) (hε : 0 < ε) :
    HasDerivAt (fun y => Activation.smoothAbs y ε) (Activation.smoothAbsDeriv
      x ε) x :=
  hasDerivAt_smoothAbs_of_ne_zero x ε (add_pos_of_nonneg_of_pos (mul_self_nonneg x) hε).ne'

end TorchLean
