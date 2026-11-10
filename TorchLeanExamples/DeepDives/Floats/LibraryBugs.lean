/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Constants
public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.RuntimeApprox.NF.CROWN
public import TorchLean.Verify.NumericalCertificate
public import TorchLean.Verify.Floats.IEEE754.Expressions
public import TorchLean.Verify.RuntimeApprox.NF.Attention
public import TorchLean.Verify.RuntimeApprox.NF.Convolution
public import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
public import TorchLean.Verify.RuntimeApprox.NF.Normalization
public import TorchLean.Verify.RuntimeApprox.NF.Optimizers
public import TorchLean.Verify.RuntimeApprox.NF.ReductionOps
public import TorchLean.Verify.Floats.IEEE754.Reductions
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Autograd -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxShift
public import TorchLean.Verify.RuntimeApprox.TransformerBlock
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Layers.Activation
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
public import FloatLib.Numerics.MixedError.Basic
import TorchLean.NN.Gradients.Activation
public import FloatLib.Numerics.Automation.Attributes
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init
public import TorchLean.Verify.RuntimeApprox.Roundoff

/-!
# Rounding bugs in other libraries' formulas

Nine sections take formulas from PyTorch, JAX, Hugging Face Transformers, the
reinforcement-learning libraries TRL and Stable-Baselines3, and diffusers, and write them with
TorchLean's scalar operations. Each has an input on which the kernel checks that the formula fails
in an IEEE format, and a formula with the same real value (for Stable-Baselines3's term, the value
without its epsilon) that does not fail there. All but `EMAStall` also state mixed roundoff bounds
in every IEEE format for the computation in which the two differ.

* `SoftMargin`: PyTorch's `soft_margin_loss` computes `log1p (exp (-y x))` and its eager
  backward `-y z / (1 + z)` with `z = exp (-y x)`. At the margin `-12` the exponential overflows
  in binary16 (at `-100` in binary32), so the loss is not finite and the gradient is NaN.
  Softplus of `-y x` and the sigmoid gradient stay finite.
* `NormalKL`: `torch.distributions.kl_divergence` of two normal distributions computes
  `(r + t - 1 - log r) / 2`. Adding the squared mean difference `t` to the variance ratio `r`
  before subtracting one rounds `t` away: in binary32 a mean shift of `2⁻¹³` gives zero where the
  divergence is `2⁻²⁷`. Its one-dimensional `MultivariateNormal` formula is negative at a scale
  ratio of `1 + 2⁻¹³`. Computing `r - 1` from `sp - sq` and adding `t` last keeps the squared mean
  difference.
* `GeluTanh`: the tanh approximation of GELU, `x (1 + tanh z) / 2`, is the default of
  `jax.nn.gelu` and is PyTorch's `gelu(approximate = "tanh")` and Transformers' `gelu_new`.
  Once `tanh z` rounds to `-1` it returns zero: at `x = -6` in binary32 and at `x = -4` in
  bfloat16. `ActivationBugs.GeluTanh.gelu` computes `x * sigmoid (2 z)`, the same function.
  TorchLean's `TorchLean.Activation.geluTanh` keeps PyTorch's order of operations, and with it the
  zero.
* `LogitClamp`: `torch.special.logit x eps` clamps into `[eps, 1 - eps]` with `1 - eps` rounded.
  In binary32 at `eps = 1e-8` that bound rounds to one, so `logit 1` is not finite where it should
  be `log ((1 - eps) / eps)`. Taking the upper clamp's value as `log1p (-eps) - log eps` keeps it.
* `KLEstimator`: the k3 estimator `exp d - d - 1` of the KL penalty in TRL's GRPO loss (verl's
  has the same order with clamps, OpenRLHF's subtracts one first), with `d` the log-ratio of the
  reference and policy log-probabilities of a token. Once `d` is small, the absolute rounding
  error of `exp d`, about the unit roundoff, exceeds the estimator, about `d² / 2`: at
  `d = 2⁻¹²` in binary32 the estimator decodes to zero. `expm1 d - d` keeps it.
* `TanhSquash`: Stable-Baselines3 and CleanRL take the change-of-variables term of a tanh-squashed
  Gaussian action as `log (1 - tanh² u + 1e-6)`. Once `tanh u` rounds to one the term is
  `log 1e-6`: at `u = 10` in binary32 about `-13.8`, where the exact term is about `-18.6`.
  `2 (log 2 - u - softplus (-2 u))`, the form of `torch.distributions.TanhTransform`, is within
  relative error twice the unit roundoff there. Most of the library's deviation is its epsilon, a
  documented safeguard: at `u = 10` the exact `1 - tanh² u` is about `8.2e-9`, far below `1e-6`.
* `LabelSmoothing`: PyTorch's label-smoothed `cross_entropy` sums the negated log-probabilities
  over the classes and then over the batch in the logits' format before dividing. In a binary16
  example every log-probability and the exact term are representable, but the batch sum exceeds
  the largest finite value, so the computed term is not finite. Dividing before each addition
  gives the exact term there (dividing first can underflow elsewhere).
* `EMAStall`: `torch.optim.swa_utils.get_ema_multi_avg_fn` keeps the average in the parameters'
  format. In a scalar model of its update in bfloat16, a step with weight `1 / 1000` toward a target
  ten units in the last place away rounds back to the average itself. The average never moves, while
  the exact average covers most of the distance. In binary32 the same update moves it, so the stall
  comes from the state's format.
* `NoiseLevel`: diffusers' `DDPMScheduler.add_noise` rounds its float32 `alpha_bar`, the product
  of `1 - beta`, to the samples' format before forming the noise level `sqrt (1 - alpha_bar)`. At
  the second timestep of the default linear schedule `alpha_bar` rounds to one in binary16, so the
  noise level is zero where it should be about `0.0148`. `sqrt (-expm1 (Σ log1p (-beta)))` never
  forms `alpha_bar` and keeps it.

The real identities, kernel-checked counterexamples and conditional mixed bounds are about the
scalar formulas written here. They do not show that the installed library kernels execute exactly
these operations.
-/

@[expose] public section

namespace TorchLean.Examples

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics

/-! ## The soft-margin loss -/

namespace LossBugs.SoftMargin

section Formulas

variable {α : Type} [Storage α] [Context α]

/-- PyTorch's `soft_margin_loss` of one prediction `x` and label `y`: `log1p (exp (-x y))`. -/
def libraryLoss (x y : α) : α := Context.log1p (MathFunctions.exp (-x * y))

/-- PyTorch's eager `soft_margin_loss_backward` of one element: `-y z / (1 + z)` with
`z = exp (-y x)`. -/
def libraryLossBackward (x y : α) : α :=
  let z := MathFunctions.exp (-y * x); -(y * z) / (1 + z)

/-- The soft-margin loss as softplus of `-y x`, the negated margin. -/
def loss (x y : α) : α := Activation.softplus (-(y * x))

/-- The derivative of `loss` in `x`: `-y sigmoid (-y x)`. -/
def lossBackward (x y : α) : α := -(y * Context.sigmoid (-(y * x)))

end Formulas

/-- Over the reals the loss is PyTorch's. -/
theorem loss_real (x y : ℝ) : loss x y = libraryLoss x y := by
  simp only [loss, libraryLoss, Activation.softplus]
  rw [show -x * y = -(y * x) by ring]
  have hlog (v : ℝ) : Context.log1p v = Real.log (1 + v) := Context.log1p_eq_log_one_add v
  simp only [hlog, MathFunctions.exp_real]
  split_ifs with hm
  · have hsplit : 1 + Real.exp (-(y * x)) =
        Real.exp (-(y * x)) * (1 + Real.exp (-(-(y * x)))) := by
      rw [mul_add, mul_one, ← Real.exp_add]
      simp [add_comm]
    rw [hsplit, Real.log_mul (Real.exp_ne_zero _) (by positivity), Real.log_exp]
  · rfl

/-- Over the reals the gradient is PyTorch's. -/
theorem lossBackward_real (x y : ℝ) : lossBackward x y = libraryLossBackward x y := by
  simp only [lossBackward, libraryLossBackward, Context.sigmoid_eq_inv_exp,
    MathFunctions.exp_real, neg_neg]
  rw [show -y * x = -(y * x) by ring]
  have hz : Real.exp (-(y * x)) * Real.exp (y * x) = 1 := by
    rw [← Real.exp_add]; simp
  have h₁ : (1 : ℝ) + Real.exp (y * x) ≠ 0 := by positivity
  have h₂ : (1 : ℝ) + Real.exp (-(y * x)) ≠ 0 := by positivity
  field_simp
  linear_combination y * hz

/-- `log (1 + exp m)` exceeds `m` by at most `exp (-m)`. -/
private theorem log_one_add_exp_mem {m : ℝ} :
    m ≤ Real.log (1 + Real.exp m) ∧ Real.log (1 + Real.exp m) ≤ m + Real.exp (-m) := by
  have hsplit : 1 + Real.exp m = Real.exp m * (1 + Real.exp (-m)) := by
    rw [mul_add, mul_one, ← Real.exp_add]; simp [add_comm]
  rw [hsplit, Real.log_mul (Real.exp_ne_zero _) (by positivity), Real.log_exp]
  have hpos : 0 < 1 + Real.exp (-m) := by positivity
  have hone : (1 : ℝ) ≤ 1 + Real.exp (-m) := by linarith [Real.exp_pos (-m)]
  exact ⟨by linarith [Real.log_nonneg hone], by linarith [Real.log_le_sub_one_of_pos hpos]⟩

/-- `exp (-12) ≤ 1 / 256`. -/
private theorem exp_neg_twelve_le : Real.exp (-12) ≤ 1 / 256 := by
  have h3 : (4 : ℝ) ≤ Real.exp 3 := by linarith [Real.add_one_le_exp (3 : ℝ)]
  have h12 : Real.exp 12 = Real.exp 3 ^ 4 := by rw [← Real.exp_nat_mul]; norm_num
  have hbig : (256 : ℝ) ≤ Real.exp 12 := by
    rw [h12]; nlinarith [pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 4) h3 4]
  rw [Real.exp_neg, inv_eq_one_div, div_le_div_iff₀ (Real.exp_pos 12) (by norm_num)]
  linarith

/-! ## Binary16 at the margin `-12` -/

/-- PyTorch's loss of `x = -12` with label one is not finite in binary16, and its gradient is
NaN. -/
theorem binary16_library :
    ExecFloat.Binary.isFinite (libraryLoss (-12 : Binary16) 1) = false ∧
      ExecFloat.Binary.isNaN (libraryLossBackward (-12 : Binary16) 1) = true := by
  constructor <;> decide +kernel

/-- `loss` decodes to `12`, within one unit of binary16 roundoff of the exact loss
`log (1 + exp 12)`, and its gradient decodes to `-1`. -/
theorem binary16_stable :
    ExecFloat.Binary.toRat? (loss (-12 : Binary16) 1) = some 12 ∧
      MixedError (ExecFloat.Binary.toReal (loss (-12 : Binary16) 1))
        (Real.log (1 + Real.exp 12)) (Model.unitRoundoffAt FloatFormat.binary16) 0 ∧
      ExecFloat.Binary.toRat? (lossBackward (-12 : Binary16) 1) = some (-1) := by
  have hvalue : ExecFloat.Binary.toRat? (loss (-12 : Binary16) 1) = some 12 := by
    decide +kernel
  refine ⟨hvalue, ?_, by decide +kernel⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  obtain ⟨hlo, hhi⟩ := log_one_add_exp_mem (m := 12)
  have hexp := exp_neg_twelve_le
  refine MixedError.of_abs_sub_le_mul (Model.unitRoundoffAt_pos _).le ?_
  rw [abs_of_pos (by linarith : (0 : ℝ) < Real.log (1 + Real.exp 12)), abs_le]
  norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary16] at hlo hhi ⊢
  constructor <;> nlinarith

/-! ## Binary32 at the margin `-100` -/

/-- PyTorch's loss of `x = -100` with label one is not finite in binary32, and its gradient is
NaN. -/
theorem binary32_library :
    ExecFloat.Binary.isFinite (libraryLoss (-100 : Binary32) 1) = false ∧
      ExecFloat.Binary.isNaN (libraryLossBackward (-100 : Binary32) 1) = true := by
  constructor <;> decide +kernel

/-- `loss` decodes to `100`, and its gradient to `-1`. -/
theorem binary32_stable :
    ExecFloat.Binary.toRat? (loss (-100 : Binary32) 1) = some 100 ∧
      ExecFloat.Binary.toRat? (lossBackward (-100 : Binary32) 1) = some (-1) := by
  constructor <;> decide +kernel

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- PyTorch's loss is finite with a mixed bound under a generated condition that `exp (-y x)` stays
below the overflow threshold. -/
roundoff mixed theorem library_loss_error :=
  fun (x y : Value) (_hformat : format.isIEEE = true)
    (_hx : ExecFloat.Binary.isFinite x = true) (_hy : ExecFloat.Binary.isFinite y = true) =>
    ExecFloat.Binary.toReal (libraryLoss x y)

/-- `loss` is finite with a mixed bound. Each branch of `softplus` exponentiates its argument
with the sign that makes it nonpositive, and the generated overflow condition is on that
exponential. -/
roundoff mixed theorem loss_error :=
  fun [Storage Value] (x y : Value) (_hformat : format.isIEEE = true)
    (_hx : ExecFloat.Binary.isFinite x = true) (_hy : ExecFloat.Binary.isFinite y = true) =>
    ExecFloat.Binary.toReal (loss x y)

end Roundoff

end LossBugs.SoftMargin

/-! ## The KL divergence of two normal distributions -/

namespace ProbBugs.NormalKL

section Formulas

variable {α : Type} [Context α]

/-- `torch.distributions.kl_divergence` of `N(mp, sp²)` from `N(mq, sq²)`:
`(r + t - 1 - log r) / 2` with `r = (sp / sq)²` and `t = ((mp - mq) / sq)²`. -/
def libraryKL (mp sp mq sq : α) : α :=
  let r := (sp / sq) * (sp / sq)
  let t := ((mp - mq) / sq) * ((mp - mq) / sq)
  (1 / 2 : α) * (r + t - 1 - MathFunctions.log r)

/-- `torch.distributions.kl_divergence` of two one-dimensional `MultivariateNormal`s with scale
factors `sp` and `sq`: `(log sq - log sp) + (x² + m² - 1) / 2` with `x = sp / sq` and
`m = (mq - mp) / sq`. -/
def libraryMultivariateKL (mp sp mq sq : α) : α :=
  let x := sp / sq
  let m := (mq - mp) / sq
  (MathFunctions.log sq - MathFunctions.log sp) + (1 / 2 : α) * (x * x + m * m - 1)

/-- The same divergence with `r - 1 = (sp - sq)(sp + sq) / sq²` and
`log r = 2 log1p ((sp - sq) / sq)`, and `t` added last. -/
def kl (mp sp mq sq : α) : α :=
  let d := sp - sq
  let m := (mp - mq) / sq
  (1 / 2 : α) * (m * m + (d * (sp + sq) / (sq * sq) - (1 + 1) * Context.log1p (d / sq)))

end Formulas

/-- The divergence of `N(mp, sp²)` from `N(mq, sq²)`. -/
noncomputable def exact (mp sp mq sq : ℝ) : ℝ :=
  Real.log (sq / sp) + (sp ^ 2 + (mp - mq) ^ 2) / (2 * sq ^ 2) - 1 / 2

/-- Over the reals PyTorch's formula is the divergence. -/
theorem library_real {mp sp mq sq : ℝ} (hp : 0 < sp) (hq : 0 < sq) :
    libraryKL mp sp mq sq = exact mp sp mq sq := by
  simp only [libraryKL, exact, MathFunctions.log]
  rw [Real.log_mul (by positivity) (by positivity), Real.log_div hp.ne' hq.ne',
    Real.log_div hq.ne' hp.ne']
  field_simp
  ring

/-- Over the reals `kl` is the divergence. -/
theorem kl_real {mp sp mq sq : ℝ} (hp : 0 < sp) (hq : 0 < sq) :
    kl mp sp mq sq = exact mp sp mq sq := by
  have hlog : Context.log1p ((sp - sq) / sq) = Real.log sp - Real.log sq := by
    rw [Context.log1p_eq_log_one_add, show 1 + (sp - sq) / sq = sp / sq by field_simp; ring,
      Real.log_div hp.ne' hq.ne']
  simp only [kl, exact, hlog]
  rw [Real.log_div hq.ne' hp.ne']
  field_simp
  ring

/-- Over the reals PyTorch's multivariate formula is the divergence. -/
theorem multivariate_real {mp sp mq sq : ℝ} (hp : 0 < sp) (hq : 0 < sq) :
    libraryMultivariateKL mp sp mq sq = exact mp sp mq sq := by
  simp only [libraryMultivariateKL, exact, MathFunctions.log]
  rw [Real.log_div hq.ne' hp.ne']
  field_simp
  ring

/-- The divergence of two normal distributions with different scales is positive. -/
theorem exact_pos {mp sp mq sq : ℝ} (hp : 0 < sp) (hq : 0 < sq) (hne : sp ≠ sq) :
    0 < exact mp sp mq sq := by
  have hr : 0 < (sp / sq) * (sp / sq) := by positivity
  have hr₁ : (sp / sq) * (sp / sq) ≠ 1 := by
    intro h
    have hx : sp / sq = 1 := by nlinarith [div_pos hp hq]
    exact hne ((div_eq_one_iff_eq hq.ne').mp hx)
  have hlog := Real.log_lt_sub_one_of_pos hr hr₁
  rw [← library_real hp hq]
  simp only [libraryKL, MathFunctions.log]
  nlinarith [mul_self_nonneg ((mp - mq) / sq)]

/-- A negative value is not within relative error `r ≤ 1` of a positive one. -/
private theorem not_mixedError_of_neg {x y r : ℝ} (hx : x < 0) (hy : 0 < y) (hr : r ≤ 1) :
    ¬ MixedError x y r 0 := fun h => by
  have hbound := h.bound
  rw [abs_of_neg (by linarith), abs_of_pos hy, add_zero] at hbound
  nlinarith

/-! ## Binary32 at a mean shift of `2⁻¹³` -/

/-- PyTorch's divergence of `N(0, 1)` from `N(2⁻¹³, 1)` decodes to zero in binary32, which is not
within any relative error below one of the divergence `2⁻²⁷`. -/
theorem binary32_library :
    ExecFloat.Binary.toRat? (libraryKL (0 : Binary32) 1 (1 / 8192) 1) = some 0 ∧
      ∀ r < 1, ¬ MixedError (ExecFloat.Binary.toReal (libraryKL (0 : Binary32) 1 (1 / 8192) 1))
        (exact 0 1 (1 / 8192) 1) r 0 := by
  have hvalue : ExecFloat.Binary.toRat? (libraryKL (0 : Binary32) 1 (1 / 8192) 1) = some 0 := by
    decide +kernel
  refine ⟨hvalue, fun r hr => ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue, Rat.cast_zero]
  exact fun h => h.ne_zero_of_ne_zero hr (by norm_num [exact]) rfl

/-- `kl` decodes to `2⁻²⁷`, the divergence itself. -/
theorem binary32_stable :
    ExecFloat.Binary.toRat? (kl (0 : Binary32) 1 (1 / 8192) 1) = some (1 / 2 ^ 27) ∧
      ExecFloat.Binary.toReal (kl (0 : Binary32) 1 (1 / 8192) 1) = exact 0 1 (1 / 8192) 1 := by
  have hvalue : ExecFloat.Binary.toRat? (kl (0 : Binary32) 1 (1 / 8192) 1) = some (1 / 2 ^ 27) := by
    decide +kernel
  refine ⟨hvalue, ?_⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  norm_num [exact]

/-! ## Binary32 at a scale ratio of `1 + 2⁻¹³`, in one dimension -/

/-- `1 + 2⁻¹³` in binary32. -/
def binary32Scale : Binary32 := 1 + 1 / 8192

/-- `binary32Scale` decodes to `8193 / 8192`. -/
theorem binary32Scale_toRat? : ExecFloat.Binary.toRat? binary32Scale = some (8193 / 8192) := by
  decide +kernel

/-- PyTorch's one-dimensional `MultivariateNormal` divergence of `N(0, 1)` from
`N(0, (1 + 2⁻¹³)²)` decodes to `-2⁻²⁷` in binary32, of the wrong sign, while `kl` decodes to
`2047 / 2³⁷`, near the divergence. -/
theorem binary32_multivariate :
    ExecFloat.Binary.toRat? (libraryMultivariateKL (0 : Binary32) 1 0 binary32Scale) =
        some (-1 / 2 ^ 27) ∧
      (∀ r ≤ 1, ¬ MixedError
        (ExecFloat.Binary.toReal (libraryMultivariateKL (0 : Binary32) 1 0 binary32Scale))
        (exact 0 1 0 (8193 / 8192)) r 0) ∧
      ExecFloat.Binary.toRat? (kl (0 : Binary32) 1 0 binary32Scale) = some (2047 / 2 ^ 37) := by
  have hvalue : ExecFloat.Binary.toRat? (libraryMultivariateKL (0 : Binary32) 1 0 binary32Scale) =
      some (-1 / 2 ^ 27) := by
    decide +kernel
  refine ⟨hvalue, fun r hr => ?_, by decide +kernel⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue]
  exact not_mixedError_of_neg (by norm_num) (exact_pos (by norm_num) (by norm_num) (by norm_num)) hr

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- PyTorch's formula: the absolute part of the bound contains the unit roundoff times `|r + t|`,
which is one when the two distributions coincide. -/
roundoff mixed theorem library_error :=
  fun (mp sp mq sq : Value) (_hformat : format.isIEEE = true)
    (_h₁ : ExecFloat.Binary.isFinite mp = true) (_h₂ : ExecFloat.Binary.isFinite sp = true)
    (_h₃ : ExecFloat.Binary.isFinite mq = true) (_h₄ : ExecFloat.Binary.isFinite sq = true)
    (_hp : 0 < ExecFloat.Binary.toReal sp) (_hq : 0 < ExecFloat.Binary.toReal sq) =>
    ExecFloat.Binary.toReal (libraryKL mp sp mq sq)

/-- `kl`: besides underflow, the bound is the unit roundoff times multiples of `((mp - mq) / sq)²`,
`|r - 1|` and `|log r|`, which vanish when the two distributions coincide. -/
roundoff mixed theorem kl_error :=
  fun (mp sp mq sq : Value) (_hformat : format.isIEEE = true)
    (_h₁ : ExecFloat.Binary.isFinite mp = true) (_h₂ : ExecFloat.Binary.isFinite sp = true)
    (_h₃ : ExecFloat.Binary.isFinite mq = true) (_h₄ : ExecFloat.Binary.isFinite sq = true)
    (_hp : 0 < ExecFloat.Binary.toReal sp) (_hq : 0 < ExecFloat.Binary.toReal sq) =>
    ExecFloat.Binary.toReal (kl mp sp mq sq)

end Roundoff

end ProbBugs.NormalKL

/-! ## The tanh approximation of GELU -/

namespace ActivationBugs.GeluTanh

open Activation (geluKappa)

section Formulas

variable {α : Type} [Storage α] [Context α]

/-- `jax.nn.gelu` with its default `approximate = True`: `x * (0.5 * (1 + tanh (c (x + k x³))))`
with `k = 0.044715`, and `c = √(2 / π)` computed in the format. -/
def libraryGelu (x : α) : α :=
  let c : α := MathFunctions.sqrt ((1 + 1) / MathFunctions.pi)
  x * ((1 / 2 : α) * (1 + MathFunctions.tanh (c * (x + geluKappa * (x * x * x)))))

/-- The same approximation as `x * sigmoid (2 c (x + k x³))`, with `c = √(2 / π)` computed in the
format. -/
def gelu (x : α) : α :=
  let c : α := MathFunctions.sqrt ((1 + 1) / MathFunctions.pi)
  let inner := c * (x + geluKappa * x * x * x)
  x * Context.sigmoid (inner + inner)

end Formulas

/-- The tanh approximation of GELU over the reals, `x * sigmoid (2 c (x + k x³))`. -/
noncomputable def exact (x : ℝ) : ℝ :=
  x / (1 + Real.exp (-(2 * (Real.sqrt (2 / Real.pi) * (x + geluKappa * x ^ 3)))))

/-- `sigmoid (2 a) = (1 + tanh a) / 2` over the reals. -/
private theorem sigmoid_two_mul (a : ℝ) :
    Context.sigmoid (a + a) = (1 + Real.tanh a) / 2 := by
  rw [Context.sigmoid_eq_exp_div, Real.exp_add, Real.tanh_eq, Real.exp_neg]
  have he := Real.exp_ne_zero a
  have hd : (1 : ℝ) + Real.exp a * Real.exp a ≠ 0 := by positivity
  field_simp
  ring

/-- Over the reals JAX's formula and `gelu` are the tanh approximation. -/
theorem library_real (x : ℝ) : libraryGelu x = exact x ∧ gelu x = exact x := by
  have hpi : (MathFunctions.pi : ℝ) = Real.pi := rfl
  have hinner : Real.sqrt (2 / Real.pi) * (x + geluKappa * x ^ 3) =
      Real.sqrt (2 / Real.pi) * (x + geluKappa * x * x * x) := by ring
  have hexact : exact x = x * Context.sigmoid
      (Real.sqrt (2 / Real.pi) * (x + geluKappa * x ^ 3) +
        Real.sqrt (2 / Real.pi) * (x + geluKappa * x ^ 3)) := by
    rw [Context.sigmoid_eq_inv_exp, exact, div_eq_mul_inv, two_mul]
  refine ⟨?_, ?_⟩
  · rw [hexact, sigmoid_two_mul]
    simp only [libraryGelu, hpi, MathFunctions.tanh_real, MathFunctions.sqrt_real]
    rw [show (1 + 1 : ℝ) = 2 by norm_num, show x * x * x = x ^ 3 by ring]
    ring
  · rw [hexact, hinner]
    simp only [gelu, hpi, MathFunctions.sqrt]
    norm_num

/-- The tanh approximation is negative at every negative input. -/
theorem exact_neg {x : ℝ} (hx : x < 0) : exact x < 0 :=
  div_neg_of_neg_of_pos hx (by positivity)

/-! ## Binary32 at `x = -6` and bfloat16 at `x = -4` -/

/-- JAX's formula decodes to zero at `x = -6` in binary32, which is not within any relative error
below one of the negative exact value. `gelu` decodes to `-3040709 / 2⁵⁵`, and PyTorch's formula
`Activation.geluTanh` to zero. -/
theorem binary32 :
    ExecFloat.Binary.toRat? (libraryGelu (-6 : Binary32)) = some 0 ∧
      (∀ r < 1, ¬ MixedError (ExecFloat.Binary.toReal (libraryGelu (-6 : Binary32)))
        (exact (-6)) r 0) ∧
      ExecFloat.Binary.toRat? (gelu (-6 : Binary32)) =
        some (-3040709 / 2 ^ 55) ∧
      ExecFloat.Binary.toRat? (Activation.geluTanh (-6 : Binary32)) = some 0 := by
  have hvalue : ExecFloat.Binary.toRat? (libraryGelu (-6 : Binary32)) = some 0 := by
    decide +kernel
  refine ⟨hvalue, fun r hr => ?_, by decide +kernel, by decide +kernel⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue, Rat.cast_zero]
  exact fun h => h.ne_zero_of_ne_zero hr (exact_neg (by norm_num)).ne rfl

/-- In bfloat16, rounding after every operation, JAX's formula decodes to zero at `x = -4`, while
`gelu` decodes to `-149 / 2²¹`. -/
theorem bfloat16 :
    ExecFloat.Binary.toRat? (libraryGelu (-4 : BFloat16)) = some 0 ∧
      (∀ r < 1, ¬ MixedError (ExecFloat.Binary.toReal (libraryGelu (-4 : BFloat16)))
        (exact (-4)) r 0) ∧
      ExecFloat.Binary.toRat? (gelu (-4 : BFloat16)) =
        some (-149 / 2 ^ 21) := by
  have hvalue : ExecFloat.Binary.toRat? (libraryGelu (-4 : BFloat16)) = some 0 := by
    decide +kernel
  refine ⟨hvalue, fun r hr => ?_, by decide +kernel⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hvalue, Rat.cast_zero]
  exact fun h => h.ne_zero_of_ne_zero hr (exact_neg (by norm_num)).ne rfl

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The gate `(1 + tanh z) / 2` of JAX's formula has a mixed bound whose absolute part carries the
error of the computed `tanh z` itself. -/
roundoff mixed theorem tanh_gate_error :=
  fun (z : Value) (_hformat : format.isIEEE = true) (_hz : ExecFloat.Binary.isFinite z = true) =>
    ExecFloat.Binary.toReal ((1 / 2 : Value) * (1 + MathFunctions.tanh z))

/-- The gate `sigmoid (z + z)` of `gelu` has a mixed bound whose absolute part
is the underflow error and the rounding of its own quotient. -/
roundoff mixed theorem gate_error :=
  fun [Storage Value] (z : Value) (_hformat : format.isIEEE = true)
    (_hz : ExecFloat.Binary.isFinite z = true) =>
    ExecFloat.Binary.toReal (Context.sigmoid (z + z))

end Roundoff

end ActivationBugs.GeluTanh

/-! ## The clamp of `torch.special.logit` -/

namespace ActivationBugs.LogitClamp

section Formulas

variable {α : Type} [Context α]

/-- `torch.special.logit x eps`: `log (v / (1 - v))` for `v` clamped into `[eps, 1 - eps]`, with
`1 - eps` rounded. -/
def libraryLogit (x eps : α) : α :=
  let v := if x < eps then eps else if x > 1 - eps then 1 - eps else x
  MathFunctions.log (v / (1 - v))

/-- The same logit with the clamped values `log eps - log1p (-eps)` and `log1p (-eps) - log eps`,
and the upper clamp tested as `1 - x < eps`. -/
def logit (x eps : α) : α :=
  if x < eps then MathFunctions.log eps - Context.log1p (-eps)
  else if 1 - x < eps then Context.log1p (-eps) - MathFunctions.log eps
  else MathFunctions.log (x / (1 - x))

end Formulas

/-- Over the reals, for `0 < eps < 1`, `logit` is `torch.special.logit`. -/
theorem logit_real {x eps : ℝ} (h₀ : 0 < eps) (h₁ : eps < 1) :
    logit x eps = libraryLogit x eps := by
  have hlog : Context.log1p (-eps) = Real.log (1 - eps) := by
    rw [Context.log1p_eq_log_one_add, ← sub_eq_add_neg]
  simp only [logit, libraryLogit, hlog, MathFunctions.log]
  have hc : (0 : ℝ) < 1 - eps := by linarith
  by_cases ha : x < eps
  · simp only [ite_eq_left ha]
    rw [Real.log_div h₀.ne' hc.ne']
  · by_cases hb : 1 - x < eps
    · have hb' : x > 1 - eps := by linarith
      simp only [ite_eq_right ha, ite_eq_left hb, ite_eq_left hb']
      rw [show 1 - (1 - eps) = eps by ring, Real.log_div hc.ne' h₀.ne']
    · have hb' : ¬ x > 1 - eps := by
        push Not at hb ⊢
        linarith
      simp only [ite_eq_right ha, ite_eq_right hb, ite_eq_right hb']

/-- `1e-8` rounded to binary32. -/
def binary32Epsilon : Binary32 := 1 / (10 ^ 8 : ℕ)

/-- PyTorch's logit of `1` at `eps = 1e-8` is not finite in binary32, while `logit` decodes to
`4828871 / 2¹⁸`, about `18.42`. -/
theorem binary32 :
    ExecFloat.Binary.isFinite (libraryLogit (1 : Binary32) binary32Epsilon) = false ∧
      ExecFloat.Binary.toRat? (logit (1 : Binary32) binary32Epsilon) = some (4828871 / 2 ^ 18) := by
  constructor <;> decide +kernel

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- At the upper clamp `logit` is finite under the generated range conditions, with relative part
`u` and absolute part `1 + u` times `u (|log (1 - eps)| + |log eps|)` plus twice the underflow
error, against `log (1 - eps) - log eps`. -/
roundoff mixed theorem clamp_error :=
  fun (eps : Value) (_hformat : format.isIEEE = true)
    (_heps : ExecFloat.Binary.isFinite eps = true) (_h₀ : 0 < ExecFloat.Binary.toReal eps) =>
    ExecFloat.Binary.toReal (Context.log1p (-eps) - MathFunctions.log eps)

end Roundoff

end ActivationBugs.LogitClamp

/-! ## The k3 estimator of the KL penalty -/

namespace LossBugs.KLEstimator

section Formulas

variable {α : Type} [Context α]

/-- The k3 estimator of TRL's GRPO loss, `exp (lr - lp) - (lr - lp) - 1`, of the policy and
reference log-probabilities `lp` and `lr` of one token. -/
def libraryK3 (lp lr : α) : α := MathFunctions.exp (lr - lp) - (lr - lp) - 1

/-- The same estimator as `expm1 (lr - lp) - (lr - lp)`. -/
def k3 (lp lr : α) : α := Context.expm1 (lr - lp) - (lr - lp)

end Formulas

/-- The k3 estimator at the log-ratio `d`, `exp d - 1 - d`. -/
noncomputable def exact (d : ℝ) : ℝ := Real.exp d - 1 - d

/-- Over the reals both formulas are `exact (lr - lp)`. -/
theorem k3_real (lp lr : ℝ) : libraryK3 lp lr = exact (lr - lp) ∧ k3 lp lr = exact (lr - lp) := by
  refine ⟨by simp only [libraryK3, exact, MathFunctions.exp_real]; ring, ?_⟩
  rw [k3, Context.expm1_eq_exp_sub_one, exact]

/-- The estimator is positive at every nonzero log-ratio. -/
theorem exact_pos {d : ℝ} (hd : d ≠ 0) : 0 < exact d := by
  have := Real.add_one_lt_exp hd
  unfold exact
  linarith

/-! ## Binary32 at a log-ratio of `2⁻¹²` -/

/-- The reference log-probability `-1 + 2⁻¹²` decodes to `-4095 / 4096`, so the log-ratio against
`-1` is `2⁻¹²`. -/
theorem binary32_input :
    ExecFloat.Binary.toRat? (-1 + 1 / 4096 : Binary32) = some (-4095 / 4096) := by
  decide +kernel

/-- TRL's estimator decodes to zero at `lp = -1`, `lr = -1 + 2⁻¹²` in binary32, which is not
within any relative error below one of the positive estimator. `k3` decodes to `2⁻²⁵`, within
relative error `2⁻¹²` of it. -/
theorem binary32 :
    ExecFloat.Binary.toRat? (libraryK3 (-1 : Binary32) (-1 + 1 / 4096)) = some 0 ∧
      (∀ r < 1, ¬ MixedError (ExecFloat.Binary.toReal (libraryK3 (-1 : Binary32) (-1 + 1 / 4096)))
        (exact (1 / 4096)) r 0) ∧
      ExecFloat.Binary.toRat? (k3 (-1 : Binary32) (-1 + 1 / 4096)) = some (1 / 2 ^ 25) ∧
      MixedError (ExecFloat.Binary.toReal (k3 (-1 : Binary32) (-1 + 1 / 4096))) (exact (1 / 4096))
        (1 / 2 ^ 12) 0 := by
  have hlib : ExecFloat.Binary.toRat? (libraryK3 (-1 : Binary32) (-1 + 1 / 4096)) = some 0 := by
    decide +kernel
  have hst : ExecFloat.Binary.toRat? (k3 (-1 : Binary32) (-1 + 1 / 4096)) = some (1 / 2 ^ 25) := by
    decide +kernel
  have hbound := Real.exp_bound (x := (1 / 4096 : ℝ)) (by norm_num [abs_of_pos]) (n := 3)
    (by norm_num)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial] at hbound
  norm_num [abs_of_pos] at hbound
  rw [abs_le] at hbound
  refine ⟨hlib, fun r hr => ?_, hst, ?_⟩
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hlib, Rat.cast_zero]
    exact fun h => h.ne_zero_of_ne_zero hr (exact_pos (by norm_num)).ne' rfl
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hst]
    refine MixedError.of_abs_sub_le_mul (by norm_num) ?_
    rw [abs_of_pos (exact_pos (by norm_num)), abs_le]
    unfold exact
    constructor <;> push_cast <;> nlinarith [hbound.1, hbound.2]

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The library estimator: the absolute part of the bound contains the unit roundoff times
`exp (lr - lp)`, about one, which does not shrink with the log-ratio. -/
roundoff mixed theorem library_error :=
  fun (lp lr : Value) (_hformat : format.isIEEE = true)
    (_h₁ : ExecFloat.Binary.isFinite lp = true) (_h₂ : ExecFloat.Binary.isFinite lr = true) =>
    ExecFloat.Binary.toReal (libraryK3 lp lr)

/-- `k3`: besides underflow, the bound is the unit roundoff times multiples of `|exp d - 1|` and
`|d|` for `d = lr - lp`, which vanish with the log-ratio. -/
roundoff mixed theorem k3_error :=
  fun (lp lr : Value) (_hformat : format.isIEEE = true)
    (_h₁ : ExecFloat.Binary.isFinite lp = true) (_h₂ : ExecFloat.Binary.isFinite lr = true) =>
    ExecFloat.Binary.toReal (k3 lp lr)

end Roundoff

end LossBugs.KLEstimator

/-! ## The tanh-squashed Gaussian log-probability -/

namespace ProbBugs.TanhSquash

open LossBugs

section Formulas

variable {α : Type} [Storage α] [Context α]

/-- Stable-Baselines3's change-of-variables term for an action `tanh u`,
`log (1 - a * a + eps)` with `a = tanh u`, also CleanRL's at action scale one. -/
def libraryCorrection (u eps : α) : α :=
  let a := MathFunctions.tanh u
  MathFunctions.log (1 - a * a + eps)

/-- `log (1 - tanh² u)` as `2 (log 2 - u - softplus (-2 u))`, the form of
`torch.distributions.TanhTransform`. -/
def correction (u : α) : α :=
  (1 + 1) * (MathFunctions.log (1 + 1) - u - Activation.softplus (-(u + u)))

end Formulas

/-- The change-of-variables term `log (1 - tanh² u)`. -/
noncomputable def exact (u : ℝ) : ℝ := Real.log (1 - Real.tanh u ^ 2)

/-- Softplus over the reals is `log (1 + exp m)`. -/
private theorem softplus_real (m : ℝ) :
    Activation.softplus m = Real.log (1 + Real.exp m) := by
  have h := SoftMargin.loss_real (-m) 1
  simp only [SoftMargin.loss, SoftMargin.libraryLoss, one_mul, neg_neg, mul_one] at h
  rw [h, MathFunctions.exp_real, Context.log1p_eq_log_one_add]

/-- Over the reals, Stable-Baselines3's term without its epsilon and `correction` are the
change-of-variables term. -/
theorem correction_real (u : ℝ) : libraryCorrection u 0 = exact u ∧ correction u = exact u := by
  refine ⟨by simp only [libraryCorrection, exact, MathFunctions.tanh_real, MathFunctions.log,
    add_zero, sq], ?_⟩
  have hp := Real.exp_pos u
  have hn := Real.exp_pos (-u)
  have hs : 0 < Real.exp u + Real.exp (-u) := by positivity
  have hprod : Real.exp u * Real.exp (-u) = 1 := by rw [← Real.exp_add]; simp
  have hsech : 1 - Real.tanh u ^ 2 = 4 / (Real.exp u + Real.exp (-u)) ^ 2 := by
    rw [Real.tanh_eq]
    field_simp
    nlinarith [hprod]
  have hsum : 1 + Real.exp (-(u + u)) = Real.exp (-u) * (Real.exp u + Real.exp (-u)) := by
    rw [mul_add, ← Real.exp_add, ← Real.exp_add, show -u + u = 0 by ring,
      show -u + -u = -(u + u) by ring, Real.exp_zero]
  simp only [correction, exact, softplus_real, MathFunctions.log]
  rw [hsech, hsum, Real.log_mul hn.ne' hs.ne', Real.log_exp,
    Real.log_div (by norm_num) (by positivity), Real.log_pow, show (4 : ℝ) = 2 ^ 2 by norm_num,
    Real.log_pow]
  push_cast
  ring_nf

/-! ## Binary32 at `u = 10` -/

/-- `1e-6` rounded to binary32, Stable-Baselines3's epsilon. -/
def binary32Epsilon : Binary32 := 1 / (10 ^ 6 : ℕ)

/-- `log (1 - tanh² 10)` lies within `[2 log 2 - 20 - 2 exp (-20), 2 log 2 - 20]`. -/
private theorem exact_ten_mem :
    exact 10 ∈ Set.Icc (2 * Real.log 2 - 20 - 2 * Real.exp (-20)) (2 * Real.log 2 - 20) := by
  rw [← (correction_real 10).2]
  simp only [correction, softplus_real, MathFunctions.log]
  have hpos : 0 < Real.exp (-(10 + 10 : ℝ)) := Real.exp_pos _
  have hlog := Real.log_le_sub_one_of_pos (by positivity : (0 : ℝ) < 1 + Real.exp (-(10 + 10)))
  have hnn := Real.log_nonneg (by linarith : (1 : ℝ) ≤ 1 + Real.exp (-(10 + 10)))
  norm_num at hlog hnn ⊢
  constructor <;> nlinarith

/-- Stable-Baselines3's term decodes to `-14486613 / 2²⁰`, about `-13.82`, at `u = 10` in binary32,
which is not within relative error `1 / 4` of `log (1 - tanh² 10)`, about `-18.61`. `correction`
decodes to `-4879471 / 2¹⁸`, within relative error twice the binary32 unit roundoff of it. -/
theorem binary32 :
    ExecFloat.Binary.toRat? (libraryCorrection (10 : Binary32) binary32Epsilon) =
        some (-14486613 / 2 ^ 20) ∧
      ¬ MixedError (ExecFloat.Binary.toReal (libraryCorrection (10 : Binary32) binary32Epsilon))
        (exact 10) (1 / 4) 0 ∧
      ExecFloat.Binary.toRat? (correction (10 : Binary32)) = some (-4879471 / 2 ^ 18) ∧
      MixedError (ExecFloat.Binary.toReal (correction (10 : Binary32))) (exact 10)
        (2 * Model.unitRoundoffAt FloatFormat.binary32) 0 := by
  have hlib : ExecFloat.Binary.toRat? (libraryCorrection (10 : Binary32) binary32Epsilon) =
      some (-14486613 / 2 ^ 20) := by
    decide +kernel
  have hst : ExecFloat.Binary.toRat? (correction (10 : Binary32)) = some (-4879471 / 2 ^ 18) := by
    decide +kernel
  obtain ⟨hlo, hhi⟩ := exact_ten_mem
  have hl₁ := Real.log_two_gt_d9
  have hl₂ := Real.log_two_lt_d9
  have he : Real.exp (-20) ≤ 1 / 10 ^ 8 := by
    have h1 : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
    have h20 : Real.exp 20 = Real.exp 1 ^ 20 := by rw [← Real.exp_nat_mul]; norm_num
    have hbig : (10 : ℝ) ^ 8 ≤ Real.exp 20 := by
      rw [h20]
      nlinarith [pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 2.7182818283) h1.le 20]
    rw [Real.exp_neg, inv_eq_one_div, div_le_div_iff₀ (Real.exp_pos 20) (by norm_num)]
    linarith
  have hneg : exact 10 < 0 := by linarith
  refine ⟨hlib, fun h => ?_, hst, ?_⟩
  · have hb := h.bound
    rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hlib, abs_of_neg hneg, add_zero, abs_le] at hb
    push_cast at hb
    nlinarith [hb.1, hb.2]
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hst]
    refine MixedError.of_abs_sub_le_mul
      (mul_pos (by norm_num) (Model.unitRoundoffAt_pos _)).le ?_
    rw [abs_of_neg hneg, abs_le]
    norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary32]
    constructor <;> nlinarith

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- `correction` is finite with a mixed bound at every finite `u` that meets the generated
conditions on its intermediate values. -/
roundoff mixed theorem correction_error :=
  fun [Storage Value] (u : Value) (_hformat : format.isIEEE = true)
    (_hu : ExecFloat.Binary.isFinite u = true) =>
    ExecFloat.Binary.toReal (correction u)

end Roundoff

end ProbBugs.TanhSquash

/-! ## The label-smoothing term of cross-entropy -/

namespace LossBugs.LabelSmoothing

section Formulas

variable {α : Type} [Context α]

/-- A scalar model of PyTorch's label-smoothing term of `cross_entropy` with mean reduction, no
weights and no ignored positions, from the log-probabilities `rows` of a batch: the negated sum
over classes of each row, summed over the rows, divided by the number of rows and multiplied by
`eps / classes`, each operation rounded in the format. -/
def librarySmoothing (eps : α) (classes : Nat) (rows : List (List α)) : α :=
  (rows.map fun row => -(row.foldl (· + ·) 0)).foldl (· + ·) 0 / (rows.length : α) *
    (eps / (classes : α))

/-- The same term as `eps` times a mean of means: each log-probability is divided by the number of
classes, and each row's mean by the number of rows, before it is added. -/
def smoothing (eps : α) (classes : Nat) (rows : List (List α)) : α :=
  eps * (rows.map fun row => row.foldl (fun acc x => acc - x / (classes : α)) 0).foldl
    (fun acc r => acc + r / (rows.length : α)) 0

end Formulas

/-- The term over the reals: `eps` times the mean over rows and classes of the negated
log-probabilities. -/
noncomputable def exact (eps : ℝ) (classes : Nat) (rows : List (List ℝ)) : ℝ :=
  eps * -(rows.map List.sum).sum / (rows.length * classes)

private theorem foldl_sub_div (a c : ℝ) (l : List ℝ) :
    l.foldl (fun acc x => acc - x / c) a = a - l.sum / c := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => simp only [List.foldl_cons, List.sum_cons, ih]; ring

private theorem foldl_add_div (a c : ℝ) (l : List ℝ) :
    l.foldl (fun acc x => acc + x / c) a = a + l.sum / c := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => simp only [List.foldl_cons, List.sum_cons, ih]; ring

private theorem sum_map_neg_sum (rows : List (List ℝ)) :
    (rows.map fun row => -row.sum).sum = -(rows.map List.sum).sum := by
  induction rows with
  | nil => simp
  | cons r rows ih => simp only [List.map_cons, List.sum_cons, ih]; ring

private theorem sum_map_sub_div (c : ℝ) (rows : List (List ℝ)) :
    (rows.map fun row => 0 - row.sum / c).sum = -(rows.map List.sum).sum / c := by
  induction rows with
  | nil => simp
  | cons r rows ih => simp only [List.map_cons, List.sum_cons, ih]; ring

/-- Over the reals both formulas are `exact`. -/
theorem smoothing_real (eps : ℝ) (classes : Nat) (rows : List (List ℝ)) :
    librarySmoothing eps classes rows = exact eps classes rows ∧
      smoothing eps classes rows = exact eps classes rows := by
  constructor
  · simp only [librarySmoothing, exact, ← List.sum_eq_foldl, sum_map_neg_sum]
    rw [div_mul_div_comm, mul_comm (rows.length : ℝ) (classes : ℝ)]
    ring
  · simp only [smoothing, exact, foldl_sub_div, foldl_add_div, zero_add, sum_map_sub_div]
    rw [div_div, mul_comm (classes : ℝ) (rows.length : ℝ)]
    ring

/-! ## Binary16, three rows of two classes -/

/-- Three rows `[0, -30000]`, the log-probabilities of the logits `0` and `-30000` to binary16
precision: each row's probabilities sum to one up to `exp (-30000)`. Each row sums to `-30000`,
which binary16 represents, and the negated rows sum to `90000`, which it does not. -/
def batch : List (List Binary16) := List.replicate 3 [0, -30000]

/-- The library model's term at `eps = 1 / 8` is not finite in binary16, while `smoothing` decodes
to `1875`, the exact term. -/
theorem binary16 :
    ExecFloat.Binary.isFinite (librarySmoothing (1 / 8 : Binary16) 2 batch) = false ∧
      ExecFloat.Binary.toRat? (smoothing (1 / 8 : Binary16) 2 batch) = some 1875 ∧
      exact (1 / 8) 2 (List.replicate 3 [0, -30000]) = 1875 := by
  refine ⟨by decide +kernel, by decide +kernel, ?_⟩
  simp [exact]
  norm_num

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The library model's sum over one row, the only reduction stated here: its generated
conditions bound every partial sum of the log-probabilities by the largest finite value. -/
roundoff mixed theorem library_row_error :=
  fun (row : List Value) (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (-(row.foldl (· + ·) 0))

/-- The scaled sum over one row: its generated conditions bound the partial sums of the
log-probabilities divided by `c`. -/
roundoff mixed theorem row_error :=
  fun (row : List Value) (c : Value) (_hformat : format.isIEEE = true)
    (_hc : ExecFloat.Binary.isFinite c = true) (_hc₀ : 0 < ExecFloat.Binary.toReal c) =>
    ExecFloat.Binary.toReal (row.foldl (fun acc x => acc - x / c) 0)

end Roundoff

end LossBugs.LabelSmoothing

/-! ## An exponential moving average that does not move -/

namespace SumBugs.EMAStall

section Formulas

variable {α : Type} [Context α]

/-- A scalar model of one update of `torch.optim.swa_utils.get_ema_multi_avg_fn`,
`torch.lerp (p, x, w)` with `w = 1 - decay`, which for `|w| < 1 / 2` is `p + w * (x - p)`, each
operation rounded in the format. ATen computes it in binary32 and rounds once on storing. -/
def libraryStep (w x p : α) : α := p + w * (x - p)

/-- `n` updates toward the fixed target `x`. -/
def steps (w x : α) (n : Nat) (p : α) : α := (libraryStep w x)^[n] p

end Formulas

/-- The moving average after `n` exact updates toward `x`. -/
noncomputable def exact (w x : ℝ) (n : Nat) (p : ℝ) : ℝ := x + (p - x) * (1 - w) ^ n

/-- Over the reals the updates are `exact`. -/
theorem steps_real (w x p : ℝ) (n : Nat) : steps w x n p = exact w x n p := by
  induction n with
  | zero => simp [steps, exact]
  | succ n ih =>
    rw [steps, Function.iterate_succ_apply', ← steps, ih]
    simp only [libraryStep, exact, pow_succ]
    ring

/-- If one update leaves `p` unchanged, every number of updates does. -/
theorem steps_of_step_eq {α : Type} [Context α] {w x p : α} (h : libraryStep w x p = p)
    (n : Nat) : steps w x n p = p :=
  Function.iterate_fixed h n

/-- In the scalar model, an update with the weight `w = 1 / 1000` of a bfloat16 average `1 / 64`
toward a target ten units in the last place above it leaves the average unchanged, so no number
of updates toward that target moves it. -/
theorem bfloat16 :
    libraryStep (1 / 1000 : BFloat16) (1 / 64 + 10 / 8192) (1 / 64) = 1 / 64 ∧
      ∀ n, steps (1 / 1000 : BFloat16) (1 / 64 + 10 / 8192) n (1 / 64) = 1 / 64 :=
  have h : libraryStep (1 / 1000 : BFloat16) (1 / 64 + 10 / 8192) (1 / 64) = 1 / 64 := by
    decide +kernel
  ⟨h, steps_of_step_eq h⟩

/-- The weight and the target decode to `131 / 2¹⁷` and `69 / 2¹²`. -/
theorem bfloat16_inputs :
    ExecFloat.Binary.toRat? (1 / 1000 : BFloat16) = some (131 / 2 ^ 17) ∧
      ExecFloat.Binary.toRat? (1 / 64 + 10 / 8192 : BFloat16) = some (69 / 2 ^ 12) := by
  constructor <;> decide +kernel

/-- After `n` exact updates with that weight the average has moved toward the target by at least
`n w / (1 + n w)` of the distance. -/
theorem exact_moved {n : Nat} :
    (69 / 2 ^ 12 - 1 / 64 : ℝ) * (n * (131 / 2 ^ 17) / (1 + n * (131 / 2 ^ 17))) ≤
      exact (131 / 2 ^ 17) (69 / 2 ^ 12) n (1 / 64) - 1 / 64 := by
  have hw₀ : (0 : ℝ) ≤ 131 / 2 ^ 17 := by norm_num
  have hw₁ : (131 / 2 ^ 17 : ℝ) ≤ 1 := by norm_num
  have hbern := one_add_mul_le_pow (by norm_num : (-2 : ℝ) ≤ 131 / 2 ^ 17) n
  have hprod : (1 - 131 / 2 ^ 17 : ℝ) ^ n * (1 + 131 / 2 ^ 17) ^ n ≤ 1 := by
    rw [← mul_pow]
    exact pow_le_one₀ (by norm_num) (by norm_num)
  have hpos : (0 : ℝ) < 1 + n * (131 / 2 ^ 17) := by positivity
  have hdecay : (1 - 131 / 2 ^ 17 : ℝ) ^ n ≤ 1 / (1 + n * (131 / 2 ^ 17)) := by
    rw [le_div_iff₀ hpos]
    have hq : 0 ≤ (1 - 131 / 2 ^ 17 : ℝ) ^ n := pow_nonneg (by norm_num) n
    nlinarith
  unfold exact
  have hgap : (0 : ℝ) < 69 / 2 ^ 12 - 1 / 64 := by norm_num
  rw [show (69 / 2 ^ 12 : ℝ) + (1 / 64 - 69 / 2 ^ 12) * (1 - 131 / 2 ^ 17) ^ n - 1 / 64 =
    (69 / 2 ^ 12 - 1 / 64) * (1 - (1 - 131 / 2 ^ 17) ^ n) by ring]
  apply mul_le_mul_of_nonneg_left _ hgap.le
  rw [show (n * (131 / 2 ^ 17) / (1 + n * (131 / 2 ^ 17)) : ℝ) = 1 - 1 / (1 + n * (131 / 2 ^ 17))
    by field_simp; ring]
  linarith

/-- In binary32 one update moves the average from `1 / 64` to `8389263 / 2²⁹`. -/
theorem binary32 :
    ExecFloat.Binary.toRat? (libraryStep (1 / 1000 : Binary32) (1 / 64 + 10 / 8192) (1 / 64)) =
      some (8389263 / 2 ^ 29) := by
  decide +kernel

end SumBugs.EMAStall

/-! ## The noise level of a diffusion schedule at small timesteps -/

namespace SumBugs.NoiseLevel

section Formulas

variable {α : Type} [Context α]

/-- diffusers' noise level at a timestep, `(1 - alpha_bar) ** 0.5`, computed in the samples'
format from `alpha_bar` already rounded to that format, as `DDPMScheduler.add_noise` does. -/
def libraryNoiseScale (alphaBar : α) : α := MathFunctions.sqrt (1 - alphaBar)

/-- The noise level from the schedule's betas as `sqrt (-expm1 (Σ log1p (-beta)))`, which never
forms `alpha_bar`. -/
def noiseScale (betas : List α) : α :=
  MathFunctions.sqrt (-(Context.expm1 (betas.foldl (fun acc b => acc + Context.log1p (-b)) 0)))

end Formulas

/-- The noise level `sqrt (1 - Π (1 - beta))`. -/
noncomputable def exact (betas : List ℝ) : ℝ := Real.sqrt (1 - (betas.map fun b => 1 - b).prod)

private theorem exp_foldl_log (a : ℝ) (l : List ℝ) (hl : ∀ b ∈ l, b < 1) :
    Real.exp (l.foldl (fun acc b => acc + Real.log (1 + -b)) a) =
      Real.exp a * (l.map fun b => 1 - b).prod := by
  induction l generalizing a with
  | nil => simp
  | cons b l ih =>
    have hb : 0 < 1 + -b := by linarith [hl b (by simp)]
    simp only [List.foldl_cons, List.map_cons, List.prod_cons]
    rw [ih _ fun c hc => hl c (by simp [hc]), Real.exp_add, Real.exp_log hb]
    ring

/-- Over the reals, the library formula at `alpha_bar = Π (1 - beta)` and `noiseScale` are the
noise level, for betas below one. -/
theorem noiseScale_real (betas : List ℝ) (h : ∀ b ∈ betas, b < 1) :
    libraryNoiseScale (betas.map fun b => 1 - b).prod = exact betas ∧
      noiseScale betas = exact betas := by
  have hlog (v : ℝ) : Context.log1p v = Real.log (1 + v) := Context.log1p_eq_log_one_add v
  have hexp (v : ℝ) : Context.expm1 v = Real.exp v - 1 := Context.expm1_eq_exp_sub_one v
  refine ⟨by simp only [libraryNoiseScale, exact, MathFunctions.sqrt_real], ?_⟩
  simp only [noiseScale, exact, MathFunctions.sqrt_real, hlog, hexp]
  rw [exp_foldl_log 0 betas h, Real.exp_zero, one_mul, neg_sub]

/-! ## Binary16 at the second timestep of diffusers' default schedule -/

/-- diffusers' float32 `alphas_cumprod[1]` of its default linear schedule, `8386763 / 2²³`,
rounded to binary16 as `.to(torch.float16)` rounds it. -/
def alphaBar : Binary16 := ((8386763 / 2 ^ 23 : ℚ) : Binary16)

/-- diffusers' float32 betas at the first two timesteps, `13743895 / 2³⁷` and `4120417 / 2³⁵`,
rounded to binary16. -/
def betas : List Binary16 :=
  [((13743895 / 2 ^ 37 : ℚ) : Binary16), ((4120417 / 2 ^ 35 : ℚ) : Binary16)]

/-- `alpha_bar` rounds to one, and the betas decode to `839 / 2²³` and `503 / 2²²`. -/
theorem inputs_toRat? :
    ExecFloat.Binary.toRat? alphaBar = some 1 ∧
      betas.map ExecFloat.Binary.toRat? = [some (839 / 2 ^ 23), some (503 / 2 ^ 22)] := by
  constructor <;> decide +kernel

/-- The library formula decodes to zero, which is not within any relative error below one of the
noise level of the float32 betas, about `0.0148`. `noiseScale` decodes to `243 / 2¹⁴`. -/
theorem binary16 :
    ExecFloat.Binary.toRat? (libraryNoiseScale alphaBar) = some 0 ∧
      (∀ r < 1, ¬ MixedError (ExecFloat.Binary.toReal (libraryNoiseScale alphaBar))
        (exact [13743895 / 2 ^ 37, 4120417 / 2 ^ 35]) r 0) ∧
      ExecFloat.Binary.toRat? (noiseScale betas) = some (243 / 2 ^ 14) := by
  have hlib : ExecFloat.Binary.toRat? (libraryNoiseScale alphaBar) = some 0 := by
    decide +kernel
  refine ⟨hlib, fun r hr => ?_, by decide +kernel⟩
  rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hlib, Rat.cast_zero]
  refine fun h => h.ne_zero_of_ne_zero hr (Real.sqrt_pos.mpr ?_).ne' rfl
  norm_num

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The library formula has a conditional mixed bound against the noise level of the decoded
`alpha_bar`, which leaves out the error of rounding `alpha_bar` itself. -/
roundoff mixed theorem library_error :=
  fun (alphaBar : Value) (_hformat : format.isIEEE = true)
    (_ha : ExecFloat.Binary.isFinite alphaBar = true) =>
    ExecFloat.Binary.toReal (libraryNoiseScale alphaBar)

/-- `noiseScale` has a conditional mixed bound whose local terms are multiples of the unit
roundoff times `|log (1 - beta)|` and `|exp (Σ log (1 - beta)) - 1|`, with underflow and
accumulated terms, propagated through the square root. -/
roundoff mixed theorem noiseScale_error :=
  fun (betas : List Value) (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (noiseScale betas)

end Roundoff

end SumBugs.NoiseLevel

end TorchLean.Examples
