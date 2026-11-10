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
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff
public import FloatLib.Numerics.Automation.Attributes
public import TorchLean.Tensor.Numeric.Real
public import TorchLean.Tensor.Tactic.Init

/-!
# Rounding bugs in other libraries' gradients and decay factors

Three sections take formulas from torchvision, xLSTM and PyTorch and write them with TorchLean's
scalar operations, as in `LibraryBugs`. Each has an input on which the kernel checks that the
formula fails in an IEEE format, and a formula with the same real value that does not fail there.
The sections also state mixed roundoff bounds in every IEEE format for the computations in which the
two differ.

* `FocalPower`: torchvision's `sigmoid_focal_loss` multiplies the cross-entropy by
  `(1 - p_t) ^ γ`. For `γ = 1 / 2` the backward of the power is `γ / sqrt (1 - p_t)`, which is
  infinite once `1 - p_t` rounds to zero. At the logit `-20` with target `0` in binary32 the
  gradient is NaN, while the derivative written as `sqrt p * (p + c / 2 * (1 - p))` is finite.
* `SumBugs.ForgetGate`: xLSTM's parallel mLSTM backend takes the forget factor between two steps
  as the exponential of a difference of prefix sums of log forget gates. In bfloat16 the prefix
  sum `-64` absorbs the log gate `-199 / 2¹²`, so the factor is `1` where it should be about
  `0.9526`. Exponentiating the log gate itself gives `61 / 64`.
* `NormBackward`: PyTorch's backward of `vector_norm` of order three multiplies `x * |x|` by
  `g / (n * n)`. In binary16 at four entries `2⁻¹³` the product underflows to zero, the quotient
  overflows, and the gradient is NaN. The form `g * (x / n) * |x / n|` gives `407 / 1024`.

The real identities, kernel-checked counterexamples and conditional mixed bounds for both
computations are about the scalar formulas written here. They do not show that the installed
library kernels execute exactly these operations.
-/

@[expose] public section

namespace TorchLean.Examples

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics

/-! ## The focal-loss gradient below exponent one -/

namespace LossBugs.FocalPower

section Formulas

variable {α : Type} [Context α]

/-- The probability of the target class, `p * t + (1 - p) * (1 - t)`, as torchvision's
`sigmoid_focal_loss` forms it from `p = sigmoid x`. -/
def pt (p t : α) : α := p * t + (1 - p) * (1 - t)

/-- The base `1 - p_t` of the modulating factor `(1 - p_t) ^ γ`. -/
def libraryBase (p t : α) : α := 1 - pt p t

/-- A scalar model of the derivative in the logit of `c * (1 - p_t) ^ (1 / 2)`, for the
cross-entropy `c`, by the chain rule through the base `q`, `p_t` and `p`: the power's backward
`1 / 2 * (1 / sqrt q)`, the products in `p_t`, the sigmoid's backward `(1 - p) * p`, and the
cross-entropy's derivative `p - t` times the factor. -/
def libraryGrad (p t c : α) : α :=
  let q := libraryBase p t
  let gq := c * (1 / (1 + 1) * (1 / MathFunctions.sqrt q))
  let gpt := -gq
  let gp := gpt * t + -(gpt * (1 - t))
  gp * (1 - p) * p + (p - t) * MathFunctions.sqrt q

/-- The same derivative at target `0`, where the base is `p`, written without a reciprocal square
root: `sqrt p * (p + c / 2 * (1 - p))`. -/
def grad (p c : α) : α := MathFunctions.sqrt p * (p + c / (1 + 1) * (1 - p))

end Formulas

/-- Over the reals, at target `0` and a positive probability, both formulas agree. -/
theorem grad_real (p c : ℝ) (hp : 0 < p) : libraryGrad p 0 c = grad p c := by
  obtain ⟨s, hs, rfl⟩ : ∃ s : ℝ, 0 < s ∧ p = s * s :=
    ⟨Real.sqrt p, Real.sqrt_pos.mpr hp, (Real.mul_self_sqrt hp.le).symm⟩
  have hbase : libraryBase (s * s) 0 = s * s := by
    simp only [libraryBase, pt]
    ring
  simp only [libraryGrad, hbase, grad, MathFunctions.sqrt_real, Real.sqrt_mul_self hs.le]
  field_simp
  ring

/-! ## Binary32 at the logit `-20` and target `0` -/

/-- The binary32 probability `sigmoid (-20)`, from TorchLean's `Context.sigmoid`. -/
def prob : Binary32 := Context.sigmoid (-20)

/-- The binary32 cross-entropy `softplus (-20)` at target `0`, from TorchLean's
`Activation.softplus`.
-/
def crossEntropy : Binary32 := Activation.softplus (-20)

/-- The probability and the cross-entropy both decode to `9282611 / 2⁵²` and the library base
`1 - p_t` decodes to zero. The library gradient is NaN at this cross-entropy and at cross-entropy
zero, while the stable gradient decodes to `10357061 / 2⁶⁶`. -/
theorem binary32 :
    ExecFloat.Binary.toRat? prob = some (9282611 / 2 ^ 52) ∧
      ExecFloat.Binary.toRat? crossEntropy = some (9282611 / 2 ^ 52) ∧
      ExecFloat.Binary.toRat? (libraryBase prob 0) = some 0 ∧
      ExecFloat.Binary.isNaN (libraryGrad prob 0 crossEntropy) = true ∧
      ExecFloat.Binary.isNaN (libraryGrad prob 0 0) = true ∧
      ExecFloat.Binary.toRat? (grad prob crossEntropy) = some (10357061 / 2 ^ 66) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> decide +kernel

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The library base `1 - p_t` from a finite probability and target, under range hypotheses: the
absolute part of the bound contains multiples of the unit roundoff times `|p_t|` and
`|(1 - p) * (1 - t)|`, both about one at target `0`, which do not shrink with the base. -/
roundoff mixed theorem library_base_error :=
  fun (p t : Value) (_hformat : format.isIEEE = true)
    (_hp : ExecFloat.Binary.isFinite p = true) (_ht : ExecFloat.Binary.isFinite t = true) =>
    ExecFloat.Binary.toReal (libraryBase p t)

/-- `grad` from a finite nonnegative probability and a finite cross-entropy, under range
hypotheses: the relative part is about twice the unit roundoff, and up to underflow and
second-order terms the absolute part is half an ulp at `sqrt p` times `|p + c / 2 * (1 - p)|` plus
multiples of the unit roundoff times `sqrt p * |c / 2 * (1 - p)|`. No term is divided by the base.
-/
roundoff mixed theorem grad_error :=
  fun (p c : Value) (_hformat : format.isIEEE = true)
    (_hp : ExecFloat.Binary.isFinite p = true) (_hp₀ : 0 ≤ ExecFloat.Binary.toReal p)
    (_hc : ExecFloat.Binary.isFinite c = true) =>
    ExecFloat.Binary.toReal (grad p c)

end Roundoff

end LossBugs.FocalPower

/-! ## Forget factors from differences of prefix sums -/

namespace SumBugs.ForgetGate

section Formulas

variable {α : Type} [Context α]

/-- The forget factor into the step after `a` in xLSTM's `parallel_stabilized_simple`: the
exponential of the difference of the prefix sums `a + b` and `a` of the log forget gates, with
`b` the log forget gate of that step. -/
def libraryDecay (a b : α) : α := MathFunctions.exp (a + b - a)

/-- The forget factor from the log forget gate itself, as a masked segment sum gives it. -/
def decay (b : α) : α := MathFunctions.exp b

end Formulas

/-- Over the reals both forget factors are `exp b`. -/
theorem decay_real (a b : ℝ) : libraryDecay a b = Real.exp b ∧ decay b = Real.exp b := by
  refine ⟨?_, rfl⟩
  rw [libraryDecay, show a + b - a = b by ring]
  rfl

/-! ## Bfloat16 after the prefix sum `-64` -/

/-- The bfloat16 log forget gate `-199 / 2¹²`, near `logsigmoid 3` at the forget bias `3`. -/
def logGate : BFloat16 := ((-199 / 2 ^ 12 : ℚ) : BFloat16)

/-- After the prefix sum `-64` the library forget factor decodes to one, although
`exp (-199 / 2¹²)` is below `31 / 32`. The stable factor decodes to `61 / 64`, within relative
error `2⁻⁸` of `exp (-199 / 2¹²)`. -/
theorem bfloat16 :
    ExecFloat.Binary.toRat? (libraryDecay (-64 : BFloat16) logGate) = some 1 ∧
      Real.exp (-199 / 2 ^ 12) < 31 / 32 ∧
      ExecFloat.Binary.toRat? (decay logGate) = some (61 / 64) ∧
      MixedError (61 / 64) (Real.exp (-199 / 2 ^ 12)) (1 / 2 ^ 8) 0 := by
  have hlo := Real.add_one_le_exp (-199 / 2 ^ 12 : ℝ)
  have hpos := Real.exp_pos (-199 / 2 ^ 12 : ℝ)
  have hhi : Real.exp (-199 / 2 ^ 12) * (199 / 2 ^ 12 + 1) ≤ 1 := by
    calc Real.exp (-199 / 2 ^ 12) * (199 / 2 ^ 12 + 1)
        ≤ Real.exp (-199 / 2 ^ 12) * Real.exp (199 / 2 ^ 12) :=
          mul_le_mul_of_nonneg_left (Real.add_one_le_exp _) hpos.le
      _ = 1 := by rw [← Real.exp_add]; norm_num
  refine ⟨by decide +kernel, ?_, by decide +kernel, ?_⟩
  · norm_num at hhi ⊢
    linarith
  · refine MixedError.of_abs_sub_le_mul (by norm_num) ?_
    rw [abs_of_pos hpos, abs_le]
    norm_num at hlo hhi ⊢
    constructor <;> linarith

/-! ## Binary32 after the prefix sum `-256` -/

/-- In binary32 the prefix sum `-256` plus the log gate `-161 / 2¹⁶` rounds to
`-256 - 80 / 2¹⁵`, a tie rounded to even. The library forget factor decodes to `8368153 / 2²³`
and the stable factor to `16736051 / 2²⁴`, `255` units of `2⁻²⁴` lower. -/
theorem binary32 :
    ExecFloat.Binary.toRat? ((-256 : Binary32) + ((-161 / 2 ^ 16 : ℚ) : Binary32)) =
        some (-256 - 80 / 2 ^ 15) ∧
      ExecFloat.Binary.toRat? (libraryDecay (-256 : Binary32) ((-161 / 2 ^ 16 : ℚ) : Binary32)) =
        some (8368153 / 2 ^ 23) ∧
      ExecFloat.Binary.toRat? (decay ((-161 / 2 ^ 16 : ℚ) : Binary32)) =
        some (16736051 / 2 ^ 24) := by
  refine ⟨?_, ?_, ?_⟩ <;> decide +kernel

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The library forget factor from a finite prefix sum `a` and a finite log gate `b`, under range
hypotheses: the relative part is `(1 + u) * (exp (u * |b| + (1 + u) * u * |a + b|) - 1) + u` for
the unit roundoff `u`, and grows with the prefix sum `a + b`. The absolute part is the underflow
error. -/
roundoff mixed theorem library_decay_error :=
  fun (a b : Value) (_hformat : format.isIEEE = true)
    (_ha : ExecFloat.Binary.isFinite a = true) (_hb : ExecFloat.Binary.isFinite b = true) =>
    ExecFloat.Binary.toReal (libraryDecay a b)

/-- `decay` from a finite log gate whose exponential is below the overflow threshold: the
relative part is the unit roundoff and the absolute part the underflow error. -/
roundoff mixed theorem decay_error :=
  fun (b : Value) (_hformat : format.isIEEE = true)
    (_hb : ExecFloat.Binary.isFinite b = true) =>
    ExecFloat.Binary.toReal (decay b)

end Roundoff

end SumBugs.ForgetGate

/-! ## The backward of a norm of order three -/

namespace SumBugs.NormBackward

section Formulas

variable {α : Type} [Context α]

/-- PyTorch's `norm_backward` for order three at an entry `x` of a vector with norm `n` and
incoming gradient `g`: `x * |x| ^ (3 - 2)` times `g / n ^ (3 - 1)`. -/
def libraryGrad (x n g : α) : α := x * MathFunctions.abs x * (g / (n * n))

/-- The same derivative from the ratio of the entry to the norm, `g * (x / n) * |x / n|`. -/
def grad (x n g : α) : α := g * (x / n) * MathFunctions.abs (x / n)

end Formulas

/-- Over the reals both formulas agree at a positive norm. -/
theorem grad_real (x n g : ℝ) (hn : 0 < n) : libraryGrad x n g = grad x n g := by
  simp only [libraryGrad, grad, LawfulContext.abs_eq, abs_div, abs_of_pos hn]
  field_simp

/-! ## Binary16 at four entries `2⁻¹³` -/

/-- An entry `2⁻¹³` of a vector of four equal entries. -/
def entry : Binary16 := ((1 / 2 ^ 13 : ℚ) : Binary16)

/-- The norm supplied as data: `1625 / 2²³`, the binary16 norm of order three of four entries
`2⁻¹³` measured in PyTorch, whose exact value is `2 ^ (-37 / 3)`, about `1.9377 * 10⁻⁴`. -/
def norm3 : Binary16 := ((1625 / 2 ^ 23 : ℚ) : Binary16)

/-- At incoming gradient one, `x * |x|` decodes to zero, `1 / (n * n)` is not finite and the
library gradient is NaN. The stable gradient decodes to `407 / 1024`, within relative error
`2⁻¹⁰` of `(x / n) ^ 2 = (2¹⁰ / 1625) ^ 2`. -/
theorem binary16 :
    ExecFloat.Binary.toRat? (entry * MathFunctions.abs entry) = some 0 ∧
      ExecFloat.Binary.isFinite (1 / (norm3 * norm3)) = false ∧
      ExecFloat.Binary.isNaN (libraryGrad entry norm3 1) = true ∧
      ExecFloat.Binary.toRat? (grad entry norm3 1) = some (407 / 1024) ∧
      MixedError (407 / 1024) ((2 ^ 10 / 1625) ^ 2) (1 / 2 ^ 10) 0 := by
  refine ⟨by decide +kernel, by decide +kernel, by decide +kernel, by decide +kernel, ?_⟩
  refine MixedError.of_abs_sub_le_mul (by norm_num) ?_
  rw [abs_of_pos (by norm_num : (0 : ℝ) < 407 / 1024 - (2 ^ 10 / 1625) ^ 2),
    abs_of_pos (by norm_num : (0 : ℝ) < (2 ^ 10 / 1625) ^ 2)]
  norm_num

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The library gradient from a finite entry, a finite positive norm and a finite incoming
gradient, under range hypotheses that include `u + η / |n * n| < 1`, `u * |n * n| + η < |n * n|`
and `|g / (n * n)|`, enlarged by the quotient's relative error, at most the largest finite value:
the relative part contains `η / |n * n|` and the absolute part `η * |g / (n * n)|`, for the unit
roundoff `u` and the underflow error `η`. -/
roundoff mixed theorem library_grad_error :=
  fun (x n g : Value) (_hformat : format.isIEEE = true)
    (_hx : ExecFloat.Binary.isFinite x = true) (_hn : ExecFloat.Binary.isFinite n = true)
    (_hn₀ : 0 < ExecFloat.Binary.toReal n) (_hg : ExecFloat.Binary.isFinite g = true) =>
    ExecFloat.Binary.toReal (libraryGrad x n g)

/-- `grad` from a finite entry, a finite positive norm and a finite incoming gradient: the
relative part is a small multiple of the unit roundoff, the absolute part is underflow errors
times `|g|` and `|x / n|`, and the range hypotheses are on `x / n`, `g * (x / n)` and the result.
-/
roundoff mixed theorem grad_error :=
  fun (x n g : Value) (_hformat : format.isIEEE = true)
    (_hx : ExecFloat.Binary.isFinite x = true) (_hn : ExecFloat.Binary.isFinite n = true)
    (_hn₀ : 0 < ExecFloat.Binary.toReal n) (_hg : ExecFloat.Binary.isFinite g = true) =>
    ExecFloat.Binary.toReal (grad x n g)

end Roundoff

end SumBugs.NormBackward

end TorchLean.Examples
