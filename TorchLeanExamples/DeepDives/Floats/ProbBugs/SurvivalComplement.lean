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
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff
public import FloatLib.Numerics.Automation.Attributes
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init
public import TorchLean.Verify.RuntimeApprox.Roundoff.Mixed

/-!
# Rounding bugs in other libraries' formulas, continued

Three sections take formulas from PyTorch, PyTorch Geometric and TensorFlow Probability and write
them with TorchLean's scalar operations, as in `LibraryBugs`. Each has an input on which the kernel
checks that the formula fails in an IEEE format, and a formula with the same real value that does
not fail there. The sections also state mixed roundoff bounds in every IEEE format for the
computation in which the two differ.

* `LogSumExpHalf`: in float16 on CPU, PyTorch's `logsumexp` and its `log_softmax` over the last
  dimension store the sum of `exp (x - max)` over the row in float16. For a row of 151936 zero
  logits, the width of the Qwen2 vocabulary, the stored sum rounds to infinity, so both results
  are not finite where they should be `± log 151936`. Keeping the sum in binary32 and storing
  only the results gives `± 1527 / 2⁷`, about `± 11.93`. At 65520 zero logits the stored sum
  already overflows, while the binary32 sum gives `log 65520` within one unit of binary16 roundoff.
* `SumBugs.Variance`: PyTorch Geometric's `VarAggregation` and `StdAggregation` compute
  `mean (x * x) - mean x * mean x`. In binary32 this gives the neighborhood
  `[4095, 4097, 4097]` the variance `-2`, where the exact value is `8 / 9`. Subtracting the mean
  first gives the binary32 value nearest `8 / 9`.
* `ProbBugs.Survival`: TensorFlow Probability's Weibull, Gumbel and extreme-value
  distributions take the survival function as `1 - cdf`, with the Weibull CDF
  `-expm1 (-t)` at `t = (x / scale) ^ k`. At `t = 32` in binary32 the CDF rounds to one, the
  survival function is zero and its logarithm is not finite. `exp (-t)` and `-t` keep them.

The real identities, kernel-checked counterexamples and conditional mixed bounds are about the
scalar formulas written here. They do not show that the installed library kernels execute exactly
these operations.
-/


@[expose] public section

namespace TorchLean.Examples

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics

/-! ## The float16 sum inside `logsumexp` and `log_softmax` -/

namespace SumBugs.LogSumExpHalf

section Formulas

variable {α : Type} [Context α]

/-- `log s + m`: the log-sum-exp of a row from its maximum `m` and the sum `s` of
`exp (x - m)` over the row. -/
def logSumExp (s m : α) : α := MathFunctions.log s + m

/-- The log-probability `x - m - log s` of an entry `x` of the same row. -/
def logSoftmax (x s m : α) : α := x - m - MathFunctions.log s

end Formulas

/-- Over the reals, `logSumExp` at the sum of `exp (x - m)` is the log-sum-exp of the row, for any
shift `m`. -/
theorem logSumExp_real (xs : List ℝ) (hxs : xs ≠ []) (m : ℝ) :
    logSumExp (xs.map fun x => Real.exp (x - m)).sum m = Real.log (xs.map Real.exp).sum := by
  have hshift : (xs.map fun x => Real.exp (x - m)).sum = Real.exp (-m) * (xs.map Real.exp).sum := by
    rw [← List.sum_map_mul_left]
    congr 1
    exact List.map_congr_left fun x _ => by rw [← Real.exp_add]; ring_nf
  have hpos : 0 < (xs.map Real.exp).sum :=
    List.sum_pos _ (fun y hy => by
      obtain ⟨x, -, rfl⟩ := List.mem_map.mp hy
      exact Real.exp_pos x) (by simpa using hxs)
  simp only [logSumExp, MathFunctions.log]
  rw [hshift, Real.log_mul (Real.exp_pos _).ne' hpos.ne', Real.log_exp]
  ring

/-- Over the reals, `logSoftmax` at the same sum is `x - log Σ exp`. -/
theorem logSoftmax_real (xs : List ℝ) (hxs : xs ≠ []) (x m : ℝ) :
    logSoftmax x (xs.map fun y => Real.exp (y - m)).sum m = x - Real.log (xs.map Real.exp).sum := by
  have h := logSumExp_real xs hxs m
  simp only [logSumExp] at h
  simp only [logSoftmax]
  linarith

/-! ## Binary16 at a flat row of the Qwen2 vocabulary -/

/-- Rounding to binary16 is finite at `65519` and overflows at `65520`, half a unit in the last
place above the largest finite value `65504`. -/
theorem binary16_overflow :
    ExecFloat.Binary.toRat? ((65519 : ℚ) : Binary16) = some 65504 ∧
      ExecFloat.Binary.isFinite ((65520 : ℚ) : Binary16) = false := by
  constructor <;> decide +kernel

/-- The stored sum of a row of `151936` zero logits, the width of the Qwen2 vocabulary, as
PyTorch's float16 CPU `logsumexp` and last-dimension `log_softmax` store it. Each `exp (x - m)` is
one, the binary32 accumulator holds `151936` exactly, and storing that in binary16 overflows to
infinity. -/
def storedSum : Binary16 := ((151936 : ℚ) : Binary16)

/-- PyTorch's computation from the stored sum: the log-sum-exp and the log-probabilities are not
finite, although the exact values are `± log 151936`, about `± 11.93`. -/
theorem binary16_library :
    ExecFloat.Binary.isFinite (logSumExp storedSum 0) = false ∧
      ExecFloat.Binary.isFinite (logSoftmax 0 storedSum 0) = false := by
  constructor <;> decide +kernel

/-- Keeping the sum in binary32 and storing only the results in binary16 gives `1527 / 2⁷` and
`-1527 / 2⁷`, about `± 11.93`. -/
theorem binary32_stable :
    (ExecFloat.Binary.toRat? (logSumExp (151936 : Binary32) 0)).map
        (fun q => ExecFloat.Binary.toRat? ((q : ℚ) : Binary16)) = some (some (1527 / 2 ^ 7)) ∧
      (ExecFloat.Binary.toRat? (logSoftmax 0 (151936 : Binary32) 0)).map
        (fun q => ExecFloat.Binary.toRat? ((q : ℚ) : Binary16)) = some (some (-1527 / 2 ^ 7)) := by
  constructor <;> decide +kernel

/-! ## Binary16 at the overflow threshold -/

/-- `log 65520` lies between `11.0901` and `11.0902`. -/
private theorem log_65520_mem : 11.0901 < Real.log 65520 ∧ Real.log 65520 < 11.0902 := by
  have hsplit : Real.log 65520 = 16 * Real.log 2 + Real.log (4095 / 4096) := by
    rw [show (65520 : ℝ) = 2 ^ 16 * (4095 / 4096) by norm_num,
      Real.log_mul (by norm_num) (by norm_num), Real.log_pow]
    push_cast
    ring
  have hlo := Real.log_two_gt_d9
  have hhi := Real.log_two_lt_d9
  have hle := Real.log_le_sub_one_of_pos (show (0 : ℝ) < 4095 / 4096 by norm_num)
  have hge := Real.one_sub_inv_le_log_of_pos (show (0 : ℝ) < 4095 / 4096 by norm_num)
  rw [hsplit]
  norm_num at hlo hhi hle hge ⊢
  constructor <;> linarith

/-- At `65520` zero logits the stored sum already overflows, while the binary32 sum's
log-sum-exp stored in binary16 decodes to `355 / 2⁵`, within one unit of binary16 roundoff of
`log 65520`. -/
theorem binary16_threshold :
    ExecFloat.Binary.isFinite (logSumExp ((65520 : ℚ) : Binary16) 0) = false ∧
      (ExecFloat.Binary.toRat? (logSumExp (65520 : Binary32) 0)).map
        (fun q => ExecFloat.Binary.toRat? ((q : ℚ) : Binary16)) = some (some (355 / 2 ^ 5)) ∧
      MixedError (355 / 2 ^ 5) (Real.log 65520) (Model.unitRoundoffAt FloatFormat.binary16) 0 := by
  refine ⟨by decide +kernel, by decide +kernel, ?_⟩
  obtain ⟨hlo, hhi⟩ := log_65520_mem
  refine MixedError.of_abs_sub_le_mul (Model.unitRoundoffAt_pos _).le ?_
  rw [abs_of_pos (by linarith : (0 : ℝ) < Real.log 65520), abs_le]
  norm_num [Model.unitRoundoffAt_eq, FloatFormat.binary16]
  constructor <;> nlinarith

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- `logSumExp` from a finite positive sum and a finite shift. The stored binary16 sums of
`binary16_library` and `binary16_threshold` are not finite, so they do not meet the hypothesis. -/
roundoff mixed theorem logSumExp_error :=
  fun (s m : Value) (_hformat : format.isIEEE = true)
    (_hs : ExecFloat.Binary.isFinite s = true) (_hs₀ : 0 < ExecFloat.Binary.toReal s)
    (_hm : ExecFloat.Binary.isFinite m = true) =>
    ExecFloat.Binary.toReal (logSumExp s m)

/-- `logSoftmax` from a finite positive sum, with the same hypothesis on the sum. -/
roundoff mixed theorem logSoftmax_error :=
  fun (x s m : Value) (_hformat : format.isIEEE = true)
    (_hx : ExecFloat.Binary.isFinite x = true)
    (_hs : ExecFloat.Binary.isFinite s = true) (_hs₀ : 0 < ExecFloat.Binary.toReal s)
    (_hm : ExecFloat.Binary.isFinite m = true) =>
    ExecFloat.Binary.toReal (logSoftmax x s m)

end Roundoff

end SumBugs.LogSumExpHalf

/-! ## The one-pass variance of PyG's variance and standard-deviation aggregations -/

namespace SumBugs.Variance

section Formulas

variable {α : Type} [Context α]

/-- The mean of a neighborhood, a sum divided by the count, as PyG's `MeanAggregation`. -/
def libraryMean (xs : List α) : α := xs.foldl (· + ·) 0 / (xs.length : α)

/-- PyG's `VarAggregation`, `mean (x * x) - mean x * mean x`, each operation rounded in the
format. `StdAggregation` takes its square root after a clamp. -/
def libraryVar (xs : List α) :
    α := libraryMean (xs.map fun x => x * x) - libraryMean xs * libraryMean xs

/-- The two-pass variance: the mean of the squared deviations from the mean. -/
def var (xs : List α) : α :=
  let m := libraryMean xs
  libraryMean (xs.map fun x => (x - m) * (x - m))

end Formulas

/-- The population variance over the reals. -/
noncomputable def exact (xs : List ℝ) : ℝ :=
  (xs.map fun x => (x - xs.sum / xs.length) ^ 2).sum / xs.length

private theorem sum_sq_sub (m : ℝ) (l : List ℝ) :
    (l.map fun x => (x - m) ^ 2).sum =
      (l.map fun x => x * x).sum - 2 * m * l.sum + l.length * m ^ 2 := by
  induction l with
  | nil => simp
  | cons x l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, ih]
    push_cast
    ring

/-- Over the reals both formulas are the population variance of a nonempty neighborhood. -/
theorem var_real (xs : List ℝ) (hxs : xs ≠ []) :
    libraryVar xs = exact xs ∧ var xs = exact xs := by
  have hn : (xs.length : ℝ) ≠ 0 := by simpa using hxs
  constructor
  · simp only [libraryVar, libraryMean, exact, ← List.sum_eq_foldl, List.length_map,
      sum_sq_sub]
    field_simp
    ring
  · simp only [var, libraryMean, exact, ← List.sum_eq_foldl, List.length_map]
    congr 1
    exact congrArg List.sum (List.map_congr_left fun x _ => by ring)

/-! ## Binary32 at three neighbours `4095, 4097, 4097` -/

/-- Three neighbours whose variance is `8 / 9`. -/
def neighbours : List Binary32 := [4095, 4097, 4097]

/-- PyG's formula decodes to `-2`, a negative variance, while the two-pass variance decodes to
the binary32 value nearest `8 / 9`. -/
theorem binary32 :
    ExecFloat.Binary.toRat? (libraryVar neighbours) = some (-2) ∧
      ExecFloat.Binary.toRat? (var neighbours) = some (14913081 / 2 ^ 24) ∧
      exact [4095, 4097, 4097] = 8 / 9 := by
  refine ⟨by decide +kernel, by decide +kernel, ?_⟩
  simp [exact]
  norm_num

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The last step of PyG's formula from the two decoded means: its absolute part is the unit
roundoff times the squared mean `m * m`, which does not shrink with the variance. The rounding
errors of the two means enter through `m₂` and `m` and are not part of this step. -/
roundoff mixed theorem library_step_error :=
  fun (m₂ m : Value) (_hformat : format.isIEEE = true)
    (_h₂ : ExecFloat.Binary.isFinite m₂ = true) (_hm : ExecFloat.Binary.isFinite m = true) =>
    ExecFloat.Binary.toReal (m₂ - m * m)

/-- The two-pass formula from a given mean `m`: under range conditions at every step, the relative
error of its fold grows from the squared deviations `(x - m) ^ 2` and the absolute error is
underflow only. -/
roundoff mixed theorem var_error :=
  fun (xs : List Value) (m : Value) (_hformat : format.isIEEE = true)
    (_hm : ExecFloat.Binary.isFinite m = true) =>
    ExecFloat.Binary.toReal (libraryMean (xs.map fun x => (x - m) * (x - m)))

end Roundoff

end SumBugs.Variance

/-! ## The survival function as one minus the CDF -/

namespace ProbBugs.Survival

section Formulas

variable {α : Type} [Context α]

/-- The Weibull CDF of TensorFlow Probability's `WeibullCDF` bijector, `-expm1 (-t)` at
`t = (x / scale) ^ k`. -/
def libraryCDF (t : α) : α := -Context.expm1 (-t)

/-- The default survival function of a distribution without its own, `1 - libraryCDF`. -/
def librarySf (t : α) : α := 1 - libraryCDF t

/-- The default log survival function, `log1p (-cdf)`. -/
def libraryLogSf (t : α) : α := Context.log1p (-libraryCDF t)

/-- The Weibull survival function `exp (-t)`. -/
def sf (t : α) : α := MathFunctions.exp (-t)

/-- The Weibull log survival function `-t`. -/
def logSf (t : α) : α := -t

end Formulas

/-- Over the reals both survival functions are `exp (-t)`. -/
theorem sf_real (t : ℝ) : librarySf t = Real.exp (-t) ∧ sf t = Real.exp (-t) := by
  refine ⟨?_, by simp only [sf, MathFunctions.exp_real]⟩
  rw [librarySf, libraryCDF,
    Context.expm1_eq_exp_sub_one]
  ring

/-- Over the reals both log survival functions are `-t`. -/
theorem logSf_real (t : ℝ) : libraryLogSf t = -t ∧ logSf t = -t := by
  refine ⟨?_, rfl⟩
  have hlog (v : ℝ) : Context.log1p v = Real.log (1 + v) := Context.log1p_eq_log_one_add v
  rw [libraryLogSf, libraryCDF,
    Context.expm1_eq_exp_sub_one, hlog,
    show 1 + -(-(Real.exp (-t) - 1)) = Real.exp (-t) by ring, Real.log_exp]

/-! ## Binary32 at `t = 32` -/

/-- At `t = 32`, Weibull shape five at `x = 2`, the binary32 CDF decodes to one, the library
survival function to zero, which is not within any relative error below one of `exp (-32)`, and
the library log survival function is not finite. `sf` decodes to `1868901 / 2 ^ 67`, within
relative error `2⁻²³` of `exp (-32)`, and `logSf` is `-32`. -/
theorem binary32 :
    ExecFloat.Binary.toRat? (libraryCDF (32 : Binary32)) = some 1 ∧
      ExecFloat.Binary.toRat? (librarySf (32 : Binary32)) = some 0 ∧
      (∀ r < 1, ¬ MixedError (ExecFloat.Binary.toReal (librarySf (32 : Binary32)))
        (Real.exp (-32)) r 0) ∧
      ExecFloat.Binary.isFinite (libraryLogSf (32 : Binary32)) = false ∧
      ExecFloat.Binary.toRat? (sf (32 : Binary32)) = some (1868901 / 2 ^ 67) ∧
      MixedError (ExecFloat.Binary.toReal (sf (32 : Binary32))) (Real.exp (-32)) (1 / 2 ^ 23) 0 ∧
      ExecFloat.Binary.toRat? (logSf (32 : Binary32)) = some (-32) := by
  have hlib : ExecFloat.Binary.toRat? (librarySf (32 : Binary32)) = some 0 := by
    decide +kernel
  have hst : ExecFloat.Binary.toRat? (sf (32 : Binary32)) = some (1868901 / 2 ^ 67) := by
    decide +kernel
  have h1 : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
  have h2 : Real.exp 1 < 2.7182818286 := Real.exp_one_lt_d9
  have h32 : Real.exp 32 = Real.exp 1 ^ 32 := by rw [← Real.exp_nat_mul]; norm_num
  have hlo : (2.7182818283 : ℝ) ^ 32 ≤ Real.exp 32 := by
    rw [h32]; exact pow_le_pow_left₀ (by norm_num) h1.le 32
  have hhi : Real.exp 32 ≤ (2.7182818286 : ℝ) ^ 32 := by
    rw [h32]; exact pow_le_pow_left₀ (Real.exp_pos 1).le h2.le 32
  have hE : 0 < Real.exp 32 := Real.exp_pos 32
  refine ⟨by decide +kernel, hlib, fun r hr => ?_, by decide +kernel, hst, ?_, by decide +kernel⟩
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hlib, Rat.cast_zero]
    exact fun h => h.ne_zero_of_ne_zero hr (Real.exp_pos _).ne' rfl
  · rw [ExecFloat.Binary.toReal_of_toRat?_eq_some hst]
    refine MixedError.of_abs_sub_le_mul (by norm_num) ?_
    rw [Real.exp_neg, abs_of_pos (inv_pos.mpr hE)]
    have hsplit : ((1868901 / 2 ^ 67 : ℚ) : ℝ) - (Real.exp 32)⁻¹ =
        ((1868901 / 2 ^ 67 : ℝ) * Real.exp 32 - 1) * (Real.exp 32)⁻¹ := by
      push_cast; field_simp
    rw [hsplit, abs_mul, abs_of_pos (inv_pos.mpr hE)]
    refine mul_le_mul_of_nonneg_right (abs_le.mpr ⟨?_, ?_⟩) (inv_nonneg.mpr hE.le)
    · norm_num at hlo ⊢; linarith
    · norm_num at hhi ⊢; linarith

/-! ## Every IEEE format -/

section Roundoff

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The library survival function: the absolute part of the bound contains the unit roundoff
times the CDF, about one in the upper tail, which does not shrink with `exp (-t)`. -/
roundoff mixed theorem library_sf_error :=
  fun (t : Value) (_hformat : format.isIEEE = true)
    (_ht : ExecFloat.Binary.isFinite t = true) =>
    ExecFloat.Binary.toReal (librarySf t)

/-- `sf`: besides underflow, the bound is the unit roundoff times `exp (-t)`. -/
roundoff mixed theorem sf_error :=
  fun (t : Value) (_hformat : format.isIEEE = true)
    (_ht : ExecFloat.Binary.isFinite t = true) =>
    ExecFloat.Binary.toReal (sf t)

end Roundoff

end ProbBugs.Survival

end TorchLean.Examples
