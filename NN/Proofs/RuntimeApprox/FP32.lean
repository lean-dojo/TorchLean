/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32
public import NN.Proofs.RuntimeApprox.Core.Tolerance

/-!
# FP32 rounding tolerances

These theorems express FloatLib's `Model.abs_roundAt_sub_le` bounds using `approxR`
(notation `≈[t]`). Each operation has an absolute-only tolerance of half an ULP at its
exact real result. The tolerance therefore depends on the result's magnitude.

`FP32` here is a rounded-real model with binary32 precision and an unbounded upper
exponent. It has no NaN or infinity values. These are not native `Float32` or LibTorch
kernel correctness theorems; transcendental operations round the exact real function.
-/

@[expose] public section

open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

namespace TorchLean.Floats
namespace FP32

open Proofs.RuntimeApprox

/-- Express a nonnegative absolute error bound as an absolute-only tolerance. -/
private theorem approxR_absOnly_of_abs_sub_le {x y eps : ℝ} (heps : 0 ≤ eps)
    (h : abs (y - x) ≤ eps) : approxR x y (ApproxTol.absOnly eps) :=
  (approxR_absOnly_iff (x := x) (y := y) (eps := eps) heps).2 h

/-- The binary32 half-ULP tolerance is nonnegative. -/
private theorem epsilon_nonneg (x : ℝ) : 0 ≤ Model.epsilonAt FloatFormat.binary32 x := by
  unfold Model.epsilonAt Model.ulpAt
  exact div_nonneg
    (ulp.nonneg (β := binaryRadix) (fexp := (Model.fexpOf FloatFormat.binary32)) (x := x))
    (by norm_num)

/-! ## Arithmetic (one real op + one rounding step) -/

/-- Rounded addition differs from the exact sum by at most half an ULP. -/
theorem add_approxR (a b : FP32) :
    approxR (a.val + b.val) (a + b).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val + b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (a.val + b.val)

/-- Rounded subtraction differs from the exact difference by at most half an ULP. -/
theorem sub_approxR (a b : FP32) :
    approxR (a.val - b.val) (a - b).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val - b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (a.val - b.val)

/-- Rounded multiplication differs from the exact product by at most half an ULP. -/
theorem mul_approxR (a b : FP32) :
    approxR (a.val * b.val) (a * b).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val * b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (a.val * b.val)

/-- Rounded division follows real division, including its totalized value at zero. -/
theorem div_approxR (a b : FP32) :
    approxR (a.val / b.val) (a / b).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val / b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (a.val / b.val)

/-! ## Transcendentals (real function + rounding) -/

/-- Rounding the exact real exp result introduces at most half an ULP of error. -/
theorem exp_approxR (a : FP32) :
    approxR (Real.exp a.val) (Numerics.MathFunctions.exp a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.exp a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.exp a.val)

/-- Rounding the exact real tanh result introduces at most half an ULP of error. -/
theorem tanh_approxR (a : FP32) :
    approxR (Real.tanh a.val) (Numerics.MathFunctions.tanh a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.tanh a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.tanh a.val)

/-- Rounded logarithm uses mathlib's total real logarithm as its reference. -/
theorem log_approxR (a : FP32) :
    approxR (Real.log a.val) (Numerics.MathFunctions.log a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.log a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.log a.val)

/-- Rounding the exact real cos result introduces at most half an ULP of error. -/
theorem cos_approxR (a : FP32) :
    approxR (Real.cos a.val) (Numerics.MathFunctions.cos a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.cos a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.cos a.val)

/-- Rounding the exact real sin result introduces at most half an ULP of error. -/
theorem sin_approxR (a : FP32) :
    approxR (Real.sin a.val) (Numerics.MathFunctions.sin a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.sin a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.sin a.val)

/-- Rounding the exact real sinh result introduces at most half an ULP of error. -/
theorem sinh_approxR (a : FP32) :
    approxR (Real.sinh a.val) (Numerics.MathFunctions.sinh a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.sinh a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.sinh a.val)

/-- Rounding the exact real cosh result introduces at most half an ULP of error. -/
theorem cosh_approxR (a : FP32) :
    approxR (Real.cosh a.val) (Numerics.MathFunctions.cosh a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.cosh a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.cosh a.val)

/-- Rounded square root uses mathlib's total real square root as its reference. -/
theorem sqrt_approxR (a : FP32) :
    approxR (Real.sqrt a.val) (Numerics.MathFunctions.sqrt a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.sqrt a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.sqrt a.val)

/-- Rounding the exact real abs result introduces at most half an ULP of error. -/
theorem abs_approxR (a : FP32) :
    approxR (abs a.val) (Numerics.MathFunctions.abs a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (abs a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (abs a.val)

/-! ## Examples (how this looks in practice) -/

section Examples

open scoped ApproxTol

variable (a b : FP32)

/-- The same `add_approxR` theorem, but written using the `≈[t]` notation. -/
example :
    (a.val + b.val) ≈[
      ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val + b.val))
    ] (a + b).val := by
  simpa using add_approxR (a := a) (b := b)

/--
An `≈[absOnly eps]` goal can always be unpacked back to a plain absolute error inequality.

This is useful when feeding the result into lemmas stated using `abs`, or into `linarith`.
-/
example :
    abs ((a + b).val - (a.val + b.val)) ≤
      Model.epsilonAt FloatFormat.binary32 (a.val + b.val) := by
  have happ :
      approxR (a.val + b.val) (a + b).val
        (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val + b.val))) :=
    add_approxR (a := a) (b := b)
  have heps :
      0 ≤ Model.epsilonAt FloatFormat.binary32 (a.val + b.val) :=
    epsilon_nonneg (x := a.val + b.val)
  -- `approxR_absOnly_iff` says `≈[absOnly eps]` is exactly `|y - x| ≤ eps`.
  have : abs ((a + b).val - (a.val + b.val)) ≤
      Model.epsilonAt FloatFormat.binary32 (a.val + b.val) :=
    (approxR_absOnly_iff (x := a.val + b.val) (y := (a + b).val)
      (eps := Model.epsilonAt FloatFormat.binary32 (a.val + b.val)) heps).1 happ
  simpa [abs_sub_comm] using this

end Examples

end FP32
end TorchLean.Floats
