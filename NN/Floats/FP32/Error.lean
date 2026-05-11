/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Notation
import Mathlib.Algebra.Order.Algebra

/-!
# `FP32` per-op error bounds

Proofs about numerical code should not have to expand the definition of “float32 rounding” every
time they use `+` or `exp`. This file provides reusable rewrite theorems for FP32
rounding-error reasoning.

We collect those lemmas for TorchLean’s `FP32` model:

- each primitive op is interpreted as “compute in `ℝ`, then round to the binary32 grid”, and
- the rounding step has a standard **half-ULP** absolute error bound.

This mirrors the standard mental model behind PyTorch/IEEE float analysis (even though our proof
backend is not a bit-level model): it’s the classic “real computation + rounding error” split.

The pattern is always:

- compute the exact real result,
- apply the `FP32` rounding operator,
- bound the rounding error by `eps₃₂ x` (half an ulp at `x`; see `NN/Floats/FP32/Notation.lean`).

Important nuance: these are **local** statements about a *single* rounding step. Whole-network
bounds typically combine these with:

- algebra/triangle-inequality steps, or
- higher-level composition lemmas in `NN/Proofs/RuntimeApprox/*`.
-/

@[expose] public section


/-!
## References

- IEEE Std 754-2019, "IEEE Standard for Floating-Point Arithmetic".
- D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic",
  ACM Computing Surveys, 1991.
- N. J. Higham, "Accuracy and Stability of Numerical Algorithms", SIAM, 2nd ed., 2002.

This module works with a proof-friendly float32 semantics ("real operation + one rounding step").
Relating this abstraction to specific hardware/libm behavior is a separate, target-specific trust
boundary handled elsewhere in TorchLean.
-/

namespace TorchLean.Floats
namespace FP32


/-! ## Per-op rounding error lemmas -/

/--
Core rounding lemma for the binary32 parameters fixed by `FP32`.

This is the “one thing we use everywhere”: once you know an operation is defined as “round the real
result”, the proof goal reduces to an instance of this lemma.

Informal: if `fl32(x)` denotes rounding `x : ℝ` to the binary32 grid, then
`|fl32(x) - x| ≤ eps₃₂(x)`.
-/
theorem round_abs_error (x : ℝ) :
    abs (round₃₂ x - x) ≤ eps₃₂ x := by
  simpa [rnd32] using
    (neural_error_bound_ulp (β := binaryRadix) (fexp := fexp32) (rnd := rnd32) x)

/--
Addition: `FP32` adds in `ℝ` and then rounds once.

This lemma isolates the rounding error introduced by that last step.

Informal: `|fl32(a+b) - (a+b)| ≤ eps₃₂(a+b)`.
-/
theorem add_abs_error (a b : FP32) :
    abs ((a + b).val - (a.val + b.val)) ≤
      eps₃₂ (a.val + b.val) := by
  -- By definition, `a + b` rounds the exact real sum.
  simpa [HAdd.hAdd, NF.instAdd, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := a.val + b.val))

/--
Subtraction: one real subtraction followed by one rounding step.

Informal: `|fl32(a-b) - (a-b)| ≤ eps₃₂(a-b)`.
-/
theorem sub_abs_error (a b : FP32) :
    abs ((a - b).val - (a.val - b.val)) ≤
      eps₃₂ (a.val - b.val) := by
  simpa [HSub.hSub, NF.instSub, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := a.val - b.val))

/--
Multiplication: one real multiplication followed by one rounding step.

Informal: `|fl32(a*b) - (a*b)| ≤ eps₃₂(a*b)`.
-/
theorem mul_abs_error (a b : FP32) :
    abs ((a * b).val - (a.val * b.val)) ≤
      eps₃₂ (a.val * b.val) := by
  simpa [HMul.hMul, NF.instMul, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := a.val * b.val))

/--
Division: one real division followed by one rounding step.

This does *not* say division is “well-conditioned”; it only isolates the rounding stage.

Informal: `|fl32(a/b) - (a/b)| ≤ eps₃₂(a/b)`.
-/
theorem div_abs_error (a b : FP32) :
    abs ((a / b).val - (a.val / b.val)) ≤
      eps₃₂ (a.val / b.val) := by
  simpa [HDiv.hDiv, NF.instDiv, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := a.val / b.val))

/-! ## Transcendentals (proof semantics) -/

/--
In `FP32`, transcendental functions are specified as: apply the real function, then round.

That is *not* how hardware/libm is implemented, but it is exactly the right abstraction for proofs:
it gives a clear mathematical meaning and a one-rounding-step error theorem.

Informal: `|fl32(exp(x)) - exp(x)| ≤ eps₃₂(exp(x))`, and similarly for the other functions below.
-/
theorem exp_abs_error (a : FP32) :
    abs ((MathFunctions.exp a).val - Real.exp a.val) ≤
      eps₃₂ (Real.exp a.val) := by
  simpa [MathFunctions.exp, NF.instMathFunctions, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := Real.exp a.val))

/--
`tanh` in the proof model: real `tanh` followed by rounding.

Informal: `|fl32(tanh(x)) - tanh(x)| ≤ eps₃₂(tanh(x))`.
-/
theorem tanh_abs_error (a : FP32) :
    abs ((MathFunctions.tanh a).val - Real.tanh a.val) ≤
      eps₃₂ (Real.tanh a.val) := by
  simpa [MathFunctions.tanh, NF.instMathFunctions, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := Real.tanh a.val))

/--
`log` in the proof model: real `log` followed by rounding.

Domain note: the inequality is about rounding error around `Real.log a.val`; it does not assert
anything about `a.val > 0`.

Informal: `|fl32(log(x)) - log(x)| ≤ eps₃₂(log(x))`.
-/
theorem log_abs_error (a : FP32) :
    abs ((MathFunctions.log a).val - Real.log a.val) ≤
      eps₃₂ (Real.log a.val) := by
  simpa [MathFunctions.log, NF.instMathFunctions, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := Real.log a.val))

/--
`cos` in the proof model: real `cos` followed by rounding.

Informal: `|fl32(cos(x)) - cos(x)| ≤ eps₃₂(cos(x))`.
-/
theorem cos_abs_error (a : FP32) :
    abs ((MathFunctions.cos a).val - Real.cos a.val) ≤
      eps₃₂ (Real.cos a.val) := by
  simpa [MathFunctions.cos, NF.instMathFunctions, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := Real.cos a.val))

/--
`sin` in the proof model: real `sin` followed by rounding.

Informal: `|fl32(sin(x)) - sin(x)| ≤ eps₃₂(sin(x))`.
-/
theorem sin_abs_error (a : FP32) :
    abs ((MathFunctions.sin a).val - Real.sin a.val) ≤
      eps₃₂ (Real.sin a.val) := by
  simpa [MathFunctions.sin, NF.instMathFunctions, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := Real.sin a.val))

/--
`sinh` in the proof model: real `sinh` followed by rounding.

Informal: `|fl32(sinh(x)) - sinh(x)| ≤ eps₃₂(sinh(x))`.
-/
theorem sinh_abs_error (a : FP32) :
    abs ((MathFunctions.sinh a).val - Real.sinh a.val) ≤
      eps₃₂ (Real.sinh a.val) := by
  simpa [MathFunctions.sinh, NF.instMathFunctions, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := Real.sinh a.val))

/--
`cosh` in the proof model: real `cosh` followed by rounding.

Informal: `|fl32(cosh(x)) - cosh(x)| ≤ eps₃₂(cosh(x))`.
-/
theorem cosh_abs_error (a : FP32) :
    abs ((MathFunctions.cosh a).val - Real.cosh a.val) ≤
      eps₃₂ (Real.cosh a.val) := by
  simpa [MathFunctions.cosh, NF.instMathFunctions, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := Real.cosh a.val))

/--
`sqrt` in the proof model: real `sqrt` followed by rounding.

Informal: `|fl32(sqrt(x)) - sqrt(x)| ≤ eps₃₂(sqrt(x))`.
-/
theorem sqrt_abs_error (a : FP32) :
    abs ((MathFunctions.sqrt a).val - Real.sqrt a.val) ≤
      eps₃₂ (Real.sqrt a.val) := by
  simpa [MathFunctions.sqrt, NF.instMathFunctions, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := Real.sqrt a.val))

/--
`abs` in the proof model: real absolute value followed by rounding.

Even though `|x|` is exact over `ℝ`, we still round the result because this is a “round after every
primitive” semantics.

Informal: `|fl32(|x|) - |x|| ≤ eps₃₂(|x|)`.
-/
theorem abs_abs_error (a : FP32) :
    abs ((MathFunctions.abs a).val - |a.val|) ≤
      eps₃₂ (|a.val|) := by
  simpa [MathFunctions.abs, NF.instMathFunctions, NF.ofReal, NF.roundR, rnd32] using
    (round_abs_error (x := |a.val|))

end FP32
end TorchLean.Floats
