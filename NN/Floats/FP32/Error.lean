/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module


import Mathlib.Algebra.Order.Algebra
public import NN.Floats.FP32.Notation
public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel

/-!
# `FP32` per-op error bounds

Proofs about numerical code should not have to expand the definition of “float32 rounding” every
time they use `+` or `exp`. This file provides reusable rewrite theorems for FP32
rounding-error reasoning.

We collect those lemmas for TorchLean’s `FP32` model:

- each primitive op is interpreted as “compute in `ℝ`, then round to the binary32 grid”, and
- the rounding step has a standard **half-ULP** absolute error bound.

This is the standard real-computation-plus-rounding-error model used in numerical analysis. It is
not a bit-level IEEE execution model.

The pattern is always:

- compute the exact real result,
- apply the `FP32` rounding operator,
- bound the rounding error by $\varepsilon_{32}(x)$, half an ulp at $x$; see
  `NN/Floats/FP32/Notation.lean`.

Important nuance: these are **local** statements about a *single* rounding step. Whole-network
bounds typically combine these with:

- algebra/triangle-inequality steps, or
- higher-level composition lemmas in `NN/Proofs/RuntimeApprox/*`.
-/

@[expose] public section

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


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
$|\operatorname{fl}_{32}(x)-x|\le\varepsilon_{32}(x)$.
-/
theorem round_abs_error (x : ℝ) :
    abs (round32 x - x) ≤ eps32 x :=
  Model.abs_roundAt_sub_le FloatFormat.binary32 x

/-- Normal binary32 rounding has relative error at most the unit roundoff $2^{-24}$. -/
theorem round_relative_error_of_normal (x : ℝ) (hx : x ≠ 0)
    (hnormal : minNormal ≤ abs x) :
    ErrorBounds.relativeError x (round32 x) hx ≤ bpow binaryRadix (-24) := by
  have hbound := Model.relativeError_roundAt_le_of_normal FloatFormat.binary32 x hx hnormal
  rw [← Model.two_mul_unitRoundoffAt, mul_div_cancel_left₀ _ two_ne_zero] at hbound
  exact hbound

/--
Addition: `FP32` adds in `ℝ` and then rounds once.

This lemma isolates the rounding error introduced by that last step.

Informally,
$|\operatorname{fl}_{32}(a+b)-(a+b)|\le\varepsilon_{32}(a+b)$.
-/
theorem add_abs_error (a b : FP32) :
    abs ((a + b).val - (a.val + b.val)) ≤
      eps32 (a.val + b.val) :=
  round_abs_error (a.val + b.val)

/--
Subtraction: one real subtraction followed by one rounding step.

Informally,
$|\operatorname{fl}_{32}(a-b)-(a-b)|\le\varepsilon_{32}(a-b)$.
-/
theorem sub_abs_error (a b : FP32) :
    abs ((a - b).val - (a.val - b.val)) ≤
      eps32 (a.val - b.val) :=
  round_abs_error (a.val - b.val)

/--
Multiplication: one real multiplication followed by one rounding step.

Informally,
$|\operatorname{fl}_{32}(ab)-ab|\le\varepsilon_{32}(ab)$.
-/
theorem mul_abs_error (a b : FP32) :
    abs ((a * b).val - (a.val * b.val)) ≤
      eps32 (a.val * b.val) :=
  round_abs_error (a.val * b.val)

/--
Division: one real division followed by one rounding step.

This does *not* say division is “well-conditioned”; it only isolates the rounding stage.

Informally,
$\left|\operatorname{fl}_{32}(a/b)-a/b\right|\le\varepsilon_{32}(a/b)$.
-/
theorem div_abs_error (a b : FP32) :
    abs ((a / b).val - (a.val / b.val)) ≤
      eps32 (a.val / b.val) :=
  round_abs_error (a.val / b.val)

/-! ## Transcendentals (proof semantics) -/

/--
In `FP32`, transcendental functions are specified as: apply the real function, then round.

That is *not* how hardware/libm is implemented, but it is exactly the right abstraction for proofs:
it gives a clear mathematical meaning and a one-rounding-step error theorem.

Informally,
$|\operatorname{fl}_{32}(\exp x)-\exp x|\le\varepsilon_{32}(\exp x)$,
and similarly for the other functions below.
-/
theorem exp_abs_error (a : FP32) :
    abs ((FloatLib.Numerics.MathFunctions.exp a).val - Real.exp a.val) ≤
      eps32 (Real.exp a.val) :=
  round_abs_error (Real.exp a.val)

/--
`tanh` in the proof model: real `tanh` followed by rounding.

Informally,
$|\operatorname{fl}_{32}(\tanh x)-\tanh x|\le\varepsilon_{32}(\tanh x)$.
-/
theorem tanh_abs_error (a : FP32) :
    abs ((FloatLib.Numerics.MathFunctions.tanh a).val - Real.tanh a.val) ≤
      eps32 (Real.tanh a.val) :=
  round_abs_error (Real.tanh a.val)

/--
`log` in the proof model: real `log` followed by rounding.

Domain note: the inequality is about rounding error around `Real.log a.val`; it does not assert
anything about `a.val > 0`.

Informally,
$|\operatorname{fl}_{32}(\log x)-\log x|\le\varepsilon_{32}(\log x)$.
-/
theorem log_abs_error (a : FP32) :
    abs ((FloatLib.Numerics.MathFunctions.log a).val - Real.log a.val) ≤
      eps32 (Real.log a.val) :=
  round_abs_error (Real.log a.val)

/--
`cos` in the proof model: real `cos` followed by rounding.

Informally,
$|\operatorname{fl}_{32}(\cos x)-\cos x|\le\varepsilon_{32}(\cos x)$.
-/
theorem cos_abs_error (a : FP32) :
    abs ((FloatLib.Numerics.MathFunctions.cos a).val - Real.cos a.val) ≤
      eps32 (Real.cos a.val) :=
  round_abs_error (Real.cos a.val)

/--
`sin` in the proof model: real `sin` followed by rounding.

Informally,
$|\operatorname{fl}_{32}(\sin x)-\sin x|\le\varepsilon_{32}(\sin x)$.
-/
theorem sin_abs_error (a : FP32) :
    abs ((FloatLib.Numerics.MathFunctions.sin a).val - Real.sin a.val) ≤
      eps32 (Real.sin a.val) :=
  round_abs_error (Real.sin a.val)

/--
`sinh` in the proof model: real `sinh` followed by rounding.

Informally,
$|\operatorname{fl}_{32}(\sinh x)-\sinh x|\le\varepsilon_{32}(\sinh x)$.
-/
theorem sinh_abs_error (a : FP32) :
    abs ((FloatLib.Numerics.MathFunctions.sinh a).val - Real.sinh a.val) ≤
      eps32 (Real.sinh a.val) :=
  round_abs_error (Real.sinh a.val)

/--
`cosh` in the proof model: real `cosh` followed by rounding.

Informally,
$|\operatorname{fl}_{32}(\cosh x)-\cosh x|\le\varepsilon_{32}(\cosh x)$.
-/
theorem cosh_abs_error (a : FP32) :
    abs ((FloatLib.Numerics.MathFunctions.cosh a).val - Real.cosh a.val) ≤
      eps32 (Real.cosh a.val) :=
  round_abs_error (Real.cosh a.val)

/--
`sqrt` in the proof model: real `sqrt` followed by rounding.

Informally,
$|\operatorname{fl}_{32}(\sqrt{x})-\sqrt{x}|\le\varepsilon_{32}(\sqrt{x})$.
-/
theorem sqrt_abs_error (a : FP32) :
    abs ((FloatLib.Numerics.MathFunctions.sqrt a).val - Real.sqrt a.val) ≤
      eps32 (Real.sqrt a.val) :=
  round_abs_error (Real.sqrt a.val)

/--
`abs` in the proof model: real absolute value followed by rounding.

Even though $|x|$ is exact over $\mathbb R$, we still round the result because this is a
“round after every primitive” semantics.

Informally,
$\left|\operatorname{fl}_{32}(|x|)-|x|\right|\le\varepsilon_{32}(|x|)$.
-/
theorem abs_abs_error (a : FP32) :
    abs ((FloatLib.Numerics.MathFunctions.abs a).val - |a.val|) ≤
      eps32 (|a.val|) :=
  round_abs_error |a.val|

end FP32
end TorchLean.Floats
