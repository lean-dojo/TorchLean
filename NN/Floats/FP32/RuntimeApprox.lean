/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Notation
public import NN.Proofs.RuntimeApprox.Core.Tolerance
import Mathlib.Algebra.Order.Algebra
import NN.Floats.FP32.Error

/-!
# Bridging `FP32` error bounds into `≈[t]`

`NN.Proofs.RuntimeApprox.Core.Tolerance` defines a small “close enough” relation:

* `approxR x y t`  (notation: `x ≈[t] y`)

Here we connect that generic notion of tolerance to our `FP32` rounding model.

In `NN.Floats.FP32.Error` we prove *per‑operation* absolute error bounds like

```
abs (approx - exact) ≤ eps₃₂ exact
```

We repackage those bounds so downstream proofs can use the uniform `≈[t]` vocabulary.

Two small conventions show up everywhere below:

* We use an **absolute-only tolerance** `ApproxTol.absOnly eps`.
  That means `x ≈[absOnly eps] y` is literally the usual bound `|y - x| ≤ eps`
  (see `approxR_absOnly_iff`). This matches the shape of our `FP32` theorems.

* The `FP32` “epsilon” is **value-dependent**: `eps = ulp(exact) / 2 = eps₃₂(exact)`.
  This is the standard round-to-nearest error model: the ulp is smaller near 0 and grows as the
  magnitude grows. Writing it as an `≈[t]` fact makes it easy to mix with other tolerances without
  inventing yet another approximation relation.
-/

@[expose] public section


namespace TorchLean.Floats
namespace FP32

open Proofs.RuntimeApprox

/-!
The next two lemmas are local rewrite helpers:

* `approxR_absOnly_of_abs_sub_le` turns a plain `abs (y - x) ≤ eps` inequality into an `approxR`.
* `eps32_nonneg` is the nonnegativity proof we need to use `approxR_absOnly_iff`.

They are private because they are only used by this file's approximation bridge.
-/

/--
Helper: turn a plain absolute-error inequality into an `approxR` with an absolute-only tolerance.

Informal: if `|y - x| ≤ eps` and `eps ≥ 0`, then `x ≈[absOnly eps] y`.
-/
private lemma approxR_absOnly_of_abs_sub_le {x y eps : ℝ} (heps : 0 ≤ eps) (h : abs (y - x) ≤ eps) :
    approxR x y (ApproxTol.absOnly eps) :=
  (approxR_absOnly_iff (x := x) (y := y) (eps := eps) heps).2 h

/--
Nonnegativity of the FP32 half-ULP scale `eps₃₂`.

This is needed to use `approxR_absOnly_iff` (which requires `0 ≤ eps`).
-/
private lemma eps32_nonneg (x : ℝ) : 0 ≤ eps₃₂ x := by
  -- Unfold to the underlying `neural_ulp` so we can reuse its nonnegativity lemma.
  unfold eps₃₂ eps32 ulp32
  exact div_nonneg
    (neuralUlp.nonneg (β := binaryRadix) (fexp := fexp32) (x := x)
      (phase := TrainingPhase.forward))
    (by norm_num)

/-! ## Arithmetic (one real op + one rounding step) -/

/--
`FP32` addition, stated as an `≈[t]` fact with `t = absOnly (ulp(exact)/2)`.

Read this as:

- “the exact real sum” is `a.val + b.val`,
- “the rounded result” is `(a + b).val`,
- and the two differ by at most half an ulp of the exact real sum.

Informal: `a.val + b.val ≈[absOnly (eps₃₂(a.val + b.val))] (a + b).val`.
-/
theorem add_approxR (a b : FP32) :
    approxR (a.val + b.val) (a + b).val
      (ApproxTol.absOnly (eps₃₂ (a.val + b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := a.val + b.val) (y := (a + b).val) (eps := _)
    (eps32_nonneg (x := a.val + b.val)) ?_
  simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using add_abs_error (a := a) (b := b)

/--
`FP32` subtraction, stated as an `≈[t]` fact with `t = absOnly (eps₃₂(exact))`.

Informal: `a.val - b.val ≈[absOnly (eps₃₂(a.val - b.val))] (a - b).val`.
-/
theorem sub_approxR (a b : FP32) :
    approxR (a.val - b.val) (a - b).val
      (ApproxTol.absOnly (eps₃₂ (a.val - b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := a.val - b.val) (y := (a - b).val) (eps := _)
    (eps32_nonneg (x := a.val - b.val)) ?_
  simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using sub_abs_error (a := a) (b := b)

/--
`FP32` multiplication, stated as an `≈[t]` fact with `t = absOnly (eps₃₂(exact))`.

Informal: `a.val * b.val ≈[absOnly (eps₃₂(a.val * b.val))] (a * b).val`.
-/
theorem mul_approxR (a b : FP32) :
    approxR (a.val * b.val) (a * b).val
      (ApproxTol.absOnly (eps₃₂ (a.val * b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := a.val * b.val) (y := (a * b).val) (eps := _)
    (eps32_nonneg (x := a.val * b.val)) ?_
  simpa using mul_abs_error (a := a) (b := b)

/--
`FP32` division, stated as an `≈[t]` fact with `t = absOnly (eps₃₂(exact))`.

Informal: `a.val / b.val ≈[absOnly (eps₃₂(a.val / b.val))] (a / b).val`.
-/
theorem div_approxR (a b : FP32) :
    approxR (a.val / b.val) (a / b).val
      (ApproxTol.absOnly (eps₃₂ (a.val / b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := a.val / b.val) (y := (a / b).val) (eps := _)
    (eps32_nonneg (x := a.val / b.val)) ?_
  simpa using div_abs_error (a := a) (b := b)

/-! ## Transcendentals (real function + rounding) -/

/--
`FP32` `exp` is `Real.exp` followed by rounding; this is the result as an `≈[t]` statement.

Informal: `exp(a.val) ≈[absOnly (eps₃₂(exp(a.val)))] (exp a).val`.
-/
theorem exp_approxR (a : FP32) :
    approxR (Real.exp a.val) (MathFunctions.exp a).val
      (ApproxTol.absOnly (eps₃₂ (Real.exp a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := Real.exp a.val) (y := (MathFunctions.exp a).val) (eps
    := _)
    (eps32_nonneg (x := Real.exp a.val)) ?_
  simpa using exp_abs_error (a := a)

/--
`FP32` `tanh` as an `≈[t]` statement.

Informal: `tanh(a.val) ≈[absOnly (eps₃₂(tanh(a.val)))] (tanh a).val`.
-/
theorem tanh_approxR (a : FP32) :
    approxR (Real.tanh a.val) (MathFunctions.tanh a).val
      (ApproxTol.absOnly (eps₃₂ (Real.tanh a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := Real.tanh a.val) (y := (MathFunctions.tanh a).val) (eps
    := _)
    (eps32_nonneg (x := Real.tanh a.val)) ?_
  simpa using tanh_abs_error (a := a)

/--
`FP32` `log` as an `≈[t]` statement (using `Real.log` as the exact reference).

Informal: `log(a.val) ≈[absOnly (eps₃₂(log(a.val)))] (log a).val`.
-/
theorem log_approxR (a : FP32) :
    approxR (Real.log a.val) (MathFunctions.log a).val
      (ApproxTol.absOnly (eps₃₂ (Real.log a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := Real.log a.val) (y := (MathFunctions.log a).val) (eps
    := _)
    (eps32_nonneg (x := Real.log a.val)) ?_
  simpa using log_abs_error (a := a)

/--
`FP32` `cos` as an `≈[t]` statement.

Informal: `cos(a.val) ≈[absOnly (eps₃₂(cos(a.val)))] (cos a).val`.
-/
theorem cos_approxR (a : FP32) :
    approxR (Real.cos a.val) (MathFunctions.cos a).val
      (ApproxTol.absOnly (eps₃₂ (Real.cos a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := Real.cos a.val) (y := (MathFunctions.cos a).val) (eps
    := _)
    (eps32_nonneg (x := Real.cos a.val)) ?_
  simpa using cos_abs_error (a := a)

/--
`FP32` `sin` as an `≈[t]` statement.

Informal: `sin(a.val) ≈[absOnly (eps₃₂(sin(a.val)))] (sin a).val`.
-/
theorem sin_approxR (a : FP32) :
    approxR (Real.sin a.val) (MathFunctions.sin a).val
      (ApproxTol.absOnly (eps₃₂ (Real.sin a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := Real.sin a.val) (y := (MathFunctions.sin a).val) (eps
    := _)
    (eps32_nonneg (x := Real.sin a.val)) ?_
  simpa using sin_abs_error (a := a)

/--
`FP32` `sinh` as an `≈[t]` statement.

Informal: `sinh(a.val) ≈[absOnly (eps₃₂(sinh(a.val)))] (sinh a).val`.
-/
theorem sinh_approxR (a : FP32) :
    approxR (Real.sinh a.val) (MathFunctions.sinh a).val
      (ApproxTol.absOnly (eps₃₂ (Real.sinh a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := Real.sinh a.val) (y := (MathFunctions.sinh a).val) (eps
    := _)
    (eps32_nonneg (x := Real.sinh a.val)) ?_
  simpa using sinh_abs_error (a := a)

/--
`FP32` `cosh` as an `≈[t]` statement.

Informal: `cosh(a.val) ≈[absOnly (eps₃₂(cosh(a.val)))] (cosh a).val`.
-/
theorem cosh_approxR (a : FP32) :
    approxR (Real.cosh a.val) (MathFunctions.cosh a).val
      (ApproxTol.absOnly (eps₃₂ (Real.cosh a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := Real.cosh a.val) (y := (MathFunctions.cosh a).val) (eps
    := _)
    (eps32_nonneg (x := Real.cosh a.val)) ?_
  simpa using cosh_abs_error (a := a)

/--
`FP32` `sqrt` as an `≈[t]` statement.

Informal: `sqrt(a.val) ≈[absOnly (eps₃₂(sqrt(a.val)))] (sqrt a).val`.
-/
theorem sqrt_approxR (a : FP32) :
    approxR (Real.sqrt a.val) (MathFunctions.sqrt a).val
      (ApproxTol.absOnly (eps₃₂ (Real.sqrt a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := Real.sqrt a.val) (y := (MathFunctions.sqrt a).val) (eps
    := _)
    (eps32_nonneg (x := Real.sqrt a.val)) ?_
  simpa using sqrt_abs_error (a := a)

/--
`FP32` `abs` as an `≈[t]` statement.

Informal: `|a.val| ≈[absOnly (eps₃₂(|a.val|))] (abs a).val`.
-/
theorem abs_approxR (a : FP32) :
    approxR (abs a.val) (MathFunctions.abs a).val
      (ApproxTol.absOnly (eps₃₂ (abs a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (x := abs a.val) (y := (MathFunctions.abs a).val) (eps := _)
    (eps32_nonneg (x := abs a.val)) ?_
  simpa using abs_abs_error (a := a)

/-! ## Examples (how this looks in practice) -/

section Examples

open scoped ApproxTol

variable (a b : FP32)

/-- The same `add_approxR` theorem, but written using the `≈[t]` notation. -/
example :
    (a.val + b.val) ≈[
      ApproxTol.absOnly (eps₃₂ (a.val + b.val))
    ] (a + b).val := by
  simpa using add_approxR (a := a) (b := b)

/--
An `≈[absOnly eps]` goal can always be unpacked back to a plain absolute error inequality.

This is useful when feeding the result into lemmas stated using `abs`, or into `linarith`.
-/
example :
    abs ((a + b).val - (a.val + b.val)) ≤
      eps₃₂ (a.val + b.val) := by
  have happ :
      approxR (a.val + b.val) (a + b).val
        (ApproxTol.absOnly (eps₃₂ (a.val + b.val))) :=
    add_approxR (a := a) (b := b)
  have heps :
      0 ≤ eps₃₂ (a.val + b.val) :=
    eps32_nonneg (x := a.val + b.val)
  -- `approxR_absOnly_iff` says `≈[absOnly eps]` is exactly `|y - x| ≤ eps`.
  have : abs ((a + b).val - (a.val + b.val)) ≤
      eps₃₂ (a.val + b.val) :=
    (approxR_absOnly_iff (x := a.val + b.val) (y := (a + b).val)
      (eps := eps₃₂ (a.val + b.val)) heps).1 happ
  simpa [abs_sub_comm] using this

end Examples

end FP32
end TorchLean.Floats
