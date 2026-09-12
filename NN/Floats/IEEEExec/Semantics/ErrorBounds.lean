/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Error
public import NN.Floats.IEEEExec.Bridge.FP32Total.Arithmetic

/-!
# `IEEE32Exec` per-op real error bounds (finite branch)

`NN.Floats.IEEEExec.Bridge.FP32Total` provides refinement theorems of the form

$$
\operatorname{toReal}(\operatorname{op}_{\mathrm{exec}}(x,y))
=\operatorname{fp32Round}\!\left(
  \operatorname{op}_{\mathrm{real}}(\operatorname{toReal}(x),\operatorname{toReal}(y))
\right),
$$

valid when the executable result stays finite (no NaN/Inf).

This file turns those equalities into the standard **half-ULP absolute error bounds** you want in
numerical proofs:

$$
\left|
\operatorname{toReal}(\operatorname{op}_{\mathrm{exec}}(x,y))
-\operatorname{op}_{\mathrm{real}}(\operatorname{toReal}(x),\operatorname{toReal}(y))
\right|
\le
\varepsilon_{32}\!\left(
  \operatorname{op}_{\mathrm{real}}(\operatorname{toReal}(x),\operatorname{toReal}(y))
\right).
$$

We intentionally do *not* provide bounds for `sin`/`cos` here: the current executable
implementation is deterministic (see `NN.Floats.IEEEExec.Exec32`), but it is an algorithmic
approximation rather than “real trig + one rounding step”, so its real-analytic error bounds live
in a separate trig-specific theory file.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

open TorchLean.Floats

noncomputable section

/-- `fp32Round` has the standard half-ULP absolute error bound. -/
theorem fp32Round_abs_error (x : ℝ) :
    _root_.abs (fp32Round x - x) ≤ eps32 x := by
  -- `fp32Round` is definitionally the `FP32` rounding operator.
  simpa [fp32Round] using (TorchLean.Floats.FP32.round_abs_error (x := x))

/-- In the normal range, executable binary32 rounding has relative error at most $2^{-24}$. -/
theorem fp32Round_relative_error_of_normal (x : ℝ) (hx : x ≠ 0)
    (hnormal : TorchLean.Floats.FP32.minNormal ≤ _root_.abs x) :
    ErrorBounds.relativeError x (fp32Round x) hx ≤ neuralBpow binaryRadix (-24) := by
  simpa [fp32Round] using
    (TorchLean.Floats.FP32.round_relative_error_of_normal x hx hnormal)

/--
Addition absolute error bound for `IEEE32Exec` on the finite branch.

Informal: if `add x y` stays finite then
$|\operatorname{toReal}(\operatorname{add}(x,y))
-(\operatorname{toReal}(x)+\operatorname{toReal}(y))|
\le\varepsilon_{32}(\operatorname{toReal}(x)+\operatorname{toReal}(y))$.
-/
theorem toReal_add_abs_error_of_isFinite (x y : IEEE32Exec)
    (hfin : isFinite (add x y) = true) :
    _root_.abs (toReal (add x y) - (toReal x + toReal y)) ≤ eps32 (toReal x + toReal y) := by
  simpa [toReal_add_eq_fp32Round_of_isFinite (x := x) (y := y) hfin] using
    fp32Round_abs_error (x := toReal x + toReal y)

/-- Subtraction has the standard half-ULP bound whenever the executable result is finite. -/
theorem toReal_sub_abs_error_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true)
    (hfin : isFinite (sub x y) = true) :
    _root_.abs (toReal (sub x y) - (toReal x - toReal y)) ≤ eps32 (toReal x - toReal y) := by
  simpa [toReal_sub_eq_fp32Round_of_isFinite (x := x) (y := y) hx hy hfin] using
    fp32Round_abs_error (x := toReal x - toReal y)

/--
Multiplication absolute error bound for `IEEE32Exec` on the finite branch.

Informal: if `mul x y` stays finite then
$|\operatorname{toReal}(\operatorname{mul}(x,y))
-\operatorname{toReal}(x)\operatorname{toReal}(y)|
\le\varepsilon_{32}(\operatorname{toReal}(x)\operatorname{toReal}(y))$.
-/
theorem toReal_mul_abs_error_of_isFinite (x y : IEEE32Exec)
    (hfin : isFinite (mul x y) = true) :
    _root_.abs (toReal (mul x y) - (toReal x * toReal y)) ≤ eps32 (toReal x * toReal y) := by
  simpa [toReal_mul_eq_fp32Round_of_isFinite (x := x) (y := y) hfin] using
    fp32Round_abs_error (x := toReal x * toReal y)

/--
Division absolute error bound for `IEEE32Exec` on the finite branch.

Informal: if `div x y` stays finite then
$\left|\operatorname{toReal}(\operatorname{div}(x,y))
-\frac{\operatorname{toReal}(x)}{\operatorname{toReal}(y)}\right|
\le\varepsilon_{32}\!\left(
  \frac{\operatorname{toReal}(x)}{\operatorname{toReal}(y)}
\right)$.
-/
theorem toReal_div_abs_error_of_isFinite (x y : IEEE32Exec)
    (hfin : isFinite (div x y) = true) :
    _root_.abs (toReal (div x y) - (toReal x / toReal y)) ≤ eps32 (toReal x / toReal y) := by
  simpa [toReal_div_eq_fp32Round_of_isFinite (x := x) (y := y) hfin] using
    fp32Round_abs_error (x := toReal x / toReal y)

/--
FMA absolute error bound for `IEEE32Exec` on the finite branch.

Informal: if `fma x y z` stays finite then
$|\operatorname{toReal}(\operatorname{fma}(x,y,z))
-(\operatorname{toReal}(x)\operatorname{toReal}(y)+\operatorname{toReal}(z))|
\le\varepsilon_{32}(
  \operatorname{toReal}(x)\operatorname{toReal}(y)+\operatorname{toReal}(z)
)$.
-/
theorem toReal_fma_abs_error_of_isFinite (x y z : IEEE32Exec)
    (hfin : isFinite (fma x y z) = true) :
    _root_.abs (toReal (fma x y z) - (toReal x * toReal y + toReal z)) ≤
      eps32 (toReal x * toReal y + toReal z) := by
  simpa [toReal_fma_eq_fp32Round_of_isFinite (x := x) (y := y) (z := z) hfin] using
    fp32Round_abs_error (x := toReal x * toReal y + toReal z)

/--
Square-root absolute error bound for `IEEE32Exec` on the finite branch.

Informal: if `sqrt x` stays finite then
$|\operatorname{toReal}(\operatorname{sqrt}(x))-\sqrt{\operatorname{toReal}(x)}|
\le\varepsilon_{32}(\sqrt{\operatorname{toReal}(x)})$.
-/
theorem toReal_sqrt_abs_error_of_isFinite (x : IEEE32Exec)
    (hfin : isFinite (sqrt x) = true) :
    _root_.abs (toReal (sqrt x) - Real.sqrt (toReal x)) ≤ eps32 (Real.sqrt (toReal x)) := by
  simpa [toReal_sqrt_eq_fp32Round_of_isFinite (x := x) hfin] using
    fp32Round_abs_error (x := Real.sqrt (toReal x))

end

end IEEE32Exec
end TorchLean.Floats.IEEE754
