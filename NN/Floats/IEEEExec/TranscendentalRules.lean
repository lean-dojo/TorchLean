/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32

/-!
## Deterministic transcendental functions (`exp`, `log`, `tanh`, `sinh`, `cosh`)

IEEE-754 fixes the representation and the basic arithmetic operations, but it does **not** mandate
exact bit-level behavior for transcendental functions like `exp` and `log`. In practice, libraries
(`libm`, SVML, CUDA math, etc.) make slightly different choices, and results can differ across
platforms.

In TorchLean, we still need executable transcendental operations in some examples and runtime
experiments. So we give `IEEE32Exec` *deterministic* definitions for a small set of functions using
fixed algorithms:

- `exp`: range reduction + a fixed-point polynomial for `2^(x/ln 2)`,
- `log`: normalization `x = m * 2^k` + a convergent atanh-series for `log m`,
- `sinh`, `cosh`: defined via `exp`,
- `tanh`: a numerically stable expression in terms of `exp`.

Here we prove the **special-value rules** for those definitions: how NaNs, infinities, signed
zeros (`+0` and `-0`), and sign checks behave. These are the facts we want in proofs without
  unfolding the full fixed-point
implementations.

We do *not* claim that the finite-value approximations match any particular hardware `expf`/`logf`;
the goal is reproducibility (and well-defined behavior) inside Lean.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

namespace IEEE32Exec

/-!
## `exp`

Special values we follow (common libm / PyTorch behavior):

- `exp(NaN) = NaN` (we quiet signaling NaNs),
- `exp(+Inf) = +Inf`,
- `exp(-Inf) = +0`.

For finite inputs, `Exec32.lean` defines a deterministic approximation. These lemmas only expose
the early "special-case" branches, so proofs do not have to unfold the fixed-point core.
-/

/-- `exp` propagates NaNs by returning the quieted NaN payload. -/
theorem exp_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    exp x = quietNaN x := by
  simp [IEEE32Exec.exp, hx]

/-- `exp(+Inf) = +Inf`. -/
theorem exp_posInf : exp posInf = posInf := by
  decide

/-- `exp(-Inf) = +0`. -/
theorem exp_negInf : exp negInf = posZero := by
  decide

/-!
## `log`

The special cases match the usual real-analytic extension used by common libraries:

- `log(NaN) = NaN` (quieted),
- `log(+Inf) = +Inf`,
- `log(0) = -Inf` (for both `+0` and `-0`),
- `log(x < 0) = NaN` (including `log(-Inf)`).

PyTorch follows these conventions on IEEE hardware; we mirror them in `IEEE32Exec.log`.
-/

/-- `log` propagates NaNs by returning the quieted NaN payload. -/
theorem log_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    log x = quietNaN x := by
  simp [IEEE32Exec.log, hx]

/-- `log(+Inf) = +Inf`. -/
theorem log_posInf : log posInf = posInf := by
  decide

/-- `log(-Inf) = NaN` (domain error for negative inputs). -/
theorem log_negInf : log negInf = canonicalNaN := by
  decide

/-- `log(+0) = -Inf`. -/
theorem log_posZero : log posZero = negInf := by
  decide

/-- `log(-0) = -Inf`. -/
theorem log_negZero : log negZero = negInf := by
  decide

/--
`log` returns `canonicalNaN` on negative finite inputs (domain error).

This captures the standard real domain restriction `log : (0, +∞) → ℝ`, transported to the float32
setting (excluding NaNs/Infs/zeros which are handled by earlier branches).
-/
theorem log_eq_canonicalNaN_of_negative (x : IEEE32Exec)
    (hxNaN : isNaN x = false) (hxInf : isInf x = false) (hxZero : isZero x = false)
    (hxSign : signBit x = true) :
    log x = canonicalNaN := by
  simp [IEEE32Exec.log, hxNaN, hxInf, hxZero, hxSign]

/-!
## `tanh`

Special values we follow:

- `tanh(NaN) = NaN` (quieted),
- `tanh(+Inf) = +1`,
- `tanh(-Inf) = -1`.

The bit patterns used below are the IEEE-754 binary32 encodings of `+1.0f` and `-1.0f`:

- `0x3F800000` = `+1`,
- `0xBF800000` = `-1`.
-/

/-- `tanh` propagates NaNs by returning the quieted NaN payload (via `chooseNaN1`). -/
theorem tanh_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    tanh x = quietNaN x := by
  simp [IEEE32Exec.tanh, hx, chooseNaN1]

/-- `tanh(+Inf) = +1`. -/
theorem tanh_posInf : tanh posInf = ofBits 0x3F800000 := by
  decide

/-- `tanh(-Inf) = -1`. -/
theorem tanh_negInf : tanh negInf = ofBits 0xBF800000 := by
  decide

/-!
## `sinh` / `cosh`

These are defined in `Exec32.lean` in terms of `exp` (with a NaN short-circuit up front), so the
special cases are the familiar ones:

- propagate NaNs (quieted),
- `sinh(±Inf) = ±Inf`,
- `cosh(±Inf) = +Inf`.
-/

/-- `sinh` propagates NaNs by returning the quieted NaN payload (via `chooseNaN1`). -/
theorem sinh_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    sinh x = quietNaN x := by
  simp [IEEE32Exec.sinh, hx, chooseNaN1]

/-- `cosh` propagates NaNs by returning the quieted NaN payload (via `chooseNaN1`). -/
theorem cosh_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    cosh x = quietNaN x := by
  simp [IEEE32Exec.cosh, hx, chooseNaN1]

/-- `sinh(+Inf) = +Inf`. -/
theorem sinh_posInf : sinh posInf = posInf := by
  decide

/-- `sinh(-Inf) = -Inf`. -/
theorem sinh_negInf : sinh negInf = negInf := by
  decide

/-- `cosh(+Inf) = +Inf`. -/
theorem cosh_posInf : cosh posInf = posInf := by
  decide

/-- `cosh(-Inf) = +Inf`. -/
theorem cosh_negInf : cosh negInf = posInf := by
  decide

end IEEE32Exec

end TorchLean.Floats.IEEE754
