/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Semantics.RealSemantics
public import NN.Floats.IEEEExec.Exec32.Transcendentals

/-!
## Deterministic transcendental functions (`exp`, `log`, `tanh`, `sinh`, `cosh`)

IEEE-754 fixes the representation and the basic arithmetic operations, but it does **not** mandate
exact bit-level behavior for transcendental functions like `exp` and `log`. In practice, libraries
(`libm`, SVML, CUDA math, etc.) make slightly different choices, and results can differ across
platforms.

In TorchLean, we still need executable transcendental operations in some examples and runtime
experiments. So we give `IEEE32Exec` *deterministic* definitions for a small set of functions using
fixed algorithms:

- `exp`: range reduction + a fixed-point polynomial for $2^{x/\log 2}$,
- `log`: normalization $x=m2^k$ + a convergent atanh-series for $\log m$,
- `sinh`, `cosh`: defined via `exp`,
- `tanh`: a small-input odd polynomial plus a bounded expression in terms of `exp`.

Here we prove the **special-value rules** for those definitions: how NaNs, infinities, signed
zeros (`+0` and `-0`), and sign checks behave. These are the facts we want in proofs without
  unfolding the full fixed-point
implementations.

We do *not* claim that the finite-value approximations match any particular hardware `expf`/`logf`;
the Lean definitions prioritize reproducibility and well-defined behavior.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

namespace IEEE32Exec

/-!
## Finite approximation contracts

The deterministic functions in this module have proved special-value behavior below. A numerical
accuracy claim on finite inputs requires a separate contract. Keeping that evidence explicit
prevents reproducible execution from being mistaken for correct rounding.
-/

/--
A uniform real-error contract for an executable unary binary32 operation on a stated domain.

For every finite input whose real value lies in `domain`, the executable result must also be finite
and differ from `spec` by at most `tolerance`.
-/
structure UnaryApproximationContract
    (impl : IEEE32Exec → IEEE32Exec) (spec : ℝ → ℝ) (domain : ℝ → Prop)
    (tolerance : ℝ) : Prop where
  tolerance_nonneg : 0 ≤ tolerance
  error_le : ∀ x xr,
    toReal? x = some xr → domain xr →
      ∃ yr, toReal? (impl x) = some yr ∧ |yr - spec xr| ≤ tolerance

namespace UnaryApproximationContract

/-- A proved approximation contract remains valid when its tolerance is weakened. -/
theorem mono {impl : IEEE32Exec → IEEE32Exec} {spec : ℝ → ℝ} {domain : ℝ → Prop}
    {tolerance tolerance' : ℝ}
    (h : UnaryApproximationContract impl spec domain tolerance)
    (hle : tolerance ≤ tolerance') :
    UnaryApproximationContract impl spec domain tolerance' where
  tolerance_nonneg := h.tolerance_nonneg.trans hle
  error_le := by
    intro x xr hx hdomain
    obtain ⟨yr, hyr, herr⟩ := h.error_le x xr hx hdomain
    exact ⟨yr, hyr, herr.trans hle⟩

end UnaryApproximationContract

/-!
## `exp`

Special values we follow (common libm / PyTorch behavior):

- $\exp(\mathrm{NaN})=\mathrm{NaN}$ (we quiet signaling NaNs),
- $\exp(+\infty)=+\infty$,
- $\exp(-\infty)=+0$.

For finite inputs, `Exec32.lean` defines a deterministic approximation. These lemmas only expose
the early "special-case" branches, so proofs do not have to unfold the fixed-point core.
-/

/-- `exp` propagates NaNs by returning the quieted NaN payload. -/
theorem exp_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    exp x = quietNaN x := by
  simp [IEEE32Exec.exp, hx]

/-- $\exp(+\infty)=+\infty$. -/
theorem exp_posInf : exp posInf = posInf := by
  decide

/-- $\exp(-\infty)=+0$. -/
theorem exp_negInf : exp negInf = posZero := by
  decide

/-!
## `log`

The special cases match the usual real-analytic extension used by common libraries:

- $\log(\mathrm{NaN})=\mathrm{NaN}$ (quieted),
- $\log(+\infty)=+\infty$,
- $\log(0)=-\infty$ (for both $+0$ and $-0$),
- $\log(x)=\mathrm{NaN}$ for $x<0$ (including $\log(-\infty)$).

PyTorch follows these conventions on IEEE hardware; we mirror them in `IEEE32Exec.log`.
-/

/-- `log` propagates NaNs by returning the quieted NaN payload. -/
theorem log_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    log x = quietNaN x := by
  simp [IEEE32Exec.log, hx]

/-- $\log(+\infty)=+\infty$. -/
theorem log_posInf : log posInf = posInf := by
  decide

/-- $\log(-\infty)=\mathrm{NaN}$ (domain error for negative inputs). -/
theorem log_negInf : log negInf = canonicalNaN := by
  decide

/-- $\log(+0)=-\infty$. -/
theorem log_posZero : log posZero = negInf := by
  decide

/-- $\log(-0)=-\infty$. -/
theorem log_negZero : log negZero = negInf := by
  decide

/--
`log` returns `canonicalNaN` on negative finite inputs (domain error).

This captures the standard real domain restriction
$\log:(0,+\infty)\to\mathbb{R}$, transported to the float32 setting (excluding NaNs, infinities,
and zeros, which are handled by earlier branches).
-/
theorem log_eq_canonicalNaN_of_negative (x : IEEE32Exec)
    (hxNaN : isNaN x = false) (hxInf : isInf x = false) (hxZero : isZero x = false)
    (hxSign : signBit x = true) :
    log x = canonicalNaN := by
  simp [IEEE32Exec.log, hxNaN, hxInf, hxZero, hxSign]

/-!
## `tanh`

Special values we follow:

- $\tanh(\mathrm{NaN})=\mathrm{NaN}$ (quieted),
- $\tanh(+\infty)=+1$,
- $\tanh(-\infty)=-1$.

The bit patterns used below are the IEEE-754 binary32 encodings of `+1.0f` and `-1.0f`:

- `0x3F800000` represents $+1$,
- `0xBF800000` represents $-1$.
-/

/-- `tanh` propagates NaNs by returning the quieted NaN payload (via `chooseNaN1`). -/
theorem tanh_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    tanh x = quietNaN x := by
  simp [IEEE32Exec.tanh, hx, chooseNaN1]

/-- $\tanh(+\infty)=+1$. -/
theorem tanh_posInf : tanh posInf = ofBits 0x3F800000 := by
  decide

/-- $\tanh(-\infty)=-1$. -/
theorem tanh_negInf : tanh negInf = ofBits 0xBF800000 := by
  decide

/-!
## `sinh` / `cosh`

These are defined in `Exec32.lean` in terms of `exp` (with a NaN short-circuit up front), so the
special cases are the familiar ones:

- propagate NaNs (quieted),
- $\sinh(\pm\infty)=\pm\infty$,
- $\cosh(\pm\infty)=+\infty$.
-/

/-- `sinh` propagates NaNs by returning the quieted NaN payload (via `chooseNaN1`). -/
theorem sinh_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    sinh x = quietNaN x := by
  simp [IEEE32Exec.sinh, hx, chooseNaN1]

/-- `cosh` propagates NaNs by returning the quieted NaN payload (via `chooseNaN1`). -/
theorem cosh_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    cosh x = quietNaN x := by
  simp [IEEE32Exec.cosh, hx, chooseNaN1]

/-- $\sinh(+\infty)=+\infty$. -/
theorem sinh_posInf : sinh posInf = posInf := by
  decide

/-- $\sinh(-\infty)=-\infty$. -/
theorem sinh_negInf : sinh negInf = negInf := by
  decide

/-- $\cosh(+\infty)=+\infty$. -/
theorem cosh_posInf : cosh posInf = posInf := by
  decide

/-- $\cosh(-\infty)=+\infty$. -/
theorem cosh_negInf : cosh negInf = posInf := by
  decide

end IEEE32Exec

end TorchLean.Floats.IEEE754
