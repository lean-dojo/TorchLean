/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32

/-!
## Deterministic trigonometric special-value rules (`sin`, `cos`)

`NN.Floats.IEEEExec.Exec32` defines deterministic, executable `IEEE32Exec.sin` and `IEEE32Exec.cos`
(implemented purely inside Lean; no delegation to the host `Float` runtime).

This file exposes the **special-case rewrite rules** (NaN/Inf/±0) so downstream proofs do not need
to unfold the whole implementation.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/-- `sin` propagates NaNs by returning the quieted payload. -/
theorem sin_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    sin x = quietNaN x := by
  simp [IEEE32Exec.sin, Trig.sin, Trig.sinCos, hx]

/-- `cos` propagates NaNs by returning the quieted payload. -/
theorem cos_eq_quietNaN_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    cos x = quietNaN x := by
  simp [IEEE32Exec.cos, Trig.cos, Trig.sinCos, hx]

/-- `sin(±Inf) = NaN`. -/
theorem sin_posInf : sin posInf = canonicalNaN := by
  decide

/-- `sin(±Inf) = NaN`. -/
theorem sin_negInf : sin negInf = canonicalNaN := by
  decide

/-- `cos(±Inf) = NaN`. -/
theorem cos_posInf : cos posInf = canonicalNaN := by
  decide

/-- `cos(±Inf) = NaN`. -/
theorem cos_negInf : cos negInf = canonicalNaN := by
  decide

/-- `sin(+0) = +0`. -/
theorem sin_posZero : sin posZero = posZero := by
  decide

/-- `sin(-0) = -0`. -/
theorem sin_negZero : sin negZero = negZero := by
  decide

/-- `cos(±0) = 1`. -/
theorem cos_posZero : cos posZero = ofBits 0x3F800000 := by
  decide

/-- `cos(±0) = 1`. -/
theorem cos_negZero : cos negZero = ofBits 0x3F800000 := by
  decide

end IEEE32Exec
end TorchLean.Floats.IEEE754

