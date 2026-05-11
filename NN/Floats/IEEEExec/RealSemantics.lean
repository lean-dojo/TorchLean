/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.RatScaling

/-!
# RealSemantics

Finite-only real semantics for `IEEE32Exec`.

We interpret a float32 bit-pattern as a real number by decoding its exact dyadic payload
`(-1)^sign * mant * 2^exp` and then mapping to `ℝ`.

The executable kernel has NaN/Inf payloads; these do not have a real interpretation. We therefore
provide:

- `toReal? : IEEE32Exec → Option ℝ`, returning `none` for NaN/Inf,
- `toReal  : IEEE32Exec → ℝ`, a totalized version mapping `none` to `0`.

Most theorems should be read under finiteness hypotheses and use `toReal?` (or lemmas that assume
`toDyadic? x = some …`).
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-- Real interpretation for finite values (undefined for NaN/Inf). -/
noncomputable def toReal? (x : IEEE32Exec) : Option ℝ :=
  match toDyadic? x with
  | some d => some (dyadicToReal d)
  | none => none

/-- Total real interpretation (maps NaN/Inf to `0`; use `toReal?` for proofs). -/
noncomputable def toReal (x : IEEE32Exec) : ℝ :=
  match toReal? x with
  | some r => r
  | none => 0

/--
Unfolding lemma for `toReal?`.

This lemma is intentionally not `[simp]`: rewriting `toReal? x` into a `match` on `toDyadic? x`
is usually not the best first step.
-/
lemma toReal?_eq (x : IEEE32Exec) :
    toReal? x =
      match toDyadic? x with
      | some d => some (dyadicToReal d)
      | none => none := by
  rfl

/-- If `toDyadic? x = some d`, then `toReal? x = some (dyadicToReal d)`. -/
@[simp] lemma toReal?_of_toDyadic?_some {x : IEEE32Exec} {d : Dyadic} (hx : toDyadic? x = some d) :
    toReal? x = some (dyadicToReal d) := by
  simp [IEEE32Exec.toReal?, hx]

/-- If `toDyadic? x = none`, then `toReal? x = none`. -/
@[simp] lemma toReal?_of_toDyadic?_none {x : IEEE32Exec} (hx : toDyadic? x = none) :
    toReal? x = none := by
  simp [IEEE32Exec.toReal?, hx]

/--
Unfolding lemma for `toReal` in terms of `toDyadic?`.

This is the form that is most convenient in algebraic proofs: it avoids an intermediate `Option`.
-/
lemma toReal_eq (x : IEEE32Exec) :
    toReal x =
      match toDyadic? x with
      | some d => dyadicToReal d
      | none => 0 := by
  cases h : toDyadic? x <;> simp [IEEE32Exec.toReal, IEEE32Exec.toReal?, h]

end -- noncomputable section

end IEEE32Exec

end TorchLean.Floats.IEEE754

