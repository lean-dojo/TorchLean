/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Arb.Oracle
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof
public import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Rational.Conversion
public import FloatLib.Numerics.Enclosure.Interval.Basic

/-!
# Arb-backed enclosures with configured endpoints

Arb/python-flint supplies the external real-enclosure claim. Exact rational endpoints are then
rounded by FloatLib's descriptor-generic software rounders and packed through the selected
`ModelCodec`. For IEEE formats, the transport theorems prove that conversion rounds outward,
including overflow to infinite endpoints.

These theorems cover endpoint conversion only. They do not state that `unary` returns an enclosure
of the requested function: Arb's answer remains unchecked. No native floating-point conversion
or software transcendental approximation participates in the endpoint conversion.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Numerics (Interval)

namespace TorchLean.Floats.Interval.Arb

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

/-- Packing the downward rational conversion preserves its extended-real lower bound. -/
theorem toEReal_roundRatQDown_le (q : Rat) (hformat : format.isIEEE = true) :
    (Binary.toModel
      (Binary.ofModel (plan := plan) (code := code) (Model.roundRatQDown format q))).toEReal ≤
        ((q : ℝ) : EReal) := by
  simpa only [Binary.toModel_ofModel] using Model.toEReal_roundRatQDown_le format q hformat

/-- Packing the upward rational conversion preserves its extended-real upper bound. -/
theorem toEReal_roundRatQUp_ge (q : Rat) (hformat : format.isIEEE = true) :
    ((q : ℝ) : EReal) ≤ (Binary.toModel
      (Binary.ofModel (plan := plan) (code := code) (Model.roundRatQUp format q))).toEReal := by
  rw [Binary.toModel_ofModel]
  have hcast : (q : ℝ) = Model.signedScaledRatToReal (q.num < 0) q.num.natAbs q.den 0 := by
    rw [Rat.cast_def]
    by_cases h : q.num < 0
    · simp [Model.signedScaledRatToReal, Model.scaledRatToReal, h, abs_of_neg h, neg_div]
    · simp [Model.signedScaledRatToReal, Model.scaledRatToReal,
        h, abs_of_nonneg (le_of_not_gt h)]
  rw [hcast]
  exact Model.le_toEReal_roundRatUp format (q.num < 0) q.num.natAbs q.den hformat q.den_nz

/-- Decode an endpoint exactly as a rational, failing on NaN or infinity. -/
def finite (x : ExecFloat (Configured.Family format code plan)) (label : String) :
    IO Rat := do
  match Binary.toRat? x with
  | some q => pure q
  | none => throw <| IO.userError s!"Expected finite endpoint for {label}, got NaN/Inf."

/--
Send exact rational input endpoints to Arb and return its claimed rational enclosure.

Non-finite inputs fail before the request. This call crosses the external oracle trust boundary;
parsing its response does not prove that the requested function lies within these bounds.
-/
def bounds (func : String) (X : Interval (ExecFloat (Configured.Family format code plan)))
    (precBits digits : Nat := 200) : IO (Rat × Rat) := do
  let loQ ← finite X.lo "lo"
  let hiQ ← finite X.hi "hi"
  let q : TorchLean.Floats.Arb.Query :=
    { func := func
      lo := toString loQ
      hi := toString hiQ
      precBits := precBits
      digits := digits }
  let r ← TorchLean.Floats.Arb.run q
  pure r.outputBall.toRatBounds

/--
Round Arb's claimed enclosure for a unary function into the input's format and storage plan.

For `format.isIEEE = true`, the endpoint transport theorems prove outward rounding, including
overflow. Enclosure of the requested function still depends on Arb's external claim.
-/
def unary (func : String) (X : Interval (ExecFloat (Configured.Family format code plan)))
    (precBits digits : Nat := 200) :
    IO (Interval (ExecFloat (Configured.Family format code plan))) := do
  let (lo, hi) ← bounds func X (precBits := precBits) (digits := digits)
  pure ⟨Binary.ofModel (Model.roundRatQDown format lo),
    Binary.ofModel (Model.roundRatQUp format hi)⟩

end TorchLean.Floats.Interval.Arb
