/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Arb.Oracle
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.IEEE754.Native
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-!
# Arb-backed enclosures with FloatLib endpoint rounding

Arb/python-flint supplies the external real-enclosure claim. Exact rational endpoints are then
rounded outward by FloatLib's descriptor-generic software rounders. The theorems below are binary32
transport corollaries of FloatLib's directed-rational bounds. No native floating-point conversion
or software transcendental approximation participates in this endpoint conversion.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace Rat

/-- Render a rational in a format that Arb's parser accepts (e.g. `-3/2`, `5`). -/
def toArbString (q : Rat) : String :=
  let n := q.num
  let d := q.den
  if d = 1 then
    toString n
  else
    s!"{n}/{d}"

end Rat

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats
open FloatLib.Floats.Formats.BinaryInterchange

namespace IEEE32Exec

/-! ## Proved outward rounding from `ℚ` to `Binary 8 23` -/

/-- Rewrite a rational cast into the signed numerator/positive-denominator form used by the
directed rational rounders. -/
private theorem rat_cast_eq_signed (q : Rat) :
    (q : ℝ) = if q.num < 0 then -((q.num.natAbs : ℝ) / (q.den : ℝ))
      else (q.num.natAbs : ℝ) / (q.den : ℝ) := by
  rw [Rat.cast_def]
  split_ifs with h
  · simp [abs_of_neg h]
    ring
  · have hn : 0 ≤ q.num := le_of_not_gt h
    simp [abs_of_nonneg hn]

/-- The lower rational endpoint conversion is an `EReal` lower bound. -/
theorem toEReal_roundRatQDown_le (q : Rat) :
    (ExecFloat.Binary.toModel
      (ExecFloat.Binary.ofModel (Model.roundRatQDown FloatFormat.binary32 q) :
        Binary 8 23)).toEReal ≤ ((q : ℝ) : EReal) := by
  have hdecode :
      ExecFloat.Binary.toModel
        (ExecFloat.Binary.ofModel (Model.roundRatQDown FloatFormat.binary32 q) :
          Binary 8 23) = Model.roundRatQDown FloatFormat.binary32 q :=
    ExecFloat.Binary.toModel_ofModel _
  exact (congrArg (Model.toEReal (fmt := FloatFormat.binary32)) hdecode).trans_le
    (Model.toEReal_roundRatQDown_le FloatFormat.binary32 q rfl)

/-- The upper rational endpoint conversion is an `EReal` upper bound. -/
theorem toEReal_roundRatQUp_ge (q : Rat) :
    ((q : ℝ) : EReal) ≤ (ExecFloat.Binary.toModel
      (ExecFloat.Binary.ofModel (Model.roundRatQUp FloatFormat.binary32 q) :
        Binary 8 23)).toEReal := by
  have hdecode :
      ExecFloat.Binary.toModel
        (ExecFloat.Binary.ofModel (Model.roundRatQUp FloatFormat.binary32 q) :
          Binary 8 23) = Model.roundRatQUp FloatFormat.binary32 q :=
    ExecFloat.Binary.toModel_ofModel _
  have hbound : ((q : ℝ) : EReal) ≤
      (Model.roundRatQUp FloatFormat.binary32 q).toEReal := by
    rw [rat_cast_eq_signed]
    simpa [Model.roundRatQUp, Model.roundRatQWithRounding, Model.roundRatUp,
      Model.signedScaledRatToReal, Model.scaledRatToReal,
      FloatLib.Floats.Formats.Flocq.bpow] using
      Model.le_toEReal_roundRatUp FloatFormat.binary32 (q.num < 0)
        q.num.natAbs q.den rfl q.den_nz
  exact hbound.trans_eq (congrArg (Model.toEReal (fmt := FloatFormat.binary32)) hdecode).symm

/-! ## Arb-backed interval endpoints for transcendentals -/

namespace Interval32

/--
Decode a float endpoint as an exact rational, failing if the value is NaN/Inf.

This is used to feed exact endpoint strings into the Arb oracle.
-/
def ensureFinite (x : Binary 8 23) (label : String) : IO Rat := do
  match ExecFloat.Binary.toRat? x with
  | some q => pure q
  | none => throw <| IO.userError s!"Expected finite binary32 for {label}, got NaN/Inf."

/--
Call Arb on the real interval `[X.lo, X.hi]` (interpreted exactly as rationals) and return the
oracle-provided rational enclosure bounds `(L,U)`.

This is the only step that crosses the trust boundary.
-/
def arbBounds (func : String) (X : Interval (Binary 8 23)) (precBits digits : Nat := 200) :
    IO (Rat × Rat) := do
  let loQ ← ensureFinite X.lo "lo"
  let hiQ ← ensureFinite X.hi "hi"
  let q : TorchLean.Floats.Arb.Query :=
    { func := func
      lo := Rat.toArbString loQ
      hi := Rat.toArbString hiQ
      precBits := precBits
      digits := digits }
  let r ← TorchLean.Floats.Arb.run q
  pure r.outputBall.toRatBounds

/--
Compute a binary32 interval enclosure for a transcendental unary `func` by:

- getting a real enclosure `[L,U]` from Arb,
- rounding endpoints outward to the binary32 grid.

The exact rational endpoints are passed directly to the proved directed-rational interface.
FloatLib's directed-rounding theorem covers the conversion, including overflow to
infinite endpoints.
-/
def arbUnary (func : String) (X : Interval (Binary 8 23)) (precBits digits : Nat := 200) :
    IO (Interval (Binary 8 23)) := do
  let (L, U) ← arbBounds func X (precBits := precBits) (digits := digits)
  let lo32 : Binary 8 23 :=
    ExecFloat.Binary.ofModel (Model.roundRatQDown FloatFormat.binary32 L)
  let hi32 : Binary 8 23 :=
    ExecFloat.Binary.ofModel (Model.roundRatQUp FloatFormat.binary32 U)
  pure ⟨lo32, hi32⟩

/-- Arb-backed `tanh` enclosure, rounded outward to binary32 endpoints. -/
@[inline] def tanhArb (X : Interval (Binary 8 23)) (precBits digits : Nat := 200) :
    IO (Interval (Binary 8 23)) :=
  arbUnary "tanh" X (precBits := precBits) (digits := digits)

/-- Arb-backed `exp` enclosure, rounded outward to binary32 endpoints. -/
@[inline] def expArb (X : Interval (Binary 8 23)) (precBits digits : Nat := 200) :
    IO (Interval (Binary 8 23)) :=
  arbUnary "exp" X (precBits := precBits) (digits := digits)

/-- Arb-backed `log` enclosure, rounded outward to binary32 endpoints. -/
@[inline] def logArb (X : Interval (Binary 8 23)) (precBits digits : Nat := 200) :
    IO (Interval (Binary 8 23)) :=
  arbUnary "log" X (precBits := precBits) (digits := digits)

/-- Arb-backed `sqrt` enclosure, rounded outward to binary32 endpoints. -/
@[inline] def sqrtArb (X : Interval (Binary 8 23)) (precBits digits : Nat := 200) :
    IO (Interval (Binary 8 23)) :=
  arbUnary "sqrt" X (precBits := precBits) (digits := digits)

end Interval32

end IEEE32Exec

end TorchLean.Floats.IEEE754
