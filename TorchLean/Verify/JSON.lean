/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Json
public import TorchLean.Verify.CROWN.BoundOps
public import TorchLean.Verify.Certificate.DecimalJSON
public import TorchLean.Verify.DecimalRounding

/-!
# JSON of verification certificates

The verification tools share these certificate readers: rationals, boxes and schema strings, read
from the exact-decimal JSON values of `TorchLean.Certificate.DecimalJson`, and checked documents of
Lean's `Json`. A reader with a `ctx` argument starts its errors with it, the path of the value read,
such as `leaf.lb[0]`.

- `parseDecimalRat` reads a JSON decimal as the exact rational it denotes, `parseDecimalRatArray` an
  array of JSON decimals, and `parseRatString` a rational written `n` or `n/d`.
- `BoxRegion` is a finite box. `BoxRegion.parse` reads one in endpoint (`lo`, `hi`) or center-radius
  (`center`, `eps`) notation as binary64 values, each endpoint rounded once outward, and
  `BoxRegion.parseEndpoints` reads endpoint notation only, as exact rationals.
- `expectFormat` checks the `format` string of an artifact.
- `accepts` and `acceptsText` accept a document that decodes to a value passing a check, and
  `accepts_sound` and `acceptsText_sound` prove that the value has the checked property.
-/

@[expose] public section

namespace TorchLean.Json

open Lean
open Lean.Json
open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Floats.Formats.BinaryInterchange.Model (IEEERoundingMode)
open Certificate.DecimalJson (Value)

/-- The exact rational of a JSON number, or of a string holding one. As in
`Certificate.DecimalJson.Decimal.toRat?`, a nonzero decimal `m * 10 ^ e` with
`|e| > 2 ^ 11 + Nat.log2 m` is an error: its magnitude is below `10 ^ (-2048)` or above `10 ^ 2048`.
-/
def parseDecimalRat (ctx : String) (v : Value) : Except String Rat :=
  match v.decimal? with
  | none => .error s!"{ctx}: expected a number"
  | some d =>
      match Certificate.DecimalJson.Decimal.toRat? d with
      | some q => .ok q
      | none => .error s!"{ctx}: decimal exponent out of range"

/-- `parseDecimalRat` returns `q` exactly when `v` holds a decimal whose `Decimal.toRat?` is `q`. -/
theorem parseDecimalRat_eq_ok_iff (ctx : String) (v : Value) (q : Rat) :
    parseDecimalRat ctx v = .ok q ↔
      ∃ d, v.decimal? = some d ∧ Certificate.DecimalJson.Decimal.toRat? d = some q := by
  unfold parseDecimalRat
  split
  · simp_all
  · split <;> simp_all

/-- The exact rationals of a JSON array of numbers, or of strings holding numbers, each read by
`parseDecimalRat`. -/
def parseDecimalRatArray (ctx : String) (v : Value) : Except String (Array Rat) := do
  (← v.array ctx).mapIdxM fun i x => parseDecimalRat s!"{ctx}[{i}]" x

/--
Parse an exact rational from its JSON string form, an integer or a fraction:

- integer: `"5"`, `"-3"`
- fraction: `"5/2"`, `"-7/10"`

The text holds only decimal digits, `-` and `/`, and the denominator must be a nonzero natural
number. Fractions carry any rational without a floating-point conversion, including values such as
`1/3` that have no finite decimal form.
-/
def parseRatString (s : String) : Except String Rat := do
  if s.isEmpty then
    throw "empty rational string"
  unless s.all fun c => c.isDigit || c == '-' || c == '/' do
    throw s!"invalid rational (expected n or n/d): '{s}'"
  match s.splitOn "/" with
  | [numStr] =>
      match numStr.toInt? with
      | some n => pure (Rat.ofInt n)
      | none => throw s!"invalid integer rational: '{s}'"
  | [numStr, denStr] =>
      let n ←
        match numStr.toInt? with
        | some n => pure n
        | none => throw s!"invalid numerator: '{numStr}'"
      let d ←
        match denStr.toNat? with
        | some d => pure d
        | none => throw s!"invalid denominator (expected Nat): '{denStr}'"
      if d = 0 then
        throw "invalid rational: denominator is 0"
      pure (mkRat n d)
  | _ =>
      throw s!"invalid rational (expected n or n/d): '{s}'"

/-- A finite axis-aligned region parsed from a verification artifact, with endpoints in `α`. -/
structure BoxRegion (α : Type) where
  /-- Declared dimension, or the inferred endpoint-array length when `dim` is absent. -/
  dim : Nat
  /-- Lower coordinate bounds. -/
  lo : Array α
  /-- Upper coordinate bounds. -/
  hi : Array α
  deriving Repr

namespace BoxRegion

/-- Check that both endpoint arrays have length `dim` and that `ordered lo hi` holds at every
coordinate. -/
def validate {α : Type} [Inhabited α] [ToString α] (ctx : String) (region : BoxRegion α)
    (ordered : α → α → Bool) : Except String Unit := do
  unless region.lo.size = region.dim && region.hi.size = region.dim do
    throw <| s!"{ctx}: dimension {region.dim} does not match endpoint lengths " ++
      s!"{region.lo.size} and {region.hi.size}"
  for i in [0:region.dim] do
    let lo := region.lo[i]!
    let hi := region.hi[i]!
    unless ordered lo hi do
      throw s!"{ctx}: invalid interval at coordinate {i}: [{lo}, {hi}]"

end BoxRegion

/--
Parse a finite axis-aligned region written as `{lo, hi}` or as `{center, eps}`.

An absent `dim` is inferred from the endpoint or center array, and a present one must be a natural
number equal to the endpoint lengths. Each endpoint is rounded once outward into host binary64.
Endpoint notation rounds `lo` down and `hi` up with `DecimalRounding.float`, and `{center, eps}`
forms `center - eps` and `center + eps` exactly from the decimals and rounds them down and up with
`Model.ofRat`. The box read therefore contains the decimal box the artifact states
(`DecimalRounding.toEReal_convertDecimalText_down_le`, `Model.toEReal_ofRat_down_le` and their
upward forms). The parser rejects negative radii, endpoints that round to non-finite values, mixed
or incomplete schemas, and coordinates whose rounded lower endpoint exceeds the rounded upper one.
Every region it returns has passed `BoxRegion.validate`. `BoxRegion.parseEndpoints` reads the
endpoint schema as exact rationals instead.
-/
def BoxRegion.parse (ctx : String) (v : Value) : Except String (BoxRegion Float) := do
  let endpoints (ctx : String) (mode : IEEERoundingMode) (v : Value) :
      Except String (Array Float) := do
    (← v.array ctx).mapIdxM fun i x =>
      match (x.decimal?.map (DecimalRounding.float mode)).filter Float.isFinite with
      | some f => .ok f
      | none => .error s!"{ctx}[{i}]: expected finite float"
  let toBinary64 (mode : IEEERoundingMode) (q : Rat) : Float :=
    ExecFloat.Binary.toFloat (ExecFloat.Binary.ofModel
      (Model.ofRat FloatFormat.binary64 q (mode := mode)) : FloatLib.Floats.Binary64)
  let declaredDim? ← (← v.optionalField? "dim" ctx).mapM (·.nat s!"{ctx}.dim")
  let lo? ← v.optionalField? "lo" ctx
  let hi? ← v.optionalField? "hi" ctx
  let center? ← v.optionalField? "center" ctx
  let eps? ← v.optionalField? "eps" ctx
  let region ← match lo?, hi?, center?, eps? with
  | some loV, some hiV, none, none =>
      let lo ← endpoints s!"{ctx}.lo" .down loV
      let hi ← endpoints s!"{ctx}.hi" .up hiV
      pure { dim := declaredDim?.getD lo.size, lo, hi }
  | none, none, some centerV, some epsV =>
      let center ← parseDecimalRatArray s!"{ctx}.center" centerV
      let radius ← parseDecimalRat s!"{ctx}.eps" epsV
      unless 0 ≤ radius do
        throw s!"{ctx}.eps: expected a nonnegative radius, got {radius}"
      pure
        { dim := declaredDim?.getD center.size
          lo := center.map fun c => toBinary64 .down (c - radius)
          hi := center.map fun c => toBinary64 .up (c + radius) }
  | some _, none, _, _ =>
      throw s!"{ctx}: field `lo` requires a matching `hi` field"
  | none, some _, _, _ =>
      throw s!"{ctx}: field `hi` requires a matching `lo` field"
  | some _, some _, _, _ =>
      throw s!"{ctx}: endpoint fields (`lo`, `hi`) cannot be combined with `center` or `eps`"
  | none, none, some _, none =>
      throw s!"{ctx}: field `center` requires a matching `eps` field"
  | none, none, none, some _ =>
      throw s!"{ctx}: field `eps` requires either `center` or endpoint fields"
  | none, none, none, none =>
      throw s!"{ctx}: expected either (`lo`, `hi`) or (`center`, `eps`)"
  region.validate ctx fun lo hi => lo.isFinite && hi.isFinite && lo <= hi
  pure region

/--
Parse the endpoint-only schema `{lo, hi}`, with an optional matching `dim` field, keeping each
endpoint as the exact rational its JSON decimal denotes.
-/
def BoxRegion.parseEndpoints (ctx : String) (v : Value) : Except String (BoxRegion Rat) := do
  if (← v.optionalField? "center" ctx).isSome || (← v.optionalField? "eps" ctx).isSome then
    throw s!"{ctx}: expected endpoint fields (`lo`, `hi`), not `center` or `eps`"
  let declaredDim? ← (← v.optionalField? "dim" ctx).mapM (·.nat s!"{ctx}.dim")
  let lo ← parseDecimalRatArray s!"{ctx}.lo" (← v.field "lo" ctx)
  let hi ← parseDecimalRatArray s!"{ctx}.hi" (← v.field "hi" ctx)
  let region : BoxRegion Rat := { dim := declaredDim?.getD lo.size, lo, hi }
  region.validate ctx fun lo hi => decide (lo ≤ hi)
  pure region

/-- Require the `format` string of an artifact to be the schema string `expected`, naming the
artifact by `ctx` in the error. -/
def expectFormat (format expected : String) (ctx : String := "top-level") : Except String Unit :=
  if format = expected then .ok ()
  else .error s!"{ctx}.format: unsupported format `{format}` (expected `{expected}`)"

/-- `expectFormat` accepts exactly the format string `expected`. -/
theorem expectFormat_eq_ok_iff (format expected ctx : String) :
    expectFormat format expected ctx = .ok () ↔ format = expected := by
  unfold expectFormat
  split <;> simp_all

/-! ## Checked documents -/

/-- `accepts decode check j` decodes `j` and runs `check` on the value. A document that fails
to decode is rejected. -/
def accepts {β : Type} (decode : Json → Except String β) (check : β → Bool) (j : Json) :
    Bool :=
  match decode j with
  | .error _ => false
  | .ok value => check value

/-- If `check` is sound for `P`, every document accepted by `accepts decode check` decodes to
a value satisfying `P`. -/
theorem accepts_sound {β : Type} {decode : Json → Except String β} {check : β → Bool}
    {P : β → Prop} (check_sound : ∀ value, check value = true → P value) {j : Json}
    (h : accepts decode check j = true) : ∃ value, decode j = .ok value ∧ P value := by
  unfold accepts at h
  cases hd : decode j with
  | error error => simp [hd] at h
  | ok value => exact ⟨value, rfl, check_sound value (by simpa [hd] using h)⟩

/-- Parse JSON text, then decode and check the document with `accepts`. Text that
`Certificate.DecimalJson.parse` refuses, such as a syntax error or a repeated object key, is
rejected. -/
def acceptsText {β : Type} (decode : Json → Except String β) (check : β → Bool)
    (source : String) : Bool :=
  (Certificate.DecimalJson.parse source).isOk &&
    match Json.parse source with
    | .error _ => false
    | .ok j => accepts decode check j

/-- If `check` is sound for `P`, any text accepted by `acceptsText decode check` parses to a
document that decodes to a value satisfying `P`. -/
theorem acceptsText_sound {β : Type} {decode : Json → Except String β} {check : β → Bool}
    {P : β → Prop} (check_sound : ∀ value, check value = true → P value) {source : String}
    (h : acceptsText decode check source = true) :
    ∃ j value, Json.parse source = .ok j ∧ decode j = .ok value ∧ P value := by
  unfold acceptsText at h
  cases hp : Json.parse source with
  | error error => simp [hp] at h
  | ok j =>
      have hj : accepts decode check j = true := by
        simp only [hp, Bool.and_eq_true] at h
        exact h.2
      obtain ⟨value, hd, hP⟩ := accepts_sound check_sound hj
      exact ⟨j, value, rfl, hd, hP⟩

end TorchLean.Json
