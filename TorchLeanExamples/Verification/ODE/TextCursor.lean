/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Data.Parsing
public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Directed.Runtime

/-!
# Text cursor primitives for verification parsers

ODE and PINN certificates both contain small mathematical expression languages. This module owns
their shared byte-position cursor, scans bounded by the remaining input, and the binary64
enclosure of a decimal literal. Their grammar is `TorchLean.Examples.ArithParse`, and each
verifier module supplies its AST and identifiers.
-/

@[expose] public section

namespace TorchLean.Examples

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

/-- A source string paired with the current raw byte position. -/
structure TextCursor where
  /-- Source text being scanned. -/
  source : String
  /-- Current raw byte position in `source`. -/
  position : String.Pos.Raw := 0

namespace TextCursor

/-- Inspect the current character without advancing the cursor. -/
@[inline] def peek (cursor : TextCursor) : Option Char :=
  String.Pos.Raw.get? cursor.source cursor.position

/-- Advance the position past the current character. -/
@[inline] def bump (cursor : TextCursor) : TextCursor :=
  { cursor with position := String.Pos.Raw.next cursor.source cursor.position }

/-- Remaining bytes plus one, suitable as a budget for a single linear scan. -/
@[inline] def remainingFuel (cursor : TextCursor) : Nat :=
  (cursor.source.rawEndPos.byteIdx - cursor.position.byteIdx) + 1

/-- Whether the cursor has reached or passed the source's raw end position. -/
@[inline] def atEnd (cursor : TextCursor) : Bool :=
  cursor.position ≥ cursor.source.rawEndPos

/-- Append characters to `accumulator` while they satisfy `predicate`, returning the string and
the cursor at the first character that fails or at the end. The fuel `remainingFuel` covers every
character left in the source. -/
def takeWhile (predicate : Char → Bool) (accumulator : String) (cursor : TextCursor) :
    String × TextCursor :=
  go (remainingFuel cursor) accumulator cursor
where
  /-- The fuel-driven scan, one character per unit of fuel. -/
  go : Nat → String → TextCursor → String × TextCursor
    | 0, accumulator, cursor => (accumulator, cursor)
    | fuel + 1, accumulator, cursor =>
        match peek cursor with
        | some char =>
            if predicate char then go fuel (accumulator.push char) (bump cursor)
            else (accumulator, cursor)
        | none => (accumulator, cursor)

/-- Skip spaces, tabs, line feeds and carriage returns (`Char.isWhitespace`) up to the next other
character or the end of the source. -/
def skipWs (cursor : TextCursor) : TextCursor :=
  (takeWhile Char.isWhitespace "" cursor).2

/-- Skip leading whitespace, then parse an unsigned decimal natural number with
`Data.NPY.parseNatValue`, which rejects an empty digit run and one of more than
256 digits. Unlike `String.toNat?`, the scan stops at a `_` separator. -/
def parseNat (cursor : TextCursor) : Except String (Nat × TextCursor) := do
  let cursor := skipWs cursor
  let (text, cursor) := takeWhile Char.isDigit "" cursor
  let some value := Data.NPY.parseNatValue text
    | throw "expected natural number"
  pure (value, cursor)

/-- Parse a finite signed decimal without scientific notation into its binary64 enclosure: the
decimal rounded toward $-\infty$ and toward $+\infty$, equal exactly when it is a binary64 value.
`Data.parseFloatString` validates the text, and an infinite bound is an error.
-/
def parseFloatBounds (cursor : TextCursor) : Except String ((Float × Float) × TextCursor) := do
  let cursor := skipWs cursor
  let (sign, cursor) :=
    match peek cursor with
    | some '-' => ("-", bump cursor)
    | _ => ("", cursor)
  let (integerText, cursor) := takeWhile (fun char => char.isDigit) "" cursor
  if integerText = "" then
    .error "expected number"
  let (fractionText, fractionDigits, cursor) :=
    match peek cursor with
    | some '.' =>
        let (digits, cursor) := takeWhile (fun char => char.isDigit) "" (bump cursor)
        ("." ++ digits, digits, cursor)
    | _ => ("", "", cursor)
  let _ ← Data.parseFloatString "number"
    (sign ++ integerText ++ fractionText)
  let numerator := Data.digitsToNat
    ((integerText ++ fractionDigits).toList.filterMap Data.digitVal?)
  let denominator := 10 ^ fractionDigits.length
  let bound (mode : Model.IEEERoundingMode) : Float :=
    ExecFloat.Binary.toFloat (ExecFloat.Binary.ofModel
      (Model.Internal.Directed.roundRat FloatFormat.binary64 mode (sign == "-") numerator
        denominator) : FloatLib.Floats.Binary64)
  let lo := bound .down
  let hi := bound .up
  if !(lo.isFinite && hi.isFinite) then
    .error "number outside the finite binary64 range"
  pure ((lo, hi), cursor)

end TextCursor

end TorchLean.Examples
