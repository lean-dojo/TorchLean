/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Aesop.BuiltinRules
import Mathlib.Tactic.Finiteness.Attr

/-!
# Text cursor primitives for verification parsers

ODE and PINN certificates both contain small mathematical expression languages. This module owns
their shared byte-position cursor, fuel-bounded scanning, and decimal literal parsing. Grammar and
AST decisions remain in the respective verifier modules.
-/

@[expose] public section

namespace NN.Verification.Util.TextCursor

/-- A source string paired with the current raw byte position. -/
structure Cursor where
  /-- Source text being scanned. -/
  source : String
  /-- Current raw byte position in `source`. -/
  position : String.Pos.Raw := 0

/-- Inspect the current character without advancing the cursor. -/
@[inline] def peek (cursor : Cursor) : Option Char :=
  String.Pos.Raw.get? cursor.source cursor.position

/-- Advance by one character, preserving the source text. -/
@[inline] def bump (cursor : Cursor) : Cursor :=
  { cursor with position := String.Pos.Raw.next cursor.source cursor.position }

/-- Remaining bytes plus one, suitable as a budget for a single linear scan. -/
@[inline] def remainingFuel (cursor : Cursor) : Nat :=
  (cursor.source.rawEndPos.byteIdx - cursor.position.byteIdx) + 1

/-- Whether the cursor has reached or passed the source's raw end position. -/
@[inline] def atEnd (cursor : Cursor) : Bool :=
  cursor.position ≥ cursor.source.rawEndPos

/-- ASCII whitespace accepted by the verification expression languages. -/
def isWhitespace (char : Char) : Bool :=
  char.isWhitespace

/-- Skip characters satisfying `predicate`, bounded by explicit recursion fuel. -/
def skipWhileFuel (predicate : Char → Bool) : Nat → Cursor → Cursor
  | 0, cursor => cursor
  | Nat.succ fuel, cursor =>
      match peek cursor with
      | some char =>
          if predicate char then skipWhileFuel predicate fuel (bump cursor) else cursor
      | none => cursor

/-- Consume characters satisfying `predicate`, bounded by explicit recursion fuel. -/
def takeWhileFuel (fuel : Nat) (predicate : Char → Bool) (accumulator : String)
    (cursor : Cursor) : String × Cursor :=
  match fuel with
  | 0 => (accumulator, cursor)
  | Nat.succ fuel =>
      match peek cursor with
      | some char =>
          if predicate char then
            takeWhileFuel fuel predicate (accumulator.push char) (bump cursor)
          else
            (accumulator, cursor)
      | none => (accumulator, cursor)

/-- Convert decimal digits to a natural number, rejecting empty or nondigit input. -/
def decimalNat (text : String) : Except String Nat :=
  if text.isEmpty || !text.toList.all Char.isDigit then
    .error "expected natural number"
  else
    .ok <| text.toList.foldl
      (fun accumulator char => accumulator * 10 + (char.toNat - '0'.toNat)) 0

/-- Parse a whitespace-prefixed unsigned decimal natural using a caller-supplied scan budget. -/
def parseNat (fuel : Nat) (cursor : Cursor) : Except String (Nat × Cursor) := do
  let cursor := skipWhileFuel isWhitespace fuel cursor
  let (text, cursor) := takeWhileFuel fuel (fun char => char.isDigit) "" cursor
  let value ← decimalNat text
  pure (value, cursor)

/-- Parse a finite signed decimal `Float` without scientific notation. -/
def parseFloat (fuel : Nat) (cursor : Cursor) : Except String (Float × Cursor) := do
  let cursor := skipWhileFuel isWhitespace fuel cursor
  let (sign, cursor) :=
    match peek cursor with
    | some '-' => (-1.0, bump cursor)
    | _ => (1.0, cursor)
  let (integerText, cursor) := takeWhileFuel fuel (fun char => char.isDigit) "" cursor
  if integerText = "" then
    .error "expected number"
  let integerValue := Float.ofNat <| integerText.toList.foldl
    (fun accumulator char => accumulator * 10 + (char.toNat - '0'.toNat)) 0
  let (fractionValue, cursor) :=
    match peek cursor with
    | some '.' =>
        let cursor := bump cursor
        let (fractionText, cursor) := takeWhileFuel fuel (fun char => char.isDigit) "" cursor
        if fractionText = "" then
          (0.0, cursor)
        else
          let numerator := fractionText.toList.foldl
            (fun accumulator char => accumulator * 10 + (char.toNat - '0'.toNat)) 0
          let denominator := Nat.pow 10 fractionText.length
          (Float.ofNat numerator / Float.ofNat denominator, cursor)
    | _ => (0.0, cursor)
  let value := sign * (integerValue + fractionValue)
  if !value.isFinite then
    throw "number is outside the finite Float range"
  pure (value, cursor)

end NN.Verification.Util.TextCursor
