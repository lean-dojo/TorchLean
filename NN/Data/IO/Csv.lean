/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Data.IO.Parsing

/-!
# CSV Loader

Small CSV helpers for TorchLean examples and regression tests.

The parser is kept narrow: unquoted delimiter-separated numeric cells only. It does not
support quoted fields, escaped delimiters, locale-specific number formats, `NaN`, or `inf`.
Keeping that grammar explicit is better than accidentally treating this as a production CSV
library.
-/

@[expose] public section

namespace TorchLean
namespace Data
namespace IO

open Internal

/--
Options for the CSV parser in this module.

The parser accepts unquoted numeric fields with literal delimiters and locale-independent
number syntax.
-/
structure CsvOptions where
  /-- Delimiter character (default: `,`). -/
  delimiter : Char := ','
  /-- If true, drop the first line before parsing rows. -/
  skipHeader : Bool := false
  /-- If true, trim ASCII whitespace around cells and around each row. -/
  trimCells : Bool := true
  /-- If true, ignore empty lines (otherwise treat them as an error). -/
  allowEmptyLines : Bool := true

namespace Internal

/--
Parse an optional exponent suffix of the form `e±NNN` or `E±NNN`.

Returns `(exp, rest)` where `exp` is an `Int` power-of-10 exponent to apply.
-/
def parseExponent (tag : String) (cs : List Char) : Except String (Int × List Char) :=
  match cs with
  | 'e' :: rest | 'E' :: rest =>
      let (negExp, rest) := parseSign rest
      let (expDigits, rest) := takeDigits rest
      if expDigits.isEmpty then
        .error (formatError tag "invalid exponent")
      else
        let e := digitsToNat expDigits
        let expInt : Int := if negExp then - (Int.ofNat e) else Int.ofNat e
        .ok (expInt, rest)
  | _ => .ok ((0 : Int), cs)

/--
Parse a numeric string into a `Float`.

Supported grammar:
- optional sign
- digits
- optional fractional part `.digits`
- optional scientific exponent `e±digits`

This parser rejects `NaN`, `inf`, locale separators, and quoted CSV cells.
-/
def parseFloatString (tag : String) (s : String) : Except String Float := do
  let s := (s.trimAscii).toString
  if s.length > maxNumericCellChars then
    .error (formatError tag s!"cell too long ({s.length} chars; max {maxNumericCellChars})")
  else
  let cs := s.toList
  if cs.isEmpty then
    .error (formatError tag "empty cell")
  else
    let (neg, cs) := parseSign cs
    let (intDigits, cs) := takeDigits cs
    let (fracDigits, cs) :=
      match cs with
      | '.' :: rest => takeDigits rest
      | _ => ([], cs)
    let (exp, cs) <- parseExponent (tag := tag) cs
    if !cs.isEmpty then
      .error (formatError tag s!"unparsed suffix: {String.ofList cs}")
    else
      let allDigits := intDigits ++ fracDigits
      if allDigits.isEmpty then
        .error (formatError tag "no digits found")
      else
        let mantissa := digitsToNat allDigits
        let decimalPlaces := fracDigits.length
        let netExp : Int := exp - (Int.ofNat decimalPlaces)
        let (expSign, expNat) :=
          match netExp with
          | Int.ofNat n => (false, n)
          | Int.negSucc n => (true, n.succ)
        let val := Float.ofScientific mantissa expSign expNat
        if val.isFinite then
          .ok (if neg then -val else val)
        else
          .error (formatError tag "numeric value is outside the finite Float range")

end Internal

open Internal

/--
Parse one CSV line into an array of floats.

Returns `none` for empty lines when `allowEmptyLines = true`.
-/
def parseCsvLine (tag : String) (options : CsvOptions) (rowIdx : Nat) (line : String) :
  Except String (Option (Array Float)) := do
  let line := if options.trimCells then (line.trimAscii).toString else line
  if line.isEmpty then
    if options.allowEmptyLines then
      pure none
    else
      .error (formatError tag s!"row {rowIdx}: empty line")
  else
    let delim := String.singleton options.delimiter
    let cells := line.splitOn delim
    let cells := if options.trimCells then cells.map (fun c => (c.trimAscii).toString) else cells
    let floats <- (cells.zipIdx).mapM (fun pair => do
      let cell := pair.fst
      let colIdx := pair.snd + 1
      if cell.isEmpty then
        .error (formatError tag s!"row {rowIdx}, col {colIdx}: empty cell")
      else
        parseFloatString (tag := s!"{tag} row {rowIdx}, col {colIdx}") cell)
    pure (some floats.toArray)

/--
Read a CSV file into an array of float rows.

This helper is intended for compact example datasets and runtime checks, not a full CSV
implementation.
-/
def readCsvFloatRows (path : System.FilePath) (options : CsvOptions := {}) :
  IO (Except String (Array (Array Float))) := do
  let content <- IO.FS.readFile path
  let lines := content.splitOn "\n"
  let lines := if options.skipHeader then lines.drop 1 else lines
  let res : Except String (Nat × Array (Array Float)) :=
    lines.foldlM (init := (0, #[])) (fun acc line => do
      let (i, rows) := acc
      let rowIdx := i + 1
      match parseCsvLine (tag := "csv") options rowIdx line with
      | .error e => .error e
      | .ok none => .ok (rowIdx, rows)
      | .ok (some row) => .ok (rowIdx, rows.push row))
  match res with
  | .error e => pure (.error e)
  | .ok (_, rows) =>
      if rows.isEmpty then
        pure (.error (formatError "csv" "no data rows"))
      else
        pure (.ok rows)

end IO
end Data
end TorchLean
