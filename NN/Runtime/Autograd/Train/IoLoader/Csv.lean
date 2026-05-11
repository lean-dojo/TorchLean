/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.IoLoader.Common

/-!
# CSV loaders

Small CSV helpers for TorchLean examples and runtime regression tests.

The parser is deliberately narrow: unquoted delimiter-separated numeric cells only. It does not
support quoted fields, escaped delimiters, locale-specific number formats, `NaN`, or `inf`.
Keeping that grammar explicit is better than accidentally treating this as a production CSV
library.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Train

open Spec
open Tensor
open IoLoader.Internal

/--
Options for the CSV parser in this module.

Limitations (by design): no quoted fields, no escaped delimiters, and no locale-aware number
parsing.
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

namespace IoLoader.Internal

/--
Parse an optional exponent suffix of the form `e±NNN` or `E±NNN`.

Returns `(exp, rest)` where `exp` is an `Int` power-of-10 exponent to apply.
-/
def parseExponent (tag : String) (cs : List Char) : Result (Int × List Char) :=
  match cs with
  | 'e' :: rest | 'E' :: rest =>
      let (negExp, rest) := parseSign rest
      let (expDigits, rest) := takeDigits rest
      if expDigits.isEmpty then
        .error (tagError tag "invalid exponent")
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
def parseFloatString (tag : String) (s : String) : Result Float := do
  let s := (s.trimAscii).toString
  if s.length > maxNumericCellChars then
    .error (tagError tag s!"cell too long ({s.length} chars; max {maxNumericCellChars})")
  else
  let cs := s.toList
  if cs.isEmpty then
    .error (tagError tag "empty cell")
  else
    let (neg, cs) := parseSign cs
    let (intDigits, cs) := takeDigits cs
    let (fracDigits, cs) :=
      match cs with
      | '.' :: rest => takeDigits rest
      | _ => ([], cs)
    let (exp, cs) <- parseExponent (tag := tag) cs
    if !cs.isEmpty then
      .error (tagError tag s!"unparsed suffix: {String.ofList cs}")
    else
      let allDigits := intDigits ++ fracDigits
      if allDigits.isEmpty then
        .error (tagError tag "no digits found")
      else
        let mantissa := digitsToNat allDigits
        let decimalPlaces := fracDigits.length
        let netExp : Int := exp - (Int.ofNat decimalPlaces)
        let (expSign, expNat) :=
          match netExp with
          | Int.ofNat n => (false, n)
          | Int.negSucc n => (true, n.succ)
        let val := Float.ofScientific mantissa expSign expNat
        .ok (if neg then -val else val)

end IoLoader.Internal

open IoLoader.Internal

/--
Parse one CSV line into a list of floats.

Returns `none` for empty lines when `allowEmptyLines = true`.
-/
def parseCsvLine (tag : String) (opts : CsvOptions) (rowIdx : Nat) (line : String) :
  Result (Option (List Float)) := do
  let line := if opts.trimCells then (line.trimAscii).toString else line
  if line.isEmpty then
    if opts.allowEmptyLines then
      pure none
    else
      .error (tagError tag s!"row {rowIdx}: empty line")
  else
    let delim := String.singleton opts.delimiter
    let cells := line.splitOn delim
    let cells := if opts.trimCells then cells.map (fun c => (c.trimAscii).toString) else cells
    let floats <- (cells.zipIdx).mapM (fun pair => do
      let cell := pair.fst
      let colIdx := pair.snd + 1
      if cell.isEmpty then
        .error (tagError tag s!"row {rowIdx}, col {colIdx}: empty cell")
      else
        parseFloatString (tag := s!"{tag} row {rowIdx}, col {colIdx}") cell)
    pure (some floats)

/--
Read a CSV file into a list of float rows.

This helper is intended for small example datasets and smoke tests, not a full CSV
implementation.
-/
def readCsvFloatRows (path : System.FilePath) (opts : CsvOptions := {}) :
  IO (Result (List (List Float))) := do
  let content <- IO.FS.readFile path
  let lines := content.splitOn "\n"
  let lines := if opts.skipHeader then lines.drop 1 else lines
  let res : Result (Nat × List (List Float)) :=
    lines.foldlM (init := (0, [])) (fun acc line => do
      let (i, rows) := acc
      let rowIdx := i + 1
      match parseCsvLine (tag := "csv") opts rowIdx line with
      | .error e => .error e
      | .ok none => .ok (rowIdx, rows)
      | .ok (some row) => .ok (rowIdx, row :: rows))
  match res with
  | .error e => pure (.error e)
  | .ok (_, rowsRev) =>
      let rows := rowsRev.reverse
      if rows.isEmpty then
        pure (.error (tagError "csv" "no data rows"))
      else
        pure (.ok rows)

/--
Read a two-column CSV file into a dataset of pairs `(x, y)`.

This is useful for small regression examples where each row is one training pair.
-/
def readCsvDatasetPairs (path : System.FilePath) (opts : CsvOptions := {}) :
  IO (Result (Dataset (Prod Float Float))) := do
  let rowsRes <- readCsvFloatRows path opts
  match rowsRes with
  | .error e => pure (.error e)
  | .ok rows =>
      let pairsRes : Result (List (Prod Float Float)) := rows.mapM (fun row => do
        match row with
        | [x, y] => .ok (x, y)
        | _ => .error (tagError "csv" "expected exactly 2 columns per row"))
      pure (pairsRes.map Dataset.ofList)

/--
Read an `n`-column CSV file into a dataset of length-`n` vectors.

Each row must have exactly `n` cells.
-/
def readCsvVectorDataset (path : System.FilePath) (n : Nat) (opts : CsvOptions := {}) :
  IO (Result (Dataset (Tensor Float (.dim n .scalar)))) := do
  let rowsRes <- readCsvFloatRows path opts
  match rowsRes with
  | .error e => pure (.error e)
  | .ok rows =>
      let tensorsRes : Result (List (Tensor Float (.dim n .scalar))) := rows.mapM (fun row =>
        vectorOfList (tag := "csv") (n := n) row)
      pure (tensorsRes.map Dataset.ofList)

end Train
end Autograd
end Runtime
