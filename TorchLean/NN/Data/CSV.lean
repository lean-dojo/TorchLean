/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Data.Parsing

/-!
# CSV Loader

Small CSV helpers for TorchLean examples and regression tests.

The grammar is narrow on purpose: delimiter-separated numeric cells without quotes, each in the
decimal syntax of `parseFloatString`. Quoted fields, escaped delimiters, locale-specific number
formats, `NaN` and `inf` are unsupported.
-/

@[expose] public section

namespace TorchLean
namespace Data
namespace CSV

/--
Options for the CSV parser in this module.

The parser accepts unquoted numeric fields with literal delimiters and locale-independent
number syntax.
-/
structure Options where
  /-- Delimiter character (default: `,`). -/
  delimiter : Char := ','
  /-- If true, drop the first line before parsing rows. -/
  skipHeader : Bool := false
  /--
  Trim rows before empty-line detection and cells before empty-cell detection.
  Numeric conversion accepts surrounding ASCII whitespace even when this option is false.
  -/
  trimCells : Bool := true
  /-- If true, ignore empty lines (otherwise treat them as an error). -/
  allowEmptyLines : Bool := true

/--
Parse one CSV line into an array of floats.

Returns `none` for empty lines when `allowEmptyLines = true`.
-/
def parseLine (tag : String) (options : Options) (rowIdx : Nat) (line : String) :
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
Read a CSV file into an array of float rows, for compact example datasets and runtime checks.

A trailing newline ends the last line, and row numbers in errors count from 1 after any skipped
header. File-system errors are raised in `IO`, and parse errors and a file without data rows come
back as `Except.error`.
-/
def read (path : System.FilePath) (options : Options := {}) :
  IO (Except String (Array (Array Float))) := do
  let content <- IO.FS.readFile path
  -- A trailing newline terminates the last line, so the empty piece after it is dropped.
  let lines := match content.splitOn "\n" with
    | [] => []
    | first :: rest => if rest.getLast? == some "" then (first :: rest).dropLast else first :: rest
  let lines := if options.skipHeader then lines.drop 1 else lines
  let res : Except String (Nat × Array (Array Float)) :=
    lines.foldlM (init := (0, #[])) (fun acc line => do
      let (i, rows) := acc
      let rowIdx := i + 1
      match parseLine (tag := "csv") options rowIdx line with
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

end CSV
end Data
end TorchLean
