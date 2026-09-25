/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Data-File Parsing

This module contains the small pieces shared by the CSV and NPY loaders:

- ASCII digit/sign parsing used by numeric CSV cells and NPY headers; and
- small string helpers for parsing Python/NumPy-style header fragments.

The parser is deliberately small. It supports TorchLean examples and regression tests, not a
full data-ingestion framework.
-/

@[expose] public section

namespace TorchLean
namespace Data
namespace IO

namespace Internal

/-- Maximum number of characters accepted for a numeric CSV cell or numeric NPY header atom. -/
def maxNumericCellChars : Nat := 256

/-- Prefix a parser error with the file format or field that produced it. -/
def formatError (context message : String) : String :=
  s!"{context}: {message}"

/-!
## Decimal scanning
-/

/-- Convert a digit character to its numeric value, or return `none` if not a digit. -/
def digitVal? (c : Char) : Option Nat :=
  if c.isDigit then
    some (c.toNat - ('0' : Char).toNat)
  else
    none

/-- Interpret a list of base-10 digits as a natural number. -/
def digitsToNat (ds : List Nat) : Nat :=
  ds.foldl (fun acc d => acc * 10 + d) 0

/-- Consume a maximal prefix of digits from a character list. -/
def takeDigits (cs : List Char) : List Nat × List Char :=
  let rec go (acc : List Nat) (cs : List Char) : List Nat × List Char :=
    match cs with
    | [] => (acc.reverse, [])
    | c :: rest =>
        match digitVal? c with
        | some d => go (d :: acc) rest
        | none => (acc.reverse, cs)
  go [] cs

/-- Parse an optional leading sign, returning whether the number is negative plus the remaining
characters. -/
def parseSign (cs : List Char) : Bool × List Char :=
  match cs with
  | '-' :: rest => (true, rest)
  | '+' :: rest => (false, rest)
  | _ => (false, cs)

/-- Parse a natural number header value (expects a pure digit string). -/
def parseNatValue (s : String) : Option Nat :=
  let s := (s.trimAscii).toString
  let cs := s.toList
  if cs.isEmpty || s.length > maxNumericCellChars then
    none
  else
    let (digits, rest) := takeDigits cs
    if rest.isEmpty && !digits.isEmpty then
      some (digitsToNat digits)
    else
      none

/-!
## Header string helpers
-/

/-- Drop characters until predicate `p` becomes true. -/
def dropUntil (p : Char -> Bool) : List Char -> List Char
  | [] => []
  | c :: rest => if p c then c :: rest else dropUntil p rest

/-- Take characters until `stop` is encountered (not including `stop`). -/
def takeUntilChar (stop : Char) : List Char -> List Char × List Char
  | [] => ([], [])
  | c :: rest =>
      if c = stop then
        ([], rest)
      else
        let (xs, rem) := takeUntilChar stop rest
        (c :: xs, rem)

/-- End of one field value in a NumPy header dictionary or standalone parser input. -/
def isHeaderValueEnd (rest : List Char) : Bool :=
  match rest.dropWhile Char.isWhitespace with
  | [] | ',' :: _ | '}' :: _ => true
  | _ => false

/-- Parse a closed quoted value from a header fragment. Escaped strings are unsupported. -/
def parseQuotedValue (s : String) : Option String :=
  match (s.trimAsciiStart).toString.toList with
  | quote :: rest =>
      if quote = '\'' || quote = '"' then
        let (value, remaining) := rest.span (fun c => c != quote)
        match remaining with
        | _ :: tail => if isHeaderValueEnd tail then some (String.ofList value) else none
        | [] => none
      else
        none
  | [] => none

/-- Parse a complete boolean header atom (`True` or `False`). -/
def parseBoolValue (s : String) : Option Bool :=
  match (s.trimAsciiStart).toString.toList with
  | 'T' :: 'r' :: 'u' :: 'e' :: rest =>
      if isHeaderValueEnd rest then some true else none
  | 'F' :: 'a' :: 'l' :: 's' :: 'e' :: rest =>
      if isHeaderValueEnd rest then some false else none
  | _ => none

/-- Find the substring after `key` in a header string, if present. -/
def fieldAfter (hdr key : String) : Option String :=
  match hdr.splitOn key with
  | _ :: after :: _ => some after
  | _ => none

/-- Find a header field value by key name (handles both `'key':` and `\"key\":` spellings). -/
def findField (hdr key : String) : Option String :=
  match fieldAfter hdr s!"'{key}':" with
  | some s => some s
  | none => fieldAfter hdr s!"\"{key}\":"

end Internal

end IO
end Data
end TorchLean
