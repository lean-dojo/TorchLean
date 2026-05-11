/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.TensorLoader

/-!
# Shared IO-loader parsing utilities

This module contains the small pieces shared by the CSV and NPY loaders:

- ASCII digit/sign parsing used by numeric CSV cells and NPY headers; and
- small string helpers for parsing Python/NumPy-style header fragments.

The helpers are intentionally modest. They support TorchLean examples and regression tests, not a
full data-ingestion framework.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Train

open Spec
open Tensor

namespace IoLoader.Internal

/-- Maximum number of characters accepted for a numeric CSV cell or numeric NPY header atom. -/
def maxNumericCellChars : Nat := 256

/-!
## Decimal scanning
-/

/-- ASCII digit test used by the numeric parser. -/
def isDigit (c : Char) : Bool :=
  let n := c.toNat
  let n0 := ('0' : Char).toNat
  let n9 := ('9' : Char).toNat
  n0 <= n && n <= n9

/-- Convert a digit character to its numeric value, or return `none` if not a digit. -/
def digitVal? (c : Char) : Option Nat :=
  if isDigit c then
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
  if cs.isEmpty then
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

/-- Parse a quoted value `'...'` or `\"...\"` from a header fragment (best-effort). -/
def parseQuotedValue (s : String) : Option String :=
  let cs := (s.trimAsciiStart).toString.toList
  match cs with
  | quote :: rest =>
      if quote = '\'' || quote = '"' then
        let (valChars, _) := takeUntilChar quote rest
        some (String.ofList valChars)
      else
        none
  | [] => none

/-- Parse a boolean header value (expects `True` or `False`). -/
def parseBoolValue (s : String) : Option Bool :=
  let s := (s.trimAsciiStart).toString
  if s.startsWith "True" then
    some true
  else if s.startsWith "False" then
    some false
  else
    none

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

end IoLoader.Internal

end Train
end Autograd
end Runtime
