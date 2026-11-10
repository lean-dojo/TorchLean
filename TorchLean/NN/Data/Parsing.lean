/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import TorchLean.NN.Data.Monad

/-!
# Data-File Parsing

The small parsers shared by the CSV and NPY loaders, the command line and the verification text
readers:

- ASCII digit and sign scanning for numeric CSV cells and NPY headers (`takeDigits`, `parseSign`,
  `parseNatValue`),
- `parseFloatString`, the one decimal-to-`Float` parser, which rounds once,
- string helpers for Python- and NumPy-style header fragments (`findField`, `parseQuotedValue`,
  `parseBoolValue`).

`readUpTo` is the bounded byte reader shared by the NPY loader and the binary checkpoint readers.

The parsers are small on purpose, sized for TorchLean examples and regression tests.
-/

@[expose] public section

namespace TorchLean
namespace Data

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
  let rec
    /-- The digits of `acc`, which holds them in reverse order, followed by the leading digits of
    `cs`, and the characters after those. -/
    go (acc : List Nat) (cs : List Char) : List Nat × List Char :=
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
def NPY.parseNatValue (s : String) : Option Nat :=
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

/-- End of one field value in a NumPy header dictionary or standalone parser input. -/
def NPY.isHeaderValueEnd (rest : List Char) : Bool :=
  match rest.dropWhile Char.isWhitespace with
  | [] | ',' :: _ | '}' :: _ => true
  | _ => false

/-- Parse a closed quoted value from a header fragment. Escaped strings are unsupported. -/
def NPY.parseQuotedValue (s : String) : Option String :=
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
def NPY.parseBoolValue (s : String) : Option Bool :=
  match (s.trimAsciiStart).toString.toList with
  | 'T' :: 'r' :: 'u' :: 'e' :: rest =>
      if isHeaderValueEnd rest then some true else none
  | 'F' :: 'a' :: 'l' :: 's' :: 'e' :: rest =>
      if isHeaderValueEnd rest then some false else none
  | _ => none

/-- Find the substring after `key` in a header string, if present. -/
def NPY.fieldAfter (hdr key : String) : Option String :=
  match hdr.splitOn key with
  | _ :: after :: _ => some after
  | _ => none

/-- Find a header field value by key name (handles both `'key':` and `\"key\":` spellings). -/
def NPY.findField (hdr key : String) : Option String :=
  match fieldAfter hdr s!"'{key}':" with
  | some s => some s
  | none => fieldAfter hdr s!"\"{key}\":"

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

The decimal value is rounded once to the nearest `Float` (`Float.ofScientific`), and `-0` keeps
its sign. Supported grammar:
- optional sign
- optional integer digits
- optional fractional part `.digits`
- optional scientific exponent `e±digits` or `E±digits`

The integer and fractional parts hold at least one digit between them, so `.5` and `5.` both parse.
Surrounding ASCII whitespace is accepted independently of `TorchLean.Data.CSV.Options.trimCells`.
`NaN`, `inf`, locale separators, quoted CSV cells, strings longer than `Data.maxNumericCellChars`
after trimming and values outside the finite `Float` range are rejected.
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

/-!
## Bounded reads
-/

/--
Read `count` bytes from `handle`, or fewer when the file ends first. Reads go in chunks of at most
65536 bytes, so an untrusted `count` never sizes a buffer before the file supplies the bytes.

The result never holds more than `count` bytes (`readUpTo_size_le`). A chunk longer than the
request is cut to the bytes still wanted, so the bound does not rest on `IO.FS.Handle.read`.
-/
def readUpTo (handle : _root_.IO.FS.Handle) (count : Nat) : _root_.IO ByteArray := do
  let rec
    /-- Append up to `remaining` more bytes to `bytes`, requesting at most 65536 bytes per read and
    stopping at the end of the file. -/
    loop (remaining : Nat) (bytes : ByteArray) : _root_.IO ByteArray := do
      if _hDone : remaining = 0 then
        pure bytes
      else
        let chunk ← handle.read (min remaining 65536).toUSize
        if _hEnd : chunk.size = 0 then
          pure bytes
        else if chunk.size ≤ remaining then
          loop (remaining - chunk.size) (bytes ++ chunk)
        else
          pure (bytes ++ chunk.extract 0 remaining)
      termination_by remaining
      decreasing_by omega
  loop count ByteArray.empty

/-- Each run of the read loop adds at most `remaining` bytes to the bytes it started with. -/
theorem readUpTo.loop_size_le (handle : _root_.IO.FS.Handle) (remaining : Nat) (bytes : ByteArray)
    {result : ByteArray} (h : MonadAttach.CanReturn (readUpTo.loop handle remaining bytes) result) :
    result.size ≤ bytes.size + remaining := by
  induction remaining using Nat.strongRecOn generalizing bytes with
  | _ remaining ih =>
    rw [readUpTo.loop.eq_1] at h
    split at h
    · cases LawfulMonadAttach.eq_of_canReturn_pure h
      omega
    · obtain ⟨chunk, -, h⟩ := LawfulMonadAttach.canReturn_bind_imp' h
      split at h
      · cases LawfulMonadAttach.eq_of_canReturn_pure h
        omega
      · split at h
        · have hRec := ih (remaining - chunk.size) (by omega) (bytes ++ chunk) h
          simp only [ByteArray.size_append] at hRec
          omega
        · cases LawfulMonadAttach.eq_of_canReturn_pure h
          simp only [ByteArray.size_append, ByteArray.size_extract]
          omega

/-- `readUpTo handle count` returns at most `count` bytes, whatever the handle supplies. -/
theorem readUpTo_size_le {handle : _root_.IO.FS.Handle} {count : Nat} {bytes : ByteArray}
    (h : MonadAttach.CanReturn (readUpTo handle count) bytes) : bytes.size ≤ count := by
  have := readUpTo.loop_size_le handle count ByteArray.empty h
  simpa using this

end Data
end TorchLean
