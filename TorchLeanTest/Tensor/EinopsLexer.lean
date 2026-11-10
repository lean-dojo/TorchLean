/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Einops.Syntax.Lexer

/-!
# Einops Lexer Unicode Policy Regression Tests

The Python identifier policy of the einops lexer reproduces the CPython 3.12.13 predicates
`str.isalnum`, `str.isidentifier`, and `str.isdecimal` for Unicode 15.0.0 through the generated
interval table in `TorchLean/Tensor/Einops/Syntax/UnicodeData.lean`. These checks compare the
compiled table lookup with expectations recorded from CPython on selected Unicode property
boundaries, decimal alphabets, and identifier edge cases.

Each expectation `(codepoint, flags, decimal)` was produced by CPython 3.12.13 with
`unicodedata.unidata_version == "15.0.0"`. For the character `c`, `flags` is
`c.isalnum() | c.isidentifier() << 1 | ("a" + c).isidentifier() << 2 | c.isdecimal() << 3` and
`decimal` is `unicodedata.decimal(c, None)`.
-/

@[expose] public section

namespace TorchLean.Test.EinopsLexer

open Einops

/-- Fail with `label` unless `condition` holds. -/
def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"lexer unicode check failed: {label}"

/-- Encode the four character predicates of a policy the way the generated table records them. -/
def policyFlags (policy : IdentifierPolicy) (char : Char) : Nat :=
  let text := String.singleton char
  (if policy.isWordChar char && char != '_' then 1 else 0) +
    (if policy.isIdentifier text then 2 else 0) +
    (if policy.isIdentifier ("a" ++ text) then 4 else 0) +
    (if policy.isDecimal text then 8 else 0)

/-- Read the raw table flags of a character. -/
def tableFlags (char : Char) : Nat :=
  (if PythonUnicode.hasFlag char PythonUnicode.alphanumericFlag then 1 else 0) +
    (if PythonUnicode.hasFlag char PythonUnicode.identifierStartFlag then 2 else 0) +
    (if PythonUnicode.hasFlag char PythonUnicode.identifierContinueFlag then 4 else 0) +
    (if PythonUnicode.hasFlag char PythonUnicode.decimalFlag then 8 else 0)

/-- CPython expectations for sampled code points, as described in the module docstring. -/
def samples : Array (Nat × Nat × Option Nat) := #[
  (0x2F, 0, none),
  (0x30, 13, some 0),
  (0x39, 13, some 9),
  (0x3A, 0, none),
  (0x41, 7, none),
  (0x5F, 6, none),
  (0xB2, 1, none),
  (0x300, 4, none),
  (0x660, 13, some 0),
  (0x669, 13, some 9),
  (0x66A, 0, none),
  (0xFF10, 13, some 0),
  (0xFF19, 13, some 9),
  (0xFF1A, 0, none),
  (0x1D7CE, 13, some 0),
  (0x1D7D7, 13, some 9),
  (0x3134F, 0, none),
  (0x31350, 7, none),
  (0x323AF, 7, none),
  (0x323B0, 0, none),
  (0xE00FF, 0, none),
  (0xE0100, 4, none),
  (0xE01EF, 4, none),
  (0xE01F0, 0, none),
  (0x10FFFF, 0, none)
]
/-- Render a code point as `U+XXXX` for failure messages. -/
def label (codepoint : Nat) : String :=
  s!"U+{String.ofList (Nat.toDigits 16 codepoint)}"

/-- Sampled code points agree with CPython on every predicate and on decimal values. -/
def checkSamples : IO Unit := do
  for (codepoint, flags, decimal) in samples do
    let char := Char.ofNat codepoint
    let name := label codepoint
    expect s!"{name} is a scalar value" (char.toNat == codepoint)
    expect s!"table flags of {name}" (tableFlags char == flags)
    expect s!"policy flags of {name}" (policyFlags IdentifierPolicy.python char == flags)
    expect s!"decimal value of {name}"
      (IdentifierPolicy.python.toNat? (String.singleton char) == decimal)

/-- Run `checkSamples`. -/
def run : IO Unit := do
  IO.println "== Tensor: einops lexer Unicode policy =="
  checkSamples
  IO.println "  einops lexer Unicode policy checks passed"

end TorchLean.Test.EinopsLexer
