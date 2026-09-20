/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Internal.Syntax.Lexer

/-!
# Einops Lexer Unicode Policy Regression Tests

The Python identifier policy of the einops lexer reproduces the CPython 3.12.13 predicates
`str.isalnum`, `str.isidentifier`, and `str.isdecimal` for Unicode 15.0.0 through the generated
interval table in `NN/Tensor/Internal/Syntax/UnicodeData.lean`. These checks compare the compiled
table lookup with expectations recorded directly from CPython on every ASCII character, on code
points spread through the Basic Multilingual Plane and the supplementary planes, and on the
boundaries of representative property runs.

Each expectation `(codepoint, flags, decimal)` was produced by CPython 3.12.13 with
`unicodedata.unidata_version == "15.0.0"`. For the character `c`, `flags` is
`c.isalnum() | c.isidentifier() << 1 | ("a" + c).isidentifier() << 2 | c.isdecimal() << 3` and
`decimal` is `unicodedata.decimal(c, None)`.
-/

@[expose] public section

namespace NN.Tests.Tensor.Lexer

open TorchLean.Tensor.Internal.Syntax

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
  (0x2F, 0, none), (0x30, 13, some 0), (0x39, 13, some 9), (0x3A, 0, none), (0x40, 0, none),
  (0x41, 7, none), (0x5A, 7, none), (0x5B, 0, none), (0x5E, 0, none), (0x5F, 6, none),
  (0x60, 0, none), (0x61, 7, none), (0x7A, 7, none), (0x7B, 0, none), (0x80, 0, none),
  (0xA9, 0, none), (0xAA, 7, none), (0xAB, 0, none), (0xB1, 0, none), (0xB2, 1, none),
  (0xB3, 1, none), (0xB4, 0, none), (0x145, 7, none), (0x20A, 7, none), (0x2CF, 7, none),
  (0x2FF, 0, none), (0x300, 4, none), (0x36F, 4, none), (0x370, 7, none), (0x394, 7, none),
  (0x459, 7, none), (0x51E, 7, none), (0x5E3, 7, none), (0x65F, 4, none), (0x660, 13, some 0),
  (0x669, 13, some 9), (0x66A, 0, none), (0x6A8, 7, none), (0x76D, 7, none), (0x832, 0, none),
  (0x8F7, 4, none), (0x965, 0, none), (0x966, 13, some 0), (0x96F, 13, some 9), (0x970, 0, none),
  (0x9BC, 4, none), (0xA81, 4, none), (0xB46, 0, none), (0xC0B, 7, none), (0xCD0, 0, none),
  (0xD95, 7, none), (0xE5A, 0, none), (0xF1F, 0, none), (0xFE4, 0, none), (0x10A9, 7, none),
  (0x116E, 7, none), (0x1233, 7, none), (0x12F8, 7, none), (0x13BD, 7, none), (0x1482, 7, none),
  (0x1547, 7, none), (0x160C, 7, none), (0x16D1, 7, none), (0x1796, 7, none), (0x185B, 7, none),
  (0x1920, 4, none), (0x19E5, 0, none), (0x1AAA, 0, none), (0x1B6F, 4, none), (0x1C34, 4, none),
  (0x1CF9, 4, none), (0x1DBE, 7, none), (0x1E83, 7, none), (0x1F48, 7, none), (0x200D, 0, none),
  (0x206F, 0, none), (0x2070, 1, none), (0x2071, 7, none), (0x20D2, 4, none), (0x215F, 1, none),
  (0x2160, 7, none), (0x2188, 7, none), (0x2189, 1, none), (0x2197, 0, none), (0x225C, 0, none),
  (0x2321, 0, none), (0x23E6, 0, none), (0x24AB, 0, none), (0x2570, 0, none), (0x2635, 0, none),
  (0x26FA, 0, none), (0x27BF, 0, none), (0x2884, 0, none), (0x2949, 0, none), (0x2A0E, 0, none),
  (0x2AD3, 0, none), (0x2B98, 0, none), (0x2C5D, 7, none), (0x2D22, 7, none), (0x2DE7, 4, none),
  (0x2EAC, 0, none), (0x2F71, 0, none), (0x3036, 0, none), (0x30FB, 0, none), (0x31C0, 0, none),
  (0x3285, 1, none), (0x334A, 0, none), (0x340F, 7, none), (0x34D4, 7, none), (0x3599, 7, none),
  (0x365E, 7, none), (0x3723, 7, none), (0x37E8, 7, none), (0x38AD, 7, none), (0x3972, 7, none),
  (0x3A37, 7, none), (0x3AFC, 7, none), (0x3BC1, 7, none), (0x3C86, 7, none), (0x3D4B, 7, none),
  (0x3E10, 7, none), (0x3ED5, 7, none), (0x3F9A, 7, none), (0x405F, 7, none), (0x4124, 7, none),
  (0x41E9, 7, none), (0x42AE, 7, none), (0x4373, 7, none), (0x4438, 7, none), (0x44FD, 7, none),
  (0x45C2, 7, none), (0x4687, 7, none), (0x474C, 7, none), (0x4811, 7, none), (0x48D6, 7, none),
  (0x499B, 7, none), (0x4A60, 7, none), (0x4B25, 7, none), (0x4BEA, 7, none), (0x4CAF, 7, none),
  (0x4D74, 7, none), (0x4DFF, 0, none), (0x4E00, 7, none), (0x4E39, 7, none), (0x4EFE, 7, none),
  (0x4FC3, 7, none), (0x5088, 7, none), (0x514D, 7, none), (0x5212, 7, none), (0x52D7, 7, none),
  (0x539C, 7, none), (0x5461, 7, none), (0x5526, 7, none), (0x55EB, 7, none), (0x56B0, 7, none),
  (0x5775, 7, none), (0x583A, 7, none), (0x58FF, 7, none), (0x59C4, 7, none), (0x5A89, 7, none),
  (0x5B4E, 7, none), (0x5C13, 7, none), (0x5CD8, 7, none), (0x5D9D, 7, none), (0x5E62, 7, none),
  (0x5F27, 7, none), (0x5FEC, 7, none), (0x60B1, 7, none), (0x6176, 7, none), (0x623B, 7, none),
  (0x6300, 7, none), (0x63C5, 7, none), (0x648A, 7, none), (0x654F, 7, none), (0x6614, 7, none),
  (0x66D9, 7, none), (0x679E, 7, none), (0x6863, 7, none), (0x6928, 7, none), (0x69ED, 7, none),
  (0x6AB2, 7, none), (0x6B77, 7, none), (0x6C3C, 7, none), (0x6D01, 7, none), (0x6DC6, 7, none),
  (0x6E8B, 7, none), (0x6F50, 7, none), (0x7015, 7, none), (0x70DA, 7, none), (0x719F, 7, none),
  (0x7264, 7, none), (0x7329, 7, none), (0x73EE, 7, none), (0x74B3, 7, none), (0x7578, 7, none),
  (0x763D, 7, none), (0x7702, 7, none), (0x77C7, 7, none), (0x788C, 7, none), (0x7951, 7, none),
  (0x7A16, 7, none), (0x7ADB, 7, none), (0x7BA0, 7, none), (0x7C65, 7, none), (0x7D2A, 7, none),
  (0x7DEF, 7, none), (0x7EB4, 7, none), (0x7F79, 7, none), (0x803E, 7, none), (0x8103, 7, none),
  (0x81C8, 7, none), (0x828D, 7, none), (0x8352, 7, none), (0x8417, 7, none), (0x84DC, 7, none),
  (0x85A1, 7, none), (0x8666, 7, none), (0x872B, 7, none), (0x87F0, 7, none), (0x88B5, 7, none),
  (0x897A, 7, none), (0x8A3F, 7, none), (0x8B04, 7, none), (0x8BC9, 7, none), (0x8C8E, 7, none),
  (0x8D53, 7, none), (0x8E18, 7, none), (0x8EDD, 7, none), (0x8FA2, 7, none), (0x9067, 7, none),
  (0x912C, 7, none), (0x91F1, 7, none), (0x92B6, 7, none), (0x937B, 7, none), (0x9440, 7, none),
  (0x9505, 7, none), (0x95CA, 7, none), (0x968F, 7, none), (0x9754, 7, none), (0x9819, 7, none),
  (0x98DE, 7, none), (0x99A3, 7, none), (0x9A68, 7, none), (0x9B2D, 7, none), (0x9BF2, 7, none),
  (0x9CB7, 7, none), (0x9D7C, 7, none), (0x9E41, 7, none), (0x9F06, 7, none), (0x9FCB, 7, none),
  (0xA090, 7, none), (0xA155, 7, none), (0xA21A, 7, none), (0xA2DF, 7, none), (0xA3A4, 7, none),
  (0xA469, 7, none), (0xA48C, 7, none), (0xA48D, 0, none), (0xA52E, 7, none), (0xA5F3, 7, none),
  (0xA6B8, 7, none), (0xA77D, 7, none), (0xA842, 7, none), (0xA907, 13, some 7), (0xA9CC, 0, none),
  (0xAA91, 7, none), (0xAB56, 7, none), (0xABFF, 0, none), (0xAC00, 7, none), (0xAC1B, 7, none),
  (0xACE0, 7, none), (0xADA5, 7, none), (0xAE6A, 7, none), (0xAF2F, 7, none), (0xAFF4, 7, none),
  (0xB0B9, 7, none), (0xB17E, 7, none), (0xB243, 7, none), (0xB308, 7, none), (0xB3CD, 7, none),
  (0xB492, 7, none), (0xB557, 7, none), (0xB61C, 7, none), (0xB6E1, 7, none), (0xB7A6, 7, none),
  (0xB86B, 7, none), (0xB930, 7, none), (0xB9F5, 7, none), (0xBABA, 7, none), (0xBB7F, 7, none),
  (0xBC44, 7, none), (0xBD09, 7, none), (0xBDCE, 7, none), (0xBE93, 7, none), (0xBF58, 7, none),
  (0xC01D, 7, none), (0xC0E2, 7, none), (0xC1A7, 7, none), (0xC26C, 7, none), (0xC331, 7, none),
  (0xC3F6, 7, none), (0xC4BB, 7, none), (0xC580, 7, none), (0xC645, 7, none), (0xC70A, 7, none),
  (0xC7CF, 7, none), (0xC894, 7, none), (0xC959, 7, none), (0xCA1E, 7, none), (0xCAE3, 7, none),
  (0xCBA8, 7, none), (0xCC6D, 7, none), (0xCD32, 7, none), (0xCDF7, 7, none), (0xCEBC, 7, none),
  (0xCF81, 7, none), (0xD046, 7, none), (0xD10B, 7, none), (0xD1D0, 7, none), (0xD295, 7, none),
  (0xD35A, 7, none), (0xD41F, 7, none), (0xD4E4, 7, none), (0xD5A9, 7, none), (0xD66E, 7, none),
  (0xD733, 7, none), (0xD7A3, 7, none), (0xD7A4, 0, none), (0xD7F8, 7, none), (0xE06F, 0, none),
  (0xE134, 0, none), (0xE1F9, 0, none), (0xE2BE, 0, none), (0xE383, 0, none), (0xE448, 0, none),
  (0xE50D, 0, none), (0xE5D2, 0, none), (0xE697, 0, none), (0xE75C, 0, none), (0xE821, 0, none),
  (0xE8E6, 0, none), (0xE9AB, 0, none), (0xEA70, 0, none), (0xEB35, 0, none), (0xEBFA, 0, none),
  (0xECBF, 0, none), (0xED84, 0, none), (0xEE49, 0, none), (0xEF0E, 0, none), (0xEFD3, 0, none),
  (0xF098, 0, none), (0xF15D, 0, none), (0xF222, 0, none), (0xF2E7, 0, none), (0xF3AC, 0, none),
  (0xF471, 0, none), (0xF536, 0, none), (0xF5FB, 0, none), (0xF6C0, 0, none), (0xF785, 0, none),
  (0xF84A, 0, none), (0xF90F, 7, none), (0xF9D4, 7, none), (0xFA99, 7, none), (0xFB5E, 7, none),
  (0xFC23, 7, none), (0xFCE8, 7, none), (0xFDAD, 7, none), (0xFE72, 1, none), (0xFF0F, 0, none),
  (0xFF10, 13, some 0), (0xFF19, 13, some 9), (0xFF1A, 0, none), (0xFF37, 7, none),
  (0xFFFC, 0, none), (0x10000, 7, none), (0x14F0F, 0, none), (0x19E1E, 0, none), (0x1D7CD, 0, none),
  (0x1D7CE, 13, some 0), (0x1D7D7, 13, some 9), (0x1D7D8, 13, some 0), (0x1D7E1, 13, some 9),
  (0x1D7E2, 13, some 0), (0x1ED2D, 1, none), (0x23C3C, 7, none), (0x28B4B, 7, none),
  (0x2DA5A, 7, none), (0x3134F, 0, none), (0x31350, 7, none), (0x323AF, 7, none),
  (0x323B0, 0, none), (0x32969, 0, none), (0x37878, 0, none), (0x3C787, 0, none),
  (0x41696, 0, none), (0x465A5, 0, none), (0x4B4B4, 0, none), (0x503C3, 0, none),
  (0x552D2, 0, none), (0x5A1E1, 0, none), (0x5F0F0, 0, none), (0x63FFF, 0, none),
  (0x68F0E, 0, none), (0x6DE1D, 0, none), (0x72D2C, 0, none), (0x77C3B, 0, none),
  (0x7CB4A, 0, none), (0x81A59, 0, none), (0x86968, 0, none), (0x8B877, 0, none),
  (0x90786, 0, none), (0x95695, 0, none), (0x9A5A4, 0, none), (0x9F4B3, 0, none),
  (0xA43C2, 0, none), (0xA92D1, 0, none), (0xAE1E0, 0, none), (0xB30EF, 0, none),
  (0xB7FFE, 0, none), (0xBCF0D, 0, none), (0xC1E1C, 0, none), (0xC6D2B, 0, none),
  (0xCBC3A, 0, none), (0xD0B49, 0, none), (0xD5A58, 0, none), (0xDA967, 0, none),
  (0xDF876, 0, none), (0xE00FF, 0, none), (0xE0100, 4, none), (0xE01EF, 4, none),
  (0xE01F0, 0, none), (0xE4785, 0, none), (0xE9694, 0, none), (0xEE5A3, 0, none),
  (0xF34B2, 0, none), (0xF83C1, 0, none), (0xFD2D0, 0, none), (0x1021DF, 0, none),
  (0x1070EE, 0, none), (0x10BFFD, 0, none), (0x10FFFF, 0, none)
]

/-- Render a code point as `U+XXXX` for failure messages. -/
def label (codepoint : Nat) : String :=
  s!"U+{String.ofList (Nat.toDigits 16 codepoint)}"

/-- On ASCII the Python predicates coincide with the ASCII policy, character by character. -/
def checkAscii : IO Unit := do
  for codepoint in [0:128] do
    let char := Char.ofNat codepoint
    let text := String.singleton char
    expect s!"ASCII flags of {label codepoint}"
      (policyFlags IdentifierPolicy.pythonUnicode char == policyFlags IdentifierPolicy.ascii char)
    expect s!"ASCII table flags of {label codepoint}"
      (tableFlags char == policyFlags IdentifierPolicy.ascii char)
    expect s!"ASCII decimal value of {label codepoint}"
      (IdentifierPolicy.pythonUnicode.toNat? text == IdentifierPolicy.ascii.toNat? text)

/-- Sampled code points agree with CPython on every predicate and on decimal values. -/
def checkSamples : IO Unit := do
  for (codepoint, flags, decimal) in samples do
    let char := Char.ofNat codepoint
    let name := label codepoint
    expect s!"{name} is a scalar value" (char.toNat == codepoint)
    expect s!"table flags of {name}" (tableFlags char == flags)
    expect s!"policy flags of {name}" (policyFlags IdentifierPolicy.pythonUnicode char == flags)
    expect s!"decimal value of {name}"
      (IdentifierPolicy.pythonUnicode.toNat? (String.singleton char) == decimal)

/-- Whole words behave like CPython's `str` predicates and `int` conversion. -/
def checkWords : IO Unit := do
  let policy := IdentifierPolicy.pythonUnicode
  expect "ASCII identifier" (policy.isIdentifier "batch_size1")
  expect "Latin identifier with diacritics" (policy.isIdentifier "höhe")
  expect "Han identifier" (policy.isIdentifier "变量")
  expect "leading underscore" (policy.isIdentifier "_x")
  expect "leading digit is rejected" (!policy.isIdentifier "1abc")
  expect "empty word is rejected" (!policy.isIdentifier "")
  expect "superscript two is alphanumeric" (policy.isWordChar '²')
  expect "superscript two cannot start an identifier" (!policy.isIdentifier "²")
  expect "superscript two cannot continue an identifier" (!policy.isIdentifier "a²")
  expect "superscript two is not decimal" (!policy.isDecimal "²")
  expect "middle dot continues an identifier" (policy.isIdentifier "a·b")
  expect "middle dot cannot start an identifier" (!policy.isIdentifier "·b")
  expect "hyphen is not a word character" (!policy.isWordChar '-')
  expect "Arabic-Indic digits are decimal" (policy.isDecimal "٣٤")
  expect "Arabic-Indic digits convert" (policy.toNat? "٣٤" == some 34)
  expect "fullwidth digits convert" (policy.toNat? "１２" == some 12)
  expect "mixed decimal alphabets convert" (policy.toNat? "1٣" == some 13)
  expect "mathematical bold digits convert" (policy.toNat? "𝟗𝟘" == some 90)
  expect "large decimal converts exactly"
    (policy.toNat? "123456789012345678901234567890" == some 123456789012345678901234567890)
  expect "letters are not decimal" (!policy.isDecimal "12a")
  expect "letters do not convert" (policy.toNat? "12a" == none)
  expect "empty word is not decimal" (!policy.isDecimal "")

def run : IO Unit := do
  IO.println "== Tensor: einops lexer Unicode policy =="
  checkAscii
  checkSamples
  checkWords
  IO.println "  einops lexer Unicode policy checks passed"

end NN.Tests.Tensor.Lexer
