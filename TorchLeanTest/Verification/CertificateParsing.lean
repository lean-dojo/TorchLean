/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.JSON
public import TorchLean.Verify.Certificate.NodeReplay
import TorchLeanExamples.Verification.Splines.PiecewisePolyCert

/-!
# Certificate parsing regressions

These regressions cover the outward rounding of `{center, eps}` input regions and of binary32 IBP
endpoints, and the rejection of negative and NaN radii. They also cover the JSON reader of
certificates with exact signed decimals (number grammar, input after the value, repeated keys,
integers, and the exact decimal checks of boxes and α vectors) and the JSON reader of spline
certificates.
-/

public section

namespace TorchLean.Test.CertificateParsing

open Lean
open FloatLib.Floats (ExecFloat)
open TorchLean.Json BoxRegion

private def check (name : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"certificate parsing: {name}"

private def parse (text : String) : Except String (BoxRegion Float) := do
  BoxRegion.parse "region" (← Certificate.DecimalJson.parse text)

private def boxRegions : IO Unit := do
  match parse r#"{"center": [1.0], "eps": -1e-20}"# with
  | .ok _ => throw <| IO.userError "certificate parsing: negative eps accepted"
  | .error _ => pure ()
  match parse r#"{"center": [1.0], "eps": "NaN"}"# with
  | .ok _ => throw <| IO.userError "certificate parsing: NaN eps accepted"
  | .error _ => pure ()
  -- 1 ± 1e-17 rounds back to 1 under round-to-nearest, but outward rounding must still move.
  let tiny ← IO.ofExcept <| parse r#"{"center": [1.0], "eps": 1e-17}"#
  check "tiny radius widens the lower endpoint" (tiny.lo[0]! < 1.0)
  check "tiny radius widens the upper endpoint" (1.0 < tiny.hi[0]!)
  -- Exact sums stay exact.
  let exact ← IO.ofExcept <| parse r#"{"center": [1, 2, 3], "eps": 0.25}"#
  check "exact lower endpoints" (exact.lo == #[0.75, 1.75, 2.75])
  check "exact upper endpoints" (exact.hi == #[1.25, 2.25, 3.25])
  let zero ← IO.ofExcept <| parse r#"{"center": [0.5], "eps": 0}"#
  check "zero radius gives a point box" (zero.lo == #[0.5] && zero.hi == #[0.5])

/-- The exact value of a finite binary32 endpoint. A nonfinite endpoint fails the test. -/
private def toRat (x : ExecFloat.Binary 8 23) : IO Rat := do
  let some q := ExecFloat.Binary.toRat? x
    | throw <| IO.userError "certificate parsing: nonfinite binary32 endpoint"
  pure q

/-- Exact values of the `k`-th lower and upper endpoints. -/
private def endpoints (box : CROWN.FlatBox (ExecFloat.Binary 8 23)) (k : Nat) :
    IO (Rat × Rat) := do
  if h : k < box.dim then
    pure (← toRat (box.lo.getScalar ⟨k, h⟩), ← toRat (box.hi.getScalar ⟨k, h⟩))
  else
    throw <| IO.userError s!"certificate parsing: IBP box has no coordinate {k}"

private def ibpEndpoints : IO Unit := do
  let j ← IO.ofExcept <|
    Certificate.DecimalJson.parse r#"{"lo": [0.1, -0.1, 1], "hi": [0.1, -0.1, 1]}"#
  let some box ← Certificate.NodeReplay.parseFlatBox? 3 j
    | throw <| IO.userError "certificate parsing: IBP box did not parse"
  for (k, decimal) in [(0, (1 : Rat) / 10), (1, -1 / 10), (2, 1)] do
    let (lo, hi) ← endpoints box k
    check s!"binary32 lower endpoint {k} is at or below the decimal" (lo ≤ decimal)
    check s!"binary32 upper endpoint {k} is at or above the decimal" (decimal ≤ hi)
  let (lo0, hi0) ← endpoints box 0
  check "inexact decimal gives a nondegenerate box" (lo0 < hi0)
  let (lo2, hi2) ← endpoints box 2
  check "exact decimal gives a point" (lo2 == hi2)

private def expectFailure (name fragment : String) (action : IO Unit) : IO Unit := do
  let error ← try
    action
    pure none
  catch e =>
    pure (some e.toString)
  match error with
  | none => throw <| IO.userError s!"certificate parsing: {name} was accepted"
  | some message => check s!"{name}: unexpected error: {message}" (message.contains fragment)

/-- The JSON text `text` as a value with exact signed decimals. A parse error fails the test. -/
private def decimalValue (text : String) : IO Certificate.DecimalJson.Value :=
  IO.ofExcept <| Certificate.DecimalJson.parse text

/-- The bits of the entries of a binary32 vector. -/
private def bitsOf {n : Nat} (v : Tensor (ExecFloat.Binary 8 23) [n]) : List Nat :=
  (List.finRange n).map fun i => (v.getScalar i).toNatBits

/-- The JSON reader of certificates refuses numbers outside the grammar of RFC 8259 section 6 and
input after the value, and reads the spellings it accepts as their exact decimals. It refuses an
object with a repeated key at any depth, comparing keys after decoding (`"x"` and `"\u0078"`).
`Value.int?` reads a number as an integer when its decimal exponent, the exponent of the token less
its count of fraction digits, is from `0` to `40`, and does not accept the integral values `1.0` and
`10e-1`. It reads the string `"7"` as `7`. -/
private def decimalDocuments : IO Unit := do
  let read (text : String) : IO Unit := discard <| decimalValue text
  for (text, fragment) in [("01", "expected end of input"), ("-01", "expected end of input"),
      ("[01]", "unexpected character in array"), ("0x10", "expected end of input"),
      ("1.", "unexpected end of input"), ("1e", "unexpected end of input"),
      ("1e+", "unexpected end of input"), ("1.5e", "unexpected end of input"),
      ("-", "unexpected end of input"), (".5", "unexpected input"), ("+1", "unexpected input"),
      ("NaN", "unexpected input"), ("Infinity", "unexpected input"), ("-Infinity", "expected 1-9"),
      ("1 2", "expected end of input"), ("[1]x", "expected end of input"),
      ("[1 2]", "unexpected character in array"), ("[1,]", "unexpected input"),
      (r#"{"a":1,}"#, "expected \"")] do
    expectFailure s!"the JSON text {text}" fragment (read text)
  for (text, rendered) in [("1E5", "1e5"), ("-2.5E+2", "-25e1"), ("-0e-0", "-0.0"),
      ("0e18446744073709551617", "0"), (" [ 1 ,\t2 ]\r\n", "[1,2]"),
      (r#"{"a":{"k":1},"b":{"k":2}}"#, r#"{"a":{"k":1},"b":{"k":2}}"#)] do
    check s!"the JSON text {text} reads as {rendered}" ((← decimalValue text).render == rendered)
  for text in [r#"{"x":1,"x":1}"#, r#"{"x":1,"x":2}"#, r#"{"x":1,"\u0078":2}"#,
      r#"{"a":{"b":1,"b":2}}"#, r#"[{"k":1,"k":1}]"#,
      r#"{"lo":[0],"hi":[1],"note":1,"note":2}"#] do
    expectFailure s!"the JSON text {text}" "repeated key" (read text)
  let integers : List (String × Option Int) := [("1e1", some 10), ("1.0e1", some 10),
    ("2.50e2", some 250), ("-5", some (-5)), ("0.0e41", some 0), ("-0", some 0),
    ("1e40", some (10 ^ 40)), ("1.0", none), ("10e-1", none), ("0e41", none), ("1e41", none),
    ("\"7\"", some 7)]
  for (text, n) in integers do
    check s!"the integer of {text} is {n}" ((← decimalValue text).int? == n)
  check "the natural number of -0 is 0"
    ((← IO.ofExcept ((← decimalValue "-0").nat "n")) == 0)
  expectFailure "the natural number -1" "expected a natural number"
    (discard <| IO.ofExcept ((← decimalValue "-1").nat "n"))

/-- Boxes and α vectors on their exact decimals: `[-1e400, 1e400]` reads as `[-∞, +∞]`, a lower
endpoint above its upper one by less than a binary64 ulp is refused, and an α entry outside
`[0, 1]` is refused, also when it rounds to `1` or to `-0` in binary64. -/
private def decimalEndpoints : IO Unit := do
  let box (text : String) : IO (Option (CROWN.FlatBox (ExecFloat.Binary 8 23))) := do
    Certificate.NodeReplay.parseFlatBox? 1 (← decimalValue text)
  let some wide ← box r#"{"lo":[-1e400],"hi":[1e400]}"#
    | throw <| IO.userError "certificate parsing: the box [-1e400, 1e400] parsed as absent"
  check "the box [-1e400, 1e400] reads as [-∞, +∞]"
    (bitsOf wide.lo == [0xFF800000] && bitsOf wide.hi == [0x7F800000])
  let some near ← box r#"{"lo":[1],"hi":[1.00000000000000001]}"#
    | throw <| IO.userError "certificate parsing: the box [1, 1.00000000000000001] parsed as absent"
  check "the box [1, 1.00000000000000001] reads as [1, the successor of 1]"
    (bitsOf near.lo == [0x3F800000] && bitsOf near.hi == [0x3F800001])
  expectFailure "the box [1.00000000000000001, 1]" "every lower bound must be <= its upper bound"
    (discard <| box r#"{"lo":[1.00000000000000001],"hi":[1]}"#)
  let alpha (text : String) :
      IO (Option (CROWN.Graph.FlatTensor (ExecFloat.Binary 8 23))) := do
    Certificate.NodeReplay.parseAlphaVec? 1 (← decimalValue s!"[{text}]")
  for text in ["1.00000000000000001", "-1e-400", "-1e-45"] do
    expectFailure s!"the α entry {text}" "requires 0 ≤ alpha ≤ 1" (discard <| alpha text)
  for (text, bits) in [("0", 0), ("-0", 0x80000000), ("-0.0", 0x80000000), ("1e-400", 0),
      ("0.5", 0x3F000000), ("0.99999999999999999", 0x3F800000), ("1", 0x3F800000)] do
    let some a ← alpha text
      | throw <| IO.userError s!"certificate parsing: the α entry {text} parsed as absent"
    check s!"the α entry {text} reads as {bits}" (bitsOf a.v == [bits])

open Examples.Splines

private def splineJson : IO Unit := do
  let validJson ← IO.ofExcept <| Json.parse
    (r#"{"format":"piecewise_poly_v0","degree":1,"xs":["0","1"],"ys":["0","1"],"# ++
     r#""pieces":[{"lo":"0","hi":"1","coeffs":["0","1"]}]}"#)
  checkJson validJson
  checkJsonBinary32 validJson
  let missingJson ← IO.ofExcept <| Json.parse
    r#"{"format":"piecewise_poly_v0","degree":1,"xs":["0","1"],"ys":["0","1"],"pieces":[]}"#
  expectFailure "JSON missing piece" "pieces length mismatch"
    (discard <| Examples.Splines.Certificate.parse missingJson)

/-- Run `boxRegions`, `ibpEndpoints`, `decimalDocuments`, `decimalEndpoints` and
`splineJson`. -/
def run : IO Unit := do
  boxRegions
  ibpEndpoints
  decimalDocuments
  decimalEndpoints
  splineJson
  IO.println
    "  Certificate parsing: rounding, radius, decimal JSON and spline JSON tests passed"

end TorchLean.Test.CertificateParsing
