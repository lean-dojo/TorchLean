/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Builder
public import TorchLean.Verify.Certificate.NodeReplay
public import TorchLean.Verify.JSON
public import TorchLeanExamples.Verification.ODE.Parse
public import TorchLeanExamples.Verification.PINN
public import TorchLeanExamples.Verification.PINN.PDEParse
public import TorchLeanTest.Verification.CertificateVerdict

/-!
# CertificatePreconditions

Checks of the readers and writers of certificate text, on side conditions that are easy for an
external producer to get wrong.

The input-region reader infers the dimension of a `{center, eps}` or `{lo, hi}` region and refuses
a malformed declared dimension, a declared dimension the endpoints disagree with, endpoints of
different lengths, a negative radius, a reversed interval and mixed schemas. The PINN reader
refuses a spacing that is not positive, a negative perturbation radius, incomplete `u_bounds` and a
reversed residual interval. α-CROWN slopes must be in $[0,1]$ and a box must not have a lower
endpoint above its upper one. ODE and PDE expressions follow their documented power and negation
semantics, and ReLU phases must be `-1`, `0` or `1`, of the right length. The power rows at $u=2$
compare bits, since an ODE or PDE product is widened outward by one binary64 value.

An interval endpoint beyond binary32 parses as the outward binary32 endpoint. An affine matrix
entry whose nearest binary32 value is infinite is refused, and an offset whose nearest binary32
value is infinite is accepted only as the infinity that bounds nothing on its side, `-∞` in `loC`
and `+∞` in `hiC`. Every finite binary32 class, `-0` included, written by the certificate writers
reads back to itself in each of the six rounding modes of `Model.IEEERoundingMode`. The decimals
`±(t - 1)`, `±t` and `±(t + 1)` at the binary32 overflow threshold `t = 2 ^ 128 - 2 ^ 103` read as
their stated bits to nearest, and only the finite ones are accepted as a matrix coefficient in
`loA`. The decimals `±1e39`, `±1e40`, `±1e-50` and `±1e-51` beyond the binary32 range, and
`±1e999999999` and `±1e-999999999`, whose exponents FloatLib's conversion clamps, read as their
stated bits in each of the six modes. The writers emit the infinite offsets as `-1e39` and `1e39`,
which read back to them, and the readers refuse NaN and the strings `"NaN"`, `"-inf"` and `"+inf"`.

`run` ends with `TorchLean.Test.CertificateVerdict.run`, the file checks of the CROWN node verdict
and the replay diagnostic.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.Formats.BinaryInterchange (Model)

open Lean
open TorchLean.CROWN
open Graph
open TorchLean.Certificate.NodeReplay TorchLean.Certificate CROWNNode
open DecimalJson (Value)
open FloatLib.Numerics.DecimalText (Decimal)

namespace TorchLean.Test
namespace CertificatePreconditions

/-- Fail the test with `msg` unless a Boolean check succeeds. -/
def expect (msg : String) (b : Bool) : IO Unit := do
  unless b do
    throw <| IO.userError msg

/-- Require an `IO` action to reject malformed certificate input. -/
def expectRejected {α : Type} (msg : String) (act : IO α) : IO Unit := do
  let mut rejected := false
  try
    let _ ← act
  catch _ =>
    rejected := true
  unless rejected do
    throw <| IO.userError msg

/-- Parse JSON for a test fixture, turning parser errors into test failures. -/
def parseJson! (s : String) : IO Json := do
  match Json.parse s with
  | .ok j => pure j
  | .error e => throw <| IO.userError s!"bad test JSON: {e}"

/-- Parse JSON with exact signed decimals for a certificate fixture, turning parser errors into
test failures. -/
def parseValue! (s : String) : IO Value := do
  match DecimalJson.parse s with
  | .ok v => pure v
  | .error e => throw <| IO.userError s!"bad test JSON: {e}"

/-- Parse a certificate input box and fail the test when the schema is rejected. -/
def parseInputRegion! (source : String) : IO (Json.BoxRegion Float) := do
  let json ← parseValue! source
  let inputResult : Except String Value :=
    match json.field "input" "top-level" with
    | .ok input => .ok input
    | .error _ => json.field "region" "top-level"
  let input ←
    match inputResult with
    | .ok input => pure input
    | .error e => throw <| IO.userError s!"input-region parser rejected a valid fixture: {e}"
  match Json.BoxRegion.parse "input" input with
  | .ok region => pure region
  | .error e => throw <| IO.userError s!"input-region parser rejected a valid fixture: {e}"

/-- Compare a computed scalar interval pair with its exact expected endpoints. -/
def expectFloatPair (msg : String) (actual : Option (Float × Float))
    (expected : Float × Float) : IO Unit := do
  match actual with
  | some pair =>
      expect msg (pair.1 == expected.1 && pair.2 == expected.2)
  | none =>
      throw <| IO.userError s!"{msg}: expression evaluation failed"

/-- Compare a computed binary64 interval pair with the bits of its expected endpoints. -/
def expectFloatBits (msg : String) (actual : Option (Float × Float)) (lo hi : UInt64) :
    IO Unit :=
  expect msg (actual.map (fun pair => (pair.1.toBits, pair.2.toBits)) == some (lo, hi))

/-- Parse and evaluate an ODE expression on the point interval $u=2$. -/
def evalODEAtTwo (source : String) : Option (Float × Float) := do
  let expr ← Examples.ODE.parseExpr source |>.toOption
  Examples.ODE.Expr.eval { t := (0.0, 0.0), u := (2.0, 2.0) } expr

/-- Parse and evaluate a PDE expression on the point interval $u=2$. -/
def evalPDEAtTwo (source : String) : Option (Float × Float) := do
  let expr ← Examples.PINN.parseExpr (fun _ => none) source |>.toOption
  Examples.PINN.Expr.eval
    { u := some (2.0, 2.0), duX := none, duY := none, d2uX := none, d2uY := none }
    expr

/-- The graph of operator `o` on one graph input per input port, its outputs the operator's. -/
def opGraph {ins outs : List Port} (o : Op ins outs) : IO IR.Graph := do
  let (b, refs) := ins.foldl (fun (b, refs) p =>
    let (b, r) := b.input refs.length p
    (b, refs ++ [r])) (({} : IR.Graph.Builder), [])
  let (b, out) := b.op o refs
  IO.ofExcept (b.build out.toArray)

/-- The entries of a binary32 vector. -/
def entries {n : Nat} (v : Tensor (Binary 8 23) [n]) : Array (Binary 8 23) :=
  Array.ofFn v.getScalar

/-- The bits of the lower and of the upper offsets of affine bounds. -/
def offsetBits (B : FlatAffineBounds (Binary 8 23)) : List Nat × List Nat :=
  ((entries B.loAff.bias).toList.map (·.toNatBits), (entries B.hiAff.bias).toList.map (·.toNatBits))

/-- The JSON text of 2 by 2 affine bounds whose numbers are `0` except the last one of `field`
(`loA`, `loC`, `hiA` or `hiC`), which is the token `coefficient`. -/
def affineText (field coefficient : String) : String :=
  let entry (f : String) : String :=
    let x := if f == field then coefficient else "0"
    if f.endsWith "A" then s!"[[0,0],[0,{x}]]" else s!"[0,{x}]"
  "{" ++ ",".intercalate (["loA", "loC", "hiA", "hiC"].map fun f =>
    "\"" ++ f ++ "\":" ++ entry f) ++ "}"

/-- The bits of one binary32 value of each finite class: both zeros, the least subnormals, the
largest subnormal, the least normal, `1`, and the largest finite values. -/
def finiteClasses : List Nat :=
  [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0x7F7FFFFF, 0xFF7FFFFF]

/-- JSON numbers beyond binary32 in boxes and affine bounds. -/
def boundary : IO Unit := do
  -- JSON decimals beyond the binary32 range. An interval endpoint rounds outward: `-1e39` to `-∞`
  -- as a lower endpoint and to `-M` as an upper one, `1e39` to `M` as a lower endpoint and to `+∞`
  -- as an upper one. `M` is the largest finite binary32 value. An affine coefficient rounds
  -- to nearest, so `±1e39` is `±∞`: refused in `loA` and `hiA` with either sign, and read in an
  -- offset only as the infinity bounding nothing on its side, `-∞` in `loC` and `+∞` in `hiC`.
  let some huge ← parseFlatBox? 3
      (← parseValue! "{\"lo\":[-1e39,-1e39,1e39],\"hi\":[1e39,-1e39,1e39]}")
    | throw <| IO.userError "an interval with decimals beyond binary32 did not parse"
  expect "a decimal beyond binary32 did not round outward to the expected endpoints"
    ((entries huge.lo).toList.map (·.toNatBits) == [0xFF800000, 0xFF800000, 0x7F7FFFFF] &&
      (entries huge.hi).toList.map (·.toNatBits) == [0x7F800000, 0xFF7FFFFF, 0x7F800000])
  for (field, huge) in [("loA", "1e39"), ("loA", "-1e39"), ("hiA", "1e39"), ("hiA", "-1e39"),
      ("loC", "1e39"), ("hiC", "-1e39")] do
    expectRejected s!"an affine coefficient {huge} in {field} was accepted" <|
      parseAffineBounds? 2 2 (← parseValue! (affineText field huge))
  for (field, huge, bits) in [("loC", "-1e39", ([0, 0xFF800000], [0, 0])),
      ("hiC", "1e39", ([0, 0], [0, 0x7F800000]))] do
    let some B ← parseAffineBounds? 2 2 (← parseValue! (affineText field huge))
      | throw <| IO.userError s!"affine bounds with {huge} in {field} parsed as absent"
    expect s!"the offset {huge} in {field} did not read as the infinity bounding nothing"
      (offsetBits B == bits)
  for field in ["loA", "loC", "hiA", "hiC"] do
    for finite in ["3e38", "-3e38"] do
      let some _ ← parseAffineBounds? 2 2 (← parseValue! (affineText field finite))
        | throw <| IO.userError s!"affine bounds with {finite} in {field} parsed as absent"

/-- The numbers of the certificate JSON, in each of the six rounding modes of
`Model.IEEERoundingMode`: every finite binary32 class written by the writers reads back to itself,
`-0` included, and four spellings of `-0` read as `-0`. A box with infinite endpoints reads back to
itself. The decimals `±(t - 1)`, `±t` and `±(t + 1)`, with `t = 2 ^ 128 - 2 ^ 103` the least
magnitude whose nearest binary32 value is infinite, are written as their digits and read back to
themselves. To nearest, `±(t - 1)` is the largest finite binary32 value of its sign, an affine
coefficient that the coefficient writer writes back to it, while `±t` and `±(t + 1)` are the
infinity of their sign. That infinity has no coefficient decimal and is refused as a matrix
coefficient in `loA`. The decimals `±1e39`, `±1e40`, `±1e-50` and `±1e-51` beyond the binary32
range, and `±1e999999999` and `±1e-999999999`, whose exponents FloatLib's conversion clamps before
it forms a power of ten, have their stated bits in each of the six modes. -/
def signedDecimals : IO Unit := do
  let modes : List Model.IEEERoundingMode :=
    [.nearest, .nearestAway, .down, .up, .towardZero, .away]
  let round32 (mode : Model.IEEERoundingMode) (d : Decimal) : Binary 8 23 :=
    Binary.ofModel (Model.convertDecimalText .binary32 mode d).value
  for bits in finiteClasses do
    let some v := coefficientValue? (Binary.ofNatBits bits : Binary 8 23)
      | throw <| IO.userError s!"the finite binary32 value {bits} has no decimal"
    let .num d ← parseValue! v.render
      | throw <| IO.userError s!"the decimal of the binary32 value {bits} did not read as a number"
    for mode in modes do
      expect s!"the decimal {v.render} of {bits} did not read back to {bits}"
        ((round32 mode d).toNatBits == bits)
  expect "the writer did not write the negative zero as -0.0"
    ((coefficientValue? (Binary.ofNatBits 0x80000000 : Binary 8 23)).map (·.render) ==
      some "-0.0")
  for bits in [0x7F800000, 0xFF800000, 0x7FC00000] do
    expect s!"the non-finite binary32 value {bits} was written as a coefficient"
      (coefficientValue? (Binary.ofNatBits bits : Binary 8 23)).isNone
  for tok in ["-0", "-0.0", "-0e3", "-0.000e-7"] do
    let .num d ← parseValue! tok
      | throw <| IO.userError s!"{tok} did not read as a number"
    for mode in modes do
      expect s!"{tok} did not read as the negative zero"
        ((round32 mode d).toNatBits == 0x80000000)
  let bitsBox (lo hi : List Nat) : FlatBox (Binary 8 23) :=
    { dim := lo.length
      lo := Tensor.ofFn fun i => Binary.ofNatBits lo[i.val]!
      hi := Tensor.ofFn fun i => Binary.ofNatBits hi[i.val]! }
  let (lo, hi) := ([0xFF800000, 0x80000000, 1, 0x7F7FFFFF], [0x7F800000, 0, 0x007FFFFF, 0x7F800000])
  let some B ← parseFlatBox? 4 (← parseValue! (flatBoxValue (bitsBox lo hi)).render)
    | throw <| IO.userError "a written box parsed as absent"
  expect "a box with infinite endpoints and signed zeros did not read back to itself"
    ((entries B.lo).toList.map (·.toNatBits) == lo && (entries B.hi).toList.map (·.toNatBits) == hi)
  let t : Nat := 2 ^ 128 - 2 ^ 103
  let affine (x : String) : String :=
    "{\"loA\":[[" ++ x ++ "]],\"loC\":[-0.0],\"hiA\":[[0]],\"hiC\":[0]}"
  for negative in [false, true] do
    let sign : Nat := if negative then 0x80000000 else 0
    for (x, bits, inRange) in [(t - 1, sign + 0x7F7FFFFF, true), (t, sign + 0x7F800000, false),
        (t + 1, sign + 0x7F800000, false)] do
      let d : Decimal := { negative, significand := x, exponent := 0 }
      let text := (Value.num d).render
      expect s!"the decimal {repr d} was not written as its digits"
        (text == (if negative then "-" else "") ++ toString x)
      let .num e ← parseValue! text
        | throw <| IO.userError s!"{text} did not read as a number"
      expect s!"{text} did not read back to its decimal" (e == d)
      let x32 := round32 .nearest e
      expect s!"{text} did not read to nearest as {bits}" (x32.toNatBits == bits)
      let v ← parseValue! (affine text)
      if inRange then
        let some w := coefficientValue? x32
          | throw <| IO.userError s!"the binary32 value {bits} of {text} has no decimal"
        let .num back ← parseValue! w.render
          | throw <| IO.userError s!"the decimal of the binary32 value {bits} did not read"
        expect s!"the decimal {w.render} of {bits} did not read back to {bits}"
          ((round32 .nearest back).toNatBits == bits)
        let some A ← parseAffineBounds? 1 1 v
          | throw <| IO.userError s!"affine bounds with the coefficient {text} parsed as absent"
        let aBits := (List.finRange A.outDim).flatMap fun i =>
          (List.finRange A.inDim).map fun j => (Tensor.get2 A.loAff.weights i j).toNatBits
        expect s!"the affine coefficient {text} did not read as {bits}"
          (aBits == [bits] && (entries A.loAff.bias).toList.map (·.toNatBits) == [0x80000000])
      else
        expect s!"the infinity {bits} of {text} was written as a coefficient"
          (coefficientValue? x32).isNone
        expectRejected s!"the affine coefficient {text} was accepted" (parseAffineBounds? 1 1 v)
  -- The tokens of a row lie beyond the same end of the binary32 range, so they round alike.
  -- Bits per direction in the order of `modes`.
  let (pinf, pmax, ninf, nmax) : Nat × Nat × Nat × Nat := (0x7F800000, 0x7F7FFFFF, 0xFF800000,
    0xFF7FFFFF)
  for (toks, bits) in [(["1e39", "1e40", "1e999999999"], [pinf, pinf, pmax, pinf, pmax, pinf]),
      (["-1e39", "-1e40", "-1e999999999"], [ninf, ninf, ninf, nmax, nmax, ninf]),
      (["1e-50", "1e-51", "1e-999999999"], [0, 0, 0, 1, 0, 1]),
      (["-1e-50", "-1e-51", "-1e-999999999"], [0x80000000, 0x80000000, 0x80000001, 0x80000000,
        0x80000000, 0x80000001])] do
    for tok in toks do
      let .num d ← parseValue! tok
        | throw <| IO.userError s!"{tok} did not read as a number"
      expect s!"{tok} did not round to {bits} in the directions {repr modes}"
        (modes.map (fun mode => (round32 mode d).toNatBits) == bits)

/-- Offsets at the infinity that bounds nothing on their side. The writers emit `-∞` in `loC` and
`+∞` in `hiC` as `-1e39` and `1e39`, which read back to them. They write no affine bounds with a
matrix entry that is not finite, a NaN offset or an offset at the other infinity. The readers refuse
the token `NaN`, the strings `"NaN"`, `"-inf"` and `"+inf"` and `null` in every field of an affine
entry and in a box, and read `±4e38` like `±1e39`. Each finite binary32 class, written as an
offset in `loC` and in `hiC`, reads back to itself. -/
def vacuousOffsets : IO Unit := do
  let (ninf, pinf, nan, one) : Nat × Nat × Nat × Nat := (0xFF800000, 0x7F800000, 0x7FC00000,
    0x3F800000)
  let vec (xs : List Nat) : Tensor (Binary 8 23) [2] :=
    Tensor.ofFn fun i => Binary.ofNatBits xs[i.val]!
  let bounds (loA hiA : Nat) (loC hiC : List Nat) : FlatAffineBounds (Binary 8 23) :=
    { inDim := 1, outDim := 2
      loAff := { weights := Tensor.full [2, 1] (Binary.ofNatBits loA), bias := vec loC }
      hiAff := { weights := Tensor.full [2, 1] (Binary.ofNatBits hiA), bias := vec hiC } }
  for (upper, bits, token) in [(false, ninf, "-1e39"), (true, pinf, "1e39")] do
    expect s!"the offset {bits} was not written as {token}"
      ((offsetValue? upper (Binary.ofNatBits bits : Binary 8 23)).map (·.render) == some token)
  let (loC, hiC) := ([ninf, 0x80000000], [pinf, 1])
  let some v := affineBoundsValue? (bounds one one loC hiC)
    | throw <| IO.userError "affine bounds with offsets -∞ in loC and +∞ in hiC were not written"
  let some B ← parseAffineBounds? 1 2 (← parseValue! v.render)
    | throw <| IO.userError "written affine bounds parsed as absent"
  expect s!"the written offsets {v.render} did not read back to themselves"
    (offsetBits B == (loC, hiC))
  for bits in finiteClasses do
    let some v := affineBoundsValue? (bounds one one [bits, 0] [0, bits])
      | throw <| IO.userError s!"affine bounds with the offset {bits} were not written"
    let some B ← parseAffineBounds? 1 2 (← parseValue! v.render)
      | throw <| IO.userError s!"affine bounds with the offset {bits} parsed as absent"
    expect s!"the offset {bits} in loC and hiC did not read back to itself"
      (offsetBits B == ([bits, 0], [0, bits]))
  for (loA, hiA, loC, hiC, what) in [(one, one, [pinf, 0], [0, 0], "+∞ in loC"),
      (one, one, [0, 0], [0, ninf], "-∞ in hiC"), (one, one, [nan, 0], [0, 0], "NaN in loC"),
      (one, one, [0, 0], [nan, 0], "NaN in hiC"), (pinf, one, [0, 0], [0, 0], "+∞ in loA"),
      (ninf, one, [0, 0], [0, 0], "-∞ in loA"), (one, pinf, [0, 0], [0, 0], "+∞ in hiA"),
      (one, ninf, [0, 0], [0, 0], "-∞ in hiA"), (nan, one, [0, 0], [0, 0], "NaN in loA"),
      (one, nan, [0, 0], [0, 0], "NaN in hiA")] do
    expect s!"affine bounds with {what} were written"
      (affineBoundsValue? (bounds loA hiA loC hiC)).isNone
  for field in ["loA", "loC", "hiA", "hiC"] do
    expectRejected s!"the token NaN in {field} was read" (parseValue! (affineText field "NaN"))
    for tok in ["\"NaN\"", "\"-inf\"", "\"+inf\"", "null"] do
      expectRejected s!"{tok} in {field} was accepted" <|
        do parseAffineBounds? 2 2 (← parseValue! (affineText field tok))
  for tok in ["\"NaN\"", "\"-inf\"", "\"+inf\"", "null"] do
    expectRejected s!"{tok} as a box endpoint was accepted" <|
      do parseFlatBox? 1 (← parseValue! ("{\"lo\":[" ++ tok ++ "],\"hi\":[1]}"))
  expectRejected "the token NaN as a box endpoint was read"
    (parseValue! "{\"lo\":[NaN],\"hi\":[1]}")
  for (field, huge) in [("loA", "4e38"), ("hiA", "-4e38"), ("loC", "4e38"), ("hiC", "-4e38")] do
    expectRejected s!"an affine coefficient {huge} in {field} was accepted" <|
      parseAffineBounds? 2 2 (← parseValue! (affineText field huge))
  for (field, huge, bits) in [("loC", "-4e38", ([0, ninf], [0, 0])),
      ("hiC", "4e38", ([0, 0], [0, pinf]))] do
    let some B ← parseAffineBounds? 2 2 (← parseValue! (affineText field huge))
      | throw <| IO.userError s!"affine bounds with {huge} in {field} parsed as absent"
    expect s!"the offset {huge} in {field} did not read as the infinity bounding nothing"
      (offsetBits B == bits)

/-- Check the input-region, PINN, α-CROWN slope, box, ODE, PDE and ReLU phase conditions of the
module docstring, then run `boundary`, `signedDecimals` and `vacuousOffsets`. -/
def run : IO Unit := do
  let centerRegion ← parseInputRegion!
    "{\"input\":{\"center\":[1.0,2.0,3.0],\"eps\":0.25}}"
  expect "center/epsilon input region did not infer its dimension"
    (centerRegion.dim == 3 && centerRegion.lo == #[0.75, 1.75, 2.75] &&
      centerRegion.hi == #[1.25, 2.25, 3.25])
  let boxRegion ← parseInputRegion!
    "{\"region\":{\"lo\":[-1.0,0.0],\"hi\":[1.0,2.0]}}"
  expect "endpoint input region did not infer its dimension"
    (boxRegion.dim == 2 && boxRegion.lo == #[-1.0, 0.0] && boxRegion.hi == #[1.0, 2.0])
  expectRejected "malformed input dimension was treated as absent" <|
    parseInputRegion! "{\"input\":{\"dim\":\"two\",\"lo\":[0.0,0.0],\"hi\":[1.0,1.0]}}"
  expectRejected "declared input dimension did not constrain endpoint lengths" <|
    parseInputRegion! "{\"input\":{\"dim\":3,\"lo\":[0.0,0.0],\"hi\":[1.0,1.0]}}"
  expectRejected "mismatched input endpoint lengths were accepted" <|
    parseInputRegion! "{\"input\":{\"lo\":[0.0],\"hi\":[1.0,2.0]}}"
  expectRejected "negative input radius was accepted" <|
    parseInputRegion! "{\"input\":{\"center\":[0.0,0.0],\"eps\":-0.5}}"
  expectRejected "reversed input interval was accepted" <|
    parseInputRegion! "{\"input\":{\"lo\":[1.0],\"hi\":[0.0]}}"
  expectRejected "mixed input-region schemas were accepted" <|
    parseInputRegion!
      "{\"input\":{\"lo\":[0.0],\"hi\":[1.0],\"center\":[0.5],\"eps\":0.5}}"
  expectRejected "endpoint input region accepted an unrelated radius" <|
    parseInputRegion! "{\"input\":{\"lo\":[0.0],\"hi\":[1.0],\"eps\":0.5}}"

  let validPinnSource :=
    "{\"pinn\":{\"pde\":\"u\",\"h\":0.1,\"eps\":0.0,\"points\":[0.0]}," ++
    "\"residual_bounds\":{\"lo\":[0.0],\"hi\":[0.0]}," ++
    "\"residual_bounds_deriv\":{\"lo\":[0.0],\"hi\":[0.0]}," ++
    "\"u_bounds\":[{\"x\":0.0," ++
    "\"u_minus\":{\"lo\":0.0,\"hi\":0.0}," ++
    "\"u\":{\"lo\":0.0,\"hi\":0.0}," ++
    "\"u_plus\":{\"lo\":0.0,\"hi\":0.0}}]}"
  let validPinnCert ← parseJson! validPinnSource
  expect "a well-formed PINN certificate was rejected"
    (Examples.PINN.parseCertificate validPinnCert).isOk
  -- Each invalid fixture changes one condition of the accepted certificate.
  for spacing in ["-0.1", "0.0"] do
    let invalidSpacing ← parseJson! <|
      validPinnSource.replace "\"h\":0.1" s!"\"h\":{spacing}"
    expect s!"a PINN certificate with spacing {spacing} was accepted"
      (!(Examples.PINN.parseCertificate invalidSpacing).isOk)
  let negativeRadius ← parseJson! <|
    validPinnSource.replace "\"eps\":0.0" "\"eps\":-0.1"
  expect "a PINN certificate with a negative perturbation radius was accepted"
    (!(Examples.PINN.parseCertificate negativeRadius).isOk)
  let missingUBounds ← parseJson! <|
    "{\"pinn\":{\"pde\":\"u\",\"h\":0.1,\"eps\":0.0,\"points\":[0.0]}," ++
    "\"residual_bounds\":{\"lo\":[0.0],\"hi\":[0.0]}," ++
    "\"residual_bounds_deriv\":{\"lo\":[0.0],\"hi\":[0.0]},\"u_bounds\":[]}"
  expect "a PINN certificate with incomplete u_bounds was accepted"
    (!(Examples.PINN.parseCertificate missingUBounds).isOk)
  let reversedPinnInterval ← parseJson! <|
    validPinnSource.replace "\"residual_bounds\":{\"lo\":[0.0],\"hi\":[0.0]}"
      "\"residual_bounds\":{\"lo\":[1.0],\"hi\":[0.0]}"
  expect "a PINN certificate with a reversed residual interval was accepted"
    (!(Examples.PINN.parseCertificate reversedPinnInterval).isOk)

  IO.println "certificate_preconditions: begin"

  let goodAlpha ← parseValue! "[0.0, 0.75]"
  let some _ ← parseAlphaVec? 2 goodAlpha
    | throw <| IO.userError "valid alpha vector was rejected"

  let badAlpha ← parseValue! "[2.0, 0.5]"
  expectRejected "alpha outside [0,1] was accepted" (parseAlphaVec? 2 badAlpha)

  let hugeAlpha ← parseValue! "[1e999, 0.5]"
  expectRejected "alpha 1e999 was accepted" (parseAlphaVec? 2 hugeAlpha)

  let reversedBox ← parseValue! "{\"lo\":[0.0,1e999],\"hi\":[1.0,2.0]}"
  expectRejected "a box with a lower endpoint above its upper one was accepted"
    (parseFlatBox? 2 reversedBox)

  expectFloatPair "ODE parser interpreted u^0 incorrectly" (evalODEAtTwo "u^0") (1.0, 1.0)
  expectFloatPair "ODE parser interpreted u^1 incorrectly" (evalODEAtTwo "u^1") (2.0, 2.0)
  expectFloatBits "ODE parser interpreted u^2 incorrectly" (evalODEAtTwo "u^2")
    0x400FFFFFFFFFFFFF 0x4010000000000001
  expectFloatBits "ODE exponentiation did not bind more tightly than unary minus"
    (evalODEAtTwo "-u^2") 0xC010000000000001 0xC00FFFFFFFFFFFFF
  expectFloatBits "ODE parser interpreted (-u)^2 incorrectly"
    (evalODEAtTwo "(-u)^2") 0x400FFFFFFFFFFFFF 0x4010000000000001
  expectFloatBits "ODE parser interpreted -(u^2) incorrectly"
    (evalODEAtTwo "-(u^2)") 0xC010000000000001 0xC00FFFFFFFFFFFFF
  expectFloatPair "ODE parser interpreted double negation incorrectly"
    (evalODEAtTwo "--u") (2.0, 2.0)

  expectFloatPair "PDE parser interpreted u^0 incorrectly" (evalPDEAtTwo "u^0") (1.0, 1.0)
  expectFloatPair "PDE parser interpreted u^1 incorrectly" (evalPDEAtTwo "u^1") (2.0, 2.0)
  expectFloatBits "PDE parser interpreted u^2 incorrectly" (evalPDEAtTwo "u^2")
    0x400FFFFFFFFFFFFF 0x4010000000000001
  expectFloatBits "PDE exponentiation did not bind more tightly than unary minus"
    (evalPDEAtTwo "-u^2") 0xC010000000000001 0xC00FFFFFFFFFFFFF
  expectFloatBits "PDE parser interpreted (-u)^2 incorrectly"
    (evalPDEAtTwo "(-u)^2") 0x400FFFFFFFFFFFFF 0x4010000000000001
  expectFloatBits "PDE parser interpreted -(u^2) incorrectly"
    (evalPDEAtTwo "-(u^2)") 0xC010000000000001 0xC00FFFFFFFFFFFFF
  expectFloatPair "PDE parser interpreted double negation incorrectly"
    (evalPDEAtTwo "--u") (2.0, 2.0)

  let reluGraph ← opGraph (Op.relu (s := [1]))
  let header := "{\"ctx\":{\"inputId\":0,\"inputDim\":1},\"ibp\":[null,null]," ++
    "\"crown\":[null,null]"
  let parsePhased (rest : String) : IO CROWNNode := do
    let json ← parseValue! (header ++ rest)
    parse (reluGraph.nodes.map outputSize) json
  let parsed ← parsePhased ",\"phases\":[null,[1]]}"
  expect "valid ReLU phases were not parsed" (parsed.phases == #[none, some #[1]])
  let omitted ← parsePhased "}"
  expect "omitted ReLU phases did not parse as absent"
    (omitted.phases.size == 2 && omitted.phases.all (·.isNone))
  expectRejected "a ReLU phase outside -1, 0, 1 was accepted"
    (parsePhased ",\"phases\":[null,[2]]}")
  expectRejected "ReLU phases of the wrong length were accepted"
    (parsePhased ",\"phases\":[null,[1,1]]}")

  boundary
  signedDecimals
  vacuousOffsets
  CertificateVerdict.run

  IO.println "certificate_preconditions: ok"

end CertificatePreconditions
end TorchLean.Test
