/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.AbCrownLeafCert
public import NN.Verification.Cert.NodeReplay
import NN.Verification.Splines.PiecewisePolyCert

/-!
# Certificate parsing regressions

Outward rounding of `{center, eps}` input regions and of binary32 IBP endpoints, rejection of
negative radii, the leaf-cover check used by `abcrown-leaf`, and direct spline certificate
validation.
-/

public section

namespace NN.Tests.Verification.CertificateParsing

open Lean
open FloatLib.Floats (ExecFloat)
open NN.Verification.Json
open NN.Verification.Cert.AbCrownLeafCert (leavesCoverRoot)

private def check (name : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"certificate parsing: {name}"

private def parse (text : String) : Except String BoxRegion := do
  parseBoxRegion "region" (← Json.parse text)

private def boxRegions : IO Unit := do
  match parse r#"{"center": [1.0], "eps": -1e-20}"# with
  | .ok _ => throw <| IO.userError "certificate parsing: negative eps accepted"
  | .error _ => pure ()
  match parse r#"{"center": [1.0], "eps": "NaN"}"# with
  | .ok _ => throw <| IO.userError "certificate parsing: NaN eps accepted"
  | .error _ => pure ()
  -- 1 ± 1e-17 rounds back to 1 under round-to-nearest; outward rounding must still move.
  let tiny ← IO.ofExcept <| parse r#"{"center": [1.0], "eps": 1e-17}"#
  check "tiny radius widens the lower endpoint" (tiny.lo[0]! < 1.0)
  check "tiny radius widens the upper endpoint" (1.0 < tiny.hi[0]!)
  -- Exact sums stay exact.
  let exact ← IO.ofExcept <| parse r#"{"center": [1, 2, 3], "eps": 0.25}"#
  check "exact lower endpoints" (exact.lo == #[0.75, 1.75, 2.75])
  check "exact upper endpoints" (exact.hi == #[1.25, 2.25, 3.25])
  let zero ← IO.ofExcept <| parse r#"{"center": [0.5], "eps": 0}"#
  check "zero radius gives a point box" (zero.lo == #[0.5] && zero.hi == #[0.5])

/-- The exact value of a finite binary32 endpoint; nonfinite endpoints fail the test. -/
private def toRat (x : ExecFloat.Binary 8 23) : IO Rat := do
  let some q := ExecFloat.Binary.toRat? x
    | throw <| IO.userError "certificate parsing: nonfinite binary32 endpoint"
  pure q

/-- Exact values of the `k`-th lower and upper endpoints. -/
private def endpoints (box : NN.MLTheory.CROWN.FlatBox (ExecFloat.Binary 8 23)) (k : Nat) :
    IO (Rat × Rat) := do
  if h : k < box.dim then
    pure (← toRat (box.lo.getScalar ⟨k, h⟩), ← toRat (box.hi.getScalar ⟨k, h⟩))
  else
    throw <| IO.userError s!"certificate parsing: IBP box has no coordinate {k}"

private def ibpEndpoints : IO Unit := do
  let j ← IO.ofExcept <| Json.parse r#"{"lo": [0.1, -0.1, 1], "hi": [0.1, -0.1, 1]}"#
  let some box ← NN.Verification.Cert.NodeReplay.parseFlatBox? 3 j
    | throw <| IO.userError "certificate parsing: IBP box did not parse"
  for (k, decimal) in [(0, (1 : Rat) / 10), (1, -1 / 10), (2, 1)] do
    let (lo, hi) ← endpoints box k
    check s!"binary32 lower endpoint {k} is at or below the decimal" (lo ≤ decimal)
    check s!"binary32 upper endpoint {k} is at or above the decimal" (decimal ≤ hi)
  let (lo0, hi0) ← endpoints box 0
  check "inexact decimal gives a nondegenerate box" (lo0 < hi0)
  let (lo2, hi2) ← endpoints box 2
  check "exact decimal gives a point" (lo2 == hi2)

private def cover (leaves : Array (Array Float × Array Float)) : IO Bool :=
  IO.ofExcept <| leavesCoverRoot #[-1, -1] #[1, 1] leaves

private def leafCover : IO Unit := do
  check "root itself covers" (← cover #[(#[-1, -1], #[1, 1])])
  check "a nested leaf does not cover" !(← cover #[(#[-0.5, -0.5], #[0.5, 0.5])])
  check "two halves cover" (← cover #[(#[-1, -1], #[0, 1]), (#[0, -1], #[1, 1])])
  check "four quadrants cover" (← cover #[
    (#[-1, -1], #[0, 0]), (#[0, -1], #[1, 0]), (#[-1, 0], #[0, 1]), (#[0, 0], #[1, 1])])
  check "a missing quadrant is found" !(← cover #[
    (#[-1, -1], #[0, 0]), (#[0, -1], #[1, 0]), (#[-1, 0], #[0, 1])])
  check "a gap between halves is found" !(← cover #[(#[-1, -1], #[0, 1]), (#[0.25, -1], #[1, 1])])
  check "overlapping leaves cover" (← cover #[(#[-1, -1], #[0.5, 1]), (#[-0.5, -1], #[1, 1])])
  check "a degenerate leaf covers nothing" !(← cover #[(#[-1, -1], #[-1, 1])])
  check "differently split halves cover" (← cover #[
    (#[-1, -1], #[0, 0]), (#[-1, 0], #[0, 1]), (#[0, -1], #[1, 1])])
  match leavesCoverRoot #[0, 0] #[1, 1]
      ((List.range 2000).toArray.map fun k =>
        let t := k.toFloat / 2000
        (#[t, t], #[t + 0.0005, t + 0.0005])) (maxCells := 1000) with
  | .ok _ => throw <| IO.userError "certificate parsing: oversized cover grid was not refused"
  | .error _ => pure ()

private def expectFailure (name fragment : String) (action : IO Unit) : IO Unit := do
  let error ← try
    action
    pure none
  catch e =>
    pure (some e.toString)
  match error with
  | none => throw <| IO.userError s!"certificate parsing: {name} was accepted"
  | some message => check s!"{name}: unexpected error: {message}" (message.contains fragment)

open NN.Verification.Splines.PiecewisePolyCert
open TorchLean

private def splineCertificates : IO Unit := do
  let piece : PolynomialPiece := { lo := 0, hi := 1, coeffs := #[0, 1] }
  let linear : Certificate :=
    { degree := 1, n := 2, xs := Tensor.from (#[0, 1] : Array Rat),
      ys := Tensor.from (#[0, 1] : Array Rat), pieces := #[piece] }
  let constant : Certificate :=
    { linear with
      degree := 0
      ys := Tensor.from (#[2, 2] : Array Rat)
      pieces := #[{ piece with coeffs := #[2] }] }
  let multi : Certificate :=
    { degree := 1, n := 3, xs := Tensor.from (#[0, 1, 2] : Array Rat),
      ys := Tensor.from (#[0, 1, 2] : Array Rat),
      pieces := #[piece, { lo := 1, hi := 2, coeffs := #[1, 1] }] }
  let empty : Certificate :=
    { degree := 0, n := 0, xs := Tensor.from (#[] : Array Rat),
      ys := Tensor.from (#[] : Array Rat), pieces := #[] }
  let singleton : Certificate :=
    { degree := 0, n := 1, xs := Tensor.from (#[0] : Array Rat),
      ys := Tensor.from (#[0] : Array Rat), pieces := #[] }
  let invalid : Array (String × String × Certificate) := #[
    ("empty knots", "length ≥ 2", empty),
    ("one knot", "length ≥ 2", singleton),
    ("missing piece", "pieces length mismatch", { linear with pieces := #[] }),
    ("extra piece", "pieces length mismatch", { linear with pieces := #[piece, piece] }),
    ("equal knots", "not strictly increasing",
      { constant with xs := Tensor.from (#[0, 0] : Array Rat) }),
    ("descending knots", "not strictly increasing",
      { constant with xs := Tensor.from (#[1, 0] : Array Rat) }),
    ("lo metadata", ".lo mismatch", { linear with pieces := #[{ piece with lo := 1/3 }] }),
    ("hi metadata", ".hi mismatch", { linear with pieces := #[{ piece with hi := 2 }] }),
    ("degree mismatch", "coeffs length mismatch", { linear with degree := 0 }),
    ("missing coefficient", "coeffs length mismatch",
      { linear with pieces := #[{ piece with coeffs := #[0] }] }),
    ("empty coefficients", "coeffs length mismatch",
      { linear with pieces := #[{ piece with coeffs := #[] }] }),
    ("endpoint mismatch", "endpoint mismatch",
      { linear with ys := Tensor.from (#[0, 2] : Array Rat) })]
  for (backend, checker) in
      [("Rat", checkCertificateRat), ("IEEE32", checkCertificateIEEE32ExecExact)] do
    for valid in #[linear, constant, multi,
        { linear with degree := 2, pieces := #[{ piece with coeffs := #[0, 1, 0] }] }] do
      checker valid
    for (name, fragment, cert) in invalid do
      expectFailure s!"{backend}: {name}" fragment (checker cert)

  -- Direct IEEE checking retains its own endpoint semantics.
  let rounded : Certificate :=
    { linear with
      ys := Tensor.from (#[16777216, 16777216] : Array Rat)
      pieces := #[{ piece with coeffs := #[16777216, 1] }] }
  checkCertificateIEEE32ExecExact rounded
  expectFailure "exact endpoint disagreement" "endpoint mismatch" (checkCertificateRat rounded)
  let nondyadic : Certificate :=
    { constant with
      ys := Tensor.from (#[1/3, 1/3] : Array Rat)
      pieces := #[{ piece with coeffs := #[1/3] }] }
  checkCertificateRat nondyadic
  expectFailure "inexact binary32 data" "not exactly representable"
    (checkCertificateIEEE32ExecExact nondyadic)

  let validJson ← IO.ofExcept <| Json.parse
    (r#"{"format":"piecewise_poly_v0","degree":1,"xs":["0","1"],"ys":["0","1"],"# ++
     r#""pieces":[{"lo":"0","hi":"1","coeffs":["0","1"]}]}"#)
  checkJson validJson
  checkJsonIEEE32ExecExact validJson
  let missingJson ← IO.ofExcept <| Json.parse
    r#"{"format":"piecewise_poly_v0","degree":1,"xs":["0","1"],"ys":["0","1"],"pieces":[]}"#
  expectFailure "JSON missing piece" "pieces length mismatch"
    (discard <| NN.Verification.Splines.PiecewisePolyCert.parse missingJson)

def run : IO Unit := do
  boxRegions
  ibpEndpoints
  leafCover
  splineCertificates
  IO.println "  Certificate parsing: rounding, radius, leaf-cover, and spline tests passed"

end NN.Tests.Verification.CertificateParsing
