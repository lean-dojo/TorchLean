/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.AbCrownLeafCert
public import NN.Verification.Cert.NodeReplay

/-!
# Certificate parsing regressions

Outward rounding of `{center, eps}` input regions and of binary32 IBP endpoints, rejection of
negative radii, and the leaf-cover check used by `abcrown-leaf`.
-/

@[expose] public section

namespace NN.Tests.Verification.CertificateParsing

open Lean
open FloatLib.Floats (ExecFloat)
open NN.Verification.Json
open NN.Verification.Cert.AbCrownLeafCert (leavesCoverRoot)

def check (name : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"certificate parsing: {name}"

def parse (text : String) : Except String BoxRegion := do
  parseBoxRegion "region" (← Json.parse text)

def boxRegions : IO Unit := do
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

def rat (x : ExecFloat.Binary 8 23) : IO Rat := do
  let some q := ExecFloat.Binary.toRat? x
    | throw <| IO.userError "certificate parsing: nonfinite binary32 endpoint"
  pure q

/-- Exact values of the `k`-th lower and upper endpoints. -/
def endpoints (box : NN.MLTheory.CROWN.FlatBox (ExecFloat.Binary 8 23)) (k : Nat) :
    IO (Rat × Rat) := do
  if h : k < box.dim then
    pure (← rat (box.lo.getScalar ⟨k, h⟩), ← rat (box.hi.getScalar ⟨k, h⟩))
  else
    throw <| IO.userError s!"certificate parsing: IBP box has no coordinate {k}"

def ibpEndpoints : IO Unit := do
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

def cover (leaves : Array (Array Float × Array Float)) : IO Bool :=
  IO.ofExcept <| leavesCoverRoot #[-1, -1] #[1, 1] leaves

def leafCover : IO Unit := do
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

def run : IO Unit := do
  boxRegions
  ibpEndpoints
  leafCover
  IO.println "  Certificate parsing: outward rounding, radius, and leaf-cover tests passed"

end NN.Tests.Verification.CertificateParsing
