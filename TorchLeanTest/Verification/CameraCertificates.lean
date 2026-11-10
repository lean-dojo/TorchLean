/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.Geometry3D.Box3D
public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init

/-!
# Parsing camera certificates

`run` parses camera certificates in the `torchlean.camera.box3d.v2` JSON format with 1, 3, 8 and
17 points and checks that each parsed certificate keeps the declared point count. It also checks
that `parseJsonCert` refuses an artifact with no points, a missing `point_count`, an unknown
format, an incomplete point triple, a count that disagrees with the points, a negative count and a
non-finite coordinate.
-/

public section

namespace TorchLean.Test.CameraCertificates

open Examples.Geometry3D

private def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"camera certificate: {label}"

private def camera : CameraP Float :=
  [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]

private def certificateJson (points : Lean.Json) (pointCount? : Option Lean.Json := none)
    (format : String := formatString) : Lean.Json :=
  Lean.Json.mkObj <|
    [("format", Lean.toJson format),
     ("image_width", Lean.toJson (64 : Nat)),
     ("image_height", Lean.toJson (64 : Nat)),
     ("tol", Lean.toJson (0 : Nat)),
     ("camera_P", Lean.toJson (Tensor.to camera (Array Float))),
     ("corners3d", points),
     ("bbox2d", Lean.toJson (#[0, 0, 64, 64] : Array Nat))] ++
    match pointCount? with
    | none => []
    | some count => [("point_count", count)]

private def expectRejected (label : String) (payload : Lean.Json) : IO Unit := do
  let accepted ← try
    let _ ← parseJsonCert payload
    pure true
  catch _ => pure false
  expect label (!accepted)

private def diagonalPoints (count : Nat) : Array Float :=
  (List.range count).toArray.flatMap fun index =>
    #[Float.ofNat (index + 1), Float.ofNat (index + 1), 1.0]

/-- Parse `points` under `format` and check that the certificate keeps `count` points. -/
private def checkCount (count : Nat) (points : Array Float) (format : String)
    (declared : Option Lean.Json) : IO Unit := do
  let cert ← parseJsonCert (certificateJson (Lean.toJson points) declared format)
  expect s!"count {count} preserved" (cert.pointCount == count)

/-- Check parsed point counts and the refusal of malformed certificates. -/
def run : IO Unit := do
  for count in [1, 3, 8, 17] do
    checkCount count (diagonalPoints count) formatString (some (Lean.toJson count))
  expectRejected "zero points without a declared count"
    (certificateJson (Lean.toJson (#[] : Array Float)))
  expectRejected "zero points with a declared count"
    (certificateJson (Lean.toJson (#[] : Array Float)) (some (Lean.toJson (0 : Nat))))
  expectRejected "eight points without a declared count"
    (certificateJson (Lean.toJson (diagonalPoints 8)))
  expectRejected "unknown format"
    (certificateJson (Lean.toJson (diagonalPoints 8)) (some (Lean.toJson (8 : Nat)))
      "torchlean.camera.box3d.v1")
  expectRejected "incomplete triple"
    (certificateJson (Lean.toJson (#[1, 1, 1, 1] : Array Nat)) (some (Lean.toJson (1 : Nat))))
  expectRejected "declared count mismatch"
    (certificateJson (Lean.toJson (#[1, 1, 1] : Array Nat)) (some (Lean.toJson (2 : Nat))))
  expectRejected "negative count"
    (certificateJson (Lean.toJson (#[1, 1, 1] : Array Nat)) (some (Lean.toJson (-1 : Int))))
  expectRejected "non-finite point"
    (certificateJson
      (.arr #[.num (Lean.JsonNumber.fromNat (10 ^ 400)), Lean.toJson (1 : Nat),
        Lean.toJson (1 : Nat)])
      (some (Lean.toJson (1 : Nat))))
  IO.println "  camera certificate point counts: passed"

end TorchLean.Test.CameraCertificates
