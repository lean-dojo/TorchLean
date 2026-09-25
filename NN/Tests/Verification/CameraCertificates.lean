/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Geometry3D.Box3D
public import NN.Tensor

/-!
# Camera certificate point counts

Exercise the dependent point-count boundary, the complete point traversal, the two format
versions, and malformed JSON, including an artifact with no points.
The numerical camera is the identity pinhole projection, so expected depths and pixels are exact.
-/

@[expose] public section

namespace NN.Tests.Verification.CameraCertificates

open TorchLean NN.Verification.Geometry3D.Box3D

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"camera certificate: {label}"

def camera : CameraP Float :=
  [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]

/-- Omitting the point count retains the ordinary eight-corner construction. -/
def cuboid : BoxCameraCert Float :=
  { width := 64
    height := 64
    tol := 0
    camera
    corners := Tensor.full [8, 3] 1
    bbox := [0, 0, 64, 64] }

example : cuboid.pointCount = 8 := rfl

/-- Soundness quantifies over the count carried by the artifact, including counts above eight. -/
example (cert : BoxCameraCert Float) (accepted : checkCert cert = true)
    (index : Fin cert.pointCount) : 0 < certProjectZ cert index :=
  (checkCert_sound accepted).corner_positive_depth index

def certificateJson (points : Lean.Json) (pointCount? : Option Lean.Json := none)
    (format : String := formatStringPoints) : Lean.Json :=
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

def expectRejected (label : String) (payload : Lean.Json) : IO Unit := do
  let accepted ← try
    let _ ← parseJsonCert payload
    pure true
  catch _ => pure false
  expect label (!accepted)

def diagonalPoints (count : Nat) : Array Float :=
  (List.range count).toArray.flatMap fun index =>
    #[Float.ofNat (index + 1), Float.ofNat (index + 1), 1.0]

/-- Parse `points` under `format`, then check every point and one defect in the final point. -/
def checkCount (count : Nat) (points : Array Float) (format : String)
    (declared : Option Lean.Json) : IO Unit := do
  let cert ← parseJsonCert (certificateJson (Lean.toJson points) declared format)
  expect s!"count {count} preserved" (cert.pointCount == count)
  expect s!"count {count} accepted" (checkCert cert)
  for index in List.finRange cert.pointCount do
    expect s!"count {count}, point {index.val} depth" (certProjectZ cert index == 1)
    expect s!"count {count}, point {index.val} x"
      (certProjectX cert index == Float.ofNat (index.val + 1))
    expect s!"count {count}, point {index.val} y"
      (certProjectY cert index == Float.ofNat (index.val + 1))
  -- Put the defect in the final point to detect an incomplete traversal.
  for badPoint in [#[1.0, 1.0, -1.0], #[65.0, 1.0, 1.0]] do
    let bad := points.extract 0 (points.size - 3) ++ badPoint
    let cert ← parseJsonCert (certificateJson (Lean.toJson bad) declared format)
    expect s!"count {count} checks its last point" (!checkCert cert)
  let narrow := { cert with bbox := ([0, 0, 0, 0] : Tensor Float [4]) }
  expect s!"count {count} checks enclosure" (!checkCert narrow)

def run : IO Unit := do
  expect "eight-point default" (cuboid.pointCount == 8 && checkCert cuboid)
  for count in [1, 3, 8, 17] do
    checkCount count (diagonalPoints count) formatStringPoints (some (Lean.toJson count))
  -- The eight-corner format accepts an omitted or explicit count of eight.
  for declared in [none, some (Lean.toJson (8 : Nat))] do
    checkCount 8 (diagonalPoints 8) formatString declared
  let single ←
    parseJsonCert (certificateJson (Lean.toJson (diagonalPoints 1)) (some (Lean.toJson 1)))
  expect "image dimensions are checked"
    (!checkCert { single with width := 0 })
  expect "the claimed box must be ordered"
    (!checkCert { single with bbox := ([2, 0, 1, 1] : Tensor Float [4]) })
  expectRejected "zero points without a declared count"
    (certificateJson (Lean.toJson (#[] : Array Float)))
  expectRejected "zero points with a declared count"
    (certificateJson (Lean.toJson (#[] : Array Float)) (some (Lean.toJson (0 : Nat))))
  expectRejected "zero points in the eight-corner format"
    (certificateJson (Lean.toJson (#[] : Array Float)) none formatString)
  expectRejected "eight-corner format with three points"
    (certificateJson (Lean.toJson (diagonalPoints 3)) none formatString)
  expectRejected "eight-corner format with a declared count of three"
    (certificateJson (Lean.toJson (diagonalPoints 3)) (some (Lean.toJson (3 : Nat))) formatString)
  expectRejected "point-count format without a declared count"
    (certificateJson (Lean.toJson (diagonalPoints 3)))
  expectRejected "unknown format"
    (certificateJson (Lean.toJson (diagonalPoints 8)) none "torchlean.camera.box3d.v0")
  expectRejected "incomplete triple"
    (certificateJson (Lean.toJson (#[1, 1, 1, 1] : Array Nat)) (some (Lean.toJson (1 : Nat))))
  expectRejected "declared count mismatch"
    (certificateJson (Lean.toJson (#[1, 1, 1] : Array Nat)) (some (Lean.toJson (2 : Nat))))
  expectRejected "negative count"
    (certificateJson (Lean.toJson (#[] : Array Nat)) (some (Lean.toJson (-1 : Int))))
  expectRejected "non-finite point"
    (certificateJson (.arr #[.str "NaN", Lean.toJson (1 : Nat), Lean.toJson (1 : Nat)])
      (some (Lean.toJson (1 : Nat))))
  IO.println "  camera certificate point counts: passed"

end NN.Tests.Verification.CameraCertificates
