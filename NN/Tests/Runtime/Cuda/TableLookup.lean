/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Floor, Device-Position Gather, Table Lookup

- `floor` agrees with `Float32.floor` bit for bit, and its tape node has a zero gradient;
- `gatherAt` agrees with the CPU tape's `indexSelect` for in-range positions, forward and table
  gradient, and follows its documented addressing for out-of-range and NaN positions;
- `tableLookup` and `layeredTableLookup` reproduce an executable `Float32` reference bit for bit
  at interior, boundary, out-of-range, integral, and NaN coordinates, and their forward and both
  gradients agree with the same function recorded on the CPU tape from existing nodes.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace TableLookup

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd
open Runtime.Autograd.LibTorch (Buffer)

/-- Bitwise comparison of two float arrays after narrowing both to float32. -/
def assertBits32 (msg : String) (got expected : FloatArray) : IO Unit := do
  unless got.size == expected.size do
    throw <| IO.userError s!"{msg}: size {got.size}, expected {expected.size}"
  for i in [:got.size] do
    let x := (got.get! i).toFloat32
    let y := (expected.get! i).toFloat32
    unless x.toBits == y.toBits do
      throw <| IO.userError
        s!"{msg}[{i}]: got {x} (bits {x.toBits}), expected {y} (bits {y.toBits})"

/-- Read a CUDA tape value as a raw float array. -/
def rawValue (t : LibTorch.Tape) (id : Nat) (s : Shape) : IO FloatArray := do
  let b ← Utils.okOrThrow (LibTorch.Tape.requireValue (t := t) id s)
  pure (Buffer.toFloatArray b)

def nan : Float := Float.ofBits 0x7ff8000000000000

def runFloor : IO Unit := do
  IO.println "== floor =="
  let inputs : Array Float := #[-2.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.9999,
    16777217.0, -16777217.0, 1.0e8, -1.0e8, 0.1, -0.1]
  let expected := FloatArray.mk <| inputs.map fun x => x.toFloat32.floor.toFloat
  let input := Buffer.ofFloatArray (FloatArray.mk inputs)
  assertBits32 "floor buffer" (Buffer.toFloatArray (Buffer.floor input)) expected
  let s : Shape := [inputs.size]
  let t0 : LibTorch.Tape := LibTorch.Tape.empty
  let (t1, xId) := LibTorch.Tape.leaf (t := t0) { s := s, buf := input } (name := some "x")
  let (t2, yId) ← Utils.okOrThrow (LibTorch.Tape.floor (s := s) (t := t1) xId)
  assertBits32 "floor tape value" (← rawValue t2 yId s) expected
  let seed : LibTorch.AnyBuffer := { s := s, buf := Buffer.full inputs.size.toUInt32 1.0 }
  let grads ← Utils.okOrThrow (LibTorch.Tape.backwardDenseAll (t := t2) yId seed)
  let some dx := grads[xId]?
    | throw <| IO.userError "floor: missing gradient"
  assertBits32 "floor gradient" (Buffer.toFloatArray dx.buf)
    (FloatArray.mk (Array.replicate inputs.size 0.0))

def runGatherAt : IO Unit := do
  IO.println "== gatherAt vs CPU indexSelect =="
  let n : Nat := 5
  let k : Nat := 4
  let sT : Shape := [n]
  let sP : Shape := [k]
  let tableVals : Array Float := #[0.125, -0.25, 0.5, 1.0, -2.0]
  let table : Tensor Float sT := (Tensor.from tableVals).reshape [n] (by dsimp; decide)
  let positions : Fin k → Fin n :=
    ![⟨2, by decide⟩, ⟨0, by decide⟩, ⟨2, by decide⟩, ⟨4, by decide⟩]
  let idx : Tensor (Fin n) [k] := Tensor.ofFn positions
  let positionsF : Tensor Float sP := Tensor.ofFn fun i => Float.ofNat (positions i).val
  let seed : Tensor Float sP := (Tensor.from #[1.0, 0.5, -0.25, 2.0]).reshape [k] (by dsimp; decide)

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, tId) := Tape.leaf (t := t0) table (name := some "table")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.indexSelect (α := Float) (s := sT) (t := t1) tId 0 k idx)
  let yCpu ← Utils.cpuValue (s := sP) t2 yId
  let gradsCpu ← Utils.okOrThrow
    (Tape.backwardDenseAll (α := Float) (t := t2) yId (Spec.SomeTensor.ofTensor seed))
  let dTCpu ← Utils.cpuGrad (s := sT) gradsCpu tId

  -- CUDA
  let c0 : LibTorch.Tape := LibTorch.Tape.empty
  let (c1, tIdc) := LibTorch.Tape.leaf (t := c0) (Utils.tensorToAnyBuffer table)
    (name := some "table")
  let (c2, pIdc) := LibTorch.Tape.leaf (t := c1) (Utils.tensorToAnyBuffer positionsF)
    (name := some "positions") (requiresGrad := false)
  let (c3, yIdc) ← Utils.okOrThrow
    (LibTorch.Tape.gatherAt (size := n) (s := sP) (t := c2) tIdc pIdc)
  let yCuda ← Utils.cudaValue (s := sP) c3 yIdc
  let gradsCuda ← Utils.okOrThrow
    (LibTorch.Tape.backwardDenseAll (t := c3) yIdc (Utils.tensorToAnyBuffer seed))
  let dTCuda ← Utils.cudaGrad (s := sT) gradsCuda tIdc

  Utils.assertTensorApprox (s := sP) "gather_at forward" yCuda yCpu (tol := 0.0)
  Utils.assertTensorApprox (s := sT) "gather_at table gradient" dTCuda dTCpu (tol := 0.0)

  -- Addressing: negative, past the end, non-integral, NaN, and a negative fraction.
  let tableBuf := Buffer.ofFloatArray (FloatArray.mk tableVals)
  let odd := Buffer.ofFloatArray (FloatArray.mk #[-3.0, 7.0, 2.75, nan, -0.5])
  assertBits32 "gather_at addressing" (Buffer.toFloatArray (Buffer.gatherAt tableBuf 5 odd 5))
    (FloatArray.mk #[0.125, -2.0, 0.5, 0.125, 0.125])

/-- The backend's clamp, `fmin (fmax x lo) hi`, which selects `lo` for a NaN input. -/
def clampRef (x lo hi : Float32) : Float32 :=
  let m := if x.isNaN then lo else if x < lo then lo else x
  if m > hi then hi else m

/--
Executable reference for one lookup in Lean core `Float32` (IEEE binary32), statement for
statement the sequence `Buffer.tableSegments` and `Buffer.tableLookupForward` run on device.
-/
def refLookup (table : Array Float32) (layers width : Nat) (u layer : Float32) : Float32 :=
  let last := Float32.ofNat (width - 1)
  let coordinate := clampRef u 0.0 last
  let column := coordinate.floor
  let weight := coordinate - column
  let nextColumn := clampRef (column + 1.0) 0.0 last
  let layerIdx := (clampRef layer 0.0 (Float32.ofNat (layers - 1))).floor
  let offset := layerIdx * Float32.ofNat width
  let lower := (offset + column).toFloat.toUInt64.toNat
  let upper := (offset + nextColumn).toFloat.toUInt64.toNat
  let lowerSample := table[lower]!
  let upperSample := table[upper]!
  lowerSample + (upperSample - lowerSample) * weight

/-- Smooth, layer-dependent samples in layer-major order. -/
def sampleData (width layers : Nat) : FloatArray := Id.run do
  let mut a := FloatArray.emptyWithCapacity (width * layers)
  for l in [0:layers] do
    for i in [0:width] do
      let x := Float.ofNat i
      let y := Float.ofNat l
      a := a.push (0.17 * x - 0.031 * x * x + 0.4 * y + 0.05)
  return a

def widthN : Nat := 7
def layersN : Nat := 3

/-- Interior, boundary, out-of-range, integral, signed-zero, and NaN grid coordinates. -/
def probeCoords : Array Float :=
  #[0.0, 0.25, 1.0, 2.5, 3.75, 5.999, 6.0, -1.5, 9.75, 4.125, 2.875, 0.001, nan, -0.0, 5.5]

/-- Layer selectors, including one past the end and one non-integral. -/
def probeLayers : Array Float :=
  #[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 5.0, 1.5]

def runLookupBitExact : IO Unit := do
  IO.println "== tableLookup bit-exact against the Float32 reference =="
  let count := probeCoords.size
  let sX : Shape := [count]
  let data := sampleData widthN layersN
  let table32 := (Array.range data.size).map fun i => (data.get! i).toFloat32
  let coordsBuf := Buffer.ofFloatArray (FloatArray.mk probeCoords)
  let layersBuf := Buffer.ofFloatArray (FloatArray.mk probeLayers)

  -- Layered table.
  let expectedLayered := FloatArray.mk <| (Array.range count).map fun i =>
    (refLookup table32 layersN widthN (probeCoords[i]!).toFloat32
      (probeLayers[i]!).toFloat32).toFloat
  let c0 : LibTorch.Tape := LibTorch.Tape.empty
  let (c1, tId) := LibTorch.Tape.leaf (t := c0)
    { s := [layersN, widthN], buf := Buffer.ofFloatArray data } (name := some "table")
  let (c2, uId) := LibTorch.Tape.leaf (t := c1) { s := sX, buf := coordsBuf } (name := some "u")
  let (c3, lId) := LibTorch.Tape.leaf (t := c2) { s := sX, buf := layersBuf }
    (name := some "layer") (requiresGrad := false)
  let (c4, yId) ← Utils.okOrThrow
    (LibTorch.Tape.layeredTableLookup (layers := layersN) (width := widthN) (s := sX) (t := c3)
      tId uId lId)
  assertBits32 "layered_table_lookup" (← rawValue c4 yId sX) expectedLayered

  -- Rank-one table: the first layer alone, every probe addressing layer 0.
  let layer0 := FloatArray.mk <| (Array.range widthN).map fun i => data.get! i
  let table0 := (Array.range widthN).map fun i => (layer0.get! i).toFloat32
  let expectedFlat := FloatArray.mk <| (Array.range count).map fun i =>
    (refLookup table0 1 widthN (probeCoords[i]!).toFloat32 0.0).toFloat
  let (d1, t0Id) := LibTorch.Tape.leaf (t := c0)
    { s := [widthN], buf := Buffer.ofFloatArray layer0 } (name := some "table")
  let (d2, u0Id) := LibTorch.Tape.leaf (t := d1) { s := sX, buf := coordsBuf } (name := some "u")
  let (d3, y0Id) ← Utils.okOrThrow
    (LibTorch.Tape.tableLookup (width := widthN) (s := sX) (t := d2) t0Id u0Id)
  assertBits32 "table_lookup" (← rawValue d3 y0Id sX) expectedFlat

def runLookupVsCpu : IO Unit := do
  IO.println "== tableLookup vs the same function on the CPU tape =="
  let width : Nat := 6
  let n : Nat := 8
  let sT : Shape := [width]
  let sX : Shape := [n]
  -- Dyadic values keep every intermediate exact in both float32 and Float.
  let tableVals : Array Float := #[0.5, -1.0, 2.0, 1.5, -0.25, 3.0]
  let coordVals : Array Float := #[0.0, 0.25, 1.5, 2.0, 3.75, 5.0, -2.0, 7.5]
  let seedVals : Array Float := #[1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 1.5, -0.75]
  let table : Tensor Float sT := (Tensor.from tableVals).reshape [width] (by dsimp; decide)
  let coords : Tensor Float sX := (Tensor.from coordVals).reshape [n] (by dsimp; decide)
  let seed : Tensor Float sX := (Tensor.from seedVals).reshape [n] (by dsimp; decide)
  let last : Float := Float.ofNat (width - 1)
  -- Host-side segment selection, the recipe the device follows.
  let segment : Fin n → Nat × Nat := fun i =>
    let u := coordVals[i.val]!
    let c := if u < 0.0 then 0.0 else if u > last then last else u
    let lower := c.floor.toUInt64.toNat
    (lower, min (lower + 1) (width - 1))
  let lowerIdx : Tensor (Fin width) [n] := Tensor.ofFn fun i =>
    ⟨(segment i).1 % width, Nat.mod_lt _ (by decide)⟩
  let upperIdx : Tensor (Fin width) [n] := Tensor.ofFn fun i =>
    ⟨(segment i).2 % width, Nat.mod_lt _ (by decide)⟩
  let column : Tensor Float sX := Tensor.ofFn fun i => Float.ofNat (segment i).1

  -- CPU: clamp, subtract the constant column, gather both samples, interpolate.
  let t0 : Tape Float := Tape.empty
  let (t1, tId) := Tape.leaf (t := t0) table (name := some "table")
  let (t2, uId) := Tape.leaf (t := t1) coords (name := some "coords")
  let (t3, cId) := Tape.leaf (t := t2) column (name := some "column") (requiresGrad := false)
  let (t4, ucId) ← Utils.okOrThrow (Tape.clamp (α := Float) (s := sX) (t := t3) uId 0.0 last)
  let (t5, wId) ← Utils.okOrThrow (Tape.sub (α := Float) (s := sX) (t := t4) ucId cId)
  let (t6, v0Id) ← Utils.okOrThrow
    (Tape.indexSelect (α := Float) (s := sT) (t := t5) tId 0 n lowerIdx)
  let (t7, v1Id) ← Utils.okOrThrow
    (Tape.indexSelect (α := Float) (s := sT) (t := t6) tId 0 n upperIdx)
  let (t8, slopeId) ← Utils.okOrThrow (Tape.sub (α := Float) (s := sX) (t := t7) v1Id v0Id)
  let (t9, scaledId) ← Utils.okOrThrow (Tape.mul (α := Float) (s := sX) (t := t8) slopeId wId)
  let (t10, yId) ← Utils.okOrThrow (Tape.add (α := Float) (s := sX) (t := t9) v0Id scaledId)
  let yCpu ← Utils.cpuValue (s := sX) t10 yId
  let gradsCpu ← Utils.okOrThrow
    (Tape.backwardDenseAll (α := Float) (t := t10) yId (Spec.SomeTensor.ofTensor seed))
  let dTCpu ← Utils.cpuGrad (s := sT) gradsCpu tId
  let dUCpu ← Utils.cpuGrad (s := sX) gradsCpu uId

  -- CUDA
  let c0 : LibTorch.Tape := LibTorch.Tape.empty
  let (c1, tIdc) := LibTorch.Tape.leaf (t := c0) (Utils.tensorToAnyBuffer table)
    (name := some "table")
  let (c2, uIdc) := LibTorch.Tape.leaf (t := c1) (Utils.tensorToAnyBuffer coords)
    (name := some "coords")
  let (c3, yIdc) ← Utils.okOrThrow
    (LibTorch.Tape.tableLookup (width := width) (s := sX) (t := c2) tIdc uIdc)
  let yCuda ← Utils.cudaValue (s := sX) c3 yIdc
  let gradsCuda ← Utils.okOrThrow
    (LibTorch.Tape.backwardDenseAll (t := c3) yIdc (Utils.tensorToAnyBuffer seed))
  let dTCuda ← Utils.cudaGrad (s := sT) gradsCuda tIdc
  let dUCuda ← Utils.cudaGrad (s := sX) gradsCuda uIdc

  Utils.assertTensorApprox (s := sX) "table_lookup forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := sT) "table_lookup table gradient" dTCuda dTCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := sX) "table_lookup coordinate gradient" dUCuda dUCpu
    (tol := 1e-6)

def runEdges : IO Unit := do
  IO.println "== tableLookup edges (width 1, empty coordinates, rejected shapes) =="
  -- A single sample: every coordinate reads it, and no coordinate has a gradient.
  let one : Shape := [1]
  let three : Shape := [3]
  let c0 : LibTorch.Tape := LibTorch.Tape.empty
  let (c1, tId) := LibTorch.Tape.leaf (t := c0)
    { s := one, buf := Buffer.ofFloatArray (FloatArray.mk #[0.75]) } (name := some "table")
  let (c2, uId) := LibTorch.Tape.leaf (t := c1)
    { s := three, buf := Buffer.ofFloatArray (FloatArray.mk #[0.0, 3.5, -2.0]) }
    (name := some "u")
  let (c3, yId) ← Utils.okOrThrow
    (LibTorch.Tape.tableLookup (width := 1) (s := three) (t := c2) tId uId)
  assertBits32 "width 1 value" (← rawValue c3 yId three) (FloatArray.mk #[0.75, 0.75, 0.75])
  let seed : LibTorch.AnyBuffer :=
    { s := three, buf := Buffer.ofFloatArray (FloatArray.mk #[1.0, 2.0, 4.0]) }
  let grads ← Utils.okOrThrow (LibTorch.Tape.backwardDenseAll (t := c3) yId seed)
  let some dT := grads[tId]?
    | throw <| IO.userError "width 1: missing table gradient"
  let some dU := grads[uId]?
    | throw <| IO.userError "width 1: missing coordinate gradient"
  assertBits32 "width 1 table gradient" (Buffer.toFloatArray dT.buf) (FloatArray.mk #[7.0])
  assertBits32 "width 1 coordinate gradient" (Buffer.toFloatArray dU.buf)
    (FloatArray.mk #[0.0, 0.0, 0.0])

  -- No coordinates: an empty result and empty gradients, with nothing read from the table.
  let none : Shape := [0]
  let (e1, eId) := LibTorch.Tape.leaf (t := c1)
    { s := none, buf := Buffer.ofFloatArray (FloatArray.mk #[]) } (name := some "u")
  let (e2, eyId) ← Utils.okOrThrow
    (LibTorch.Tape.tableLookup (width := 1) (s := none) (t := e1) tId eId)
  unless (← rawValue e2 eyId none).size == 0 do
    throw <| IO.userError "empty coordinates: expected an empty value"
  let emptySeed : LibTorch.AnyBuffer := { s := none, buf := Buffer.zeros 0 }
  let emptyGrads ← Utils.okOrThrow (LibTorch.Tape.backwardDenseAll (t := e2) eyId emptySeed)
  let some edT := emptyGrads[tId]?
    | throw <| IO.userError "empty coordinates: missing table gradient"
  assertBits32 "empty coordinates table gradient" (Buffer.toFloatArray edT.buf)
    (FloatArray.mk #[0.0])

  -- Rejected before any buffer is read: no samples, and more samples than float32 addresses.
  match LibTorch.Tape.tableLookup (width := 0) (s := three) (t := c2) tId uId with
  | .ok _ => throw <| IO.userError "width 0: expected an error"
  | .error msg =>
      unless (msg.splitOn "no samples").length > 1 do
        throw <| IO.userError s!"width 0: unexpected error: {msg}"
  match LibTorch.Tape.layeredTableLookup (layers := 4097) (width := 4096) (s := three)
      (t := c2) tId uId uId with
  | .ok _ => throw <| IO.userError "2^24 bound: expected an error"
  | .error msg =>
      unless (msg.splitOn "2^24").length > 1 do
        throw <| IO.userError s!"2^24 bound: unexpected error: {msg}"

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: floor, gather_at, table lookup ==="
  runFloor
  runGatherAt
  runLookupBitExact
  runLookupVsCpu
  runEdges

end TableLookup
end Cuda
end Tests
