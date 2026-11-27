/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanTest.Harness.CUDA
public import TorchLean.CUDA.LibTorch.TableLookup

/-!
# Piecewise-linear table lookup on device buffers

`Buffer.tableSegments`, `Buffer.tableLookupForward` and `Buffer.tableLookupBackward` against a
host reference, in binary32 and binary64. Every table sample, coordinate and cotangent of the
fixture is a small dyadic number, so each operation is exact in both formats and the device values
must equal the reference whatever order a repeated position accumulates in. The coordinates cover
both clamps, the nodes of the grid, its last node, interior points and a repeated segment, and the
layer indices cover both layer clamps. A NaN coordinate and a NaN layer index must each give a NaN
value. `Buffer.gatherAt` is checked directly at a NaN, both infinities, negative, fractional and
out-of-range positions, and past `2^24` entries in binary32, where the largest index has no binary32
representation and only the clamp on the integer index keeps the selection in the table. The checks
of `Buffer.tableSegments?` are tested on an empty table, a table past the binary32 position bound
and a layer buffer of another dtype.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA
namespace TableLookup

open TorchLean.LibTorch

/-- The reference lookup: the values, the coordinate cotangent and the table cotangent. -/
def reference (layers width : Nat) (table coords layer seed : Array Float) :
    Array Float × Array Float × Array Float := Id.run do
  let last := Float.ofNat (width - 1)
  let lastLayer := Float.ofNat (layers - 1)
  let mut values := #[]
  let mut dCoords := #[]
  let mut dTable := Array.replicate (layers * width) 0.0
  for i in [0:coords.size] do
    let c := coords[i]!
    let u := max 0.0 (min c last)
    let column := u.floor
    let w := u - column
    let next := min (column + 1.0) last
    let offset := (max 0.0 (min layer[i]! lastLayer)).floor * Float.ofNat width
    let lower := (offset + column).toUInt64.toNat
    let upper := (offset + next).toUInt64.toNat
    let slope := table[upper]! - table[lower]!
    let g := seed[i]!
    values := values.push (table[lower]! + slope * w)
    dCoords := dCoords.push (if 0.0 < c && c < last then g * slope else 0.0)
    dTable := dTable.modify lower (· + g * (1.0 - w))
    dTable := dTable.modify upper (· + g * w)
  return (values, dCoords, dTable)

/-- A device buffer of `dtype` holding `values`. -/
def buffer (dtype : Dtype) (values : Array Float) : Buffer :=
  Buffer.ofFloatArray ⟨values⟩ dtype

/-- The layered lookup and its pullback on the device against `reference`. -/
def check (dtype : Dtype) : IO Unit := do
  let layers := 2
  let width := 4
  let table : Array Float := #[1.0, 3.0, 2.0, -4.0, 0.5, 0.25, 8.0, -1.0]
  let coords : Array Float := #[-1.5, 0.0, 0.25, 1.0, 1.5, 1.75, 2.5, 3.0, 7.0, 0.75]
  let layer : Array Float := #[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 5.0, -2.0]
  let seed : Array Float := #[1.0, 2.0, -1.0, 0.5, 4.0, -2.0, 1.0, 3.0, 0.25, 8.0]
  let count := coords.size.toUInt32
  let size := (layers * width).toUInt32
  let tableBuf := buffer dtype table
  let coordsBuf := buffer dtype coords
  let seg ← IO.ofExcept <| (Buffer.tableSegments? layers.toUInt32 width.toUInt32 coordsBuf
    (some (buffer dtype layer)) count).mapError IO.userError
  let values := Buffer.tableLookupForward tableBuf size seg count
  let (dCoords, dTable) :=
    Buffer.tableLookupBackward tableBuf size seg coordsBuf width.toUInt32 (buffer dtype seed) count
  let (values', dCoords', dTable') := reference layers width table coords layer seed
  Harness.assertFloatArrayEq s!"table_lookup {repr dtype} forward"
    (Buffer.toFloatArray values) ⟨values'⟩
  Harness.assertFloatArrayEq s!"table_lookup {repr dtype} coordinate pullback"
    (Buffer.toFloatArray dCoords) ⟨dCoords'⟩
  Harness.assertFloatArrayEq s!"table_lookup {repr dtype} table pullback"
    (Buffer.toFloatArray dTable) ⟨dTable'⟩
  -- One layer and no layer buffer: the first layer of the table alone.
  let first := table.extract 0 width
  let flat ← IO.ofExcept <| (Buffer.tableSegments? 1 width.toUInt32 coordsBuf none count).mapError
    IO.userError
  let (flat', _, _) := reference 1 width first coords (coords.map fun _ => 0.0) seed
  Harness.assertFloatArrayEq s!"table_lookup {repr dtype} one layer"
    (Buffer.toFloatArray
      (Buffer.tableLookupForward (buffer dtype first) width.toUInt32 flat count)) ⟨flat'⟩

/-- A NaN coordinate and a NaN layer index each give a NaN value, in a layered lookup. -/
def checkNaN (dtype : Dtype) : IO Unit := do
  let nan : Float := 0.0 / 0.0
  let table := buffer dtype #[1.0, 3.0, 2.0, -4.0, 0.5, 0.25, 8.0, -1.0]
  let coords := buffer dtype #[nan, 1.5, 1.5]
  let layer := buffer dtype #[0.0, nan, 1.0]
  let seg ← IO.ofExcept <| (Buffer.tableSegments? 2 4 coords (some layer) 3).mapError IO.userError
  let values := Buffer.toFloatArray (Buffer.tableLookupForward table 8 seg 3)
  unless (values.get! 0).isNaN && (values.get! 1).isNaN && values.get! 2 == 4.125 do
    throw <| IO.userError s!"table_lookup {repr dtype} NaN: got {values}"

/-- `gatherAt` at the edges of its addressing, in binary32 and binary64, and past `2^24` entries in
binary32. -/
def checkGatherAt (dtype : Dtype) : IO Unit := do
  let nan : Float := 0.0 / 0.0
  let inf : Float := 1.0 / 0.0
  let positions := #[nan, -inf, -1.0, -0.0, 0.75, 1.5, 3.0, 3.5, inf, 1.0e30]
  let table := buffer dtype #[10.0, 20.0, 30.0, 40.0]
  Harness.assertFloatArrayEq s!"gather_at {repr dtype} edges"
    (Buffer.toFloatArray (Buffer.gatherAt table 4 (buffer dtype positions) positions.size.toUInt32))
    ⟨#[10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 40.0, 40.0, 40.0, 40.0]⟩
  let ones := buffer dtype (positions.map fun _ => 1.0)
  Harness.assertFloatArrayEq s!"scatter_add_at {repr dtype} edges"
    (Buffer.toFloatArray (Buffer.scatterAddAt 4 (buffer dtype positions) ones
      positions.size.toUInt32)) ⟨#[5.0, 1.0, 0.0, 4.0]⟩

/-- Past `2^24` entries, `size - 1 = 2^24 + 3` rounds to `size` in binary32: a position clamped
only in binary32 would leave the table. -/
def checkGatherAtLarge : IO Unit := do
  let size : UInt32 := 2 ^ 24 + 4
  let inf : Float := 1.0 / 0.0
  let last := Buffer.scatterAddAt size (buffer .float32 #[inf]) (buffer .float32 #[1.0]) 1
  Harness.assertFloatArrayEq "gather_at past 2^24 entries"
    (Buffer.toFloatArray
      (Buffer.gatherAt last size (buffer .float32 #[inf, Float.ofNat (2 ^ 24 + 3), 0.0]) 3))
    ⟨#[1.0, 1.0, 0.0]⟩

/-- Fail unless `tableSegments?` rejects the arguments. -/
def rejects (label : String) (layers width : UInt32) (coords : Buffer) (layer : Option Buffer) :
    IO Unit :=
  match Buffer.tableSegments? layers width coords layer 1 with
  | .error _ => pure ()
  | .ok _ => throw <| IO.userError s!"table_lookup: {label} was accepted"

/-- Runs the table lookup checks. -/
def run : IO Unit := do
  check .float32
  check .float64
  checkNaN .float32
  checkNaN .float64
  checkGatherAt .float32
  checkGatherAt .float64
  checkGatherAtLarge
  let coords := buffer .float32 #[0.5]
  rejects "an empty table" 1 0 coords none
  rejects "a table without layers" 0 4 coords none
  rejects "a table past the binary32 position bound" 2 (2 ^ 24) coords none
  rejects "a layer buffer of another dtype" 2 4 coords (some (buffer .float64 #[0.0]))
  IO.println "  table lookup: device values and pullbacks match the host reference ✓"

end TableLookup
end CUDA
end TorchLean.Test
