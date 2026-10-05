/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops.Core

/-!
# CUDA Tape Operations: Piecewise-Linear Table Lookup

A function tabulated at `width` uniformly spaced grid points is evaluated at a grid coordinate
`u` by linear interpolation between the two samples around `u`, after clamping `u` to
`[0, width - 1]`. This is the uniform-grid analogue of `np.interp`, which is how calibration
curves, inverted sensor-response tables, and activation lookups are evaluated. A layered table
stores one such function per layer and selects each element's layer from a second buffer.

With `i = ⌊u⌋` and `w = u - i`, the value is `v₀ + (v₁ - v₀) · w` for the samples `v₀ = table[i]`
and `v₁ = table[min (i + 1) (width - 1)]`. The forward composes numerical primitives that each
round once in float32, so the result is `KernelSpec.tableLookupSpec` bit for bit. The backward is
TorchLean's own: the coordinate receives the slope of its segment, zero where the clamp is active
as for `clamp`, and the table receives the two interpolation weights scattered to the samples
they multiplied.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

open Spec TorchLean

namespace Buffer

/-- The samples each element interpolates between, computed once and reused by the backward. -/
structure TableSegments where
  /-- Flat position of the lower sample. -/
  lower : Buffer
  /-- Flat position of the upper sample; the lower one again on the last node of a layer. -/
  upper : Buffer
  /-- Interpolation weight toward the upper sample, in `[0, 1)`. -/
  weight : Buffer
  /-- Intermediates that the forward value releases once it has consumed them. -/
  workspace : Array Buffer := #[]

/--
Locate the segment of each of `count` coordinates inside a table of `layers × width` samples.

Coordinates clamp to `[0, width - 1]` and the optional layer indices to `[0, layers - 1]` with the
backend's `clamp`, whose NaN selects the lower bound. Every operation here is exact in float32
while `layers * width ≤ 2^24`; the tape operation enforces that bound.
-/
@[no_expose] def tableSegments (layers width : UInt32) (coords : Buffer) (layer : Option Buffer)
    (count : UInt32) : TableSegments :=
  let last : Float := Float.ofNat (width.toNat - 1)
  let coordinate := clamp coords 0.0 last
  let column := floor coordinate
  let weight := releaseThen coordinate (sub coordinate column)
  let ones := full count 1.0
  let next := releaseThen ones (add column ones)
  let nextColumn := releaseThen next (clamp next 0.0 last)
  match layer with
  | none => { lower := column, upper := nextColumn, weight := weight }
  | some layerIdx =>
      let lastLayer : Float := Float.ofNat (layers.toNat - 1)
      let clampedLayer := clamp layerIdx 0.0 lastLayer
      let integralLayer := releaseThen clampedLayer (floor clampedLayer)
      let offset := releaseThen integralLayer (scale integralLayer (Float.ofNat width.toNat))
      let lower := add offset column
      let upper := add offset nextColumn
      { lower := lower, upper := upper, weight := weight
        workspace := #[column, nextColumn, offset] }

/--
Interpolate the table at located segments: `v₀ + (v₁ - v₀) · w`, one rounding per operation.

The segments' workspace is released through the result, so the forward value is the last use of
those intermediates.
-/
@[no_expose] def tableLookupForward (table : Buffer) (size : UInt32) (seg : TableSegments)
    (count : UInt32) : Buffer :=
  let lowerSample := gatherAt table size seg.lower count
  let upperSample := gatherAt table size seg.upper count
  let slope := sub upperSample lowerSample
  let scaled := mul slope seg.weight
  let value := add lowerSample scaled
  releaseManyThen (#[lowerSample, upperSample, slope, scaled] ++ seg.workspace) value

/--
VJP of `tableLookupForward`, returning the coordinate and table cotangents.

The coordinate cotangent is the segment slope times the upstream cotangent, zeroed where the
coordinate clamp was active (the open-interval rule of `clampBwd`). The table cotangent scatters
`(1 - w)` and `w` times the upstream cotangent to the lower and upper samples.
-/
@[no_expose] def tableLookupBackward (table : Buffer) (size : UInt32) (seg : TableSegments)
    (coords : Buffer) (width : UInt32) (dLdy : Buffer) (count : UInt32) : Buffer × Buffer :=
  let last : Float := Float.ofNat (width.toNat - 1)
  let lowerSample := gatherAt table size seg.lower count
  let upperSample := gatherAt table size seg.upper count
  let slope := sub upperSample lowerSample
  let interior := mul dLdy slope
  let dCoords := releaseManyThen #[lowerSample, upperSample, slope, interior]
    (clampBwd coords interior 0.0 last)
  let ones := full count 1.0
  let lowerWeight := sub ones seg.weight
  let dLower := mul dLdy lowerWeight
  let dUpper := mul dLdy seg.weight
  let lowerPart := scatterAddAt size seg.lower dLower count
  let upperPart := scatterAddAt size seg.upper dUpper count
  let dTable := releaseManyThen #[ones, lowerWeight, dLower, dUpper, lowerPart, upperPart]
    (add lowerPart upperPart)
  (dCoords, dTable)

end Buffer

namespace Tape

/-- Shared implementation of both lookups; `layerId` supplies per-element layer indices. -/
@[no_expose] def tableLookupCore (layers width : Nat) {s : Shape} (t : Tape)
    (tableId coordsId : Nat) (layerId : Option Nat) : Result (Tape × Nat) := do
  if width = 0 then
    throw "autograd: table_lookup: the table has no samples"
  if layers = 0 then
    throw "autograd: table_lookup: the table has no layers"
  if layers * width > 2 ^ 24 then
    throw "autograd: table_lookup: float32 positions address at most 2^24 samples exactly"
  let tableShape : Shape := match layerId with
    | none => .dim width .scalar
    | some _ => .dim layers (.dim width .scalar)
  let table ← requireValue (t := t) tableId tableShape
  let coords ← requireValue (t := t) coordsId s
  let layer ← match layerId with
    | none => pure none
    | some id => some <$> requireValue (t := t) id s
  let layers32 ← AnyBuffer.natToU32Checked layers
  let width32 ← AnyBuffer.natToU32Checked width
  let size32 ← AnyBuffer.natToU32Checked (layers * width)
  let count32 ← AnyBuffer.numelU32 s
  let seg := Buffer.tableSegments layers32 width32 coords layer count32
  let y := Buffer.tableLookupForward table size32 seg count32
  let node : Node :=
    { name := some (if layerId.isSome then "layered_table_lookup" else "table_lookup")
      value := { s := s, buf := y }
      requiresGrad := (t.getNode? tableId).any (·.requiresGrad) ||
        (t.getNode? coordsId).any (·.requiresGrad)
      parents := #[tableId, coordsId] ++ (match layerId with | none => #[] | some id => #[id])
      cleanup := #[seg.lower, seg.upper, seg.weight]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny s
        let (dCoords, dTable) :=
          Buffer.tableLookupBackward table size32 seg coords width32 dLdy.buf count32
        pure #[
          (tableId, { s := tableShape, buf := dTable }),
          (coordsId, { s := s, buf := dCoords })] }
  pure (t.addNode node)

/-- Piecewise-linear lookup of a rank-one table of `width` samples at coordinates of shape `s`. -/
@[inline] def tableLookup {width : Nat} {s : Shape} (t : Tape) (tableId coordsId : Nat) :
    Result (Tape × Nat) :=
  tableLookupCore 1 width (s := s) t tableId coordsId none

/--
Piecewise-linear lookup of a `layers × width` table, selecting each element's layer from the
integral values of the tape value `layerId`, which has shape `s` and receives no gradient.
-/
@[inline] def layeredTableLookup {layers width : Nat} {s : Shape} (t : Tape)
    (tableId coordsId layerId : Nat) : Result (Tape × Nat) :=
  tableLookupCore layers width (s := s) t tableId coordsId (some layerId)

end Tape

end LibTorch
end Autograd
end Runtime
