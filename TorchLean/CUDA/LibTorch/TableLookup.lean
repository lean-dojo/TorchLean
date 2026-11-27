/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.CUDA.LibTorch.Buffer
public import TorchLean.CUDA.LibTorch.Kernels

/-!
# Piecewise-linear table lookup on device buffers

A function tabulated at `width` uniformly spaced grid points is evaluated at a grid coordinate
`u` by linear interpolation between the two samples around `u`, after clamping `u` to
`[0, width - 1]`. This is the uniform-grid analogue of `np.interp`, which is how calibration
curves, inverted sensor-response tables and activation lookups are evaluated. A layered table
stores one such function per layer and selects each element's layer from a second buffer.

With `i = ⌊u⌋` and `w = u - i`, the value is `v₀ + (v₁ - v₀) · w` for the samples `v₀ = table[i]`
and `v₁ = table[min (i + 1) (width - 1)]`. `Buffer.tableSegments` locates the samples,
`Buffer.tableLookupForward` interpolates and `Buffer.tableLookupBackward` is the pullback: the
coordinate receives the slope of its segment, zero where the clamp is active as for `clamp`, and
the table receives the two interpolation weights scattered to the samples they multiplied.

The forward composes `clamp`, `floor`, `add`, `sub`, `mul`, `scale` and `gatherAt`. The positions
and the weight are exact while `layers * width ≤ 2^24` in binary32 and `2^53` in binary64, which
`Buffer.tableSegments?` checks, so only the interpolation rounds, once per operation in the
buffer's dtype. A NaN coordinate or layer index gives a NaN value. Every buffer is a
native binary32 or binary64 buffer of one dtype, and buffers of a configured format have no
lookup.
-/

@[expose] public section

namespace TorchLean
namespace LibTorch
namespace Buffer

/-- The samples each element interpolates between, computed once and reused by the backward. -/
structure TableSegments where
  /-- Flat position of the lower sample. -/
  lower : Buffer
  /-- Flat position of the upper sample, the lower one again on the last node of a layer. -/
  upper : Buffer
  /-- Interpolation weight toward the upper sample, in `[0, 1)`. -/
  weight : Buffer
  /-- Intermediates that the forward value releases once it has consumed them. -/
  workspace : Array Buffer := #[]

/-- Release the three buffers of `seg` and return `keep`. The workspace belongs to the forward
value, which releases it. -/
def TableSegments.releaseThen (seg : TableSegments) (keep : @& Buffer) : Buffer :=
  releaseManyThen #[seg.lower, seg.upper, seg.weight] keep

/--
Locate the segment of each of `count` coordinates inside a table of `layers × width` samples.

Coordinates clamp to `[0, width - 1]` and the optional layer indices to `[0, layers - 1]` with
`clamp`, which keeps a NaN. A NaN coordinate or layer index has a NaN weight, and so a NaN value;
its positions are NaN, which `gatherAt` reads as `0`. The positions are exact only within the bound
that `tableSegments?` checks.
-/
@[no_expose] def tableSegments (layers width : UInt32) (coords : Buffer) (layer : Option Buffer)
    (count : UInt32) : TableSegments :=
  let last : Float := Float.ofNat (width.toNat - 1)
  let coordinate := clamp coords 0.0 last
  let column := floor coordinate
  let weight := releaseThen coordinate (sub coordinate column)
  let ones := full count 1.0 (dtype coords)
  let next := releaseThen ones (add column ones)
  let nextColumn := releaseThen next (clamp next 0.0 last)
  match layer with
  | none => { lower := column, upper := nextColumn, weight := weight }
  | some layerIdx =>
      let lastLayer : Float := Float.ofNat (layers.toNat - 1)
      let clampedLayer := clamp layerIdx 0.0 lastLayer
      let integralLayer := releaseThen clampedLayer (floor clampedLayer)
      let offset := releaseThen integralLayer (scale integralLayer (Float.ofNat width.toNat))
      -- `0` at every layer index but a NaN, which it carries into the weight.
      let carrier := scale offset 0.0
      let layerWeight := releaseManyThen #[weight, carrier] (add weight carrier)
      let lower := add offset column
      let upper := add offset nextColumn
      { lower := lower, upper := upper, weight := layerWeight
        workspace := #[column, nextColumn, offset] }

/-- The largest sample count whose positions are exact in `dtype`, and `none` for a configured
format, which has no lookup. -/
def tablePositionBound : Dtype → Option Nat
  | .float32 => some (2 ^ 24)
  | .float64 => some (2 ^ 53)
  | .encoded => none

/-- `tableSegments` after the checks it relies on: a nonempty table, native buffers of one dtype,
and a sample count whose positions are exact in that dtype. -/
def tableSegments? (layers width : UInt32) (coords : Buffer) (layer : Option Buffer)
    (count : UInt32) : Except String TableSegments := do
  if width = 0 then
    throw "table_lookup: the table has no samples"
  if layers = 0 then
    throw "table_lookup: the table has no layers"
  let some bound := tablePositionBound (dtype coords)
    | throw "table_lookup: a binary32 or binary64 coordinate buffer is required"
  if layers.toNat * width.toNat > bound then
    throw s!"table_lookup: positions address at most {bound} samples exactly"
  if let some l := layer then
    if dtype l != dtype coords then
      throw "table_lookup: the layer and coordinate buffers differ in dtype"
  pure (tableSegments layers width coords layer count)

/--
Interpolate the table at located segments: `v₀ + (v₁ - v₀) · w`, one rounding per operation.

The segments' workspace is released through the result, so the forward value is the last use of
those intermediates. `lower`, `upper` and `weight` stay with the caller for the backward.
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
coordinate clamp was active (the open-interval rule of `clampBackward`). The table cotangent
scatters `(1 - w)` and `w` times the upstream cotangent to the lower and upper samples.
-/
@[no_expose] def tableLookupBackward (table : Buffer) (size : UInt32) (seg : TableSegments)
    (coords : Buffer) (width : UInt32) (dLdy : Buffer) (count : UInt32) : Buffer × Buffer :=
  let last : Float := Float.ofNat (width.toNat - 1)
  let lowerSample := gatherAt table size seg.lower count
  let upperSample := gatherAt table size seg.upper count
  let slope := sub upperSample lowerSample
  let interior := mul dLdy slope
  let dCoords := releaseManyThen #[lowerSample, upperSample, slope, interior]
    (clampBackward coords interior 0.0 last)
  let ones := full count 1.0 (dtype dLdy)
  let lowerWeight := sub ones seg.weight
  let dLower := mul dLdy lowerWeight
  let dUpper := mul dLdy seg.weight
  let lowerPart := scatterAddAt size seg.lower dLower count
  let upperPart := scatterAddAt size seg.upper dUpper count
  let dTable := releaseManyThen #[ones, lowerWeight, dLower, dUpper, lowerPart, upperPart]
    (add lowerPart upperPart)
  (dCoords, dTable)

end Buffer
end LibTorch
end TorchLean
