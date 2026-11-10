/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Fiber
public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Conversion
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Tensor reads for `roundoff`

The `roundoff` rules here read last-axis rows, fibers along an axis, batch slices, maps over a
leading shape (`mapLeading`, `zipEach`) and shape casts. With them the analysis reads these
producers at any axis of any shape without unfolding their recursion over the shape. It reads a
`Tensor.castShape`, a `▸` transport or a `cast` along a tensor type equality at the coordinate
transported back along the shape equality.

Reads through runtime index lists apply to every tensor. `Tensor.get?` validates its index list
with `Shape.Coord.ofList?`, and an index list taken from a coordinate reads that entry, also when
the coordinate's shape is spelled differently from the tensor's. A leading index of the leading
axis selects a slice. An accumulation `update t l (getAtOrZero t l + x)` changes only the
coordinate that `l` names, so a loop that accumulates this way is read at one coordinate of its
accumulator.
-/

public section

attribute [roundoff_rule]
  TorchLean.Tensor.row_apply
  TorchLean.Tensor.fiber_apply
  TorchLean.Tensor.batchSlice_apply
  TorchLean.Tensor.mapLeading_apply
  TorchLean.Tensor.zipEach_apply
  TorchLean.Tensor.row_updateLast
  TorchLean.Shape.Coord.getLast_updateLast
  TorchLean.Shape.Coord.updateLast_updateLast
  TorchLean.Tensor.castShape_apply
  TorchLean.Tensor.cast_apply
  TorchLean.Tensor.eqRec_eq_castShape
  TorchLean.Tensor.Rep.map_id
  TorchLean.Tensor.to_list_eq_data
  TorchLean.Tensor.get?.eq_1
  TorchLean.Tensor.getAtOrZero_coord_toList
  TorchLean.Tensor.getAtOrZero_cons_val
  TorchLean.Tensor.getAtOrZero_scalar_nil
  TorchLean.Tensor.update_add_apply
