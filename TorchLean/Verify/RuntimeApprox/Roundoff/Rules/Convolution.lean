/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Layers.Pooling
public import TorchLean.Tensor.Window
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Convolution and pooling reads for `roundoff`

Convolution and pooling read their tensors through runtime index lists and loop over index boxes
with `TorchLean.Conv.foldlIndices`. With the index-list reads of
`TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor`, this module registers the reads that let
`roundoff` follow them for any spatial rank, kernel, stride, padding and dilation. A box loop is a
fold over the coordinates of the box. The output extents are read by their lemmas, which keeps their
producers folded and their spelling the same as in the tensor types. Max pooling reads the input at
the cell its winner selects, and average pooling divides its window sum once by the count
`TorchLean.Tensor.poolCount`, which is positive when the count includes padding.
-/

public section

namespace TorchLean.Roundoff

/-- The cast count of a pool window that includes padding is positive. -/
theorem cast_poolCount_true_pos {s : Shape} {dims : List (Fin s.rank)}
    {kernel stride padLow padHigh dilation : List Nat} {ceilMode : Bool}
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    (0 : ℝ) < (Tensor.poolCount true o : ℕ) :=
  Nat.cast_pos.mpr (Tensor.poolCount_true_pos o)

end TorchLean.Roundoff

attribute [roundoff_rule]
  TorchLean.Conv.foldlIndices_eq_foldl
  TorchLean.Conv.outSpatialDilated_apply
  TorchLean.ConvTranspose.outSpatial_apply
  TorchLean.Pool.outSpatial_apply
  TorchLean.Roundoff.cast_poolCount_true_pos

attribute [roundoff_rule]
  TorchLean.Tensor.conv_apply
  TorchLean.Tensor.convInput_eq_convSource
  TorchLean.Tensor.convTranspose_apply
  TorchLean.Tensor.convTransposeInput_eq_convTransposeSource
  TorchLean.Tensor.getOrZero_eq_ofIndices?
  TorchLean.Tensor.convBackwardInput_apply
  TorchLean.Tensor.convBackwardWeight_apply
  TorchLean.Tensor.convTransposeBackwardInput_apply
  TorchLean.Tensor.convTransposeBackwardWeight_apply
  TorchLean.Tensor.poolInput_def
  TorchLean.Tensor.pool_sum_apply
  TorchLean.Tensor.pool_avg_apply
  TorchLean.Tensor.pool_max_apply
  TorchLean.Tensor.pool_min_apply
  TorchLean.Tensor.poolBackward_max_apply
  TorchLean.Tensor.poolBackward_min_apply
  TorchLean.Tensor.poolBackward_sum_apply
  TorchLean.Tensor.poolBackward_avg_apply
  TorchLean.Tensor.smoothMaxPool_apply
  TorchLean.Tensor.smoothMaxPoolPivot_def
  TorchLean.Tensor.smoothMaxPoolShift_def
  TorchLean.Tensor.smoothMaxPoolSum_def
  TorchLean.Tensor.smoothMaxPoolJVP_apply
  TorchLean.Tensor.smoothMaxPoolBackward_apply
  TorchLean.Tensor.adaptiveInput_def
  TorchLean.Tensor.adaptivePool_sum_apply
  TorchLean.Tensor.adaptivePool_avg_apply
  TorchLean.Tensor.adaptivePool_max_apply
  TorchLean.Tensor.adaptivePool_min_apply
