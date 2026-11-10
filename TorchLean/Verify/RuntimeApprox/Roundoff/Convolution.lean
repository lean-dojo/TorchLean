/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Constants
public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.RuntimeApprox.NF.CROWN
public import TorchLean.Verify.NumericalCertificate
public import TorchLean.Verify.Floats.IEEE754.Expressions
public import TorchLean.Verify.RuntimeApprox.NF.Attention
public import TorchLean.Verify.RuntimeApprox.NF.Convolution
public import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
public import TorchLean.Verify.RuntimeApprox.NF.Normalization
public import TorchLean.Verify.RuntimeApprox.NF.Optimizers
public import TorchLean.Verify.RuntimeApprox.NF.ReductionOps
public import TorchLean.Verify.Floats.IEEE754.Reductions
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Autograd -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxShift
public import TorchLean.Verify.RuntimeApprox.TransformerBlock
public import TorchLean.Tensor.Layers.Pooling
public import TorchLean.Tensor.Window
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff
public import FloatLib.Numerics.Automation.Attributes
public import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
public import Mathlib.Data.Nat.Factorial.DoubleFactorial
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init
public import TorchLean.Verify.RuntimeApprox.Roundoff.Fold
public import TorchLean.Verify.RuntimeApprox.Roundoff.Mixed
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Convolution
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor

/-!
# Rounding error of convolution and pooling

`roundoff` follows convolution and pooling for any spatial rank, kernel, stride, padding and
dilation, through the reads of `TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Convolution` and the
index-list reads of `TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor`. A loop over an index box
is a fold over the coordinates of the box, and a lookup at the index list of a coordinate reads that
entry, so a sum over an index box is an ordinary fold of the entries. A loop that adds values
into a tensor at runtime index lists is read at one coordinate, where only the additions whose
index list names that coordinate act.

The forward maps of grouped, dilated convolution with asymmetric zero padding (on a channels-first
sample and on any layout), of average pooling and of smooth max pooling over any axes, and the
average-pooling VJP over any axes, have mixed bounds for every IEEE format under range conditions
on the values they compute. Max pooling is exact: it returns an input entry, or zero for a window
with no input cell.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open Conv

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-! ## Convolution -/

/-- Grouped, dilated convolution with asymmetric zero padding, at any spatial rank: the
contraction over the group's input channels and the kernel cells keeps its accumulated mixed
error, and adding the bias rounds once more. -/
roundoff mixed theorem grouped_conv_error :=
  fun [Storage Value] {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter inSpatial : Tensor Nat [d]} (groups : Nat)
    (weights : Tensor Value (outC :: inC :: kernel.data.toList))
    (bias : Tensor Value [outC])
    (input : Tensor Value (inC :: inSpatial.data.toList))
    (index : Shape.Coord (outC :: (outSpatialDilated inSpatial kernel stride
      dilation paddingBefore paddingAfter).data.toList))
    (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (grouped (stride := stride) (dilation := dilation)
      (paddingBefore := paddingBefore) (paddingAfter := paddingAfter) groups weights bias input
      index)

/-! ## Pooling -/

/-- Average pooling over any axes that counts padding divides the window's sum by the count
`poolCount true`, which is positive (`Tensor.poolCount_true_pos`). The count is not cast, so the
quotient carries the sum's mixed error and rounds once. -/
roundoff mixed theorem avg_pool_error :=
  fun [Storage Value] {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool)
    (x : Tensor Value s)
    (index : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (Tensor.pool (.avg true) x dims kernel stride padLow
      padHigh dilation ceilMode index)

/-- Average pooling that counts only input cells divides by the count `poolCount false`, which is
zero for a window over padding only, so the bound also asks `u * |n| < |n|` of the count `n`, which
fails at zero, as the division by the cast count does. -/
roundoff mixed theorem avg_pool_exclude_pad_error :=
  fun [Storage Value] {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool)
    (x : Tensor Value s)
    (index : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (Tensor.pool (.avg false) x dims kernel stride padLow
      padHigh dilation ceilMode index)

/-- Max pooling over any axes returns the input entry at the cell `poolIndices` selects, or zero
for a window with no input cell, so it is exact. -/
roundoff theorem max_pool_exact :=
  fun [Storage Value] {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool)
    (x : Tensor Value s)
    (index : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal
      (Tensor.pool .max x dims kernel stride padLow padHigh dilation ceilMode index)

/-- Smooth max pooling over any axes has a mixed bound against its log-sum-exp in real arithmetic:
the pivot plus the logarithm of the sum of the shifted exponentials, times `1 / beta`. -/
roundoff mixed theorem smooth_max_pool_error :=
  fun [Storage Value] {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) (beta : Value)
    (x : Tensor Value s)
    (index : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal
      (Tensor.smoothMaxPool beta x dims kernel stride padLow padHigh dilation ceilMode index)

/-- The average-pooling VJP over any axes adds each output gradient, divided by the count of its
window, into the input cells of its window. At one input coordinate only the windows that contain
it act, and the error there is absolute. -/
roundoff mixed theorem avg_pool_backward_error :=
  fun [Storage Value] {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode countIncludePad : Bool)
    (x : Tensor Value s)
    (g : Tensor Value (s.pool dims kernel stride padLow padHigh dilation ceilMode))
    (index : s.Coord) (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal (Tensor.poolBackward (.avg countIncludePad) x dims kernel stride padLow
      padHigh dilation ceilMode g index)

/-! ## Convolution on any layout -/

/-- Convolution with dimension numbers, input and kernel dilation, feature and batch groups and
asymmetric zero padding, at any number of spatial axes: the fold over the kernel's input features
and window keeps its accumulated mixed error. -/
roundoff mixed theorem conv_error :=
  fun [Storage Value] {s t : Shape} {r : Nat} (dims : Shape.ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (groups batchGroups : Nat)
    (x : Tensor Value s) (w : Tensor Value t)
    (index : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord)
    (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal
      (Tensor.conv x w dims stride padLow padHigh inputDilation dilation groups batchGroups index)

end TorchLean.Roundoff
