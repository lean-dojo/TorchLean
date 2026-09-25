/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# Convolution Specifications

This file defines channels-first convolution and transpose convolution over an arbitrary spatial
rank. The core specification handles one sample; batched interfaces map it over their leading
dimensions.

PyTorch analogy: the grouped, dilated core corresponds to `torch.nn.Conv{d}d` with:

- arbitrary positive `groups`,
- per-axis `dilation` and `stride`,
- independent zero padding before and after each spatial axis,
- and the usual output-size formula (floor division, like PyTorch):

For each axis `a : Fin d`:

`out[a] = (in[a] + padBefore[a] + padAfter[a] - (dilation[a]*(kernel[a]-1) + 1)) / stride[a] + 1`

(`Shape.slidingWindowOutDimDilated`), and `0` when the padded input is shorter than the dilated
kernel extent. Unlike PyTorch, a dilation of `0` is accepted and gives kernel extent `1`.

The weight tensor has shape `(outC × inC × kernel[0] × ... × kernel[d-1])` and the bias has
shape `(outC)`.

Implementation notes:

- Convolution uses natural nested loops (outer axes first) and one `foldl` accumulator. This fixes
  an explicit evaluation order for executable scalar models.
- Padding semantics are implemented via `getAtOrZero` plus an explicit guard for the
  left/top/front padding region (to avoid negative indices, which `Nat` cannot represent).
-/

@[expose] public section

open TorchLean

namespace Spec
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-! ## Index helpers -/

namespace Conv
namespace Internal

/-- Fold over every coordinate of the rectangular index box `dims`.

Convolution sums over a kernel whose rank is a variable, so the sum is a fold over a list of extents
rather than nested `Fin` loops. Pooling windows use the same fold. Coordinates are visited in
row-major order. -/
def foldlIndices {β : Type} (dims : List Nat) (init : β) (f : β → List Nat → β) : β :=
  match dims with
  | [] => f init []
  | n :: ns =>
      (List.finRange n).foldl
        (fun acc i => foldlIndices ns acc (fun acc is => f acc (i.val :: is))) init

/--
Given:
- an output index tuple `outIdx`,
- a kernel index tuple `kIdx`,
- per-axis `stride` and `padding`,
compute the corresponding *input* index tuple (into the unpadded input),
or return `none` if we are in the left/top/front padding region on some axis.

Right/bottom/back padding is handled by `getAtOrZero` when the computed index is out of bounds.
-/
def mkInputIdx?
    (outIdx kIdx stride padding : List Nat) : Option (List Nat) :=
  match outIdx, kIdx, stride, padding with
  | [], [], [], [] => some []
  | o :: os, k :: ks, s :: ss, p :: ps =>
      let q := o * s + k
      if _h : q < p then
        none
      else
        match mkInputIdx? os ks ss ps with
        | none => none
        | some rest => some ((q - p) :: rest)
  | _, _, _, _ => none

/-- Convolution input-index map with per-axis dilation. -/
def mkDilatedInputIdx?
    (outIdx kIdx stride dilation paddingBefore : List Nat) : Option (List Nat) :=
  match outIdx, kIdx, stride, dilation, paddingBefore with
  | [], [], [], [], [] => some []
  | o :: os, k :: ks, s :: ss, d :: ds, p :: ps =>
      let q := o * s + k * d
      if _h : q < p then
        none
      else
        match mkDilatedInputIdx? os ks ss ds ps with
        | none => none
        | some rest => some ((q - p) :: rest)
  | _, _, _, _, _ => none

/-- Unit dilation reduces the dilated convolution index map to the dense index map. -/
theorem mkDilatedInputIdx?_replicate_one
    (outIdx kernelIdx stride padding : List Nat) :
    mkDilatedInputIdx? outIdx kernelIdx stride (List.replicate stride.length 1) padding =
      mkInputIdx? outIdx kernelIdx stride padding := by
  induction outIdx generalizing kernelIdx stride padding with
  | nil =>
      cases kernelIdx <;> cases stride <;> cases padding <;>
        simp [mkDilatedInputIdx?, mkInputIdx?]
  | cons out outIdx ih =>
      cases kernelIdx with
      | nil => cases stride <;> cases padding <;> simp [mkDilatedInputIdx?, mkInputIdx?]
      | cons kernel kernelIdx =>
          cases stride with
          | nil => cases padding <;> simp [mkDilatedInputIdx?, mkInputIdx?]
          | cons step stride =>
              cases padding with
              | nil => simp [mkDilatedInputIdx?, mkInputIdx?]
              | cons pad padding =>
                  simp only [List.length_cons, List.replicate_succ, mkDilatedInputIdx?,
                    mkInputIdx?, mul_one]
                  split <;> simp_all

/--
Given:
- an output index tuple `outIdx`,
- a kernel index tuple `kIdx`,
- per-axis `stride` and `padding`,
compute the corresponding *input* index tuple for transpose convolution, or `none` if
the equality `out + padding = in * stride + k` cannot be satisfied on some axis.

Implementation detail: for each axis we solve

`in = (out + padding - k) / stride`

and require divisibility (`% stride = 0`) plus `out + padding ≥ k`.
Out-of-bounds input indices are handled by `getAtOrZero` at the call site.
-/
def mkTransposeInputIdx?
    (outIdx kIdx stride padding : List Nat) : Option (List Nat) :=
  match outIdx, kIdx, stride, padding with
  | [], [], [], [] => some []
  | o :: os, k :: ks, s :: ss, p :: ps =>
      if s = 0 then
        none
      else
        let q := o + p
        if _h : q < k then
          none
        else
          let r := q - k
          if _hs : r % s = 0 then
            match mkTransposeInputIdx? os ks ss ps with
            | none => none
            | some rest => some ((r / s) :: rest)
          else
            none
  | _, _, _, _ => none

end Internal
end Conv

/-! ## Spec definition -/

/-- Parameters for an arbitrary-rank dense convolution, in channels-first layout. -/
structure ConvSpec (d inC outC : Nat) (kernel stride padding : Tensor Nat [d])
    (α : Type) [TorchLean.Storage α] where
  /-- Kernel weights, shape `(outC, inC, kernel[0], ..., kernel[d-1])`. -/
  kernel : Tensor α (Shape.ofList (outC :: inC :: kernel.data.toList))
  /-- Bias, shape `(outC)`. -/
  bias   : Tensor α [outC]

/-- Output spatial sizes for grouped/dilated convolution with asymmetric zero padding. -/
def convOutSpatialDilated {d : Nat}
    (inSpatial kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]) :
    Tensor Nat [d] :=
  Tensor.ofFn fun axis =>
    Shape.slidingWindowOutDimDilated (inSpatial.getScalar axis) (kernel.getScalar axis)
      (stride.getScalar axis) (dilation.getScalar axis) (paddingBefore.getScalar axis)
      (paddingAfter.getScalar axis)

/-- Output spatial sizes, one extent for each spatial axis. -/
def convOutSpatial {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) : Tensor Nat [d] :=
  Tensor.ofFn (fun a =>
    Shape.slidingWindowOutDim
      (inSpatial.getScalar a) (kernel.getScalar a) (stride.getScalar a) (padding.getScalar a))

/-- Dilated convolution geometry reduces to the symmetric, unit-dilation case. -/
@[simp]
theorem convOutSpatialDilated_one_symmetric {d : Nat}
    (input kernel stride padding : Tensor Nat [d]) :
    convOutSpatialDilated input kernel stride (Tensor.full [d] 1) padding padding =
      convOutSpatial input kernel stride padding := by
  apply Tensor.ext_vector
  intro i
  simp only [convOutSpatialDilated, convOutSpatial, Tensor.getScalar_ofFn]
  by_cases hk : kernel.getScalar i = 0
  · simp [Shape.slidingWindowOutDimDilated,
      Shape.dilatedKernelExtent, Shape.slidingWindowOutDim, hk]
  · have hkpos : 1 ≤ kernel.getScalar i := Nat.one_le_iff_ne_zero.mpr hk
    simp [Shape.slidingWindowOutDimDilated,
      Shape.dilatedKernelExtent, Shape.slidingWindowOutDim, hk,
      Nat.sub_add_cancel hkpos]
    grind

/-- A unit kernel with unit stride and no padding preserves every spatial extent. -/
theorem convOutSpatial_unit {d : Nat} (spatial : Tensor Nat [d]) :
    convOutSpatial spatial (Tensor.full [d] 1) (Tensor.full [d] 1)
      (Tensor.full [d] 0) = spatial := by
  apply Tensor.ext_vector
  intro i
  by_cases hSpatial : spatial.getScalar i = 0
  · simp [convOutSpatial, Shape.slidingWindowOutDim, hSpatial]
  · have hPos : 1 ≤ spatial.getScalar i :=
      Nat.one_le_iff_ne_zero.mpr hSpatial
    simpa [convOutSpatial, Shape.slidingWindowOutDim, hSpatial,
      Nat.not_lt.mpr hPos] using Nat.sub_add_cancel hPos

/--
Unit-stride convolution with an odd kernel and padding equal to the kernel radius preserves every
spatial extent.
-/
theorem convOutSpatial_same {d : Nat} (spatial radius : Tensor Nat [d]) :
    convOutSpatial spatial (radius.map fun p => 2 * p + 1) (Tensor.full [d] 1) radius =
      spatial := by
  apply Tensor.ext_vector
  intro i
  by_cases hSpatial : spatial.getScalar i = 0
  · simp [convOutSpatial, Shape.slidingWindowOutDim, hSpatial]
  · simp [convOutSpatial, Shape.slidingWindowOutDim]
    grind

/-- Output spatial shape `Shape.ofList [out0, ..., out(d-1)]`. -/
def convOutShape {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) : Shape :=
  Shape.ofList (convOutSpatial inSpatial kernel stride padding).data.toList

/-- The grouped bilinear contraction shared by arbitrary-rank convolutions. -/
def Conv.Internal.convCoreWith
    {d inC outC : Nat} {kernel inSpatial outSpatial : Tensor Nat [d]}
    (channelsPerOutput : Nat)
    (inputChannel : Fin outC → Fin channelsPerOutput → Nat)
    (inputIndex? : List Nat → List Nat → Option (List Nat))
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.data.toList)))
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor α (Shape.ofList (outC :: outSpatial.data.toList)) :=
  Tensor.dim fun outChannel =>
    Tensor.generate outSpatial.data.toList fun outIndex =>
      (List.finRange channelsPerOutput).foldl (fun acc localChannel =>
        let inChannel := inputChannel outChannel localChannel
        Conv.Internal.foldlIndices kernel.data.toList acc fun acc kernelIndex =>
          let inputValue : α :=
            match inputIndex? outIndex kernelIndex with
            | none => 0
            | some inputIndex => getAtOrZero input (inChannel :: inputIndex)
          let kernelValue : α :=
            getAtOrZero weights (outChannel.val :: inChannel :: kernelIndex)
          acc + inputValue * kernelValue) 0

/--
The grouped, dilated contraction for an arbitrary-rank convolution, with weights in the
block-diagonal dense layout `(outC, inC, k...)`.

Output channel `oc` belongs to group `oc / (outC / groups)` and reads only the input channels of
that group, so only the block-diagonal part of `weights` is ever consulted. PyTorch stores grouped
weights packed as `(outC, inC / groups, k...)`; use `groupedConvPackedCoreSpec` for that layout,
or `groupedConvDenseWeights` to expand a packed tensor into this one.

The definition is total. It is meaningful only when `groups ∣ inC` and `groups ∣ outC`; otherwise
the trailing `inC % groups` input channels are never read. `groups = 0` reads no channels at all.
-/
def groupedConvCoreSpec
    {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (groups : Nat)
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.data.toList)))
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor α (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation paddingBefore
        paddingAfter).data.toList)) :=

  let inChannelsPerGroup := inC / groups
  let outChannelsPerGroup := outC / groups
  Conv.Internal.convCoreWith inChannelsPerGroup
    (fun outChannel localChannel =>
      (outChannel.val / outChannelsPerGroup) * inChannelsPerGroup + localChannel.val)
    (fun outIndex kernelIndex =>
      Conv.Internal.mkDilatedInputIdx? outIndex kernelIndex stride.data.toList dilation.data.toList
        paddingBefore.data.toList)
    weights input

/--
The grouped bilinear contraction with weights packed per group, so that the channel axis of
`weights` is indexed by the position of an input channel inside its group rather than by the
global input channel.
-/
def Conv.Internal.groupedConvCoreWith
    {d inC outC weightC : Nat} {kernel inSpatial outSpatial : Tensor Nat [d]}
    (channelsPerOutput : Nat)
    (inputChannel : Fin outC → Fin channelsPerOutput → Nat)
    (inputIndex? : List Nat → List Nat → Option (List Nat))
    (weights : Tensor α (Shape.ofList (outC :: weightC :: kernel.data.toList)))
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor α (Shape.ofList (outC :: outSpatial.data.toList)) :=
  Tensor.dim fun outChannel =>
    Tensor.generate outSpatial.data.toList fun outIndex =>
      (List.finRange channelsPerOutput).foldl (fun acc localChannel =>
        let inChannel := inputChannel outChannel localChannel
        Conv.Internal.foldlIndices kernel.data.toList acc fun acc kernelIndex =>
          let inputValue : α :=
            match inputIndex? outIndex kernelIndex with
            | none => 0
            | some inputIndex => getAtOrZero input (inChannel :: inputIndex)
          let kernelValue : α :=
            getAtOrZero weights (outChannel.val :: localChannel.val :: kernelIndex)
          acc + inputValue * kernelValue) 0

/--
The grouped, dilated contraction with weights in PyTorch's packed layout
`(outC, inC / groups, k...)`, which is how `torch.nn.Conv{1,2,3}d(groups=g).weight` is stored.

Output channel `oc` belongs to group `g = oc / (outC / groups)` and reads input channels
`g * (inC / groups) + j` for `j < inC / groups`, weighting each by `weights[oc, j, k...]`.

The definition is total but only meaningful when `groups ∣ inC` and `groups ∣ outC`. When the
divisibility fails the trailing `inC % groups` input channels are never read, and `groups = 0`
reads no channels at all (the output is then the bias alone once it is added).
-/
def groupedConvPackedCoreSpec
    {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (groups : Nat)
    (weights : Tensor α (Shape.ofList (outC :: inC / groups :: kernel.data.toList)))
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor α (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation paddingBefore
        paddingAfter).data.toList)) :=

  let inChannelsPerGroup := inC / groups
  let outChannelsPerGroup := outC / groups
  Conv.Internal.groupedConvCoreWith inChannelsPerGroup
    (fun outChannel localChannel =>
      (outChannel.val / outChannelsPerGroup) * inChannelsPerGroup + localChannel.val)
    (fun outIndex kernelIndex =>
      Conv.Internal.mkDilatedInputIdx? outIndex kernelIndex stride.data.toList dilation.data.toList
        paddingBefore.data.toList)
    weights input

/--
Expand packed grouped weights `(outC, inC / groups, k...)` into the block-diagonal dense layout
`(outC, inC, k...)` read by `groupedConvCoreSpec` and `groupedConvSpec`.

Entry `[oc, ic, k...]` is `weights[oc, ic - g * (inC / groups), k...]` when input channel `ic`
lies in the group `g` of output channel `oc`, and `0` otherwise. This is the remap a PyTorch
checkpoint needs before it can be fed to the dense-layout grouped convolution.
-/
def groupedConvDenseWeights
    {d inC outC : Nat} {kernel : Tensor Nat [d]}
    (groups : Nat)
    (weights : Tensor α (Shape.ofList (outC :: inC / groups :: kernel.data.toList))) :
    Tensor α (Shape.ofList (outC :: inC :: kernel.data.toList)) :=
  let inChannelsPerGroup := inC / groups
  let outChannelsPerGroup := outC / groups
  Tensor.dim fun outChannel =>
    Tensor.dim fun inChannel =>
      Tensor.generate kernel.data.toList fun kernelIndex =>
        let group := outChannel.val / outChannelsPerGroup
        if inChannel.val / inChannelsPerGroup = group then
          getAtOrZero weights
            (outChannel.val :: (inChannel.val - group * inChannelsPerGroup) :: kernelIndex)
        else
          0

/-- The bilinear kernel/input contraction underlying an arbitrary-rank dense convolution. -/
def convCoreSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.data.toList)))
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor α
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).data.toList)) :=
  Conv.Internal.convCoreWith inC (fun _ inChannel => inChannel.val)
    (fun outIndex kernelIndex =>
      Conv.Internal.mkInputIdx? outIndex kernelIndex stride.data.toList padding.data.toList)
    weights input

/--
The grouped convolution contraction reduces to dense convolution for one group, unit dilation,
and symmetric padding. The explicit cast transports the dilated output shape across the geometric
specialization theorem.
 -/
theorem castShape_groupedConvCoreSpec_one_symmetric
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.data.toList)))
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor.castShape
        (groupedConvCoreSpec (dilation := Tensor.full [d] 1) (paddingBefore := padding)
          (paddingAfter := padding) (stride := stride) 1 weights input)
        (congrArg (fun spatial => Shape.ofList (outC :: spatial.data.toList))
          (convOutSpatialDilated_one_symmetric inSpatial kernel stride padding)) =
      convCoreSpec (stride := stride) (padding := padding) weights input := by
  unfold groupedConvCoreSpec convCoreSpec
  simp only [Nat.div_one]
  have hStrideLength : stride.data.toList.length = d := by
    simp
  have hFillStride : (Tensor.full [d] 1).data.toList =
      List.replicate stride.data.toList.length 1 := by
    rw [hStrideLength]
    unfold Tensor.full Tensor.Internal.Rep.const Tensor.Internal.Rep.data
      Tensor.Internal.Rep.ofFlatFn
    rw [TorchLean.Storage.toArray_ofFn]
    simp
  have hInputIndex :
      (fun outIndex kernelIndex =>
          Conv.Internal.mkDilatedInputIdx? outIndex kernelIndex stride.data.toList
            (Tensor.full [d] 1).data.toList padding.data.toList) =
        fun outIndex kernelIndex =>
          Conv.Internal.mkInputIdx? outIndex kernelIndex stride.data.toList
            padding.data.toList := by
    funext outIndex kernelIndex
    rw [hFillStride]
    exact Conv.Internal.mkDilatedInputIdx?_replicate_one
      outIndex kernelIndex stride.data.toList padding.data.toList
  have hInputChannel :
      (fun (outChannel : Fin outC) (localChannel : Fin inC) =>
          (outChannel.val / outC) * inC + localChannel.val) =
        fun _ localChannel => localChannel.val := by
    funext outChannel localChannel
    simp [Nat.div_eq_of_lt outChannel.isLt]
  apply Tensor.ext_getSpec
  intro index
  simp only [get_spec_castShape]
  rw [convOutSpatialDilated_one_symmetric]
  rw [Nat.div_one inC]
  rw [hInputChannel, hInputIndex]

/-- Broadcast one channel value over a supplied spatial shape. -/
def Conv.Internal.convBiasBroadcastWith
    {d outC : Nat} (outSpatial : Tensor Nat [d]) (bias : Tensor α [outC]) :
    Tensor α (Shape.ofList (outC :: outSpatial.data.toList)) :=
  Tensor.dim fun outChannel =>
    Tensor.generate outSpatial.data.toList fun _ => getAtOrZero bias [outChannel.val]

/-- Broadcast a convolution bias over every output spatial position. -/
def convBiasBroadcastSpec
    {d outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (bias : Tensor α [outC]) :
    Tensor α
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).data.toList)) :=
  Conv.Internal.convBiasBroadcastWith (convOutSpatial inSpatial kernel stride padding) bias

/-- Add a channel bias to a dilated convolution output. -/
def convBiasBroadcastDilatedSpec
    {d outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter inSpatial : Tensor Nat [d]}
    (bias : Tensor α [outC]) :
    Tensor α (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation paddingBefore
        paddingAfter).data.toList)) :=
  Conv.Internal.convBiasBroadcastWith
    (convOutSpatialDilated inSpatial kernel stride dilation paddingBefore paddingAfter) bias

/-- Dilated-output bias broadcasting reduces to dense bias broadcasting in the symmetric case. -/
theorem castShape_convBiasBroadcastDilatedSpec_one_symmetric
    {d outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (bias : Tensor α [outC]) :
    Tensor.castShape
        (convBiasBroadcastDilatedSpec (kernel := kernel) (stride := stride)
          (dilation := Tensor.full [d] 1) (paddingBefore := padding) (paddingAfter := padding)
          (inSpatial := inSpatial) bias)
        (congrArg (fun spatial => Shape.ofList (outC :: spatial.data.toList))
          (convOutSpatialDilated_one_symmetric inSpatial kernel stride padding)) =
      convBiasBroadcastSpec (kernel := kernel) (stride := stride) (padding := padding)
        (inSpatial := inSpatial) bias := by
  apply Tensor.ext_getSpec
  intro index
  simp only [get_spec_castShape]
  unfold convBiasBroadcastDilatedSpec convBiasBroadcastSpec
  rw [convOutSpatialDilated_one_symmetric]

/--
Numerical semantics for grouped, dilated convolution with asymmetric zero padding, with weights in
the block-diagonal dense layout `(outC, inC, k...)`.

This is the entry point used by the IR evaluator and the lowering passes, whose payloads carry a
dense `ConvSpec` kernel. PyTorch checkpoints store grouped weights packed as
`(outC, inC / groups, k...)`; either expand them with `groupedConvDenseWeights` or use
`groupedConvPackedSpec` directly. Both forms assume `groups ∣ inC` and `groups ∣ outC`.
-/
def groupedConvSpec
    {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (groups : Nat)
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.data.toList)))
    (bias : Tensor α [outC])
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor α (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation paddingBefore
        paddingAfter).data.toList)) :=
  addSpec (groupedConvCoreSpec groups weights input)
    (convBiasBroadcastDilatedSpec bias)

/--
Grouped, dilated convolution with asymmetric zero padding and PyTorch's packed weight layout
`(outC, inC / groups, k...)`.

PyTorch analogue: `torch.nn.functional.conv{1,2,3}d(input, weight, bias, stride, padding,
dilation, groups)` on one unbatched sample, with `weight` used as stored. The divisibility
requirement `groups ∣ inC ∧ groups ∣ outC` is documented on `groupedConvPackedCoreSpec`.
-/
def groupedConvPackedSpec
    {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (groups : Nat)
    (weights : Tensor α (Shape.ofList (outC :: inC / groups :: kernel.data.toList)))
    (bias : Tensor α [outC])
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor α (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation paddingBefore
        paddingAfter).data.toList)) :=
  addSpec (groupedConvPackedCoreSpec groups weights input)
    (convBiasBroadcastDilatedSpec bias)

/--
Arbitrary-rank dense convolution on a single channels-first input (no batch dimension).

Mathematically, for output channel `oc` and output spatial index `o : Tensor Nat [d]`:

`y[oc,o] = Σ_{ic, k} x_pad[ic, o*stride + k] * W[oc,ic,k] + b[oc]`

where `k` ranges over the kernel window and `x_pad` is `input` with zero-padding.
-/
def convSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor α
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).data.toList)) :=
  addSpec (convCoreSpec layer.kernel input) (convBiasBroadcastSpec layer.bias)

/-- Grouped convolution reduces to dense convolution for its canonical dense configuration. -/
theorem castShape_groupedConvSpec_one_symmetric
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.data.toList)))
    (bias : Tensor α [outC])
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor.castShape
        (groupedConvSpec (dilation := Tensor.full [d] 1) (paddingBefore := padding)
          (paddingAfter := padding) (stride := stride) 1 weights bias input)
        (congrArg (fun spatial => Shape.ofList (outC :: spatial.data.toList))
          (convOutSpatialDilated_one_symmetric inSpatial kernel stride padding)) =
      convSpec (stride := stride) (padding := padding) { kernel := weights, bias := bias }
        input := by
  unfold groupedConvSpec convSpec
  rw [Tensor.castShape_addSpec]
  rw [castShape_groupedConvCoreSpec_one_symmetric]
  rw [castShape_convBiasBroadcastDilatedSpec_one_symmetric]

/-- Directional derivative formula for convolution in its kernel, bias, and input arguments. -/
def convJvpSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer tangentLayer : ConvSpec d inC outC kernel stride padding α)
    (input tangentInput : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor α
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).data.toList)) :=
  addSpec
    (addSpec (convCoreSpec tangentLayer.kernel input)
      (convCoreSpec layer.kernel tangentInput))
    (convBiasBroadcastSpec tangentLayer.bias)


/-- Gradient of convolution output w.r.t. the kernel weights (given `gradOutput`). -/
def convKernelDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (_layer : ConvSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList)))
    (gradOutput :
      Tensor α
        (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).data.toList)))
    : Tensor α (Shape.ofList (outC :: inC :: kernel.data.toList)) :=

  let outSpatial := convOutSpatial inSpatial kernel stride padding
  let outDims : List Nat := outSpatial.data.toList
  let kDims : List Nat := kernel.data.toList
  let strideDims : List Nat := stride.data.toList
  let padDims : List Nat := padding.data.toList

  Tensor.dim (fun out_ch =>
    Tensor.dim (fun in_ch =>
      Tensor.generate kDims (fun kIdx =>
        let totalSum : α :=
          Conv.Internal.foldlIndices outDims 0 (fun acc outIdx =>
            let inputVal : α :=
              match Conv.Internal.mkInputIdx? outIdx kIdx strideDims padDims with
              | none => 0
              | some inIdx => getAtOrZero input (in_ch.val :: inIdx)
            let gradVal : α :=
              getAtOrZero gradOutput (out_ch.val :: outIdx)
            acc + inputVal * gradVal
          )
        totalSum
      )
    )
  )

/-- Gradient of convolution output w.r.t. the bias (sum over spatial positions). -/
def convBiasDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (_layer : ConvSpec d inC outC kernel stride padding α)
    (_input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList)))
    (gradOutput :
      Tensor α
        (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).data.toList)))
    : Tensor α [outC] :=

  let outSpatial := convOutSpatial inSpatial kernel stride padding
  let outDims : List Nat := outSpatial.data.toList

  Tensor.dim (fun out_ch =>
    let totalSum : α :=
      Conv.Internal.foldlIndices outDims 0 (fun acc outIdx =>
        acc + getAtOrZero gradOutput (out_ch.val :: outIdx)
      )
    Tensor.scalar totalSum
  )


/--
Gradient of convolution output w.r.t. the input (the "input-gradient" / transpose-convolution map).

This is stated once for arbitrary spatial rank `d`; there are no rank-specific variants.
-/
def convInputDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding α)
    (_input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList)))
    (gradOutput :
      Tensor α
        (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).data.toList)))
    : Tensor α (Shape.ofList (inC :: inSpatial.data.toList)) :=

  let kDims : List Nat := kernel.data.toList
  let strideDims : List Nat := stride.data.toList
  let padDims : List Nat := padding.data.toList
  let inDims : List Nat := inSpatial.data.toList

  Tensor.dim (fun in_ch =>
    Tensor.generate inDims (fun inIdx =>
      let totalSum : α :=
        (List.finRange outC).foldl (fun acc out_ch =>
          Conv.Internal.foldlIndices kDims acc (fun acc kIdx =>
            let contrib : α :=
              match Conv.Internal.mkTransposeInputIdx? inIdx kIdx strideDims padDims with
              | none => 0
              | some outIdx =>
                  let gradVal := getAtOrZero gradOutput (out_ch.val :: outIdx)
                  let kernelVal := getAtOrZero layer.kernel (out_ch.val :: in_ch.val :: kIdx)
                  gradVal * kernelVal
            acc + contrib
          )
        ) 0
      totalSum
    )
  )


/-- Named reverse-mode result of a convolution.

The kernel, bias, and input gradients retain their respective tensor shapes. Named fields identify
the three contributions in the backward pass and its adjoint theorem. -/
structure ConvGradients (α : Type) [TorchLean.Storage α]
    (kernelShape biasShape inputShape : Shape) where
  /-- Gradient with respect to the kernel weights. -/
  kernelGradient : Tensor α kernelShape
  /-- Gradient with respect to the per-output-channel bias. -/
  biasGradient : Tensor α biasShape
  /-- Gradient with respect to the layer input. -/
  inputGradient : Tensor α inputShape
deriving Repr

/-- Convolution backward pass: the kernel, bias, and input gradients under their own names. -/
def convBackwardSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList)))
    (gradOutput :
      Tensor α
        (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).data.toList)))
    :
    ConvGradients α (Shape.ofList (outC :: inC :: kernel.data.toList)) [outC]
      (Shape.ofList (inC :: inSpatial.data.toList)) :=
  { kernelGradient := convKernelDerivSpec layer input gradOutput
    biasGradient := convBiasDerivSpec layer input gradOutput
    inputGradient := convInputDerivSpec layer input gradOutput }

/-! ## Transpose convolution -/

/--
Parameters for an arbitrary-rank transpose convolution, in channels-first layout.

PyTorch analogy: this is `torch.nn.ConvTranspose{d}d` with:

- `output_padding = 0`,
- `dilation = 1`,
- `groups = 1`,
- per-axis `stride` and `padding`,
- and weight layout `(inC, outC, k0, ..., k(d-1))`.
-/
structure ConvTransposeSpec (d inC outC : Nat) (kernel stride padding : Tensor Nat [d])
    (α : Type) [TorchLean.Storage α] where
  /-- Kernel weights, shape `(inC, outC, kernel[0], ..., kernel[d-1])`. -/
  kernel : Tensor α (Shape.ofList (inC :: outC :: kernel.data.toList))
  /-- Bias, shape `(outC)`. -/
  bias   : Tensor α [outC]

/--
Output size along one transpose-convolution axis with `output_padding = 0`.

For positive input, kernel, and stride this is
`(input - 1) * stride + kernel - 2 * padding`. A zero input, kernel, or stride is treated as an
invalid axis and has size zero; excessive padding also saturates the final subtraction at zero.
The addition precedes subtraction intentionally: Nat subtraction in
`(input - 1) * stride - 2 * padding + kernel` does not represent the integer formula.
-/
def convTransposeOutDim (inDim kDim stride padding : Nat) : Nat :=
  if inDim = 0 || kDim = 0 || stride = 0 then
    0
  else
    (inDim - 1) * stride + kDim - 2 * padding

/-- Output spatial sizes (`Tensor Nat [d]`) for transpose convolution (`output_padding = 0`). -/
def convTransposeOutSpatial {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) :
    Tensor Nat [d] :=
  Tensor.ofFn (fun a =>
    convTransposeOutDim (inSpatial.getScalar a) (kernel.getScalar a) (stride.getScalar a)
      (padding.getScalar a))

/--
Arbitrary-rank transpose convolution on a single channels-first input (no batch dimension).

 For output channel `oc` and output spatial index `o : Tensor Nat [d]` we define:

`y[oc,o] = Σ_{ic, k} x[ic, (o + padding - k) / stride] * W[ic,oc,k] + b[oc]`

where each axis must satisfy `out + padding ≥ k` and divisibility by `stride` (`% stride = 0`).
-/
def convTransposeSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : ConvTransposeSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList))) :
    Tensor α
      (Shape.ofList (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).data.toList))
    :=

  let outSpatial := convTransposeOutSpatial inSpatial kernel stride padding
  let outDims : List Nat := outSpatial.data.toList
  let kDims : List Nat := kernel.data.toList
  let strideDims : List Nat := stride.data.toList
  let padDims : List Nat := padding.data.toList

  Tensor.dim (fun out_ch =>
    Tensor.generate outDims (fun outIdx =>
      let totalSum : α :=
        (List.finRange inC).foldl (fun acc in_ch =>
          Conv.Internal.foldlIndices kDims acc (fun acc kIdx =>
            let inputVal : α :=
              match Conv.Internal.mkTransposeInputIdx? outIdx kIdx strideDims padDims with
              | none => 0
              | some inIdx => getAtOrZero input (in_ch.val :: inIdx)
            let kernelVal : α :=
              getAtOrZero layer.kernel (in_ch.val :: out_ch.val :: kIdx)
            acc + inputVal * kernelVal
          )
        ) 0
      totalSum + getAtOrZero layer.bias [out_ch.val]
    )
  )


/-- Gradient of transpose convolution output w.r.t. the kernel weights (given `gradOutput`). -/
def convTransposeKernelDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList)))
    (gradOutput :
      Tensor α
        (Shape.ofList
          (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).data.toList)))
    : Tensor α (Shape.ofList (inC :: outC :: kernel.data.toList)) :=

  let inDims : List Nat := inSpatial.data.toList
  let kDims : List Nat := kernel.data.toList
  let strideDims : List Nat := stride.data.toList
  let padDims : List Nat := padding.data.toList

  Tensor.dim (fun in_ch =>
    Tensor.dim (fun out_ch =>
      Tensor.generate kDims (fun kIdx =>
        let totalSum : α :=
          Conv.Internal.foldlIndices inDims 0 (fun acc inIdx =>
            match Conv.Internal.mkInputIdx? inIdx kIdx strideDims padDims with
            | none => acc
            | some outIdx =>
                let x : α := getAtOrZero input (in_ch.val :: inIdx)
                let g : α := getAtOrZero gradOutput (out_ch.val :: outIdx)
                acc + x * g
          )
        totalSum
      )
    )
  )


/-- Gradient of transpose convolution output w.r.t. the bias (sum over spatial positions). -/
def convTransposeBiasDerivSpec
    {d outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (gradOutput :
      Tensor α
        (Shape.ofList
          (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).data.toList)))
    : Tensor α [outC] :=

  let outSpatial := convTransposeOutSpatial inSpatial kernel stride padding
  let outDims : List Nat := outSpatial.data.toList

  Tensor.dim (fun out_ch =>
    let totalSum : α :=
      Conv.Internal.foldlIndices outDims 0 (fun acc outIdx =>
        acc + getAtOrZero gradOutput (out_ch.val :: outIdx)
      )
    Tensor.scalar totalSum
  )


/-- Gradient of transpose convolution output w.r.t. the input (given `gradOutput`). -/
def convTransposeInputDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (weights : Tensor α (Shape.ofList (inC :: outC :: kernel.data.toList)))
    (gradOutput :
      Tensor α
        (Shape.ofList
          (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).data.toList)))
    : Tensor α (Shape.ofList (inC :: inSpatial.data.toList)) :=

  let inDims : List Nat := inSpatial.data.toList
  let kDims : List Nat := kernel.data.toList
  let strideDims : List Nat := stride.data.toList
  let padDims : List Nat := padding.data.toList

  Tensor.dim (fun in_ch =>
    Tensor.generate inDims (fun inIdx =>
      let totalSum : α :=
        (List.finRange outC).foldl (fun acc out_ch =>
          Conv.Internal.foldlIndices kDims acc (fun acc kIdx =>
            match Conv.Internal.mkInputIdx? inIdx kIdx strideDims padDims with
            | none => acc
            | some outIdx =>
                let w : α := getAtOrZero weights (in_ch.val :: out_ch.val :: kIdx)
                let g : α := getAtOrZero gradOutput (out_ch.val :: outIdx)
                acc + w * g
          )
        ) 0
      totalSum
    )
  )


/-- Transpose convolution backward pass, reported with the same named fields as the forward
convolution's `ConvGradients`. Only the kernel layout differs: transpose weights are stored
`(inC, outC, ...)`. -/
def convTransposeBackwardSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : ConvTransposeSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.data.toList)))
    (gradOutput :
      Tensor α
        (Shape.ofList
          (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).data.toList)))
    :
    ConvGradients α (Shape.ofList (inC :: outC :: kernel.data.toList)) [outC]
      (Shape.ofList (inC :: inSpatial.data.toList)) :=
  { kernelGradient := convTransposeKernelDerivSpec input gradOutput
    biasGradient := convTransposeBiasDerivSpec gradOutput
    inputGradient := convTransposeInputDerivSpec layer.kernel gradOutput }

end Spec
