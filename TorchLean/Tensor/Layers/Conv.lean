/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Arithmetic
public import TorchLean.Tensor.Window
public import TorchLean.Tensor.Rep.Traversal

/-!
# Convolution Specifications on a channels-first sample

The general convolution operators are `TorchLean.Tensor.conv`, `convTranspose` and their gradients
`convBackwardInput`, `convBackwardWeight`, `convTransposeBackwardInput` and
`convTransposeBackwardWeight` (`TorchLean.Tensor.Window`), on any layout and any
number of spatial axes. This module defines the channels-first, unbatched convolution that the IR
evaluator, the CROWN convolution matrices, the convolution derivative proofs and the lowering
proofs are stated for:

- `grouped`: grouped, dilated convolution with asymmetric zero padding and block-diagonal
  dense weights `(outC × inC × kernel[0] × ... × kernel[d-1])`, the IR payload layout.
- `forward`: its dense, undilated, symmetric case, with its JVP and its kernel, bias and input
  gradients. `TorchLean.Tensor.Layers.Conv.Window` proves that its contraction
  `contract` is `TorchLean.Tensor.conv` of the input with a batch axis of one, and that its
  input and kernel gradients are `convBackwardInput` and `convBackwardWeight`.

`Window` is the kernel, stride and symmetric padding that convolution, transpose convolution and
pooling layers share. Its output grids are `Window.convOutputSpatial` and
`Window.transposedOutputSpatial` here and `Window.poolOutputSpatial` in
`TorchLean.Tensor.Layers.Pooling`.

For each axis `a : Fin d`:

`out[a] = (in[a] + padBefore[a] + padAfter[a] - (dilation[a]*(kernel[a]-1) + 1)) / stride[a] + 1`

(`Shape.slidingWindowOutDimDilated`), and `0` when the stride is zero, the kernel is empty or the
padded input is shorter than the dilated kernel extent. A dilation of `0` gives kernel extent `1`.

`grouped` and `forward` compute the cross-correlation of `torch.nn.Conv2d` (PyTorch 2.8 docs),
with no kernel flip: at kernel offset `k` output `o` reads input
`o * stride + k * dilation - padBefore`. We did this by hand from that page, on `d` axes. With `p`
on both sides, the extent above is the page's
`⌊(H + 2p - dilation (k - 1) - 1) / stride + 1⌋`. `ConvTranspose.outSpatial` is the size from
`torch.nn.ConvTranspose2d` at `output_padding = 0` and `dilation = 1`, `(H - 1) * stride - 2p + k`,
cut at `0` by `Nat` subtraction. `Conv.mapLeading_grouped_eq_conv` puts `grouped` on a batch axis
as the forward value of `Op.conv` when `groups` divides `outC`. In
`scripts/checks/test_ir_capture.py`, `test_convolutions_execute_with_their_payload` of
`PyTorchCaptureTests` compares `Op.conv` with PyTorch on integer data, where every summation order
gives the same sum.

Implementation notes:

- The forward contraction folds one accumulator over the input channels and then the kernel
  coordinates in row-major order (outer axes first), which fixes the evaluation order for
  executable scalar models.
- Padding reads go through `getAtOrZero`, and the index maps return `none` in the leading
  (left/top/front) padding, where the input index would be negative and `Nat` cannot represent it.
-/

@[expose] public section

namespace TorchLean
open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-! ## Index helpers -/

namespace Conv

/-- Fold over every coordinate of the rectangular index box `dims`, in row-major order.

The kernel rank of a convolution is a variable, so the convolution sum is a fold over a list of
extents. -/
def foldlIndices {β : Type} (dims : List Nat) (init : β) (f : β → List Nat → β) : β :=
  match dims with
  | [] => f init []
  | n :: ns =>
      (List.finRange n).foldl
        (fun acc i => foldlIndices ns acc (fun acc is => f acc (i.val :: is))) init

/-- `foldlIndices` folds over the coordinates of the index box in row-major order, each passed as
its list of indices. -/
theorem foldlIndices_eq_foldl {β : Type} :
    (dims : List Nat) → (init : β) → (f : β → List Nat → β) →
      foldlIndices dims init f =
        (Tensor.Rep.coordinates dims).foldl
          (fun acc coordinate => f acc (Shape.Coord.toList dims coordinate)) init
  | [], init, f => by
      rw [Tensor.Rep.foldl_coordinates_nil, foldlIndices]
      rfl
  | n :: dims, init, f => by
      rw [foldlIndices, Tensor.Rep.foldl_coordinates_cons]
      congr 1
      funext acc index
      rw [foldlIndices_eq_foldl dims]
      rfl

/--
The input index (into the unpadded input) that output index `outIdx` reads at kernel offset `kIdx`
with per-axis `stride` and `padding`: `o * s + k - p` on each axis, for output coordinate `o`,
kernel coordinate `k`, stride `s` and padding `p`. The result is `none` when some axis falls in
the leading (left/top/front) padding.

Trailing (right/bottom/back) padding is left to `getAtOrZero`, which reads zero out of bounds.
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

/-- `mkInputIdx?` with per-axis dilation: the kernel offset is scaled, so each axis reads
`o * stride + k * dilation - paddingBefore`, or `none` in the leading padding. -/
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
theorem mkDilatedInputIdx?_one
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
The input index that output index `outIdx` reads at kernel offset `kIdx` in a transpose
convolution with per-axis `stride` and `padding`: on each axis the solution `in` of
`out + padding = in * stride + k`.

For each axis we take `in = (out + padding - k) / stride`, which needs a nonzero stride,
`out + padding ≥ k` and divisibility (`% stride = 0`). If some axis fails, the result is `none`.
The call site reads out-of-bounds input indices through `getAtOrZero`.
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

end Conv

/-! ## Spec definition -/

/-- Parameters for an arbitrary-rank dense convolution, in channels-first layout. -/
structure Conv (d inC outC : Nat) (kernel stride padding : Tensor Nat [d])
    (α : Type) [Storage α] where
  /-- Kernel weights, shape `(outC, inC, kernel[0], ..., kernel[d-1])`. -/
  kernel : Tensor α (outC :: inC :: kernel.data.toList)
  /-- Bias, shape `(outC)`. -/
  bias   : Tensor α [outC]

/-- Output spatial sizes for grouped/dilated convolution with asymmetric zero padding. -/
def Conv.outSpatialDilated {d : Nat}
    (inSpatial kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]) :
    Tensor Nat [d] :=
  Tensor.ofFn fun axis =>
    Shape.slidingWindowOutDimDilated (inSpatial.getScalar axis) (kernel.getScalar axis)
      (stride.getScalar axis) (dilation.getScalar axis) (paddingBefore.getScalar axis)
      (paddingAfter.getScalar axis)

/-- Each output extent of a dilated convolution is the dilated sliding-window extent of its axis. -/
theorem Conv.outSpatialDilated_apply {d : Nat}
    (inSpatial kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d])
    (axis : Shape.Coord [d]) :
    outSpatialDilated inSpatial kernel stride dilation paddingBefore paddingAfter axis =
      Shape.slidingWindowOutDimDilated (inSpatial.getScalar axis.1) (kernel.getScalar axis.1)
        (stride.getScalar axis.1) (dilation.getScalar axis.1) (paddingBefore.getScalar axis.1)
        (paddingAfter.getScalar axis.1) :=
  Tensor.ofFn_apply _ axis

/-- The bilinear contraction shared by arbitrary-rank convolutions: each output entry folds from
`0` over the `channelsPerOutput` local channels and then the kernel coordinates in row-major order.

`inputChannel` picks the input channel of a local channel. `weightChannel` picks the matching
weight channel and defaults to `inputChannel`, the index for dense weights, while packed grouped
weights `(outC, inC / groups, k...)` are indexed by the local channel. Both layouts accumulate in
the same order. -/
def Conv.contractWith
    {d inC outC weightC : Nat} {kernel inSpatial outSpatial : Tensor Nat [d]}
    (channelsPerOutput : Nat)
    (inputChannel : Fin outC → Fin channelsPerOutput → Nat)
    (inputIndex? : List Nat → List Nat → Option (List Nat))
    (weights : Tensor α (outC :: weightC :: kernel.data.toList))
    (input : Tensor α (inC :: inSpatial.data.toList))
    (weightChannel : Fin outC → Fin channelsPerOutput → Nat := inputChannel) :
    Tensor α (outC :: outSpatial.data.toList) :=
  Tensor.dim fun outChannel =>
    Tensor.generate outSpatial.data.toList fun outIndex =>
      (List.finRange channelsPerOutput).foldl (fun acc localChannel =>
        let inChannel := inputChannel outChannel localChannel
        Conv.foldlIndices kernel.data.toList acc fun acc kernelIndex =>
          let inputValue : α :=
            match inputIndex? outIndex kernelIndex with
            | none => 0
            | some inputIndex => getAtOrZero input (inChannel :: inputIndex)
          let kernelValue : α :=
            getAtOrZero weights
              (outChannel.val :: weightChannel outChannel localChannel :: kernelIndex)
          acc + inputValue * kernelValue) 0

/--
The grouped, dilated contraction for an arbitrary-rank convolution, with weights in the
block-diagonal dense layout `(outC, inC, k...)`.

Output channel `oc` belongs to group `oc / (outC / groups)` and reads only the input channels of
that group, so only the block-diagonal part of `weights` is ever consulted.

The usual reading of grouped convolution needs `groups > 0`, `groups ∣ inC` and `groups ∣ outC`.
Outside those conditions the definition keeps natural-number division and zero-filled out-of-range
reads, so trailing channels can be read: with `inC = outC = 5` and `groups = 2`, output channel `4`
reads input channel `4`. With `groups = 0` it reads no channels.
-/
def Conv.groupedContract
    {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (groups : Nat)
    (weights : Tensor α (outC :: inC :: kernel.data.toList))
    (input : Tensor α (inC :: inSpatial.data.toList)) :
    Tensor α (outC ::
      (outSpatialDilated inSpatial kernel stride dilation paddingBefore
        paddingAfter).data.toList) :=

  let inChannelsPerGroup := inC / groups
  let outChannelsPerGroup := outC / groups
  Conv.contractWith inChannelsPerGroup
    (fun outChannel localChannel =>
      (outChannel.val / outChannelsPerGroup) * inChannelsPerGroup + localChannel.val)
    (fun outIndex kernelIndex =>
      Conv.mkDilatedInputIdx? outIndex kernelIndex stride.data.toList dilation.data.toList
        paddingBefore.data.toList)
    weights input

/-- The bilinear kernel/input contraction underlying an arbitrary-rank dense convolution. -/
def Conv.contract
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (weights : Tensor α (outC :: inC :: kernel.data.toList))
    (input : Tensor α (inC :: inSpatial.data.toList)) :
    Tensor α
      (outC :: (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding).data.toList) :=
  Conv.contractWith inC (fun _ inChannel => inChannel.val)
    (fun outIndex kernelIndex =>
      Conv.mkInputIdx? outIndex kernelIndex stride.data.toList padding.data.toList)
    weights input

/-- The grouped convolution contraction with one group, unit dilation and equal padding on both
sides is the dense contraction `contract`. -/
theorem Conv.groupedContract_one
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor α (outC :: inC :: kernel.data.toList))
    (input : Tensor α (inC :: inSpatial.data.toList)) :
    groupedContract (dilation := Tensor.ones [d]) (paddingBefore := padding)
        (paddingAfter := padding) (stride := stride) 1 weights input =
      contract (stride := stride) (padding := padding) weights input := by
  unfold groupedContract contract
  simp only [Nat.div_one]
  have hStrideLength : stride.data.toList.length = d := by
    simp
  have hFillStride : (Tensor.ones [d] : Tensor Nat [d]).data.toList =
      List.replicate stride.data.toList.length 1 := by
    rw [hStrideLength]
    unfold Tensor.ones Tensor.full Tensor.Rep.data
      Tensor.Rep.ofFlatFn
    rw [Storage.toArray_ofFn]
    simp
  have hInputIndex :
      (fun outIndex kernelIndex =>
          Conv.mkDilatedInputIdx? outIndex kernelIndex stride.data.toList
            (Tensor.ones [d]).data.toList padding.data.toList) =
        fun outIndex kernelIndex =>
          Conv.mkInputIdx? outIndex kernelIndex stride.data.toList
            padding.data.toList := by
    funext outIndex kernelIndex
    rw [hFillStride]
    exact Conv.mkDilatedInputIdx?_one
      outIndex kernelIndex stride.data.toList padding.data.toList
  have hInputChannel :
      (fun (outChannel : Fin outC) (localChannel : Fin inC) =>
          (outChannel.val / outC) * inC + localChannel.val) =
        fun _ localChannel => localChannel.val := by
    funext outChannel localChannel
    simp [Nat.div_eq_of_lt outChannel.isLt]
  rw [Nat.div_one inC]
  rw [hInputChannel, hInputIndex]
  rw [Nat.div_one outC, hInputChannel]

/-- Broadcast the bias of each channel over the spatial shape `outSpatial`. -/
def Conv.biasBroadcastWith
    {d outC : Nat} (outSpatial : Tensor Nat [d]) (bias : Tensor α [outC]) :
    Tensor α (outC :: outSpatial.data.toList) :=
  Tensor.dim fun outChannel =>
    Tensor.generate outSpatial.data.toList fun _ => getAtOrZero bias [outChannel.val]

/--
Numerical semantics for grouped, dilated convolution with asymmetric zero padding, with weights in
the block-diagonal dense layout `(outC, inC, k...)`: `Conv.groupedContract` plus the broadcast bias.

The CROWN convolution bounds and their proofs are stated for it, and
`TorchLean.CROWN.Op.flat_conv` relates it to `Op.conv` on packed weights. PyTorch checkpoints
store grouped weights packed as `(outC, inC / groups, k...)`, the layout `TorchLean.Tensor.conv`
reads. The grouped interpretation assumes `groups ∣ inC` and `groups ∣ outC`.
-/
def Conv.grouped
    {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (groups : Nat)
    (weights : Tensor α (outC :: inC :: kernel.data.toList))
    (bias : Tensor α [outC])
    (input : Tensor α (inC :: inSpatial.data.toList)) :
    Tensor α (outC ::
      (outSpatialDilated inSpatial kernel stride dilation paddingBefore
        paddingAfter).data.toList) :=
  groupedContract groups weights input +
    Conv.biasBroadcastWith
      (outSpatialDilated inSpatial kernel stride dilation paddingBefore paddingAfter) bias

/--
Arbitrary-rank dense convolution on a single channels-first input (no batch dimension). For output
channel `oc` and output spatial index `o : Tensor Nat [d]`,

`y[oc,o] = Σ_{ic, k} x_pad[ic, o*stride + k] * W[oc,ic,k] + b[oc]`

where `k` ranges over the kernel window and `x_pad` is `input` with zero padding.
-/
def Conv.forward
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (input : Tensor α (inC :: inSpatial.data.toList)) :
    Tensor α
      (outC :: (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding).data.toList) :=
  contract layer.kernel input +
    biasBroadcastWith (outSpatialDilated inSpatial kernel stride (Tensor.ones [d]) padding padding)
      layer.bias

/-- Grouped convolution with one group, unit dilation and equal padding on both sides is the dense
convolution `forward`. -/
theorem Conv.grouped_one
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor α (outC :: inC :: kernel.data.toList))
    (bias : Tensor α [outC])
    (input : Tensor α (inC :: inSpatial.data.toList)) :
    grouped (dilation := Tensor.ones [d]) (paddingBefore := padding)
        (paddingAfter := padding) (stride := stride) 1 weights bias input =
      forward (stride := stride) (padding := padding) { kernel := weights, bias := bias }
        input := by
  unfold grouped forward
  rw [groupedContract_one]

/-- The directional derivative of `forward` along `tangentLayer` and `tangentInput`: the
contraction of the tangent kernel with `input`, plus that of the kernel with `tangentInput`, plus
the broadcast tangent bias. -/
def Conv.jvp
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer tangentLayer : Conv d inC outC kernel stride padding α)
    (input tangentInput : Tensor α (inC :: inSpatial.data.toList)) :
    Tensor α
      (outC :: (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding).data.toList) :=
  (contract tangentLayer.kernel input +
      contract layer.kernel tangentInput) +
    biasBroadcastWith (outSpatialDilated inSpatial kernel stride (Tensor.ones [d]) padding padding)
      tangentLayer.bias


/-- The kernel gradient of `forward` for the cotangent `gradOutput`: entry `[oc, ic, k]` sums,
over the output positions `o`, the input value that offset `k` reads at `o` times
`gradOutput[oc, o]`. -/
def Conv.kernelDeriv
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (_layer : Conv d inC outC kernel stride padding α)
    (input : Tensor α (inC :: inSpatial.data.toList))
    (gradOutput :
      Tensor α
        (outC :: (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList))
    : Tensor α (outC :: inC :: kernel.data.toList) :=

  let outSpatial := outSpatialDilated inSpatial kernel stride (Tensor.ones [d]) padding padding
  let outDims : List Nat := outSpatial.data.toList
  let kDims : List Nat := kernel.data.toList
  let strideDims : List Nat := stride.data.toList
  let padDims : List Nat := padding.data.toList

  Tensor.dim (fun out_ch =>
    Tensor.dim (fun in_ch =>
      Tensor.generate kDims (fun kIdx =>
        let totalSum : α :=
          Conv.foldlIndices outDims 0 (fun acc outIdx =>
            let inputVal : α :=
              match Conv.mkInputIdx? outIdx kIdx strideDims padDims with
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

/-- The bias gradient of `forward`: each channel of `gradOutput` summed over the output
positions. -/
def Conv.biasDeriv
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (_layer : Conv d inC outC kernel stride padding α)
    (_input : Tensor α (inC :: inSpatial.data.toList))
    (gradOutput :
      Tensor α
        (outC :: (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList))
    : Tensor α [outC] :=

  let outSpatial := outSpatialDilated inSpatial kernel stride (Tensor.ones [d]) padding padding
  let outDims : List Nat := outSpatial.data.toList

  Tensor.dim (fun out_ch =>
    let totalSum : α :=
      Conv.foldlIndices outDims 0 (fun acc outIdx =>
        acc + getAtOrZero gradOutput (out_ch.val :: outIdx)
      )
    Tensor.scalar totalSum
  )


/--
The input gradient of `forward`, the transpose-convolution map of `gradOutput`: entry `[ic, i]`
sums `gradOutput[oc, o] * W[oc, ic, k]` over the channels `oc` and the kernel offsets `k` with
`i + padding = o * stride + k`. One definition covers every spatial rank `d`.
-/
def Conv.inputDeriv
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (_input : Tensor α (inC :: inSpatial.data.toList))
    (gradOutput :
      Tensor α
        (outC :: (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList))
    : Tensor α (inC :: inSpatial.data.toList) :=

  let kDims : List Nat := kernel.data.toList
  let strideDims : List Nat := stride.data.toList
  let padDims : List Nat := padding.data.toList
  let inDims : List Nat := inSpatial.data.toList

  Tensor.dim (fun in_ch =>
    Tensor.generate inDims (fun inIdx =>
      let totalSum : α :=
        (List.finRange outC).foldl (fun acc out_ch =>
          Conv.foldlIndices kDims acc (fun acc kIdx =>
            let contrib : α :=
              match Conv.mkTransposeInputIdx? inIdx kIdx strideDims padDims with
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


/-- The kernel, bias and input gradients of a convolution, each in the shape of its argument.
`Conv.jvp_backward_adjoint` pairs each field with the matching
tangent. -/
structure Conv.Gradients (α : Type) [Storage α]
    (kernelShape biasShape inputShape : Shape) where
  /-- Gradient with respect to the kernel weights. -/
  kernelGradient : Tensor α kernelShape
  /-- Gradient with respect to the per-output-channel bias. -/
  biasGradient : Tensor α biasShape
  /-- Gradient with respect to the layer input. -/
  inputGradient : Tensor α inputShape
deriving Repr

/-- Convolution backward pass: the kernel, bias, and input gradients under their own names. -/
def Conv.backward
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (input : Tensor α (inC :: inSpatial.data.toList))
    (gradOutput :
      Tensor α
        (outC :: (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList))
    :
    Gradients α (outC :: inC :: kernel.data.toList) [outC]
      (inC :: inSpatial.data.toList) :=
  { kernelGradient := kernelDeriv layer input gradOutput
    biasGradient := biasDeriv layer input gradOutput
    inputGradient := inputDeriv layer input gradOutput }

/-! ## Transpose convolution geometry -/

/-- Output spatial sizes (`Tensor Nat [d]`) for transpose convolution with symmetric padding, no
output padding and no dilation: `Shape.convTransposeOutDim` along each axis. -/
def ConvTranspose.outSpatial {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) :
    Tensor Nat [d] :=
  Tensor.ofFn (fun a =>
    Shape.convTransposeOutDim (inSpatial.getScalar a) (kernel.getScalar a) (stride.getScalar a)
      (padding.getScalar a) (padding.getScalar a) 0 1)

/-- Each output extent of a transpose convolution is `Shape.convTransposeOutDim` of its axis. -/
theorem ConvTranspose.outSpatial_apply {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d])
    (axis : Shape.Coord [d]) :
    outSpatial inSpatial kernel stride padding axis =
      Shape.convTransposeOutDim (inSpatial.getScalar axis.1) (kernel.getScalar axis.1)
        (stride.getScalar axis.1) (padding.getScalar axis.1) (padding.getScalar axis.1) 0 1 :=
  Tensor.ofFn_apply _ axis

/-! ## Window geometry -/

/-- Kernel extent, stride and symmetric zero padding along each of `d` spatial axes, the geometry
that convolution, transpose convolution and pooling layers share. -/
structure Window (d : Nat) where
  /-- Kernel extent along each spatial axis. -/
  kernelSize : Tensor Nat [d]
  /-- Step along each spatial axis. -/
  stride : Tensor Nat [d] := Tensor.ones [d]
  /-- Zero padding on each side of each spatial axis. -/
  padding : Tensor Nat [d] := Tensor.zeros [d]

namespace Window

/-- Output grid of a convolution with this window, unit dilation and equal padding on both sides
(`Conv.outSpatialDilated`). An axis with an empty kernel, a zero stride or a padded input shorter
than the kernel has extent `0`. -/
def convOutputSpatial {d : Nat} (window : Window d) (input : Tensor Nat [d]) : Tensor Nat [d] :=
  Conv.outSpatialDilated input window.kernelSize window.stride (Tensor.ones [d]) window.padding
    window.padding

/-- Output grid of a transpose convolution with this window, no output padding and unit dilation
(`ConvTranspose.outSpatial`). -/
def transposedOutputSpatial {d : Nat} (window : Window d) (input : Tensor Nat [d]) :
    Tensor Nat [d] :=
  ConvTranspose.outSpatial input window.kernelSize window.stride window.padding

/-- The unit-stride window whose kernel extent is `2 * r + 1` and whose padding is `r` on an axis of
radius `r`. -/
def samePadding {d : Nat} (radius : Tensor Nat [d]) : Window d :=
  { kernelSize := radius.map (fun extent => 2 * extent + 1)
    stride := Tensor.ones [d]
    padding := radius }

/-- A same-padding convolution preserves every input extent, including an empty one. -/
theorem convOutputSpatial_samePadding {d : Nat} (input radius : Tensor Nat [d]) :
    (samePadding radius).convOutputSpatial input = input := by
  apply Tensor.ext_vector
  intro i
  simp only [convOutputSpatial, samePadding, Conv.outSpatialDilated, Tensor.getScalar_ofFn]
  by_cases hInput : input.getScalar i = 0
  · simp [Shape.slidingWindowOutDimDilated, Shape.dilatedKernelExtent, hInput]
    omega
  · simp [Shape.slidingWindowOutDimDilated, Shape.dilatedKernelExtent]
    grind

/-- A convolution with a unit kernel, unit stride and no padding preserves every input extent. -/
theorem convOutputSpatial_unit {d : Nat} (spatial : Tensor Nat [d]) :
    ({ kernelSize := Tensor.ones [d] } : Window d).convOutputSpatial spatial = spatial := by
  apply Tensor.ext_vector
  intro i
  simp only [convOutputSpatial, Conv.outSpatialDilated, Tensor.getScalar_ofFn]
  by_cases hSpatial : spatial.getScalar i = 0
  · simp [Shape.slidingWindowOutDimDilated, Shape.dilatedKernelExtent, hSpatial]
  · have hPos : 1 ≤ spatial.getScalar i := Nat.one_le_iff_ne_zero.mpr hSpatial
    simp [Shape.slidingWindowOutDimDilated, Shape.dilatedKernelExtent, Nat.not_lt.mpr hPos,
      Nat.sub_add_cancel hPos]

end Window

end TorchLean
