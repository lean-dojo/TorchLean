/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.GroupWithZero.Nat
public import NN.Spec.Core.TensorOps

@[expose] public section


open TorchLean

namespace Spec
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-!
# Spatial Pooling

Dimension-polymorphic pooling specs for spatial tensors and channels-first tensors.
-/

/-!
## Arbitrary-rank pooling (channels-first, no batch)

These operators define pooling over an arbitrary spatial rank `d`.

Conventions:
- Input is channels-first: shape `[C] ++ spatialDims`.
- Pooling is applied independently per channel.
- `kernel`, `stride`, and `padding` are per-axis vectors (`Tensor Nat [d]`).
- Padding is symmetric. Average pooling counts padded positions as zeros. Max pooling ignores
  padded positions; a window with no input position is outside PyTorch's valid max-pool domain and
  is totalized to zero by the scalar-polymorphic TorchLean spec.

PyTorch comparisons (conceptual, without batch axis):
- `maxPoolSpec` corresponds to PyTorch's rank-specific `max_pool1d`, `max_pool2d`, and
  `max_pool3d` operations.
- `avgPoolSpec` corresponds to PyTorch's rank-specific `avg_pool1d`, `avg_pool2d`, and
  `avg_pool3d` operations.
-/

/-!
### Layer configs + output shapes
-/

/--
Witness that an arbitrary-rank max-pooling configuration has nonzero kernel and stride on every
axis.

The structure has no fields: it is a phantom type whose indices force every construction site to
supply the nonzero-kernel and nonzero-stride proofs, and it is the value threaded through the
runtime, lowering, and verification code paths. The pooling specs themselves do not eliminate
these proofs. A zero kernel or stride is instead excluded at the shape level: `poolOutDim` returns
an empty output axis for a zero kernel, so no window is ever evaluated, and the `getD 0` fallback
inside `Pooling.Internal.maxPoolValue` is unreachable for the shapes this witness admits.

The tensors are indices of the type rather than duplicate structure fields, so a value cannot
advertise one configuration while its type describes another.
-/
structure MaxPoolSpec (d : Nat)
    (kernel stride padding : Tensor Nat [d])
    (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0)
    (hStride : ∀ i : Fin d, stride.getScalar i ≠ 0) where

/-- Witness that an arbitrary-rank average-pooling configuration has nonzero kernel and stride.
Like `MaxPoolSpec`, it is a field-free phantom type; see that docstring for what the proofs do. -/
structure AvgPoolSpec (d : Nat)
    (kernel stride padding : Tensor Nat [d])
    (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0)
    (hStride : ∀ i : Fin d, stride.getScalar i ≠ 0) where

/--
Output spatial sizes with symmetric padding.

Pooling follows the usual floor-mode sliding-window formula, but an empty input axis, empty kernel,
or padding larger than half the kernel gives an empty output axis. The last condition is part of the
pooling contract used by PyTorch and by TorchLean's native implementations; it is not a restriction
on convolution.
-/
def poolOutDim (input kernel stride padding : Nat) : Nat :=
  if input = 0 || kernel = 0 || padding > kernel / 2 then
    0
  else
    Shape.slidingWindowOutDim input kernel stride padding

/--
Output spatial sizes without padding.

An invalid axis (empty input, zero kernel, zero stride, or a kernel larger than the input) has size
zero.
-/
def poolOutSpatial {d : Nat} (inSpatial kernel stride : Tensor Nat [d]) : Tensor Nat [d] :=
  Tensor.ofFn (fun i =>
    poolOutDim (inSpatial.getScalar i) (kernel.getScalar i) (stride.getScalar i) 0)

/-- Apply `poolOutDim` independently to each spatial axis. -/
def poolOutSpatialPad {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) :
    Tensor Nat [d] :=
  Tensor.ofFn (fun i =>
    poolOutDim (inSpatial.getScalar i) (kernel.getScalar i) (stride.getScalar i)
      (padding.getScalar i))

/-- Pooling over the complete spatial extent produces one value on every spatial axis. -/
theorem poolOutSpatialPad_global {d : Nat} (spatial : Tensor Nat [d])
    (hSpatial : ∀ i : Fin d, spatial.getScalar i ≠ 0) :
    poolOutSpatialPad spatial spatial (Tensor.full [d] 1) (Tensor.full [d] 0) =
      Tensor.full [d] 1 := by
  apply Tensor.ext_vector
  intro i
  have hNonzero : spatial.getScalar i ≠ 0 := hSpatial i
  simp [poolOutSpatialPad, poolOutDim, Shape.slidingWindowOutDim, hNonzero]

/-- Output shape for single-channel arbitrary-rank pooling (no padding). -/
def poolOutShape {d : Nat} (inSpatial kernel stride : Tensor Nat [d]) : Shape :=
  Shape.ofList (poolOutSpatial inSpatial kernel stride).data.toList

/-- Output shape for channels-first arbitrary-rank pooling (no padding; channels preserved). -/
def poolMultiOutShape {d : Nat} (inC : Nat) (inSpatial kernel stride : Tensor Nat [d]) : Shape :=
  Shape.ofList (inC :: (poolOutSpatial inSpatial kernel stride).data.toList)

/-- Output shape for single-channel arbitrary-rank pooling with symmetric padding. -/
def poolOutShapePad {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) : Shape :=
  Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).data.toList

/-- Output shape for channels-first arbitrary-rank pooling with symmetric padding (channels
preserved). -/
def poolMultiOutShapePad {d : Nat} (inC : Nat) (inSpatial kernel stride padding : Tensor Nat [d])
    : Shape :=
  Shape.ofList (inC :: (poolOutSpatialPad inSpatial kernel stride padding).data.toList)

namespace Pooling
namespace Internal

/-- Choose the input-space pivot whose scaled value is maximal. -/
def smoothMaxPivotStep (beta current candidate : α) : α :=
  if beta > 0 then Max.max current candidate else Min.min current candidate

/-- Fold over every coordinate of a rectangular index box given by `dims`.

Pooling walks a window whose rank is only known at runtime, so the walk is a fold over a list of
extents rather than nested `Fin` loops. The accumulator sees each coordinate list once, in row-major
order. -/
def foldlIndices' {β : Type} (dims : List Nat) (init : β) (f : β → List Nat → β) : β :=
  match dims with
  | [] => f init []
  | n :: ns =>
      (List.range n).foldl (fun acc i =>
        foldlIndices' ns acc (fun acc' is => f acc' (i :: is))) init

/-- Coordinate of a window cell in the padded input, or `none` on a rank mismatch.

Per axis this is `output * stride + window`, the position the cell occupies before padding is
removed. -/
def paddedCoords? (outIdxs winIdxs stride : List Nat) : Option (List Nat) :=
  match outIdxs, winIdxs, stride with
  | [], [], [] => some []
  | o :: os, w :: ws, s :: ss =>
      match paddedCoords? os ws ss with
      | some rest => some ((o * s + w) :: rest)
      | none => none
  | _, _, _ => none

/-- Turn a padded-input coordinate back into a real input coordinate.

Returns `none` when the coordinate lands inside the padding, which is how the callers tell a genuine
input cell from a fabricated one; average and max pooling then treat that answer differently. -/
def unpadCoords? (padded padding : List Nat) : Option (List Nat) :=
  match padded, padding with
  | [], [] => some []
  | x :: xs, p :: ps =>
      if _h : x < p then
        none
      else
        match unpadCoords? xs ps with
        | some rest => some ((x - p) :: rest)
        | none => none
  | _, _ => none

/-- Whether a runtime coordinate list is in range for `dims`, and of the right length. -/
def coordsInBounds (idx dims : List Nat) : Bool :=
  match idx, dims with
  | [], [] => true
  | i :: is, d :: ds => decide (i < d) && coordsInBounds is ds
  | _, _ => false

/--
Input lookup for average/smooth pooling.

For average-style pooling, padded cells contribute numeric zero and are still counted by the
denominator chosen by the surrounding pooling spec. We keep this separate from
`getPaddedMaxInputVal?`, where padded cells must be ignored rather than treated as zero.
-/
def getPaddedAverageInputVal
    {d : Nat} {inSpatial : Tensor Nat [d]}
    (input : Tensor α (Shape.ofList inSpatial.data.toList))
    (outIdxs winIdxs : List Nat)
    (stride padding : List Nat) : α :=
  match paddedCoords? outIdxs winIdxs stride with
  | none => 0
  | some padded =>
      match unpadCoords? padded padding with
      | none => 0
      | some orig => getAtOrZero input orig

/--
Input lookup for hard max-pooling.

Unlike average pooling, max pooling does not insert numeric zero for an individual padded cell:
PyTorch's valid max-pool configurations behave as though those cells were `-∞`. TorchLean keeps
the spec scalar-polymorphic by returning `none` for padded coordinates and ignoring them in the
max fold. `poolOutSpatialPad` rejects empty input axes, empty kernels, and padding beyond PyTorch's
half-kernel restriction, so every emitted output window contains at least one input coordinate.
-/
def getPaddedMaxInputVal?
    {d : Nat} {inSpatial : Tensor Nat [d]}
    (input : Tensor α (Shape.ofList inSpatial.data.toList))
    (outIdxs winIdxs : List Nat)
    (stride padding : List Nat) : Option α :=
  match paddedCoords? outIdxs winIdxs stride with
  | none => none
  | some padded =>
      match unpadCoords? padded padding with
      | none => none
      | some orig =>
          if coordsInBounds orig inSpatial.data.toList then
            some (getAtOrZero input orig)
          else
            none

/-- Number of cells in a pooling window, the denominator average pooling counts with. -/
def kernelProd (kernel : List Nat) : Nat :=
  kernel.foldl (fun acc k => acc * k) 1

/-- Start of adaptive-pooling bin `i`: `floor(i * input / output)`. -/
def adaptiveStart (input output i : Nat) : Nat :=
  (i * input) / output

/-- End of adaptive-pooling bin `i`: `ceil((i + 1) * input / output)`. -/
def adaptiveEnd (input output i : Nat) : Nat :=
  ((i + 1) * input + output - 1) / output

/--
Every adaptive-pooling bin is nonempty when the input and output extents are positive.

Both hypotheses are needed: an empty input axis gives `start = end = 0`, and `output = 0` makes
both quotients zero under Lean's total division. This is the fact that justifies the `hInput` and
`hOutput` arguments of `adaptiveAvgPoolSpec` and `adaptiveMaxPoolSpec`: the average never divides
by zero and the max fold always sees at least one element.
-/
theorem adaptiveStart_lt_adaptiveEnd (input output i : Nat)
    (hInput : input ≠ 0) (hOutput : output ≠ 0) :
    adaptiveStart input output i < adaptiveEnd input output i := by
  unfold adaptiveStart adaptiveEnd
  have hle : i * input / output * output ≤ i * input := Nat.div_mul_le_self _ _
  rw [Nat.lt_iff_add_one_le, Nat.le_div_iff_mul_le (Nat.pos_of_ne_zero hOutput), Nat.add_mul,
    Nat.one_mul, Nat.succ_mul]
  generalize i * input = m at *
  generalize m / output * output = q at *
  have hpos : 1 ≤ input := Nat.pos_of_ne_zero hInput
  have hstep : m + output + 1 ≤ m + input + output := by
    rw [Nat.add_right_comm m input output]
    exact Nat.add_le_add_left hpos _
  exact le_trans (Nat.add_le_add_right hle output) (Nat.le_sub_of_add_le hstep)

/-- Per-axis bin starts for one adaptive-pooling output coordinate. -/
def adaptiveStarts (input output index : List Nat) : List Nat :=
  match input, output, index with
  | i :: is, o :: os, x :: xs => adaptiveStart i o x :: adaptiveStarts is os xs
  | _, _, _ => []

/-- Per-axis bin extents for one adaptive-pooling output coordinate.

Adaptive pooling is the case where the window is not fixed: bin widths vary with the coordinate when
the input extent is not a multiple of the output extent, which is exactly why the walk is over a
computed extent list. -/
def adaptiveWindowDims (input output index : List Nat) : List Nat :=
  match input, output, index with
  | i :: is, o :: os, x :: xs =>
      (adaptiveEnd i o x - adaptiveStart i o x) :: adaptiveWindowDims is os xs
  | _, _, _ => []

/-- Add two coordinate lists axiswise, dropping any tail the shorter list does not cover. -/
def addCoords (left right : List Nat) : List Nat :=
  match left, right with
  | x :: xs, y :: ys => (x + y) :: addCoords xs ys
  | _, _ => []

/-- Average of one adaptive-pooling bin.

`adaptiveStart_lt_adaptiveEnd` is what keeps this honest: with positive input and output extents the
bin is nonempty, so the division below is never a division by zero. -/
def adaptiveAvgPoolValue
    {d : Nat} {inSpatial : Tensor Nat [d]}
    (input : Tensor α (Shape.ofList inSpatial.data.toList))
    (outSpatial : Tensor Nat [d]) (outIdxs : List Nat) : α :=
  let starts := adaptiveStarts inSpatial.data.toList outSpatial.data.toList outIdxs
  let window := adaptiveWindowDims inSpatial.data.toList outSpatial.data.toList outIdxs
  let sum := foldlIndices' window (0 : α) (fun acc offset =>
    acc + getAtOrZero input (addCoords starts offset))
  sum / (kernelProd window : Nat)

/-- Maximum over one adaptive-pooling bin.

The `getD 0` fallback would only be reached on an empty bin, which `adaptiveStart_lt_adaptiveEnd`
rules out for the shapes the public specs accept. -/
def adaptiveMaxPoolValue
    {d : Nat} {inSpatial : Tensor Nat [d]}
    (input : Tensor α (Shape.ofList inSpatial.data.toList))
    (outSpatial : Tensor Nat [d]) (outIdxs : List Nat) : α :=
  let starts := adaptiveStarts inSpatial.data.toList outSpatial.data.toList outIdxs
  let window := adaptiveWindowDims inSpatial.data.toList outSpatial.data.toList outIdxs
  let best? := foldlIndices' window none (fun best offset =>
    let value := getAtOrZero input (addCoords starts offset)
    match best with
    | none => some value
    | some current => if value > current then some value else best)
  -- Positive input and output extents make every adaptive window nonempty.
  best?.getD 0

/-- Maximum over one fixed pooling window, ignoring padded cells.

Padded cells are skipped rather than treated as zero, since a zero would win the max over an
all-negative window. `poolOutSpatialPad` guarantees at least one real cell per window. -/
def maxPoolValue
    {d : Nat} {inSpatial : Tensor Nat [d]}
    (input : Tensor α (Shape.ofList inSpatial.data.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let best? := foldlIndices' kernel none (fun best winIdxs =>
    match getPaddedMaxInputVal? (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding), best with
    | none, _ => best
    | some v, none => some v
    | some v, some b => if v > b then some v else best)
  -- The default makes this helper total; valid pooling shapes always select an input value.
  best?.getD 0

/--
Selected-branch tangent for one hard max-pooling window.

The tangent follows the same winner selected by `maxPoolValue`. At a tie this is a deterministic
generalized-derivative convention, not the mathematical directional derivative of `max`.
-/
def maxPoolSelectedTangentValue
    {d : Nat} {inSpatial : Tensor Nat [d]}
    (input tangent : Tensor α (Shape.ofList inSpatial.data.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let best? := foldlIndices' kernel none (fun best winIdxs =>
    match getPaddedMaxInputVal? (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding), best with
    | none, _ => best
    | some v, none => some (winIdxs, v)
    | some v, some (_, b) => if v > b then some (winIdxs, v) else best)
  match best? with
  | none => 0
  | some (bestWin, _) =>
      match paddedCoords? outIdxs bestWin stride with
      | none => 0
      | some padded =>
          match unpadCoords? padded padding with
          | none => 0
          | some orig =>
              if coordsInBounds orig inSpatial.data.toList then
                getAtOrZero tangent orig
              else
                0

/-- Average over one fixed pooling window, counting padded cells as zero.

This is PyTorch's `count_include_pad = True` behaviour: the denominator is the whole window, so a
padded edge pulls the average toward zero rather than being renormalized away. -/
def avgPoolValue
    {d : Nat} {inSpatial : Tensor Nat [d]}
    (input : Tensor α (Shape.ofList inSpatial.data.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let sum := foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    acc + getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding))
  sum / (kernelProd kernel : α)

/-- Input-space pivot whose scaled value is maximal over a nonempty arbitrary-rank pooling
window. -/
def smoothMaxPoolPivot
    {d : Nat} {inSpatial : Tensor Nat [d]}
    (beta : α)
    (input : Tensor α (Shape.ofList inSpatial.data.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let firstIdx := List.replicate d 0
  let first := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
    (input := input) (outIdxs := outIdxs) (winIdxs := firstIdx) (stride := stride)
    (padding := padding)
  foldlIndices' kernel first (fun pivot winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    smoothMaxPivotStep beta pivot x)

/-- Evaluate one arbitrary-rank smooth-max window with the sign-aware input-space pivot. -/
def smoothMaxPoolValue
    {d : Nat} {inSpatial : Tensor Nat [d]}
    (beta : α)
    (input : Tensor α (Shape.ofList inSpatial.data.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let pivot := smoothMaxPoolPivot (d := d) (inSpatial := inSpatial) beta input outIdxs
    kernel stride padding
  let sumExp := foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    acc + MathFunctions.exp (beta * (x - pivot)))
  let invTemp : α := 1 / beta
  pivot + MathFunctions.log sumExp * invTemp

/--
Directional derivative of the smooth log-sum-exp pooling value.

For `y = beta⁻¹ log Σ exp(beta*xᵢ)`, the directional derivative is
`Σ softmax(beta*xᵢ) * dxᵢ`, using the same zero-padding and stable input-pivot convention as
`smoothMaxPoolValue`.
-/
def smoothMaxPoolJvpValue
    {d : Nat} {inSpatial : Tensor Nat [d]}
    (beta : α)
    (input tangent : Tensor α (Shape.ofList inSpatial.data.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let pivot := smoothMaxPoolPivot (d := d) (inSpatial := inSpatial) beta input outIdxs
    kernel stride padding
  let sumExp := foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    acc + MathFunctions.exp (beta * (x - pivot)))
  foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    let dx := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := tangent) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    acc + (MathFunctions.exp (beta * (x - pivot)) / sumExp) * dx)

end Internal
end Pooling

/-!
### Forward (single-channel spatial tensor)
-/

/-- arbitrary-rank max pooling on a spatial tensor (no explicit channel axis). -/
def maxPoolSpatialSpec
    {d : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList inSpatial.data.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).data.toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.data.toList
  let strideL := stride.data.toList
  let paddingL := padding.data.toList

  Tensor.generate outSpatial.data.toList (fun outIdxs =>
    Pooling.Internal.maxPoolValue (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/--
Selected-branch linearization for arbitrary-rank hard max-pooling on a spatial tensor.

Away from ties this is the ordinary JVP. At ties it follows the first row-major primal maximizer,
matching the VJP convention but not claiming an analytic directional derivative.
-/
def maxPoolSpatialLinearizationSpec
    {d : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input tangent : Tensor α (Shape.ofList inSpatial.data.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).data.toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.data.toList
  let strideL := stride.data.toList
  let paddingL := padding.data.toList

  Tensor.generate outSpatial.data.toList (fun outIdxs =>
    Pooling.Internal.maxPoolSelectedTangentValue (d := d) (inSpatial := inSpatial)
      (input := input) (tangent := tangent) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/-- arbitrary-rank average pooling on a spatial tensor (no explicit channel axis). -/
def avgPoolSpatialSpec
    {d : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (_layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList inSpatial.data.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).data.toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.data.toList
  let strideL := stride.data.toList
  let paddingL := padding.data.toList

  Tensor.generate outSpatial.data.toList (fun outIdxs =>
    Pooling.Internal.avgPoolValue (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/-!
### Backward (single-channel spatial tensor)

These are the VJPs of the forward pooling specs above.

Conventions:
- For max pooling, ties are broken by **first occurrence** in row-major order.
- For max pooling, padded cells are ignored, modeling PyTorch's `-∞` padding without requiring a
  scalar-polymorphic infinity constant.
- For average pooling, gradients are evenly distributed across the full kernel window
  (`count_include_pad=true` behavior when padding is present).
-/

/--
Backward/VJP for `maxPoolSpatialSpec`.

Each output gradient is propagated to the argmax location in the corresponding input window.
Ties keep the first position in row-major order.
-/
def maxPoolSpatialBackwardSpec
    {d : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList inSpatial.data.toList))
    (gradOutput :
      Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).data.toList)) :
    Tensor α (Shape.ofList inSpatial.data.toList) :=

  let _ := layer
  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let outDims := outSpatial.data.toList
  let kernelL := kernel.data.toList
  let strideL := stride.data.toList
  let paddingL := padding.data.toList

  let gradInit : Tensor α (Shape.ofList inSpatial.data.toList) :=
    Tensor.generate inSpatial.data.toList (fun _ => 0)

  Pooling.Internal.foldlIndices' outDims gradInit (fun accGrad outIdxs =>
    let best? : Option (List Nat × α) :=
      Pooling.Internal.foldlIndices' kernelL none (fun best winIdxs =>
        match Pooling.Internal.getPaddedMaxInputVal? (d := d) (inSpatial := inSpatial)
          (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := strideL)
          (padding := paddingL), best with
        | none, _ => best
        | some curr, none => some (winIdxs, curr)
        | some curr, some (_, bestVal) =>
            if curr > bestVal then some (winIdxs, curr) else best)
    let gOut : α := getAtOrZero gradOutput outIdxs
    match best? with
    | none => accGrad
    | some (bestWin, _) =>
        match Pooling.Internal.paddedCoords? outIdxs bestWin strideL with
        | none => accGrad
        | some padded =>
            match Pooling.Internal.unpadCoords? padded paddingL with
            | none => accGrad
            | some orig =>
                if Pooling.Internal.coordsInBounds orig inSpatial.data.toList then
                  let current : α := getAtOrZero accGrad orig
                  updateTensorSpec accGrad orig (current + gOut)
                else
                  accGrad)

/--
Backward/VJP for `avgPoolSpatialSpec` (single-channel).

Each output gradient is evenly distributed across its kernel window.
-/
def avgPoolSpatialBackwardSpec
    {d : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (_layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (gradOutput :
      Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).data.toList)) :
    Tensor α (Shape.ofList inSpatial.data.toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let outDims := outSpatial.data.toList
  let kernelL := kernel.data.toList
  let strideL := stride.data.toList
  let paddingL := padding.data.toList
  let poolSize : α := (Pooling.Internal.kernelProd kernelL : Nat)

  let gradInit : Tensor α (Shape.ofList inSpatial.data.toList) :=
    Tensor.generate inSpatial.data.toList (fun _ => 0)

  Pooling.Internal.foldlIndices' outDims gradInit (fun accGrad outIdxs =>
    let gOut : α := getAtOrZero gradOutput outIdxs
    Pooling.Internal.foldlIndices' kernelL accGrad (fun acc winIdxs =>
      match Pooling.Internal.paddedCoords? outIdxs winIdxs strideL with
      | none => acc
      | some padded =>
          match Pooling.Internal.unpadCoords? padded paddingL with
          | none => acc
          | some orig =>
              let current : α := getAtOrZero acc orig
              updateTensorSpec acc orig (current + gOut / poolSize)))

/-!
### Forward (channels-first: `C × spatial...`)
-/

/-- arbitrary-rank max pooling on a channels-first tensor: shape `[C] ++ spatial`. -/
def maxPoolSpec
    {d C : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList (C :: inSpatial.data.toList))) :
    Tensor α
      (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).data.toList)) :=

  Tensor.dim (fun c =>
    maxPoolSpatialSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (get input c))

/-- arbitrary-rank hard max-pool selected-branch linearization, applied channel-wise. -/
def maxPoolLinearizationSpec
    {d C : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input tangent : Tensor α (Shape.ofList (C :: inSpatial.data.toList))) :
    Tensor α
      (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).data.toList)) :=

  Tensor.dim (fun c =>
    maxPoolSpatialLinearizationSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (get input c) (get tangent c))

/-- arbitrary-rank average pooling on a channels-first tensor: shape `[C] ++ spatial`. -/
def avgPoolSpec
    {d C : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList (C :: inSpatial.data.toList))) :
    Tensor α
      (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).data.toList)) :=

  Tensor.dim (fun c =>
    avgPoolSpatialSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (get input c))

/-!
### Adaptive pooling

Adaptive pooling partitions every input axis into a requested number of bins. Unlike fixed-window
pooling, its window sizes depend on the output index. The same definition handles sequence,
image, volume, and higher-rank tensors.
-/

/-- Adaptive average pooling on a channels-first tensor of arbitrary spatial rank.

The two hypotheses make every bin nonempty (`Pooling.Internal.adaptiveStart_lt_adaptiveEnd`), so
the per-bin average never divides by zero. They are not eliminated inside the definition. -/
def adaptiveAvgPoolSpec
    {d C : Nat} {inSpatial outSpatial : Tensor Nat [d]}
    (input : Tensor α (Shape.ofList (C :: inSpatial.data.toList)))
    (_hInput : ∀ i : Fin d, inSpatial.getScalar i ≠ 0)
    (_hOutput : ∀ i : Fin d, outSpatial.getScalar i ≠ 0) :
    Tensor α (Shape.ofList (C :: outSpatial.data.toList)) :=
  Tensor.dim (fun c =>
    Tensor.generate outSpatial.data.toList (fun outIdxs =>
      Pooling.Internal.adaptiveAvgPoolValue (d := d) (inSpatial := inSpatial)
        (input := get input c) outSpatial outIdxs))

/-- Adaptive max pooling on a channels-first tensor of arbitrary spatial rank.

The two hypotheses make every bin nonempty (`Pooling.Internal.adaptiveStart_lt_adaptiveEnd`), so
the `getD 0` fallback in `Pooling.Internal.adaptiveMaxPoolValue` is never taken. -/
def adaptiveMaxPoolSpec
    {d C : Nat} {inSpatial outSpatial : Tensor Nat [d]}
    (input : Tensor α (Shape.ofList (C :: inSpatial.data.toList)))
    (_hInput : ∀ i : Fin d, inSpatial.getScalar i ≠ 0)
    (_hOutput : ∀ i : Fin d, outSpatial.getScalar i ≠ 0) :
    Tensor α (Shape.ofList (C :: outSpatial.data.toList)) :=
  Tensor.dim (fun c =>
    Tensor.generate outSpatial.data.toList (fun outIdxs =>
      Pooling.Internal.adaptiveMaxPoolValue (d := d) (inSpatial := inSpatial)
        (input := get input c) outSpatial outIdxs))

/-!
### Backward (channels-first: `C × spatial...`)
-/

/-- Multi-channel VJP for `maxPoolSpec` (apply spatial backward per channel). -/
def maxPoolBackwardSpec
    {d C : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList (C :: inSpatial.data.toList)))
    (gradOutput :
      Tensor α
        (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).data.toList))) :
    Tensor α (Shape.ofList (C :: inSpatial.data.toList)) :=

  Tensor.dim (fun c =>
    maxPoolSpatialBackwardSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (get input c) (get gradOutput c))

/-- Multi-channel VJP for `avgPoolSpec` (apply spatial backward per channel). -/
def avgPoolBackwardSpec
    {d C : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (gradOutput :
      Tensor α
        (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).data.toList))) :
    Tensor α (Shape.ofList (C :: inSpatial.data.toList)) :=

  Tensor.dim (fun _c =>
    avgPoolSpatialBackwardSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (get gradOutput _c))

/-!
### Smooth max pooling (log-sum-exp surrogate)
-/

/--
Smooth log-sum-exp max pooling on a spatial tensor (no explicit channel axis).

The temperature parameter must be nonzero because the forward expression contains `1 / beta`.
-/
def smoothMaxPoolSpatialSpec
    {d : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (hBeta : beta ≠ 0)
    (input : Tensor α (Shape.ofList inSpatial.data.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).data.toList) :=

  let _ := hBeta
  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.data.toList
  let strideL := stride.data.toList
  let paddingL := padding.data.toList

  Tensor.generate outSpatial.data.toList (fun outIdxs =>
    Pooling.Internal.smoothMaxPoolValue (d := d) (inSpatial := inSpatial) (beta := beta)
      (input := input) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/--
Forward-mode JVP for arbitrary-rank smooth max-pooling on a spatial tensor.

For the log-sum-exp surrogate this is the softmax-weighted sum of the input tangent over each
window. It is the forward-mode counterpart of `smoothMaxPoolSpatialBackwardSpec`.
-/
def smoothMaxPoolSpatialJvpSpec
    {d : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (hBeta : beta ≠ 0)
    (input tangent : Tensor α (Shape.ofList inSpatial.data.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).data.toList) :=

  let _ := hBeta
  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.data.toList
  let strideL := stride.data.toList
  let paddingL := padding.data.toList

  Tensor.generate outSpatial.data.toList (fun outIdxs =>
    Pooling.Internal.smoothMaxPoolJvpValue (d := d) (inSpatial := inSpatial) (beta := beta)
      (input := input) (tangent := tangent) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/--
Smooth log-sum-exp max pooling on a channels-first tensor (channel-wise application).

The temperature parameter must be nonzero because the forward expression contains `1 / beta`.
-/
def smoothMaxPoolSpec
    {d C : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (hBeta : beta ≠ 0)
    (input : Tensor α (Shape.ofList (C :: inSpatial.data.toList))) :
    Tensor α
      (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).data.toList)) :=

  Tensor.dim (fun c =>
    smoothMaxPoolSpatialSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer beta hBeta (get input c))

/-- arbitrary-rank smooth max-pool JVP on a channels-first tensor (channel-wise application). -/
def smoothMaxPoolJvpSpec
    {d C : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (hBeta : beta ≠ 0)
    (input tangent : Tensor α (Shape.ofList (C :: inSpatial.data.toList))) :
    Tensor α
      (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).data.toList)) :=

  Tensor.dim (fun c =>
    smoothMaxPoolSpatialJvpSpec (α := α) (d := d) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding)
      layer (beta := beta) (hBeta := hBeta)
      (input := get input c) (tangent := get tangent c))

/-!
### Smooth max pooling backward
-/

/--
Backward/VJP for `smoothMaxPoolSpatialSpec` (log-sum-exp surrogate).

For a window `x₁,…,xₙ`, the surrogate is:

`y = (1/beta) * log(∑ exp(beta*xᵢ))`

and the VJP distributes upstream gradient proportionally to `exp(beta*xᵢ)`.
The implementation evaluates the equivalent max/min-shifted weights so large finite inputs do not
overflow before normalization.
-/
def smoothMaxPoolSpatialBackwardSpec
    {d : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (hBeta : beta ≠ 0)
    (input : Tensor α (Shape.ofList inSpatial.data.toList))
    (gradOutput :
      Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).data.toList)) :
    Tensor α (Shape.ofList inSpatial.data.toList) :=

  let _ := hBeta
  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let outDims := outSpatial.data.toList
  let kernelL := kernel.data.toList
  let strideL := stride.data.toList
  let paddingL := padding.data.toList
  let coeff : α := 1

  let gradInit : Tensor α (Shape.ofList inSpatial.data.toList) :=
    Tensor.generate inSpatial.data.toList (fun _ => 0)

  Pooling.Internal.foldlIndices' outDims gradInit (fun accGrad outIdxs =>
    let pivot : α :=
      Pooling.Internal.smoothMaxPoolPivot (d := d) (inSpatial := inSpatial) beta input outIdxs
        kernelL strideL paddingL
    let sumExp : α :=
      Pooling.Internal.foldlIndices' kernelL (0 : α) (fun acc winIdxs =>
        let x :=
          Pooling.Internal.getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
            (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs)
            (stride := strideL) (padding := paddingL)
        acc + MathFunctions.exp (beta * (x - pivot)))
    let gOut : α := getAtOrZero gradOutput outIdxs
    Pooling.Internal.foldlIndices' kernelL accGrad (fun acc winIdxs =>
      match Pooling.Internal.paddedCoords? outIdxs winIdxs strideL with
      | none => acc
      | some padded =>
          match Pooling.Internal.unpadCoords? padded paddingL with
          | none => acc
          | some orig =>
              let x :=
                Pooling.Internal.getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
                  (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs)
                  (stride := strideL) (padding := paddingL)
              let expVal := MathFunctions.exp (beta * (x - pivot))
              let w : α := coeff * (expVal / sumExp)
              let current : α := getAtOrZero acc orig
              updateTensorSpec acc orig (current + gOut * w)))

/-- Multi-channel VJP for `smoothMaxPoolSpec` (apply spatial backward per channel). -/
def smoothMaxPoolBackwardSpec
    {d C : Nat} {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (hBeta : beta ≠ 0)
    (input : Tensor α (Shape.ofList (C :: inSpatial.data.toList)))
    (gradOutput :
      Tensor α
        (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).data.toList))) :
    Tensor α (Shape.ofList (C :: inSpatial.data.toList)) :=

  Tensor.dim (fun c =>
    smoothMaxPoolSpatialBackwardSpec (α := α) (d := d) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding)
      layer (beta := beta) (hBeta := hBeta)
      (input := get input c) (gradOutput := get gradOutput c))
end Spec
