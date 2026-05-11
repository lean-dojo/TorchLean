/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Layers.Utils

/-!
# Pooling layers (spec layer)

This file defines a small set of pooling operators in TorchLean's spec layer.

PyTorch analogies:

- `max_pool2d_*` corresponds to `torch.nn.functional.max_pool2d`.
- `avg_pool2d_*` corresponds to `torch.nn.functional.avg_pool2d`.
- `adaptive_*` corresponds to `torch.nn.functional.adaptive_{avg,max}_pool2d` (output size fixed,
  pooling regions vary per output position).

We also include a smooth log-sum-exp surrogate for max pooling. It is useful when you want an
everywhere-differentiable approximation for proofs or analysis, without changing the rest of the
pooling API.
-/

@[expose] public section


namespace Spec
open Tensor
open Spec (Image MultiChannelImage getValueAtPosition extractWindow)

variable {α : Type} [Context α]

/--
MaxPool2d configuration.

The spec uses a fixed kernel `(kH,kW)` and a single stride value (applied to both height and width).
We require `kH ≠ 0`, `kW ≠ 0`, and `stride ≠ 0` so windows are nonempty and the output-shape
arithmetic is well-defined.

PyTorch analogy: `F.max_pool2d(x, kernel_size=(kH,kW), stride=stride)`.
-/
structure MaxPool2DSpec (kH kW stride: ℕ) (h1 : kH ≠ 0) (h2 : kW ≠ 0) (hStride : stride ≠ 0) where
  /-- kernel Height. -/
  kernelHeight : ℕ := kH
  /-- kernel Width. -/
  kernelWidth : ℕ := kW
  /-- Stride. -/
  stride : ℕ := stride

/--
AvgPool2d configuration.

We treat the pooling window as `kH*kW` elements and divide by that count.
This corresponds to PyTorch's default behavior when no padding is present.
-/
structure AvgPool2DSpec (kH kW stride : ℕ) (h1 : kH ≠ 0) (h2 : kW ≠ 0) (hStride : stride ≠ 0) where
  /-- kernel Height. -/
  kernelHeight : ℕ := kH
  /-- kernel Width. -/
  kernelWidth : ℕ := kW
  /-- Stride. -/
  stride : ℕ := stride

/--
Output shape for a 2D pooling op (single-channel) with no padding.

This uses the standard "valid" pooling formula:

`outH = floor((inH - kH)/stride) + 1`, `outW = floor((inW - kW)/stride) + 1`.

PyTorch analogy: `ceil_mode=false` with no padding.
-/
def pool2dOutShape (inH inW kH kW stride : ℕ) : Shape :=
  let outH := (inH - kH) / stride + 1
  let outW := (inW - kW) / stride + 1
  .dim outH (.dim outW .scalar)

/-- Output shape for multi-channel 2D pooling (channels preserved). -/
def pool2dMultiOutShape (inC inH inW kH kW stride : ℕ) : Shape :=
  let outH := (inH - kH) / stride + 1
  let outW := (inW - kW) / stride + 1
  .dim inC (.dim outH (.dim outW .scalar))

/--
Output shape for a 2D pooling op (single-channel) with symmetric padding.

`padding` means we use the usual PyTorch output-size formula for an input extended by `padding`
cells on each side. Hard max-pooling ignores padded cells (the PyTorch `-∞` convention), while
average-pooling below explicitly includes padded zeros.
-/
def pool2dOutShapePad (inH inW kH kW stride padding : ℕ) : Shape :=
  let outH := (inH + 2 * padding - kH) / stride + 1
  let outW := (inW + 2 * padding - kW) / stride + 1
  .dim outH (.dim outW .scalar)

/-- Output shape for multi-channel 2D pooling with symmetric padding (channels preserved). -/
def pool2dMultiOutShapePad (inC inH inW kH kW stride padding : ℕ) : Shape :=
  let outH := (inH + 2 * padding - kH) / stride + 1
  let outW := (inW + 2 * padding - kW) / stride + 1
  .dim inC (.dim outH (.dim outW .scalar))

/-!
## Smooth max pooling

`max_pool2d_spec` uses `max` and is non-differentiable (ties and kink points).

For proofs that need everywhere differentiability, we provide a smooth surrogate
based on log-sum-exp over each pooling window:

  `smooth_max(x₁,…,xₙ) = (1 / β) * log (∑ exp (β * xᵢ))`

This is the standard log-sum-exp surrogate and is intended for `β ≠ 0`.
-/

/--
Smooth max-pooling (single-channel) using a log-sum-exp surrogate.

This is useful in proof settings that want a differentiable alternative to `max_pool2d_spec`.
For large `beta`, the output approaches hard max pooling.
-/
def smoothMaxPool2dSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input : Image inH inW α) :
  Image ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let window := extractWindow kW kH input (i.val * layer.stride) (j.val * layer.stride)
      let expWindow :=
        mapSpec (s := Shape.dim kH (Shape.dim kW Shape.scalar))
          (fun x => MathFunctions.exp (beta * x)) window
      have instH : Shape.valid_axis_inst 0 (Shape.dim kH (Shape.dim kW Shape.scalar)) :=
        Shape.validAxisInstZeroAlt h1
      have instW : Shape.valid_axis_inst 0 (Shape.dim kW Shape.scalar) :=
        Shape.validAxisInstZeroAlt h2
      let sumH := reduceSumAuto 0 expWindow
      have hcast : (Shape.dim kW Shape.scalar) = shapeAfterSum (Shape.dim kH (Shape.dim kW
        Shape.scalar)) 0 := by
        simp [shapeAfterSum]
      let sumH' := tensorCast (Shape.dim kW Shape.scalar) hcast.symm sumH
      let sumAll := reduceSumAuto 0 sumH'
      match sumAll with
      | Tensor.scalar s =>
          let invTemp : α := 1 / beta
          Tensor.scalar (MathFunctions.log s * invTemp)))

/-- Smooth max-pooling (multi-channel): apply `smooth_max_pool2d_spec` per channel. -/
def smoothMaxPool2dMultiSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α :=
  Tensor.dim (fun c => smoothMaxPool2dSpec (layer := layer) (beta := beta) (getAtSpec input c))

/--
Forward-mode JVP for smooth max-pooling (single-channel).

For each pooling window this is the differential of the log-sum-exp surrogate,
`Σᵢ softmax(beta*xᵢ) * dxᵢ`. This mirrors the VJP weights below but pushes an input tangent
forward instead of pulling an output cotangent backward.
-/
def smoothMaxPool2dJvpSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input tangent : Image inH inW α) :
  Image ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let window := extractWindow kW kH input (i.val * layer.stride) (j.val * layer.stride)
      let tangentWindow := extractWindow kW kH tangent (i.val * layer.stride) (j.val * layer.stride)
      let expWindow :=
        mapSpec (s := Shape.dim kH (Shape.dim kW Shape.scalar))
          (fun x => MathFunctions.exp (beta * x)) window
      let weighted :=
        map2Spec (fun e dx => e * dx) expWindow tangentWindow
      have instH : Shape.valid_axis_inst 0 (Shape.dim kH (Shape.dim kW Shape.scalar)) :=
        Shape.validAxisInstZeroAlt h1
      have instW : Shape.valid_axis_inst 0 (Shape.dim kW Shape.scalar) :=
        Shape.validAxisInstZeroAlt h2
      let sumExpH := reduceSumAuto 0 expWindow
      let sumWeightedH := reduceSumAuto 0 weighted
      have hcast : (Shape.dim kW Shape.scalar) = shapeAfterSum
          (Shape.dim kH (Shape.dim kW Shape.scalar)) 0 := by
        simp [shapeAfterSum]
      let sumExpH' := tensorCast (Shape.dim kW Shape.scalar) hcast.symm sumExpH
      let sumWeightedH' := tensorCast (Shape.dim kW Shape.scalar) hcast.symm sumWeightedH
      let sumExp := reduceSumAuto 0 sumExpH'
      let sumWeighted := reduceSumAuto 0 sumWeightedH'
      match sumExp, sumWeighted with
      | Tensor.scalar denom, Tensor.scalar num => Tensor.scalar (num / denom)))

/-- Multi-channel JVP for smooth max-pooling (channel-wise application). -/
def smoothMaxPool2dMultiJvpSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0}
  {h2 : kW ≠ 0} {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input tangent : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α :=
  Tensor.dim (fun c =>
    smoothMaxPool2dJvpSpec (layer := layer) (beta := beta)
      (input := getAtSpec input c) (tangent := getAtSpec tangent c))

/--
MaxPool2d forward pass (single-channel).

This takes the maximum over each `kH×kW` window sampled with the given stride.
The return type encodes the standard output spatial size formula.
-/
def maxPool2dSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input : Image inH inW α) :
  Image ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let window := extractWindow kW kH input (i.val * layer.stride) (j.val * layer.stride)
      have inst : Shape.valid_axis_inst 0 (Shape.dim kH (Shape.dim kW Shape.scalar)) :=
              Shape.validAxisInstZeroAlt h1
      let window_max := reduceMaxAuto 0 window
      have h1_eq : ((Shape.dim kW Shape.scalar)) =
        (shapeAfterSum (Shape.dim kH (Shape.dim kW Shape.scalar)) 0) := by
        simp [shapeAfterSum]
      have inst2 : Shape.valid_axis_inst 0 (Shape.dim kW Shape.scalar) :=
        Shape.validAxisInstZeroAlt h2
      let window_max' := tensorCast ((Shape.dim kW Shape.scalar)) h1_eq.symm window_max
      let window_max_2 := reduceMaxAuto 0 window_max'
      window_max_2))

-- Multi-channel max pooling forward pass
/-- MaxPool2d forward pass (multi-channel): apply `max_pool2d_spec` per channel. -/
def maxPool2dMultiSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α :=
  Tensor.dim (fun c => maxPool2dSpec layer (getAtSpec input c))

/--
Forward-mode JVP for hard max-pooling (single-channel).

The tangent is read at the argmax chosen by the primal input. At ties the first row-major
maximizer is used, matching `maxPool2dBackwardSpec` and PyTorch's index convention.
-/
def maxPool2dJvpSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input tangent : Image inH inW α) :
  Image ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α :=
  Tensor.dim (fun out_i =>
    Tensor.dim (fun out_j =>
      let window := extractWindow kW kH input (out_i.val * layer.stride) (out_j.val * layer.stride)
      let max_pos : Fin kH × Fin kW :=
        (List.finRange kH).foldl (fun best_pos (di : Fin kH) =>
          (List.finRange kW).foldl (fun best_pos_inner (dj : Fin kW) =>
            let current_val := getAtSpec (getAtSpec window di) dj
            let best_val := getAtSpec (getAtSpec window best_pos.1) best_pos.2
            match current_val, best_val with
            | Tensor.scalar curr, Tensor.scalar best =>
                if curr > best then (di, dj) else best_pos_inner
          ) best_pos
        ) (⟨0, Nat.zero_lt_of_ne_zero h1⟩, ⟨0, Nat.zero_lt_of_ne_zero h2⟩)
      let inp_i := out_i.val * layer.stride + max_pos.1.val
      let inp_j := out_j.val * layer.stride + max_pos.2.val
      if h_inp_i : inp_i < inH then
        if h_inp_j : inp_j < inW then
          getAtSpec (getAtSpec tangent ⟨inp_i, h_inp_i⟩) ⟨inp_j, h_inp_j⟩
        else
          Tensor.scalar 0
      else
        Tensor.scalar 0))

/-- Multi-channel JVP for hard max-pooling (channel-wise application). -/
def maxPool2dMultiJvpSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input tangent : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α :=
  Tensor.dim (fun c =>
    maxPool2dJvpSpec (layer := layer)
      (input := getAtSpec input c) (tangent := getAtSpec tangent c))

-- Average pooling forward pass
/--
AvgPool2d forward pass (single-channel).

We sum all values in the window and divide by `kH*kW`.
PyTorch analogy: `avg_pool2d` with `count_include_pad=true` only matters for *padded* pooling;
for the unpadded case it matches the usual definition.
-/
def avgPool2dSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
  (input : Image inH inW α) :
  Image ((inH - kH) / stride + 1)
        ((inW - kW) / stride + 1) α :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let window := extractWindow kW kH input (i.val * layer.stride) (j.val * layer.stride)
      have inst : Shape.valid_axis_inst 0 (Shape.dim kH (Shape.dim kW Shape.scalar)) :=
              Shape.validAxisInstZeroAlt h1
      have inst2 : Shape.valid_axis_inst 0 (Shape.dim kW Shape.scalar) :=
              Shape.validAxisInstZeroAlt h2
      let sumHeight := reduceSumAuto 0 window
      have h1_eq : (Shape.dim kW Shape.scalar) =
        shapeAfterSum (Shape.dim kH (Shape.dim kW Shape.scalar)) 0 := by
        simp [shapeAfterSum]
      let sumHeight' := tensorCast (Shape.dim kW Shape.scalar) h1_eq.symm sumHeight
      let sumTotal := reduceSumAuto 0 sumHeight'
      divSpec sumTotal (Tensor.scalar (kH * kW))))

-- Multi-channel average pooling forward pass
/-- AvgPool2d forward pass (multi-channel): apply `avg_pool2d_spec` per channel. -/
def avgPool2dMultiSpec {kH kW inH inW inC stride : ℕ} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  {hStride : stride ≠ 0}
  (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
  (input : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α :=
  Tensor.dim (fun c => avgPool2dSpec (α := α) (h1 := h1) (h2 := h2) (layer := layer)
    (input := getAtSpec input c))

-- Adaptive pooling that pools to a specific output size
/-- Spec record for adaptive average pooling to a fixed output size. -/
structure AdaptiveAvgPool2DSpec (outH outW : ℕ) where
  /-- output Height. -/
  outputHeight : ℕ := outH
  /-- output Width. -/
  outputWidth : ℕ := outW

/-- Spec record for adaptive max pooling to a fixed output size. -/
structure AdaptiveMaxPool2DSpec (outH outW : ℕ) where
  /-- output Height. -/
  outputHeight : ℕ := outH
  /-- output Width. -/
  outputWidth : ℕ := outW

/-!
## Adaptive pooling

PyTorch defines adaptive pooling by partitioning the input into `out` bins.
For output index `i`, the pooling region is:

- `start = floor(i * in / out)`
- `end   = ceil((i+1) * in / out)`

This matters when `in` is not divisible by `out`: region sizes vary by at most 1.
-/

/-- Adaptive-pooling region start index: `floor(i * in / out)` (PyTorch definition). -/
def adaptiveStart (inSize outSize i : Nat) : Nat :=
  (i * inSize) / outSize

/-- Adaptive-pooling region end index: `ceil((i+1) * in / out)` (PyTorch definition). -/
def adaptiveEnd (inSize outSize i : Nat) : Nat :=
  -- `ceil(a/b) = (a + b - 1) / b` for naturals.
  ((i + 1) * inSize + outSize - 1) / outSize

-- Adaptive average pooling forward pass
/--
AdaptiveAvgPool2d forward pass.

Unlike fixed-kernel pooling, adaptive pooling chooses a window for each output position so that
the whole input is covered by `outH×outW` bins.
This follows the PyTorch start/end formula (see the section comment above).
-/
def adaptiveAvgPool2dSpec {inH inW inC : ℕ} (outH outW : ℕ)
  (_layer : AdaptiveAvgPool2DSpec outH outW)
  (input : MultiChannelImage inC inH inW α)
  (_h_inH : inH > 0 := by norm_num)
  (_h_inW : inW > 0 := by norm_num)
  (_h_outH : outH > 0 := by norm_num)
  (_h_outW : outW > 0 := by norm_num) :
  MultiChannelImage inC outH outW α :=

  Tensor.dim (fun c =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        -- Pooling region for this output position (PyTorch definition).
        let start_i := adaptiveStart inH outH i.val
        let start_j := adaptiveStart inW outW j.val
        let end_i := adaptiveEnd inH outH i.val
        let end_j := adaptiveEnd inW outW j.val
        let actual_kH := end_i - start_i
        let actual_kW := end_j - start_j

        -- Extract and sum the region
        let region_sum :=
          (List.range actual_kH).foldl (fun acc_i di =>
            (List.range actual_kW).foldl (fun acc_j dj =>
              let pos_i := start_i + di
              let pos_j := start_j + dj
              if h_i : pos_i < inH then
                if h_j : pos_j < inW then
                  let val := getAtSpec (getAtSpec (getAtSpec input c) ⟨pos_i, h_i⟩) ⟨pos_j,
                    h_j⟩
                  match acc_j, val with
                  | Tensor.scalar acc, Tensor.scalar v => Tensor.scalar (acc + v)
                else acc_j
              else acc_j
            ) acc_i
          ) (Tensor.scalar 0)

        -- Divide by the actual region size
        divSpec region_sum (Tensor.scalar (actual_kH * actual_kW)))))

-- Adaptive max pooling forward pass
/--
AdaptiveMaxPool2d forward pass (same binning as adaptive avg, but with `max` instead of `mean`).

We intentionally do not use a numeric sentinel value to seed the max fold; we seed from the first
element of the region via `getValueAtPosition`. That keeps the spec meaningful across different
scalar backends.
-/
def adaptiveMaxPool2dSpec {inH inW inC : ℕ} (outH outW : ℕ)
  (_layer : AdaptiveMaxPool2DSpec outH outW)
  (input : MultiChannelImage inC inH inW α)
  (_h_inH : inH > 0 := by norm_num)
  (_h_inW : inW > 0 := by norm_num)
  (_h_outH : outH > 0 := by norm_num)
  (_h_outW : outW > 0 := by norm_num) :
  MultiChannelImage inC outH outW α :=

  Tensor.dim (fun c =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        -- Pooling region for this output position (PyTorch definition).
        let start_i := adaptiveStart inH outH i.val
        let start_j := adaptiveStart inW outW j.val
        let end_i := adaptiveEnd inH outH i.val
        let end_j := adaptiveEnd inW outW j.val
        let actual_kH := end_i - start_i
        let actual_kW := end_j - start_j

        -- Find max in the region
        -- We seed the fold with the first element instead of using a sentinel like `-1000`.
        -- This keeps the spec correct for arbitrary scalar types and scales.
        let init : Tensor α .scalar :=
          -- `getValueAtPosition` performs the bounds check for us, so we don't have to thread a
          -- proof that `start_i < inH` and `start_j < inW` through the code.
          -- Under the stated positivity assumptions this is always in-bounds.
          getValueAtPosition (getAtSpec input c) start_i start_j

        (List.range actual_kH).foldl (fun acc_i di =>
          (List.range actual_kW).foldl (fun acc_j dj =>
            let pos_i := start_i + di
            let pos_j := start_j + dj
            if h_i : pos_i < inH then
              if h_j : pos_j < inW then
                let val := getAtSpec (getAtSpec (getAtSpec input c) ⟨pos_i, h_i⟩) ⟨pos_j, h_j⟩
                match acc_j, val with
                | Tensor.scalar acc, Tensor.scalar v => Tensor.scalar (max acc v)
              else acc_j
            else acc_j
          ) acc_i
        ) init)))

/--
Backward/VJP for `max_pool2d_spec`.

This propagates each output gradient to the argmax location inside the corresponding window.
Tie-breaking: if multiple values in the window are equal to the maximum, we keep the *first*
position in row-major order (same convention as PyTorch's max-pool indices).
-/
def maxPool2dBackwardSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (_layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input : Image inH inW α)
  (grad_output : Image ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α) :
  Image inH inW α :=

  -- Initialize input gradient to zero
  let input_grad_init : Image inH inW α := createZeroImage inH inW

  let outH := (inH - kH) / stride + 1
  let outW := (inW - kW) / stride + 1

  -- For each output position, find max position in input and propagate gradient
  (List.finRange outH).foldl (fun acc_grad (out_i : Fin outH) =>
    (List.finRange outW).foldl (fun acc_grad_inner (out_j : Fin outW) =>

      -- Extract window starting at (out_i * stride, out_j * stride)
      let window := extractWindow kW kH input (out_i.val * stride) (out_j.val * stride)

      -- Find position of maximum in the window
      let max_pos : Fin kH × Fin kW :=
        (List.finRange kH).foldl (fun best_pos (di : Fin kH) =>
          (List.finRange kW).foldl (fun best_pos_inner (dj : Fin kW) =>
            let current_val := getAtSpec (getAtSpec window di) dj
            let best_val    := getAtSpec (getAtSpec window best_pos.1) best_pos.2
            match current_val, best_val with
            | Tensor.scalar curr, Tensor.scalar best =>
              if curr > best then (di, dj) else best_pos_inner
          ) best_pos
        ) (⟨0, Nat.zero_lt_of_ne_zero h1⟩, ⟨0, Nat.zero_lt_of_ne_zero h2⟩)

      -- Gradient value to propagate
      let grad_val := getAtSpec (getAtSpec grad_output out_i) out_j

      -- Compute absolute input indices of the maximum
      let inp_i := out_i.val * stride + max_pos.1.val
      let inp_j := out_j.val * stride + max_pos.2.val

      -- Bounds checks for inH, inW
      if h_inp_i : inp_i < inH then
        if h_inp_j : inp_j < inW then
          let current_grad := getAtSpec (getAtSpec acc_grad_inner ⟨inp_i, h_inp_i⟩) ⟨inp_j,
            h_inp_j⟩
          let new_grad     := addSpec current_grad grad_val
          updateTensorSpec acc_grad_inner [inp_i, inp_j] (Tensor.toScalar new_grad)
        else acc_grad_inner
      else acc_grad_inner

    ) acc_grad
  ) input_grad_init

/-- Multi-channel max-pooling backward (channel-wise application of `max_pool2d_backward_spec`). -/
def maxPool2dMultiBackwardSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input : MultiChannelImage inC inH inW α)
  (grad_output :
    MultiChannelImage inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α) :
  MultiChannelImage inC inH inW α :=
  Tensor.dim (fun c =>
    maxPool2dBackwardSpec (α := α) (_layer := layer)
      (input := getAtSpec input c) (grad_output := getAtSpec grad_output c))


-- Backward pass for average pooling
/--
Backward/VJP for `avg_pool2d_spec` (single-channel).

Each output gradient is evenly distributed across its corresponding input window.
-/
def avgPool2dBackwardSpec {kH kW inH inW stride : ℕ} (_h1 : kH ≠ 0) (_h2 : kW ≠ 0)
  {hStride : stride ≠ 0}
  (_layer : AvgPool2DSpec kH kW stride _h1 _h2 hStride)
  (grad_output : Image ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α) :
  Image inH inW α :=

  -- Initialize input gradient to zero
  let input_grad_init : Image inH inW α := createZeroImage inH inW

  let outH := (inH - kH) / stride + 1
  let outW := (inW - kW) / stride + 1
  let pool_size := kH * kW

  -- For each output position, distribute its gradient evenly across the corresponding input window.
  (List.finRange outH).foldl (fun acc_grad (out_i : Fin outH) =>
    (List.finRange outW).foldl (fun acc_grad_inner (out_j : Fin outW) =>
      let grad_val := getAtSpec (getAtSpec grad_output out_i) out_j
      let distributed_grad := divSpec grad_val (Tensor.scalar pool_size)

      (List.finRange kH).foldl (fun acc_di (di : Fin kH) =>
        (List.finRange kW).foldl (fun acc_dj (dj : Fin kW) =>
          let inp_i := out_i.val * stride + di.val
          let inp_j := out_j.val * stride + dj.val
          if h_inp_i : inp_i < inH then
            if h_inp_j : inp_j < inW then
              let current_grad := getAtSpec (getAtSpec acc_dj ⟨inp_i, h_inp_i⟩) ⟨inp_j, h_inp_j⟩
              let new_grad := addSpec current_grad distributed_grad
              updateTensorSpec acc_dj [inp_i, inp_j] (Tensor.toScalar new_grad)
            else acc_dj
          else acc_dj
        ) acc_di
      ) acc_grad_inner
    ) acc_grad
  ) input_grad_init

/-!
## Padded pooling (symmetric padding)

For max-pooling, padded locations are not real input elements and are ignored when selecting the
maximum. This is the scalar-polymorphic way to model PyTorch's `-∞` max-pool padding without adding
a backend-specific infinity constant to `Context α`.

For average pooling, this corresponds to including padded zeros in the average (PyTorch's default
`count_include_pad = true`).
-/

/-- Remove symmetric zero-padding from a single-channel image. -/
def unpadImage {α : Type} [Context α] {H W padding : ℕ}
    (img : Image (H + 2 * padding) (W + 2 * padding) α) : Image H W α :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      have hi0 : i.val + padding < H + padding := Nat.add_lt_add_right i.isLt padding
      have hj0 : j.val + padding < W + padding := Nat.add_lt_add_right j.isLt padding
      have hleH : H + padding ≤ H + 2 * padding := by
        have hle : padding ≤ 2 * padding := by
          simp [two_mul]
        exact Nat.add_le_add_left hle H
      have hleW : W + padding ≤ W + 2 * padding := by
        have hle : padding ≤ 2 * padding := by
          simp [two_mul]
        exact Nat.add_le_add_left hle W
      have hi : i.val + padding < H + 2 * padding := Nat.lt_of_lt_of_le hi0 hleH
      have hj : j.val + padding < W + 2 * padding := Nat.lt_of_lt_of_le hj0 hleW
      getAtSpec (getAtSpec img ⟨i.val + padding, hi⟩) ⟨j.val + padding, hj⟩))

/-- Remove symmetric zero-padding from a multi-channel image (channel-wise `unpad_image`). -/
def unpadMultiChannel {α : Type} [Context α] {C H W padding : ℕ}
    (img : MultiChannelImage C (H + 2 * padding) (W + 2 * padding) α) : MultiChannelImage C H W α :=
  Tensor.dim (fun c =>
    unpadImage (α := α) (H := H) (W := W) (padding := padding) (getAtSpec img c))

/-- Multi-channel max-pooling forward pass with PyTorch-style padding (`-∞` outside bounds). -/
def maxPool2dMultiSpecPad {kH kW inH inW inC stride padding : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
    {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input : MultiChannelImage inC inH inW α) :
    MultiChannelImage inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) / stride
      + 1) α :=
  Tensor.dim (fun c =>
    Tensor.dim (fun oh =>
      Tensor.dim (fun ow =>
        let best? :=
          (List.range kH).foldl (fun rowBest ky =>
            (List.range kW).foldl (fun best kx =>
              let ph := oh.val * layer.stride + ky
              let pw := ow.val * layer.stride + kx
              if ph < padding then
                best
              else if pw < padding then
                best
              else
                let ih := ph - padding
                let iw := pw - padding
                if hIh : ih < inH then
                  if hIw : iw < inW then
                    let valT := getAtSpec (getAtSpec (getAtSpec input c) ⟨ih, hIh⟩) ⟨iw, hIw⟩
                    match valT with
                    | Tensor.scalar v =>
                        match best with
                        | none => some v
                        | some b => if v > b then some v else some b
                  else best
                else best
            ) rowBest
          ) none
        Tensor.scalar (best?.getD 0))))

/--
Forward-mode JVP for padded hard max-pooling.

Padding cells are ignored exactly as in `maxPool2dMultiSpecPad`, so the tangent is taken from the
primal winner among real input locations only. If a window contains no real input cells, the
forward value and tangent are both `0`.
-/
def maxPool2dMultiJvpSpecPad {kH kW inH inW inC stride padding : ℕ} {h1 : kH ≠ 0}
    {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input tangent : MultiChannelImage inC inH inW α) :
    MultiChannelImage inC ((inH + 2 * padding - kH) / stride + 1)
      ((inW + 2 * padding - kW) / stride + 1) α :=
  Tensor.dim (fun c =>
    Tensor.dim (fun oh =>
      Tensor.dim (fun ow =>
        let best? :=
          (List.range kH).foldl (fun rowBest ky =>
            (List.range kW).foldl (fun best kx =>
              let ph := oh.val * layer.stride + ky
              let pw := ow.val * layer.stride + kx
              if ph < padding then
                best
              else if pw < padding then
                best
              else
                let ih := ph - padding
                let iw := pw - padding
                if hIh : ih < inH then
                  if hIw : iw < inW then
                    let valT := getAtSpec (getAtSpec (getAtSpec input c) ⟨ih, hIh⟩) ⟨iw, hIw⟩
                    match valT with
                    | Tensor.scalar v =>
                        match best with
                        | none => some (ih, iw, v)
                        | some (_, _, b) => if v > b then some (ih, iw, v) else best
                  else best
                else best
            ) rowBest
          ) none
        match best? with
        | none => Tensor.scalar 0
        | some (ih, iw, _) =>
            if hIh : ih < inH then
              if hIw : iw < inW then
                getAtSpec (getAtSpec (getAtSpec tangent c) ⟨ih, hIh⟩) ⟨iw, hIw⟩
              else
                Tensor.scalar 0
            else
              Tensor.scalar 0)))

/-- Multi-channel average pooling forward pass with symmetric zero padding. -/
def avgPool2dMultiSpecPad {kH kW inH inW inC stride padding : ℕ} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
    {hStride : stride ≠ 0}
    (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
    (input : MultiChannelImage inC inH inW α) :
    MultiChannelImage inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) / stride
      + 1) α :=
  -- PyTorch note: this matches `count_include_pad=true` (the padded zeros are part of the average).
  let inputPad : MultiChannelImage inC (inH + 2 * padding) (inW + 2 * padding) α :=
    padMultiChannel (inC := inC) (inH := inH) (inW := inW) input padding
  avgPool2dMultiSpec (h1 := h1) (h2 := h2) (layer := layer) (input := inputPad)

/-- Multi-channel max-pooling backward pass with PyTorch-style padding (`-∞` outside bounds). -/
def maxPool2dMultiBackwardSpecPad {kH kW inH inW inC stride padding : ℕ} {h1 : kH ≠ 0} {h2 : kW
  ≠ 0}
    {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input : MultiChannelImage inC inH inW α)
    (grad_output :
    MultiChannelImage inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1) α) :
    MultiChannelImage inC inH inW α :=
  let outH := (inH + 2 * padding - kH) / stride + 1
  let outW := (inW + 2 * padding - kW) / stride + 1
  let grad_init : MultiChannelImage inC inH inW α :=
    Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0)))
  (List.range inC).foldl (fun acc cNat =>
    if hC : cNat < inC then
      (List.range outH).foldl (fun accH oh =>
        (List.range outW).foldl (fun accW ow =>
          let best? :=
            (List.range kH).foldl (fun rowBest ky =>
              (List.range kW).foldl (fun best kx =>
                let ph := oh * layer.stride + ky
                let pw := ow * layer.stride + kx
                if ph < padding then
                  best
                else if pw < padding then
                  best
                else
                  let ih := ph - padding
                  let iw := pw - padding
                  if hIh : ih < inH then
                    if hIw : iw < inW then
                      let valT := getAtSpec (getAtSpec (getAtSpec input ⟨cNat, hC⟩) ⟨ih, hIh⟩)
                        ⟨iw, hIw⟩
                      match valT with
                      | Tensor.scalar v =>
                          match best with
                          | none => some (ih, iw, v)
                          | some (_, _, b) => if v > b then some (ih, iw, v) else best
                    else best
                  else best
              ) rowBest
            ) none
          match best? with
          | none => accW
          | some (ih, iw, _) =>
              if hOh : oh < outH then
                if hOw : ow < outW then
                  let gOutT :=
                    getAtSpec (getAtSpec (getAtSpec grad_output ⟨cNat, hC⟩) ⟨oh, hOh⟩)
                      ⟨ow, hOw⟩
                  match gOutT with
                  | Tensor.scalar gOut =>
                      let idx := [cNat, ih, iw]
                      let current : α := getAtOrZero accW idx
                      updateTensorSpec accW idx (current + gOut)
                else accW
              else accW
        ) accH
      ) acc
    else acc
  ) grad_init

/-- Multi-channel average-pooling backward pass with symmetric padding (backprop then unpad). -/
def avgPool2dMultiBackwardSpecPad {kH kW inH inW inC stride padding : ℕ} (h1 : kH ≠ 0) (h2 : kW
  ≠ 0)
    {hStride : stride ≠ 0}
    (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
    (grad_output :
      MultiChannelImage inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1) α) :
    MultiChannelImage inC inH inW α :=
  let gradPad : MultiChannelImage inC (inH + 2 * padding) (inW + 2 * padding) α :=
    Tensor.dim (fun c =>
      avgPool2dBackwardSpec (α := α) h1 h2 layer (getAtSpec grad_output c))
  unpadMultiChannel (α := α) (C := inC) (H := inH) (W := inW) (padding := padding) gradPad

-- Smooth max pooling backward pass (log-sum-exp surrogate)
/-- Backward/VJP for `smooth_max_pool2d_spec` (log-sum-exp surrogate). -/
def smoothMaxPool2dBackwardSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (_layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input : Image inH inW α)
  (grad_output : Image ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α) :
  Image inH inW α :=
  -- This is the VJP of the log-sum-exp surrogate:
  --   smooth_max(x) = (1/beta) * log(sum(exp(beta*x))).
  -- The gradient distributes `grad_output` proportionally to `exp(beta*x)` inside each window.
  let input_grad_init : Image inH inW α := createZeroImage inH inW
  let outH := (inH - kH) / stride + 1
  let outW := (inW - kW) / stride + 1
  let coeff : α := 1
  (List.finRange outH).foldl (fun acc_grad (out_i : Fin outH) =>
    (List.finRange outW).foldl (fun acc_grad_inner (out_j : Fin outW) =>
      let window := extractWindow kW kH input (out_i.val * stride) (out_j.val * stride)
      let expWindow :=
        mapSpec (s := Shape.dim kH (Shape.dim kW Shape.scalar))
          (fun x => MathFunctions.exp (beta * x)) window
      have instH : Shape.valid_axis_inst 0 (Shape.dim kH (Shape.dim kW Shape.scalar)) :=
        Shape.validAxisInstZeroAlt h1
      have instW : Shape.valid_axis_inst 0 (Shape.dim kW Shape.scalar) :=
        Shape.validAxisInstZeroAlt h2
      let sumH := reduceSumAuto 0 expWindow
      have hcast : (Shape.dim kW Shape.scalar) = shapeAfterSum (Shape.dim kH (Shape.dim kW
        Shape.scalar)) 0 := by
        simp [shapeAfterSum]
      let sumH' := tensorCast (Shape.dim kW Shape.scalar) hcast.symm sumH
      let sumAll := reduceSumAuto 0 sumH'
      match sumAll with
      | Tensor.scalar sumExp =>
          let gOut := getAtSpec (getAtSpec grad_output out_i) out_j
          -- Distribute gradient over the pooling window.
          (List.finRange kH).foldl (fun acc_di (di : Fin kH) =>
            (List.finRange kW).foldl (fun acc_dj (dj : Fin kW) =>
              let inp_i := out_i.val * stride + di.val
              let inp_j := out_j.val * stride + dj.val
              if h_inp_i : inp_i < inH then
                if h_inp_j : inp_j < inW then
                  let expVal := getAtSpec (getAtSpec expWindow di) dj
                  match expVal with
                  | Tensor.scalar eVal =>
                      let w : α := coeff * (eVal / sumExp)
                      let contrib : Tensor α .scalar :=
                        match gOut with
                        | Tensor.scalar g => Tensor.scalar (g * w)
                      let current := getAtSpec (getAtSpec acc_dj ⟨inp_i, h_inp_i⟩) ⟨inp_j,
                        h_inp_j⟩
                      let new := addSpec current contrib
                      updateTensorSpec acc_dj [inp_i, inp_j] (Tensor.toScalar new)
                else acc_dj
              else acc_dj
            ) acc_di
          ) acc_grad_inner
    ) acc_grad
  ) input_grad_init

/-- Multi-channel backward for `smooth_max_pool2d_multi_spec` (apply per channel). -/
def smoothMaxPool2dMultiBackwardSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input : MultiChannelImage inC inH inW α)
  (grad_output : MultiChannelImage inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1) α) :
  MultiChannelImage inC inH inW α :=
  Tensor.dim (fun c =>
    smoothMaxPool2dBackwardSpec (_layer := layer) (beta := beta)
      (input := getAtSpec input c) (grad_output := getAtSpec grad_output c))

/-!
## Generic N-D pooling (channels-first, no batch)

These operators generalize the existing 2D pooling specs to an arbitrary spatial rank `d`.

Conventions:
- Input is channels-first: shape `[C] ++ spatialDims`.
- Pooling is applied independently per channel (like the existing 2D specs).
- `kernel`, `stride`, and `padding` are per-axis vectors (`Vector Nat d`).
- Padding is symmetric and uses zeros.

PyTorch comparisons (conceptual, without batch axis):
- `max_pool_spec` corresponds to `torch.nn.functional.max_poolNd`.
- `avg_pool_spec` corresponds to `torch.nn.functional.avg_poolNd`.
-/

/-!
### Layer configs + output shapes
-/

/-- Kernel/stride/padding configuration for N-D max pooling. -/
structure MaxPoolSpec (d : Nat)
    (kernel stride padding : Vector Nat d)
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
    (hStride : ∀ i : Fin d, stride.get i ≠ 0) where
  /-- Kernel sizes per spatial axis (outermost to innermost). -/
  kernelSizes : Vector Nat d := kernel
  /-- Strides per spatial axis (outermost to innermost). -/
  strideSizes : Vector Nat d := stride
  /-- Symmetric zero padding per spatial axis (outermost to innermost). -/
  paddingSizes : Vector Nat d := padding

/-- Kernel/stride/padding configuration for N-D average pooling. -/
structure AvgPoolSpec (d : Nat)
    (kernel stride padding : Vector Nat d)
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
    (hStride : ∀ i : Fin d, stride.get i ≠ 0) where
  /-- Kernel sizes per spatial axis (outermost to innermost). -/
  kernelSizes : Vector Nat d := kernel
  /-- Strides per spatial axis (outermost to innermost). -/
  strideSizes : Vector Nat d := stride
  /-- Symmetric zero padding per spatial axis (outermost to innermost). -/
  paddingSizes : Vector Nat d := padding

/-- "Valid" output spatial sizes (no padding): `out = (in - k) / stride + 1` per axis. -/
def poolOutSpatial {d : Nat} (inSpatial kernel stride : Vector Nat d) : Vector Nat d :=
  Vector.ofFn (fun i =>
    (inSpatial.get i - kernel.get i) / stride.get i + 1)

/-- Padded output spatial sizes: `out = (in + 2*pad - k) / stride + 1` per axis. -/
def poolOutSpatialPad {d : Nat} (inSpatial kernel stride padding : Vector Nat d) : Vector Nat d :=
  Vector.ofFn (fun i =>
    (inSpatial.get i + 2 * padding.get i - kernel.get i) / stride.get i + 1)

/-- Output shape for single-channel N-D pooling (no padding). -/
def poolOutShape {d : Nat} (inSpatial kernel stride : Vector Nat d) : Shape :=
  Shape.ofList (poolOutSpatial inSpatial kernel stride).toList

/-- Output shape for channels-first N-D pooling (no padding; channels preserved). -/
def poolMultiOutShape {d : Nat} (inC : Nat) (inSpatial kernel stride : Vector Nat d) : Shape :=
  Shape.ofList (inC :: (poolOutSpatial inSpatial kernel stride).toList)

/-- Output shape for single-channel N-D pooling with symmetric padding. -/
def poolOutShapePad {d : Nat} (inSpatial kernel stride padding : Vector Nat d) : Shape :=
  Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList

/-- Output shape for channels-first N-D pooling with symmetric padding (channels preserved). -/
def poolMultiOutShapePad {d : Nat} (inC : Nat) (inSpatial kernel stride padding : Vector Nat d)
    : Shape :=
  Shape.ofList (inC :: (poolOutSpatialPad inSpatial kernel stride padding).toList)

namespace Private

def tensorOfDims (dims : List Nat) (f : List Nat → α) : Tensor α (Shape.ofList dims) :=
  match dims with
  | [] => Tensor.scalar (f [])
  | _n :: ns =>
      Tensor.dim (fun i =>
        tensorOfDims ns (fun is => f (i.val :: is)))

def foldlIndices' {β : Type} (dims : List Nat) (init : β) (f : β → List Nat → β) : β :=
  match dims with
  | [] => f init []
  | n :: ns =>
      (List.range n).foldl (fun acc i =>
        foldlIndices' ns acc (fun acc' is => f acc' (i :: is))) init

def paddedCoords? (outIdxs winIdxs stride : List Nat) : Option (List Nat) :=
  match outIdxs, winIdxs, stride with
  | [], [], [] => some []
  | o :: os, w :: ws, s :: ss =>
      match paddedCoords? os ws ss with
      | some rest => some ((o * s + w) :: rest)
      | none => none
  | _, _, _ => none

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
    {d : Nat} {inSpatial : Vector Nat d}
    (input : Tensor α (Shape.ofList inSpatial.toList))
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

Unlike average-pooling, max-pooling should not insert a numeric zero for padded cells: PyTorch's
max-pool semantics treat padding as `-∞`. TorchLean keeps the spec scalar-polymorphic by returning
`none` for padded coordinates and letting the max fold ignore them.
-/
def getPaddedMaxInputVal?
    {d : Nat} {inSpatial : Vector Nat d}
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs winIdxs : List Nat)
    (stride padding : List Nat) : Option α :=
  match paddedCoords? outIdxs winIdxs stride with
  | none => none
  | some padded =>
      match unpadCoords? padded padding with
      | none => none
      | some orig =>
          if coordsInBounds orig inSpatial.toList then
            some (getAtOrZero input orig)
          else
            none

def kernelProd (kernel : List Nat) : Nat :=
  kernel.foldl (fun acc k => acc * k) 1

def maxPoolValue
    {d : Nat} {inSpatial : Vector Nat d}
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let best? := foldlIndices' kernel none (fun best winIdxs =>
    match getPaddedMaxInputVal? (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding), best with
    | none, _ => best
    | some v, none => some v
    | some v, some b => if v > b then some v else best)
  best?.getD 0

/--
Directional derivative of hard max-pooling for one N-D window.

The derivative is taken along the same winner selected by `maxPoolValue`. At ties we keep the first
winner in row-major order, matching the VJP convention below and PyTorch's index convention.
-/
def maxPoolJvpValue
    {d : Nat} {inSpatial : Vector Nat d}
    (input tangent : Tensor α (Shape.ofList inSpatial.toList))
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
              if coordsInBounds orig inSpatial.toList then
                getAtOrZero tangent orig
              else
                0

def avgPoolValue
    {d : Nat} {inSpatial : Vector Nat d}
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let sum := foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    acc + getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding))
  sum / (kernelProd kernel : α)

def smoothMaxPoolValue
    {d : Nat} {inSpatial : Vector Nat d}
    (beta : α)
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let sumExp := foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    acc + MathFunctions.exp (beta * x))
  let invTemp : α := 1 / beta
  MathFunctions.log sumExp * invTemp

/--
Directional derivative of the smooth log-sum-exp pooling value.

For `y = beta⁻¹ log Σ exp(beta*xᵢ)`, the directional derivative is
`Σ softmax(beta*xᵢ) * dxᵢ`, using the same zero-padding convention as `smoothMaxPoolValue`.
-/
def smoothMaxPoolJvpValue
    {d : Nat} {inSpatial : Vector Nat d}
    (beta : α)
    (input tangent : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let sumExp := foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    acc + MathFunctions.exp (beta * x))
  foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    let dx := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := tangent) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    acc + (MathFunctions.exp (beta * x) / sumExp) * dx)

end Private

/-!
### Forward (single-channel spatial tensor)
-/

/-- N-D max pooling on a spatial tensor (no explicit channel axis). -/
def maxPoolSpatialSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList inSpatial.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  Private.tensorOfDims outSpatial.toList (fun outIdxs =>
    Private.maxPoolValue (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/--
Forward-mode JVP for N-D hard max-pooling on a spatial tensor.

The derivative follows the same primal argmax as `maxPoolSpatialSpec`; at ties it keeps the first
row-major maximizer. This is the correct directional derivative for TorchLean's chosen subgradient
convention and matches the VJP tie policy.
-/
def maxPoolSpatialJvpSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input tangent : Tensor α (Shape.ofList inSpatial.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  Private.tensorOfDims outSpatial.toList (fun outIdxs =>
    Private.maxPoolJvpValue (d := d) (inSpatial := inSpatial)
      (input := input) (tangent := tangent) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/-- N-D average pooling on a spatial tensor (no explicit channel axis). -/
def avgPoolSpatialSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList inSpatial.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  Private.tensorOfDims outSpatial.toList (fun outIdxs =>
    Private.avgPoolValue (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/-!
### Backward (single-channel spatial tensor)

These are the VJPs of the forward pooling specs above.

Conventions:
- For max pooling, ties are broken by **first occurrence** in row-major order (same as the 2D spec).
- For max pooling, padded cells are ignored, modeling PyTorch's `-∞` padding without requiring a
  scalar-polymorphic infinity constant.
- For average pooling, gradients are evenly distributed across the full kernel window
  (`count_include_pad=true` behavior when padding is present).
-/

/--
Backward/VJP for `max_pool_spatial_spec`.

Each output gradient is propagated to the argmax location in the corresponding input window.
Ties keep the first position in row-major order.
-/
def maxPoolSpatialBackwardSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (grad_output :
      Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList)) :
    Tensor α (Shape.ofList inSpatial.toList) :=

  let _ := layer
  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let outDims := outSpatial.toList
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  let grad_init : Tensor α (Shape.ofList inSpatial.toList) :=
    Private.tensorOfDims inSpatial.toList (fun _ => 0)

  Private.foldlIndices' outDims grad_init (fun acc_grad outIdxs =>
    let best? : Option (List Nat × α) :=
      Private.foldlIndices' kernelL none (fun best winIdxs =>
        match Private.getPaddedMaxInputVal? (d := d) (inSpatial := inSpatial)
          (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := strideL)
          (padding := paddingL), best with
        | none, _ => best
        | some curr, none => some (winIdxs, curr)
        | some curr, some (_, bestVal) =>
            if curr > bestVal then some (winIdxs, curr) else best)
    let gOut : α := getAtOrZero grad_output outIdxs
    match best? with
    | none => acc_grad
    | some (bestWin, _) =>
        match Private.paddedCoords? outIdxs bestWin strideL with
        | none => acc_grad
        | some padded =>
            match Private.unpadCoords? padded paddingL with
            | none => acc_grad
            | some orig =>
                if Private.coordsInBounds orig inSpatial.toList then
                  let current : α := getAtOrZero acc_grad orig
                  updateTensorSpec acc_grad orig (current + gOut)
                else
                  acc_grad)

/--
Backward/VJP for `avg_pool_spatial_spec` (single-channel).

Each output gradient is evenly distributed across its kernel window.
-/
def avgPoolSpatialBackwardSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (grad_output :
      Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList)) :
    Tensor α (Shape.ofList inSpatial.toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let outDims := outSpatial.toList
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList
  let poolSize : α := (Private.kernelProd kernelL : Nat)

  let grad_init : Tensor α (Shape.ofList inSpatial.toList) :=
    Private.tensorOfDims inSpatial.toList (fun _ => 0)

  Private.foldlIndices' outDims grad_init (fun acc_grad outIdxs =>
    let gOut : α := getAtOrZero grad_output outIdxs
    Private.foldlIndices' kernelL acc_grad (fun acc winIdxs =>
      match Private.paddedCoords? outIdxs winIdxs strideL with
      | none => acc
      | some padded =>
          match Private.unpadCoords? padded paddingL with
          | none => acc
          | some orig =>
              let current : α := getAtOrZero acc orig
              updateTensorSpec acc orig (current + gOut / poolSize)))

/-!
### Forward (channels-first: `C × spatial...`)
-/

/-- N-D max pooling on a channels-first tensor: shape `[C] ++ spatial`. -/
def maxPoolSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList (C :: inSpatial.toList))) :
    Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=

  Tensor.dim (fun c =>
    maxPoolSpatialSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (getAtSpec input c))

/-- N-D hard max-pool JVP on a channels-first tensor (channel-wise application). -/
def maxPoolJvpSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input tangent : Tensor α (Shape.ofList (C :: inSpatial.toList))) :
    Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=

  Tensor.dim (fun c =>
    maxPoolSpatialJvpSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (getAtSpec input c) (getAtSpec tangent c))

/-- N-D average pooling on a channels-first tensor: shape `[C] ++ spatial`. -/
def avgPoolSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList (C :: inSpatial.toList))) :
    Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=

  Tensor.dim (fun c =>
    avgPoolSpatialSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (getAtSpec input c))

/-!
### Backward (channels-first: `C × spatial...`)
-/

/-- Multi-channel VJP for `max_pool_spec` (apply spatial backward per channel). -/
def maxPoolBackwardSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList (C :: inSpatial.toList)))
    (grad_output :
      Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList))) :
    Tensor α (Shape.ofList (C :: inSpatial.toList)) :=

  Tensor.dim (fun c =>
    maxPoolSpatialBackwardSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (getAtSpec input c) (getAtSpec grad_output c))

/-- Multi-channel VJP for `avg_pool_spec` (apply spatial backward per channel). -/
def avgPoolBackwardSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (grad_output :
      Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList))) :
    Tensor α (Shape.ofList (C :: inSpatial.toList)) :=

  Tensor.dim (fun _c =>
    avgPoolSpatialBackwardSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (getAtSpec grad_output _c))

/-!
### Smooth max pooling (log-sum-exp surrogate)
-/

/-- Smooth log-sum-exp max pooling on a spatial tensor (no explicit channel axis). -/
def smoothMaxPoolSpatialSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input : Tensor α (Shape.ofList inSpatial.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  Private.tensorOfDims outSpatial.toList (fun outIdxs =>
    Private.smoothMaxPoolValue (d := d) (inSpatial := inSpatial) (beta := beta)
      (input := input) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/--
Forward-mode JVP for N-D smooth max-pooling on a spatial tensor.

For the log-sum-exp surrogate this is the softmax-weighted sum of the input tangent over each
window. It is the forward-mode counterpart of `smoothMaxPoolSpatialBackwardSpec`.
-/
def smoothMaxPoolSpatialJvpSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input tangent : Tensor α (Shape.ofList inSpatial.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  Private.tensorOfDims outSpatial.toList (fun outIdxs =>
    Private.smoothMaxPoolJvpValue (d := d) (inSpatial := inSpatial) (beta := beta)
      (input := input) (tangent := tangent) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/-- Smooth log-sum-exp max pooling on a channels-first tensor (channel-wise application). -/
def smoothMaxPoolSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input : Tensor α (Shape.ofList (C :: inSpatial.toList))) :
    Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=

  Tensor.dim (fun c =>
    smoothMaxPoolSpatialSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer beta (getAtSpec input c))

/-- N-D smooth max-pool JVP on a channels-first tensor (channel-wise application). -/
def smoothMaxPoolJvpSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input tangent : Tensor α (Shape.ofList (C :: inSpatial.toList))) :
    Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=

  Tensor.dim (fun c =>
    smoothMaxPoolSpatialJvpSpec (α := α) (d := d) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding)
      layer (beta := beta)
      (input := getAtSpec input c) (tangent := getAtSpec tangent c))

/-!
### Smooth max pooling backward
-/

/--
Backward/VJP for `smooth_max_pool_spatial_spec` (log-sum-exp surrogate).

For a window `x₁,…,xₙ`, the surrogate is:

`y = (1/beta) * log(∑ exp(beta*xᵢ))`

and the VJP distributes upstream gradient proportionally to `exp(beta*xᵢ)`.
-/
def smoothMaxPoolSpatialBackwardSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (grad_output :
      Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList)) :
    Tensor α (Shape.ofList inSpatial.toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let outDims := outSpatial.toList
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList
  let coeff : α := 1

  let grad_init : Tensor α (Shape.ofList inSpatial.toList) :=
    Private.tensorOfDims inSpatial.toList (fun _ => 0)

  Private.foldlIndices' outDims grad_init (fun acc_grad outIdxs =>
    let sumExp : α :=
      Private.foldlIndices' kernelL (0 : α) (fun acc winIdxs =>
        let x :=
          Private.getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
            (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs)
            (stride := strideL) (padding := paddingL)
        acc + MathFunctions.exp (beta * x))
    let gOut : α := getAtOrZero grad_output outIdxs
    Private.foldlIndices' kernelL acc_grad (fun acc winIdxs =>
      match Private.paddedCoords? outIdxs winIdxs strideL with
      | none => acc
      | some padded =>
          match Private.unpadCoords? padded paddingL with
          | none => acc
          | some orig =>
              let x :=
                Private.getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
                  (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs)
                  (stride := strideL) (padding := paddingL)
              let expVal := MathFunctions.exp (beta * x)
              let w : α := coeff * (expVal / sumExp)
              let current : α := getAtOrZero acc orig
              updateTensorSpec acc orig (current + gOut * w)))

/-- Multi-channel VJP for `smooth_max_pool_spec` (apply spatial backward per channel). -/
def smoothMaxPoolBackwardSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input : Tensor α (Shape.ofList (C :: inSpatial.toList)))
    (grad_output :
      Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList))) :
    Tensor α (Shape.ofList (C :: inSpatial.toList)) :=

  Tensor.dim (fun c =>
    smoothMaxPoolSpatialBackwardSpec (α := α) (d := d) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding)
      layer (beta := beta)
      (input := getAtSpec input c) (grad_output := getAtSpec grad_output c))

/-!
### Friendly aliases
-/

/-- Alias for `max_pool_spec`. -/
abbrev maxPool := @maxPoolSpec

/-- Alias for `avg_pool_spec`. -/
abbrev avgPool := @avgPoolSpec

/-- Alias for `smooth_max_pool_spec`. -/
abbrev smoothMaxPool := @smoothMaxPoolSpec

/-- Alias for `max_pool_backward_spec`. -/
abbrev maxPoolBackward := @maxPoolBackwardSpec

/-- Alias for `avg_pool_backward_spec`. -/
abbrev avgPoolBackward := @avgPoolBackwardSpec

/-- Alias for `smooth_max_pool_backward_spec`. -/
abbrev smoothMaxPoolBackward := @smoothMaxPoolBackwardSpec

end Spec
