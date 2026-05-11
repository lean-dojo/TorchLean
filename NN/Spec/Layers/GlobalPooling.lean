/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Utils

/-!
# Global pooling (spec layer)

Global pooling reduces spatial dimensions (H×W) either to `1×1` (retain channel axis) or
to a flat vector of size `inC`. This file provides both average and max variants, together with
explicit backward rules.

We tried to mimic PyTorch closely:

- The common pattern is `AdaptiveAvgPool2d((1,1))` / `AdaptiveMaxPool2d((1,1))`, then flatten to a
  length-`C` vector before a classifier.
- We usually work with a single image `(C,H,W)` (no batch dimension) here to keep the API small.

Forward generalizes cleanly (and we intentionally structure the code that way):

- Global pooling is "reduce each channel over (H,W)".
- The helpers `global_pool2d_1x1` and `global_pool2d_flat` already capture the reusable shape and
  indexing discipline; the only thing that changes between avg/max/min/etc. is the
  `reduce : Image inH inW α → Tensor α .scalar`.

Max-pooling subtlety:

- If there are multiple spatial positions achieving the same maximum, the backward pass needs a
  tie-breaking convention. This file provides both:
  - a "mask all max positions" rule (sending the full gradient to every max), and
  - a "distributed" rule (split the gradient evenly among max positions).
  PyTorch's exact tie behavior is an implementation detail; the important thing is to make the
  choice explicit in the spec.

Why the backward does not unify for free:

- Different reductions have genuinely different adjoints. Average pooling sends the upstream
  gradient uniformly to every spatial position; max/min pooling routes gradients only to the
  argmax/argmin set and must choose a tie convention.
- So while the forward can be abstracted over a `reduce`, a fully generic backward would need
  extra structure (basically "a reduce + its VJP"). That is why we keep explicit backward specs
  for the concrete ops we care about.

-/

@[expose] public section


namespace Spec
open Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
## Layer tags

Global pooling has no trainable parameters. We still keep a compact "layer spec" record so call sites
can carry a tag (and so the API matches the style of other layer files).
-/

-- Global Average Pooling layer specification
/-- Tag structure for global average pooling (no trainable parameters). -/
structure GlobalAvgPool2DSpec where
  -- Global pooling has no parameters, just the operation type

-- Global Max Pooling layer specification
/-- Tag structure for global max pooling (no trainable parameters). -/
structure GlobalMaxPool2DSpec where
  -- Global pooling has no parameters, just the operation type

/-- Output shape for global pooling that keeps a `1 x 1` spatial grid: `(C,H,W) -> (C,1,1)`. -/
def globalPool2dOutShape (inC : ℕ) : Shape :=
  .dim inC (.dim 1 (.dim 1 .scalar))

/-- Output shape for global pooling that flattens spatial dims away: `(C,H,W) -> (C)`. -/
def globalPool2dFlatOutShape (inC : ℕ) : Shape :=
  .dim inC .scalar

/-!
## Helper: reduce a single channel over its spatial grid

This is the shared "walk the (H,W) grid" loop used by avg/max pooling.
-/

/-- Reduce a single channel `Image inH inW α` down to a scalar using a fold over `(H,W)`. -/
def reduceSpatial {α : Type} (inH inW : ℕ)
  (init : α) (f : α → α → α) (channel_data : Image inH inW α) : α :=
  -- `List.finRange` gives us in-bounds indices, so we can index directly with `Fin`.
  (List.finRange inH).foldl (fun acc_h i =>
    (List.finRange inW).foldl (fun acc_w j =>
      let val := getAtSpec (getAtSpec channel_data i) j
      match val with
      | Tensor.scalar v => f acc_w v
    ) acc_h
  ) init

/-- Compute the exact spatial maximum of a non-empty `(inH × inW)` channel. -/
def channelSpatialMax {α : Type} [Max α] {inH inW : ℕ}
  (hH : inH ≠ 0) (hW : inW ≠ 0)
  (channel_data : Image inH inW α) : α :=
  -- We intentionally avoid "fake -infinity" sentinels. Starting from the first element keeps the
  -- definition correct for any scalar type that supports `Max`.
  let i0 : Fin inH := ⟨0, Nat.pos_of_ne_zero hH⟩
  let j0 : Fin inW := ⟨0, Nat.pos_of_ne_zero hW⟩
  let init : α :=
    match getAtSpec (getAtSpec channel_data i0) j0 with
    | Tensor.scalar v => v
  reduceSpatial inH inW init (Max.max · ·) channel_data

/-- Alias for `reduce_spatial` (kept to make call sites read like "reduce this channel"). -/
def globalChannelReduce {α : Type} (inH inW : ℕ)
  (channel_data : Image inH inW α)
  (init : α) (f : α → α → α) : α :=
  reduceSpatial inH inW init f channel_data

/-!
## Helper: "wrap a scalar result back into an image"

PyTorch mental picture: after pooling you conceptually have a scalar per channel; these helpers
put that scalar back into a scalar tensor shape.
-/

/-- Broadcast a scalar into a `1 x 1` image. -/
def broadcastScalar1x1 {α : Type} (v : Tensor α .scalar) : Image 1 1 α :=
  Tensor.dim (fun _ => Tensor.dim (fun _ => v))

/-- Generic global pooling helper producing `(C,1,1)`. -/
def globalPool2d1x1 {α : Type} [Zero α] (inC inH inW : ℕ)
  (reduce : Image inH inW α → Tensor α .scalar)
  (input : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC 1 1 α :=
  Tensor.dim (fun c => broadcastScalar1x1 (reduce (getAtSpec input c)))

/-- Generic global pooling helper producing `(C)`. -/
def globalPool2dFlat {α : Type} [Zero α] (inC inH inW : ℕ)
  (reduce : Image inH inW α → Tensor α .scalar)
  (input : MultiChannelImage inC inH inW α) :
  Tensor α (.dim inC .scalar) :=
  Tensor.dim (fun c => reduce (getAtSpec input c))

/-!
## Forward specs

These are the layer-level forward meanings, written in the same style as PyTorch.
-/

/-- Global average pooling: `(C,H,W) -> (C,1,1)`. -/
def globalAvgPool2dSpec {α : Type} [Zero α] [Add α] [HDiv α ℕ α] [OfNat α 0] [OfNat α 1] {inC inH
  inW : ℕ} (_h1 : inH ≠ 0) (_h2 : inW ≠ 0)
  (_layer : GlobalAvgPool2DSpec)
  (input : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC 1 1 α :=
  globalPool2d1x1 inC inH inW
    (fun channel_data =>
      let sum := globalChannelReduce inH inW channel_data 0 (· + ·)
      let avg := sum / (inH * inW)
      Tensor.scalar avg)
    input

/-- Global max pooling: `(C,H,W) -> (C,1,1)`. -/
def globalMaxPool2dSpec {α : Type} [Numbers α] [Max α] [Zero α] {inC inH inW : ℕ} (h1 : inH ≠ 0)
  (h2 : inW ≠ 0)
  (_layer : GlobalMaxPool2DSpec)
  (input : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC 1 1 α :=
  globalPool2d1x1 inC inH inW
    (fun channel_data =>
      let maxv := channelSpatialMax (α := α) (inH := inH) (inW := inW) h1 h2 channel_data
      Tensor.scalar maxv)
    input

/-- Global average pooling (flattened): `(C,H,W) -> (C)`. -/
def globalAvgPool2dFlatSpec {α : Type} [Coe Nat α] [Div α] [Zero α] [Add α] [Zero α] [One α]
  {inC inH inW : ℕ} (_h1 : inH ≠ 0) (_h2 : inW ≠ 0)
  (_layer : GlobalAvgPool2DSpec)
  (input : MultiChannelImage inC inH inW α) :
  Tensor α (.dim inC .scalar) :=
  globalPool2dFlat inC inH inW
    (fun channel_data =>
      let sum := globalChannelReduce inH inW channel_data 0 (· + ·)
      let avg := sum / (((inH * inW) : α))
      Tensor.scalar avg)
    input

/-- Global max pooling (flattened): `(C,H,W) -> (C)`. -/
def globalMaxPool2dFlatSpec {α : Type} [Numbers α] [Max α] [Zero α] {inC inH inW : ℕ} (h1 : inH
  ≠ 0) (h2 : inW ≠ 0)
  (_layer : GlobalMaxPool2DSpec)
  (input : MultiChannelImage inC inH inW α) :
  Tensor α (.dim inC .scalar) :=
  globalPool2dFlat inC inH inW
    (fun channel_data =>
      let maxv := channelSpatialMax (α := α) (inH := inH) (inW := inW) h1 h2 channel_data
      Tensor.scalar maxv)
    input

/-!
## Backward/VJP specs

These are reverse-mode rules that match the intended math:

- avg pooling: distribute the upstream gradient evenly over all `(H,W)` positions;
- max pooling: route the upstream gradient to the max locations (with a tie convention).
-/

/-- Backward/VJP for global average pooling `(C,1,1)` output. -/
def globalAvgPool2dBackwardSpec {inC inH inW : ℕ} (_h1 : inH ≠ 0) (_h2 : inW ≠ 0)
  (_layer : GlobalAvgPool2DSpec)
  (grad_output : MultiChannelImage inC 1 1 α) :
  MultiChannelImage inC inH inW α :=

  let spatial_size := inH * inW

  Tensor.dim (fun c =>
    -- Get the gradient value for this channel
    let grad_val := getAtSpec (getAtSpec (getAtSpec grad_output c) ⟨0, by norm_num⟩) ⟨0, by
      norm_num⟩

    -- Distribute gradient evenly across all spatial positions
    let distributed_grad := divSpec grad_val (Tensor.scalar spatial_size)

    -- Create tensor filled with the distributed gradient
    Tensor.dim (fun i =>
      Tensor.dim (fun j => distributed_grad)))

/-- Backward/VJP for flattened global average pooling `(C)` output. -/
def globalAvgPool2dFlatBackwardSpec {inC inH inW : ℕ} (_h1 : inH ≠ 0) (_h2 : inW ≠ 0)
  (_layer : GlobalAvgPool2DSpec)
  (grad_output : Tensor α (.dim inC .scalar)) :
  MultiChannelImage inC inH inW α :=

  let spatial_size := inH * inW

  Tensor.dim (fun c =>
    -- Get the gradient value for this channel
    let grad_val := getAtSpec grad_output c

    -- Distribute gradient evenly across all spatial positions
    let distributed_grad := divSpec grad_val (Tensor.scalar spatial_size)

    -- Create tensor filled with the distributed gradient
    Tensor.dim (fun _i =>
      Tensor.dim (fun _j => distributed_grad)))

/-- Backward/VJP for global max pooling `(C,1,1)` output.

Tie convention: every spatial position equal to the maximum receives the full upstream gradient.
-/
def globalMaxPool2dBackwardSpec {inC inH inW : ℕ} (h1 : inH ≠ 0) (h2 : inW ≠ 0)
  (_layer : GlobalMaxPool2DSpec)
  (input : MultiChannelImage inC inH inW α)
  (grad_output : MultiChannelImage inC 1 1 α) :
  MultiChannelImage inC inH inW α :=

  Tensor.dim (fun c =>
    let channel_data := getAtSpec input c
    let grad_val := getAtSpec (getAtSpec (getAtSpec grad_output c) ⟨0, by norm_num⟩) ⟨0, by
      norm_num⟩

    -- Find the position(s) with maximum value
    let global_max : Tensor α .scalar :=
      Tensor.scalar (channelSpatialMax (α := α) (inH := inH) (inW := inW) h1 h2 channel_data)

    -- Create gradient tensor with grad_val only at max positions, zero elsewhere
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let val := getAtSpec (getAtSpec channel_data i) j
        match val, global_max with
        | Tensor.scalar v, Tensor.scalar max_v =>
          if v == max_v then grad_val else Tensor.scalar 0)))

/-- Backward/VJP for flattened global max pooling `(C)` output.

Tie convention: every spatial position equal to the maximum receives the full upstream gradient.
-/
def globalMaxPool2dFlatBackwardSpec {inC inH inW : ℕ} (h1 : inH ≠ 0) (h2 : inW ≠ 0)
  (_layer : GlobalMaxPool2DSpec)
  (input : MultiChannelImage inC inH inW α)
  (grad_output : Tensor α (.dim inC .scalar)) :
  MultiChannelImage inC inH inW α :=

  Tensor.dim (fun c =>
    let channel_data := getAtSpec input c
    let grad_val := getAtSpec grad_output c

    -- Find the position(s) with maximum value
    let global_max : Tensor α .scalar :=
      Tensor.scalar (channelSpatialMax (α := α) (inH := inH) (inW := inW) h1 h2 channel_data)

    -- Create gradient tensor with grad_val only at max positions, zero elsewhere
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let val := getAtSpec (getAtSpec channel_data i) j
        match val, global_max with
        | Tensor.scalar v, Tensor.scalar max_v =>
          if v == max_v then grad_val else Tensor.scalar 0)))

/-- Alternative max-pooling backward that splits the gradient evenly across max positions.

This is often a nicer mathematical choice when the max is not unique.
-/
def globalMaxPool2dBackwardDistributedSpec {inC inH inW : ℕ} (h1 : inH ≠ 0) (h2 : inW ≠ 0)
  (_layer : GlobalMaxPool2DSpec)
  (input : MultiChannelImage inC inH inW α)
  (grad_output : MultiChannelImage inC 1 1 α) :
  MultiChannelImage inC inH inW α :=

  Tensor.dim (fun c =>
    let channel_data := getAtSpec input c
    let grad_val := getAtSpec (getAtSpec (getAtSpec grad_output c) ⟨0, by norm_num⟩) ⟨0, by
      norm_num⟩

    -- Find the maximum value
    let global_max : Tensor α .scalar :=
      Tensor.scalar (channelSpatialMax (α := α) (inH := inH) (inW := inW) h1 h2 channel_data)

    -- Count positions with maximum value
    let max_count :=
      (List.finRange inH).foldl (fun acc_count i =>
        (List.finRange inW).foldl (fun acc_count_inner j =>
          let val := getAtSpec (getAtSpec channel_data i) j
          match val, global_max with
          | Tensor.scalar v, Tensor.scalar max_v =>
            if v == max_v then acc_count_inner + 1 else acc_count_inner
        ) acc_count
      ) 0

    -- Distribute gradient among all max positions.
    let distributed_grad := divSpec grad_val (Tensor.scalar max_count)

    -- Create gradient tensor
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let val := getAtSpec (getAtSpec channel_data i) j
        match val, global_max with
        | Tensor.scalar v, Tensor.scalar max_v =>
          if v == max_v then distributed_grad else Tensor.scalar 0)))

end Spec
