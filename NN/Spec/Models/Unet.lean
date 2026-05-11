/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Pooling

/-!
# Unet

U-Net (2-level) model.

This file defines a small U-Net style architecture (a single downsample + upsample):

- down path: two `Conv2d(3x3, stride=1, padding=1) + ReLU` blocks,
- downsample: `MaxPool2d(kernel=2, stride=2)`,
- bottleneck: two more conv blocks,
- upsample: `ConvTranspose2d(kernel=2, stride=2)`,
- skip connection: concatenate channels and run two conv blocks,
- output head: `Conv2d(1x1)` to map `baseC -> outC`.

PyTorch mental model:
- this matches the common "U-Net block diagram" but written without a batch axis, so our tensor
  convention is `(C,H,W)` rather than `(N,C,H,W)`;
- the skip connection concatenates on the channel axis (in PyTorch with a batch axis that would be
  `torch.cat([skip, up], dim=1)`; here it is `concat_dim0_spec` because channels are axis `0`).

Shape notes:
- the 3x3 conv blocks are set up to preserve `H×W` (stride=1, padding=1),
- the pool/upsample pair is the usual `2x` down then `2x` up, but for odd spatial sizes the
  `ConvTranspose2d` formula can produce an off-by-one; we surface this as explicit equalities
  (`h_upH`, `h_upW`) so the caller can pick compatible `inH,inW` (typically even).

References:
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI
  2015).

PyTorch docs (for API intuition, not semantics):
- `torch.nn.Conv2d`: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- `torch.nn.MaxPool2d`: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
- `torch.nn.ConvTranspose2d`:
  https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
-/

@[expose] public section


namespace Models

open Spec
open Tensor
open Activation

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
## Configuration

Architectural hyperparameters live in a dedicated config record.

PyTorch mental model:
- this mirrors the way you would pass `kernel_size/stride/padding` to `nn.Conv2d`,
  `nn.MaxPool2d`, and `nn.ConvTranspose2d`, plus the base channel width.
-/

/-- U-Net (2-level) architectural hyperparameters (spec layer). -/
structure UNet2Config where
  /-- `kernel_size` for the max-pool layer (typical: `2`). -/
  poolKernel : Nat := 2
  /-- `stride` for the max-pool layer (typical: `2`). -/
  poolStride : Nat := 2

  /-- `kernel_size` for the 2D conv blocks (typical: `3`). -/
  convKernel : Nat := 3
  /-- `stride` for the 2D conv blocks (typical: `1`). -/
  convStride : Nat := 1
  /-- symmetric zero `padding` for the 2D conv blocks (typical: `1`). -/
  convPadding : Nat := 1

  /-- `kernel_size` for the transposed-convolution upsampler (typical: `2`). -/
  upKernel : Nat := 2
  /-- `stride` for the transposed-convolution upsampler (typical: `2`). -/
  upStride : Nat := 2
  /-- `padding` for the transposed-convolution upsampler (typical: `0`). -/
  upPadding : Nat := 0

  /-- `kernel_size` for the final output head conv (typical: `1`). -/
  headKernel : Nat := 1
  /-- `stride` for the final output head conv (typical: `1`). -/
  headStride : Nat := 1
  /-- `padding` for the final output head conv (typical: `0`). -/
  headPadding : Nat := 0

  /-- Base channel count (typical: `64`). -/
  baseC : Nat := 64

/-- Well-formedness conditions for `UNet2Config` (the few nonzero facts needed by layer specs). -/
structure UNet2Config.WF (cfg : UNet2Config) : Prop where
  poolK_ne0 : cfg.poolKernel ≠ 0
  poolStride_ne0 : cfg.poolStride ≠ 0
  convK_ne0 : cfg.convKernel ≠ 0
  upK_ne0 : cfg.upKernel ≠ 0
  headK_ne0 : cfg.headKernel ≠ 0
  baseC_pos : cfg.baseC > 0

/-- Canonical "classic U-Net-ish" defaults for our 2-level spec. -/
def unet2DefaultConfig : UNet2Config := {}

/-- `unet2DefaultConfig` satisfies the nonzero facts required by the spec layer. -/
theorem unet2DefaultConfig_wf : unet2DefaultConfig.WF := by
  refine
    { poolK_ne0 := by decide
      poolStride_ne0 := by decide
      convK_ne0 := by decide
      upK_ne0 := by decide
      headK_ne0 := by decide
      baseC_pos := by decide }

/-- Output height after `MaxPool2d(kernel=2, stride=2)` (no padding). -/
abbrev UNetDownH (cfg : UNet2Config) (inH : Nat) : Nat :=
  (inH - cfg.poolKernel) / cfg.poolStride + 1

/-- Output width after `MaxPool2d(kernel=2, stride=2)` (no padding). -/
abbrev UNetDownW (cfg : UNet2Config) (inW : Nat) : Nat :=
  (inW - cfg.poolKernel) / cfg.poolStride + 1

/-- Output height after `MaxPool2d(2,2)` then `ConvTranspose2d(2,2)` (with `padding=0`). -/
abbrev UNetUpH (cfg : UNet2Config) (inH : Nat) : Nat :=
  ((UNetDownH cfg inH - 1) * cfg.upStride - 2 * cfg.upPadding + cfg.upKernel)

/-- Output width after `MaxPool2d(2,2)` then `ConvTranspose2d(2,2)` (with `padding=0`). -/
abbrev UNetUpW (cfg : UNet2Config) (inW : Nat) : Nat :=
  ((UNetDownW cfg inW - 1) * cfg.upStride - 2 * cfg.upPadding + cfg.upKernel)

/--
2-level U-Net parameter record (spec).

This is a compact U-Net with one downsample and one upsample step:
- two conv + ReLU blocks at full resolution (with a skip),
- max-pooling, then two conv + ReLU blocks at the lower resolution,
- a transposed-conv upsampler,
- channel concatenation with the skip feature map,
- two more conv + ReLU blocks,
- a final `1×1` conv head.

Shape convention: tensors are `(C,H,W)` (no batch axis).

PyTorch analogue: a small U-Net built from `nn.Conv2d`, `nn.MaxPool2d`, `nn.ConvTranspose2d`,
and `torch.cat` along the channel axis.
-/
structure UNet2Spec (cfg : UNet2Config) (inC outC inH inW : Nat) (α : Type)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (h_inC : inC ≠ 0) (hCfg : cfg.WF) where
  /-- First 3×3 conv in the first down block (`inC -> baseC`). -/
  down1_1 :
    Conv2DSpec inC cfg.baseC cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α h_inC
      hCfg.convK_ne0 hCfg.convK_ne0
  /-- Second 3×3 conv in the first down block (`baseC -> baseC`). -/
  down1_2 :
    Conv2DSpec cfg.baseC cfg.baseC cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α
      (Nat.ne_of_gt hCfg.baseC_pos) hCfg.convK_ne0 hCfg.convK_ne0

  /-- First 3×3 conv in the bottleneck block (`baseC -> 2*baseC`). -/
  down2_1 :
    Conv2DSpec cfg.baseC (2 * cfg.baseC) cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α
      (Nat.ne_of_gt hCfg.baseC_pos) hCfg.convK_ne0 hCfg.convK_ne0
  /-- Second 3×3 conv in the bottleneck block (`2*baseC -> 2*baseC`). -/
  down2_2 :
    Conv2DSpec (2 * cfg.baseC) (2 * cfg.baseC) cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α
      (Nat.ne_of_gt (Nat.mul_pos (by decide : 0 < 2) hCfg.baseC_pos)) hCfg.convK_ne0 hCfg.convK_ne0

  /-- Transposed-convolution upsampler (`2*baseC -> baseC`, `kernel=2`, `stride=2`). -/
  upT :
    ConvTranspose2DSpec (2 * cfg.baseC) cfg.baseC cfg.upKernel cfg.upKernel cfg.upStride cfg.upPadding α
      (Nat.mul_pos (by decide : 0 < 2) hCfg.baseC_pos) hCfg.upK_ne0 hCfg.upK_ne0

  /-- First 3×3 conv after skip concatenation (`(baseC+baseC) -> baseC`). -/
  up1_1 :
    Conv2DSpec (cfg.baseC + cfg.baseC) cfg.baseC cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α
    (by
      -- `baseC + baseC ≥ baseC > 0`
      have : 0 < cfg.baseC + cfg.baseC :=
        Nat.lt_of_lt_of_le hCfg.baseC_pos (Nat.le_add_right cfg.baseC cfg.baseC)
      exact Nat.ne_of_gt this)
    hCfg.convK_ne0 hCfg.convK_ne0
  /-- Second 3×3 conv after skip concatenation (`baseC -> baseC`). -/
  up1_2 :
    Conv2DSpec cfg.baseC cfg.baseC cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α
      (Nat.ne_of_gt hCfg.baseC_pos) hCfg.convK_ne0 hCfg.convK_ne0

  /-- Final 1×1 conv head (`baseC -> outC`). -/
  out1x1 :
    Conv2DSpec cfg.baseC outC cfg.headKernel cfg.headKernel cfg.headStride cfg.headPadding α
      (Nat.ne_of_gt hCfg.baseC_pos) hCfg.headK_ne0 hCfg.headK_ne0

/-!
## Gradients

This U-Net is small enough that we can write a fully explicit backward pass in a "mirror the
forward" style: rebuild the same intermediates, then walk back through them using the existing
layer-level backward specs.

Key details:
- `concat_dim0_spec` is split via `concat_dim0_backward_spec`,
- pooling backward uses `max_pool2d_multi_backward_spec`,
- ReLU is handled via elementwise gating `dZ = dY ⊙ ReLU'(Z)`.

PyTorch analogy:
- each `conv2d_backward_spec` call corresponds to the gradients PyTorch computes for
  `Conv2d(weight,bias)`;
- `max_pool2d_multi_backward_spec` corresponds to max-pool backward using the argmax locations from
  the forward (our spec computes it from the inputs).
-/

/--
Parameter-gradient container for `UNet2Spec`.

This mirrors the parameter layout of `UNet2Spec`, recording kernel and bias gradients for each
convolution and transposed-convolution layer.
-/
structure UNet2Grads (cfg : UNet2Config) (inC outC inH inW : Nat) (α : Type) where
  /-- d down 1 1 kernel. -/
  d_down1_1_kernel :
    Tensor α (.dim cfg.baseC (.dim inC (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- d down 1 1 bias. -/
  d_down1_1_bias   : Tensor α (.dim cfg.baseC .scalar)
  /-- d down 1 2 kernel. -/
  d_down1_2_kernel :
    Tensor α (.dim cfg.baseC (.dim cfg.baseC (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- d down 1 2 bias. -/
  d_down1_2_bias   : Tensor α (.dim cfg.baseC .scalar)
  /-- d down 2 1 kernel. -/
  d_down2_1_kernel :
    Tensor α (.dim (2 * cfg.baseC) (.dim cfg.baseC (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- d down 2 1 bias. -/
  d_down2_1_bias   : Tensor α (.dim (2 * cfg.baseC) .scalar)
  /-- d down 2 2 kernel. -/
  d_down2_2_kernel :
    Tensor α (.dim (2 * cfg.baseC) (.dim (2 * cfg.baseC) (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- d down 2 2 bias. -/
  d_down2_2_bias   : Tensor α (.dim (2 * cfg.baseC) .scalar)
  /-- d up T kernel. -/
  d_upT_kernel     : Spec.ConvTransposeKernel cfg.baseC (2 * cfg.baseC) cfg.upKernel cfg.upKernel α
  /-- d up T bias. -/
  d_upT_bias       : Tensor α (.dim cfg.baseC .scalar)
  /-- d up 1 1 kernel. -/
  d_up1_1_kernel   :
    Tensor α (.dim cfg.baseC (.dim (cfg.baseC + cfg.baseC) (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- d up 1 1 bias. -/
  d_up1_1_bias     : Tensor α (.dim cfg.baseC .scalar)
  /-- d up 1 2 kernel. -/
  d_up1_2_kernel   :
    Tensor α (.dim cfg.baseC (.dim cfg.baseC (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- d up 1 2 bias. -/
  d_up1_2_bias     : Tensor α (.dim cfg.baseC .scalar)
  /-- d out 1 x 1 kernel. -/
  d_out1x1_kernel  :
    Tensor α (.dim outC (.dim cfg.baseC (.dim cfg.headKernel (.dim cfg.headKernel .scalar))))
  /-- d out 1 x 1 bias. -/
  d_out1x1_bias    : Tensor α (.dim outC .scalar)

/--
Forward pass for `UNet2Spec`.

Inputs/outputs use `MultiChannelImage` tensors of shape `(C,H,W)` (no batch axis).

The many `h_*` equalities are shape-rewrite hints: layer specs compute output sizes using explicit
arithmetic (matching PyTorch's formulas), and these equalities let callers assert "this 3×3 conv
preserves spatial size" or "pool then upsample returns to the original size" for a particular
choice of `inH,inW` (typically even).
-/
def UNet2Spec.forward
  {cfg : UNet2Config} {inC outC inH inW : Nat}
  {h_inC : inC ≠ 0} {hCfg : cfg.WF}
  (m : UNet2Spec (α := α) cfg inC outC inH inW h_inC hCfg)
  (x : MultiChannelImage inC inH inW α)
  (h_convH :
    ((inH + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) = inH)
  (h_convW :
    ((inW + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) = inW)
  (h_convH_down :
    ((UNetDownH cfg inH + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) = UNetDownH cfg inH)
  (h_convW_down :
    ((UNetDownW cfg inW + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) = UNetDownW cfg inW)
  (h_upH : UNetUpH cfg inH = inH)
  (h_upW : UNetUpW cfg inW = inW)
  (h_outH : ((inH + 2 * cfg.headPadding - cfg.headKernel) / cfg.headStride + 1) = inH)
  (h_outW : ((inW + 2 * cfg.headPadding - cfg.headKernel) / cfg.headStride + 1) = inW) :
  MultiChannelImage outC inH inW α :=

  -- The `h_*` equalities are there for one reason: many of the layer specs compute output shapes
  -- with explicit arithmetic (matching PyTorch's formulas), and we sometimes want to treat a
  -- "shape-preserving" conv as literally returning `(C,inH,inW)`. These equalities are how we
  -- rewrite between the computed shape and the intended shape.

  -- Down block 1 (spatial preserved because conv is 3x3, stride=1, padding=1).
  let s1_raw :=
    reluSpec (conv2dSpec (α := α) m.down1_1 x)
  let s1 : MultiChannelImage cfg.baseC inH inW α :=
    rwMultiChannelImage (α := α) s1_raw (by rfl) h_convH h_convW

  let skip1_raw :=
    reluSpec (conv2dSpec (α := α) m.down1_2 s1)
  let skip1 : MultiChannelImage cfg.baseC inH inW α :=
    rwMultiChannelImage (α := α) skip1_raw (by rfl) h_convH h_convW

  -- Downsample (PyTorch analogy: `nn.MaxPool2d(kernel_size=2, stride=2)`).
  let pool : MaxPool2DSpec cfg.poolKernel cfg.poolKernel cfg.poolStride hCfg.poolK_ne0 hCfg.poolK_ne0
      hCfg.poolStride_ne0 :=
    {}

  let downH := UNetDownH cfg inH
  let downW := UNetDownW cfg inW

  let pooled : MultiChannelImage cfg.baseC downH downW α :=
    maxPool2dMultiSpec (α := α) (layer := pool) skip1

  -- Down block 2
  let b1_raw :=
    reluSpec (conv2dSpec (α := α) m.down2_1 pooled)
  let b1 : MultiChannelImage (2 * cfg.baseC) downH downW α :=
    rwMultiChannelImage (α := α) b1_raw (by rfl) h_convH_down h_convW_down

  let bottleneck_raw :=
    reluSpec (conv2dSpec (α := α) m.down2_2 b1)
  let bottleneck : MultiChannelImage (2 * cfg.baseC) downH downW α :=
    rwMultiChannelImage (α := α) bottleneck_raw (by rfl) h_convH_down h_convW_down

  -- Upsample (PyTorch analogy: `nn.ConvTranspose2d(kernel_size=2, stride=2, padding=0)`).
  let upRaw : MultiChannelImage cfg.baseC (UNetUpH cfg inH) (UNetUpW cfg inW) α :=
    convTranspose2dSpec (inC := 2 * cfg.baseC) (outC := cfg.baseC)
      (kH := cfg.upKernel) (kW := cfg.upKernel) (stride := cfg.upStride) (padding := cfg.upPadding)
      (inH := downH) (inW := downW) m.upT bottleneck

  let up : MultiChannelImage cfg.baseC inH inW α :=
    rwMultiChannelImage (α := α) upRaw (by rfl) h_upH h_upW

  -- Skip connection: concatenate channels (no batch axis in this file, so channels are axis 0).
  let merged : MultiChannelImage (cfg.baseC + cfg.baseC) inH inW α :=
    concatDim0Spec (t1 := skip1) (t2 := up)

  -- Up block
  let u1_raw :=
    reluSpec (conv2dSpec (α := α) m.up1_1 merged)
  let u1 : MultiChannelImage cfg.baseC inH inW α :=
    rwMultiChannelImage (α := α) u1_raw (by rfl) h_convH h_convW

  let u2_raw :=
    reluSpec (conv2dSpec (α := α) m.up1_2 u1)
  let u2 : MultiChannelImage cfg.baseC inH inW α :=
    rwMultiChannelImage (α := α) u2_raw (by rfl) h_convH h_convW

  -- Output
  let out_raw :=
    conv2dSpec (α := α) m.out1x1 u2
  rwMultiChannelImage (α := α) out_raw (by rfl) h_outH h_outW

/--
Backward pass for `UNet2Spec.forward`.

Given:
- the model parameters `m`,
- the forward input image `x`,
- an upstream gradient `grad_output = dL/dy`,
returns:
- parameter gradients (`UNet2Grads`), and
- the gradient w.r.t. the input image (`dL/dx`).

Implementation note: this is an explicit "recompute intermediates then walk backward" spec (no
mutable tape), mirroring the math behind PyTorch autograd and standard conv/pool backward rules.
-/
def UNet2Spec.backward
  {cfg : UNet2Config} {inC outC inH inW : Nat}
  {h_inC : inC ≠ 0} {hCfg : cfg.WF}
  (m : UNet2Spec (α := α) cfg inC outC inH inW h_inC hCfg)
  (x : MultiChannelImage inC inH inW α)
  (grad_output : MultiChannelImage outC inH inW α)
  (h_convH :
    ((inH + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) = inH)
  (h_convW :
    ((inW + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) = inW)
  (h_convH_down :
    ((UNetDownH cfg inH + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) = UNetDownH cfg inH)
  (h_convW_down :
    ((UNetDownW cfg inW + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) = UNetDownW cfg inW)
  (h_upH : UNetUpH cfg inH = inH)
  (h_upW : UNetUpW cfg inW = inW)
  (h_outH : ((inH + 2 * cfg.headPadding - cfg.headKernel) / cfg.headStride + 1) = inH)
  (h_outW : ((inW + 2 * cfg.headPadding - cfg.headKernel) / cfg.headStride + 1) = inW) :
  (UNet2Grads cfg inC outC inH inW α × MultiChannelImage inC inH inW α) :=

  -- Forward reconstruction (mirrors `UNet2Spec.forward`).
  -- We reconstruct intermediates because the backward rules (pooling / ReLU / conv) need the
  -- forward inputs (and in the case of max-pool, the values to determine which entries "won").
  let conv_down1_1 := conv2dSpec (α := α) m.down1_1 x
  let s1_raw := reluSpec conv_down1_1
  let s1 : MultiChannelImage cfg.baseC inH inW α :=
    rwMultiChannelImage (α := α) s1_raw (by rfl) h_convH h_convW

  let conv_down1_2 := conv2dSpec (α := α) m.down1_2 s1
  let skip1_raw := reluSpec conv_down1_2
  let skip1 : MultiChannelImage cfg.baseC inH inW α :=
    rwMultiChannelImage (α := α) skip1_raw (by rfl) h_convH h_convW

  let pool : MaxPool2DSpec cfg.poolKernel cfg.poolKernel cfg.poolStride hCfg.poolK_ne0 hCfg.poolK_ne0
      hCfg.poolStride_ne0 :=
    {}

  let downH := UNetDownH cfg inH
  let downW := UNetDownW cfg inW

  let pooled : MultiChannelImage cfg.baseC downH downW α :=
    maxPool2dMultiSpec (α := α) (layer := pool) skip1

  let conv_down2_1 := conv2dSpec (α := α) m.down2_1 pooled
  let b1_raw := reluSpec conv_down2_1
  let b1 : MultiChannelImage (2 * cfg.baseC) downH downW α :=
    rwMultiChannelImage (α := α) b1_raw (by rfl) h_convH_down h_convW_down

  let conv_down2_2 := conv2dSpec (α := α) m.down2_2 b1
  let bottleneck_raw := reluSpec conv_down2_2
  let bottleneck : MultiChannelImage (2 * cfg.baseC) downH downW α :=
    rwMultiChannelImage (α := α) bottleneck_raw (by rfl) h_convH_down h_convW_down

  let upRaw : MultiChannelImage cfg.baseC (UNetUpH cfg inH) (UNetUpW cfg inW) α :=
    convTranspose2dSpec (inC := 2 * cfg.baseC) (outC := cfg.baseC)
      (kH := cfg.upKernel) (kW := cfg.upKernel) (stride := cfg.upStride) (padding := cfg.upPadding)
      (inH := downH) (inW := downW) m.upT bottleneck

  let up : MultiChannelImage cfg.baseC inH inW α :=
    rwMultiChannelImage (α := α) upRaw (by rfl) h_upH h_upW

  let merged : MultiChannelImage (cfg.baseC + cfg.baseC) inH inW α :=
    concatDim0Spec (t1 := skip1) (t2 := up)

  let conv_up1_1 := conv2dSpec (α := α) m.up1_1 merged
  let u1_raw := reluSpec conv_up1_1
  let u1 : MultiChannelImage cfg.baseC inH inW α :=
    rwMultiChannelImage (α := α) u1_raw (by rfl) h_convH h_convW

  let conv_up1_2 := conv2dSpec (α := α) m.up1_2 u1
  let u2_raw := reluSpec conv_up1_2
  let u2 : MultiChannelImage cfg.baseC inH inW α :=
    rwMultiChannelImage (α := α) u2_raw (by rfl) h_convH h_convW

  let out_raw := conv2dSpec (α := α) m.out1x1 u2

  -- Backward starts here.
  -- For each ReLU, we backprop through it using the standard gate:
  -- `dZ = dY ⊙ ReLU'(Z)` where `Z` is the pre-activation tensor.
  let grad_out_raw :
      MultiChannelImage outC
        ((inH + 2 * cfg.headPadding - cfg.headKernel) / cfg.headStride + 1)
        ((inW + 2 * cfg.headPadding - cfg.headKernel) / cfg.headStride + 1) α :=
    rwMultiChannelImage (α := α) grad_output (by rfl) h_outH.symm h_outW.symm

  let (d_out1x1_kernel, d_out1x1_bias, d_u2) :=
    conv2dBackwardSpec (α := α)
      (inC := cfg.baseC) (outC := outC) (kH := cfg.headKernel) (kW := cfg.headKernel)
      (stride := cfg.headStride) (padding := cfg.headPadding)
      (inH := inH) (inW := inW)
      (h1 := Nat.ne_of_gt hCfg.baseC_pos) (h2 := hCfg.headK_ne0) (h3 := hCfg.headK_ne0)
      m.out1x1 u2 grad_out_raw

  let d_u2_raw :
      MultiChannelImage cfg.baseC
        ((inH + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1)
        ((inW + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) α :=
    rwMultiChannelImage (α := α) d_u2 (by rfl) h_convH.symm h_convW.symm

  let d_conv_up1_2 := mulSpec d_u2_raw (reluDerivSpec conv_up1_2)

  let (d_up1_2_kernel, d_up1_2_bias, d_u1) :=
    conv2dBackwardSpec (α := α)
      (inC := cfg.baseC) (outC := cfg.baseC) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := inH) (inW := inW)
      (h1 := Nat.ne_of_gt hCfg.baseC_pos) (h2 := hCfg.convK_ne0) (h3 := hCfg.convK_ne0)
      m.up1_2 u1 d_conv_up1_2

  let d_u1_raw :
      MultiChannelImage cfg.baseC
        ((inH + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1)
        ((inW + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) α :=
    rwMultiChannelImage (α := α) d_u1 (by rfl) h_convH.symm h_convW.symm

  let d_conv_up1_1 := mulSpec d_u1_raw (reluDerivSpec conv_up1_1)

  let (d_up1_1_kernel, d_up1_1_bias, d_merged) :=
    conv2dBackwardSpec (α := α)
      (inC := cfg.baseC + cfg.baseC) (outC := cfg.baseC) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := inH) (inW := inW)
      (h1 := by
        have : 0 < cfg.baseC + cfg.baseC :=
          Nat.lt_of_lt_of_le hCfg.baseC_pos (Nat.le_add_right cfg.baseC cfg.baseC)
        exact Nat.ne_of_gt this)
      (h2 := hCfg.convK_ne0) (h3 := hCfg.convK_ne0)
      m.up1_1 merged d_conv_up1_1

  -- Split concat backward: merged = concat(skip1, up).
  -- Channel-concat is linear, so its backward just splits the incoming gradient into the two
  -- channel ranges.
  let (d_skip1_from_merge, d_up) :=
    concatDim0BackwardSpec (α := α) (n := cfg.baseC) (m := cfg.baseC)
      (s := .dim inH (.dim inW .scalar))
      d_merged

  let d_upRaw : MultiChannelImage cfg.baseC (UNetUpH cfg inH) (UNetUpW cfg inW) α :=
    rwMultiChannelImage (α := α) d_up (by rfl) h_upH.symm h_upW.symm

  let (d_upT_kernel, d_upT_bias, d_bottleneck) :=
    convTranspose2dBackwardSpec
      (inC := 2 * cfg.baseC) (outC := cfg.baseC) (kH := cfg.upKernel) (kW := cfg.upKernel)
      (stride := cfg.upStride) (padding := cfg.upPadding)
      (inH := downH) (inW := downW)
      (h1 := Nat.mul_pos (by decide : 0 < 2) hCfg.baseC_pos) (h2 := hCfg.upK_ne0) (h3 := hCfg.upK_ne0)
      m.upT bottleneck d_upRaw

  let d_bottleneck_raw : MultiChannelImage (2 * cfg.baseC)
      ((UNetDownH cfg inH + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1)
      ((UNetDownW cfg inW + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) α :=
    rwMultiChannelImage (α := α) d_bottleneck (by rfl) h_convH_down.symm h_convW_down.symm

  let d_conv_down2_2 := mulSpec d_bottleneck_raw (reluDerivSpec conv_down2_2)

  let (d_down2_2_kernel, d_down2_2_bias, d_b1) :=
    conv2dBackwardSpec (α := α)
      (inC := 2 * cfg.baseC) (outC := 2 * cfg.baseC) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := downH) (inW := downW)
      (h1 := Nat.ne_of_gt (Nat.mul_pos (by decide : 0 < 2) hCfg.baseC_pos)) (h2 := hCfg.convK_ne0)
      (h3 := hCfg.convK_ne0)
      m.down2_2 b1 d_conv_down2_2

  let d_b1_raw : MultiChannelImage (2 * cfg.baseC)
      ((UNetDownH cfg inH + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1)
      ((UNetDownW cfg inW + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) α :=
    rwMultiChannelImage (α := α) d_b1 (by rfl) h_convH_down.symm h_convW_down.symm

  let d_conv_down2_1 := mulSpec d_b1_raw (reluDerivSpec conv_down2_1)

  let (d_down2_1_kernel, d_down2_1_bias, d_pooled) :=
    conv2dBackwardSpec (α := α)
      (inC := cfg.baseC) (outC := 2 * cfg.baseC) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := downH) (inW := downW)
      (h1 := Nat.ne_of_gt hCfg.baseC_pos) (h2 := hCfg.convK_ne0) (h3 := hCfg.convK_ne0)
      m.down2_1 pooled d_conv_down2_1

  -- Pool backward.
  -- MaxPool backward routes gradient back to the (per-window) argmax location.
  let d_skip1_from_pool :=
    maxPool2dMultiBackwardSpec (α := α) (layer := pool) (input := skip1) (grad_output :=
      d_pooled)

  let d_skip1_total := addSpec d_skip1_from_pool d_skip1_from_merge
  let d_skip1_raw : MultiChannelImage cfg.baseC
      ((inH + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1)
      ((inW + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) α :=
    rwMultiChannelImage (α := α) d_skip1_total (by rfl) h_convH.symm h_convW.symm

  let d_conv_down1_2 := mulSpec d_skip1_raw (reluDerivSpec conv_down1_2)

  let (d_down1_2_kernel, d_down1_2_bias, d_s1) :=
    conv2dBackwardSpec (α := α)
      (inC := cfg.baseC) (outC := cfg.baseC) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := inH) (inW := inW)
      (h1 := Nat.ne_of_gt hCfg.baseC_pos) (h2 := hCfg.convK_ne0) (h3 := hCfg.convK_ne0)
      m.down1_2 s1 d_conv_down1_2

  let d_s1_raw : MultiChannelImage cfg.baseC
      ((inH + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1)
      ((inW + 2 * cfg.convPadding - cfg.convKernel) / cfg.convStride + 1) α :=
    rwMultiChannelImage (α := α) d_s1 (by rfl) h_convH.symm h_convW.symm

  let d_conv_down1_1 := mulSpec d_s1_raw (reluDerivSpec conv_down1_1)

  let (d_down1_1_kernel, d_down1_1_bias, d_x) :=
    conv2dBackwardSpec (α := α)
      (inC := inC) (outC := cfg.baseC) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := inH) (inW := inW)
      (h1 := h_inC) (h2 := hCfg.convK_ne0) (h3 := hCfg.convK_ne0)
      m.down1_1 x d_conv_down1_1

  let grads : UNet2Grads cfg inC outC inH inW α :=
    { d_down1_1_kernel := d_down1_1_kernel
      d_down1_1_bias := d_down1_1_bias
      d_down1_2_kernel := d_down1_2_kernel
      d_down1_2_bias := d_down1_2_bias
      d_down2_1_kernel := d_down2_1_kernel
      d_down2_1_bias := d_down2_1_bias
      d_down2_2_kernel := d_down2_2_kernel
      d_down2_2_bias := d_down2_2_bias
      d_upT_kernel := d_upT_kernel
      d_upT_bias := d_upT_bias
      d_up1_1_kernel := d_up1_1_kernel
      d_up1_1_bias := d_up1_1_bias
      d_up1_2_kernel := d_up1_2_kernel
      d_up1_2_bias := d_up1_2_bias
      d_out1x1_kernel := d_out1x1_kernel
      d_out1x1_bias := d_out1x1_bias }

  (grads, d_x)

end Models
