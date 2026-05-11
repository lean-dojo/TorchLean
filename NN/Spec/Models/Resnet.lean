/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.GlobalPooling
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Normalization
public import NN.Spec.Layers.Pooling

/-!
# ResNet (spec model)

Defines a small ResNet‑style architecture with residual/skip connections.

PyTorch analogy: this mirrors the high-level structure of `torchvision.models.resnet*`:

- convolution + normalization + ReLU,
- residual blocks with an identity/projection shortcut,
- global average pooling,
- a final linear classifier.

Important scope note:

- Residual blocks below keep stride fixed to `1`, so spatial resolution is preserved across
  `layer1..layer4`. (Standard ResNet down-samples at the start of `layer2..layer4`; adding that is
  possible, but it complicates the type-level shape discipline.)
- Channel changes can be handled via an optional projection shortcut (`shortcut_conv`) plus
  optional shortcut batch-norm parameters. Builders such as `ResNetSpec.zeroInit` gate this behind
  `ResNetConfig.useProjectionShortcuts` (default `false`).
- If `shortcut_conv = none`, we still define a total shortcut by falling back to identity / channel
  padding / channel slicing.

Torchvision compatibility note:

- In standard ResNet (as implemented in `torchvision.models.resnet*`), whenever input/output
  channels differ, the shortcut path is a learnable **1×1 projection** (typically with BatchNorm).
- In TorchLean, that corresponds to enabling `cfg.useProjectionShortcuts = true` when constructing a
  `ResNetSpec` via helpers like `ResNetSpec.zeroInit`, or otherwise supplying `shortcut_conv := some`
  in the block parameters.

This is a *spec* model: operations are written in terms of `Spec.Tensor` and layer specs from
`NN/Spec/Layers/*`.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/--
ResNet architectural hyperparameters (simplified, spec-layer).

PyTorch mental model:
- `torchvision.models.resnet.ResNet` fixes a few "stem" choices (7×7 conv, stride 2, etc.) and
  varies the per-stage widths and block counts based on a small config.

TorchLean’s spec ResNet is intentionally smaller in scope:
- blocks keep `stride=1` so spatial resolution stays constant inside `layer1..layer4`,
- we still expose the stem / stage widths / stage block counts as explicit configuration so the
  model definition does not hide numeric architecture choices in its types.
-/
structure ResNetConfig where
  /-- Output channels of the initial conv stem (typical: `64`). -/
  stemOutChannels : Nat := 64
  /-- Kernel size of the initial conv stem (typical: `7`). -/
  stemKernel : Nat := 7
  /-- Stride of the initial conv stem (typical: `2`). -/
  stemStride : Nat := 2
  /-- Symmetric padding of the initial conv stem (typical: `3`). -/
  stemPadding : Nat := 3
  /-- MaxPool kernel size (typical: `3`). -/
  poolKernel : Nat := 3
  /-- MaxPool stride (typical: `2`). -/
  poolStride : Nat := 2

  /-- Stage 1 output channels (typical: `64`). -/
  stage1OutChannels : Nat := 64
  /-- Stage 1 block count (typical: `2` for ResNet-18). -/
  stage1Blocks : Nat := 2

  /-- Stage 2 output channels (typical: `128`). -/
  stage2OutChannels : Nat := 128
  /-- Stage 2 block count (typical: `2` for ResNet-18). -/
  stage2Blocks : Nat := 2

  /-- Stage 3 output channels (typical: `256`). -/
  stage3OutChannels : Nat := 256
  /-- Stage 3 block count (typical: `2` for ResNet-18). -/
  stage3Blocks : Nat := 2

  /-- Stage 4 output channels (typical: `512`). -/
  stage4OutChannels : Nat := 512
  /-- Stage 4 block count (typical: `2` for ResNet-18). -/
  stage4Blocks : Nat := 2

  /--
  If `true`, the **first** block in each stage whose channel count changes will use a learned
  `1×1` projection shortcut (and optional shortcut BN params) when constructed by helpers like
  `ResNetSpec.zeroInit`.

  If `false` (default), we omit the projection parameters and rely on the **total** fallback
  shortcut used by `ResNetBlockSpec.forward` when `shortcut_conv = none`:
  identity / channel padding / channel slicing.

  Note: this simplified Spec ResNet keeps the main-path stride fixed to `1`, so the projection
  shortcut (when enabled) also uses `stride = 1` here.
  -/
  useProjectionShortcuts : Bool := false

/--
Well-formedness conditions for `ResNetConfig`.

We keep these separate from the data record so "PyTorch-like configs" stay ergonomic, while still
letting the spec model use the nonzero facts needed by some layer specs.
-/
structure ResNetConfig.WF (cfg : ResNetConfig) : Prop where
  stemOut_ne0 : cfg.stemOutChannels ≠ 0
  stemK_ne0 : cfg.stemKernel ≠ 0
  poolK_ne0 : cfg.poolKernel ≠ 0
  poolStride_ne0 : cfg.poolStride ≠ 0
  c1_ne0 : cfg.stage1OutChannels ≠ 0
  c2_ne0 : cfg.stage2OutChannels ≠ 0
  c3_ne0 : cfg.stage3OutChannels ≠ 0
  c4_ne0 : cfg.stage4OutChannels ≠ 0

/-- Torchvision-style ResNet-18 hyperparameters (for our simplified spec). -/
def resnet18Config : ResNetConfig :=
  { stemOutChannels := 64
    stemKernel := 7
    stemStride := 2
    stemPadding := 3
    poolKernel := 3
    poolStride := 2
    stage1OutChannels := 64
    stage1Blocks := 2
    stage2OutChannels := 128
    stage2Blocks := 2
    stage3OutChannels := 256
    stage3Blocks := 2
    stage4OutChannels := 512
    stage4Blocks := 2 }

/-- `resnet18Config` satisfies the nonzero facts required by the spec layer. -/
theorem resnet18Config_wf : resnet18Config.WF := by
  refine
    { stemOut_ne0 := by decide
      stemK_ne0 := by decide
      poolK_ne0 := by decide
      poolStride_ne0 := by decide
      c1_ne0 := by decide
      c2_ne0 := by decide
      c3_ne0 := by decide
      c4_ne0 := by decide }

/--
`resnet18Config`, but with torchvision-style projection shortcuts enabled for stage transitions.

This keeps the same block counts and widths, but changes the *default* shortcut used by
`ResNetSpec.zeroInit` when channel counts change:
- `useProjectionShortcuts = false`: pad/slice fallback when `shortcut_conv = none` (default).
- `useProjectionShortcuts = true`: build a learned `1×1` shortcut conv (and shortcut BN params).
-/
def resnet18ConfigWithProjections : ResNetConfig :=
  { resnet18Config with useProjectionShortcuts := true }

theorem resnet18ConfigWithProjections_wf : resnet18ConfigWithProjections.WF := by
  refine
    { stemOut_ne0 := by
        simpa [resnet18ConfigWithProjections] using resnet18Config_wf.stemOut_ne0
      stemK_ne0 := by
        simpa [resnet18ConfigWithProjections] using resnet18Config_wf.stemK_ne0
      poolK_ne0 := by
        simpa [resnet18ConfigWithProjections] using resnet18Config_wf.poolK_ne0
      poolStride_ne0 := by
        simpa [resnet18ConfigWithProjections] using resnet18Config_wf.poolStride_ne0
      c1_ne0 := by
        simpa [resnet18ConfigWithProjections] using resnet18Config_wf.c1_ne0
      c2_ne0 := by
        simpa [resnet18ConfigWithProjections] using resnet18Config_wf.c2_ne0
      c3_ne0 := by
        simpa [resnet18ConfigWithProjections] using resnet18Config_wf.c3_ne0
      c4_ne0 := by
        simpa [resnet18ConfigWithProjections] using resnet18Config_wf.c4_ne0 }

/-- Output-size identity for a `3×3` conv with `stride=1` and `padding=1` (Nat-level formula). -/
theorem conv3x3_outSize_eq (n : Nat) (hn : n ≠ 0) :
    ((n + 2 * 1 - 3) / 1 + 1) = n := by
  -- `hn` rules out the `n=0` edge case where the saturated subtraction would break the identity.
  grind

/-- Output-size identity for a `1×1` conv with `stride=1` and `padding=0` (Nat-level formula). -/
theorem conv1x1_outSize_eq (n : Nat) (hn : n ≠ 0) :
    ((n + 2 * 0 - 1) / 1 + 1) = n := by
  grind

/-- Any Nat expression of the form `n + 1` is strictly positive. -/
lemma pos_add_one (n : Nat) : 0 < n + 1 := by
  -- `n + 1` reduces to `Nat.succ n`.
  exact Nat.succ_pos n

/-- Any Nat expression of the form `n + 1` is not `0`. -/
lemma ne_zero_add_one (n : Nat) : n + 1 ≠ 0 := by
  -- `n + 1` reduces to `Nat.succ n`.
  exact Nat.succ_ne_zero n

/-- A basic residual block (two 3×3 convolutions, each followed by BatchNorm; ReLU after the first
BN and after the residual addition).

This corresponds to the "basic block" used in ResNet-18/34.

PyTorch analogy (schematic):

`y = relu( bn2(conv2( relu(bn1(conv1(x))) )) + shortcut(x) )`
-/
structure ResNetBlockSpec (α : Type) (inChannels outChannels : Nat) (h1 : inChannels ≠ 0) (h2 :
  outChannels ≠ 0) where
  /-- conv 1. -/
  conv1 : Conv2DSpec inChannels outChannels 3 3 1 1 α h1 (by norm_num) (by norm_num)
  -- First convolution
  /-- conv 2. -/
  conv2 : Conv2DSpec outChannels outChannels 3 3 1 1 α h2 (by norm_num) (by norm_num)
  -- Second convolution
  /-- bn 1 gamma. -/
  bn1_gamma : Tensor α (.dim outChannels .scalar)        -- BatchNorm gamma for conv1
  /-- bn 1 beta. -/
  bn1_beta : Tensor α (.dim outChannels .scalar)         -- BatchNorm beta for conv1
  /-- bn 2 gamma. -/
  bn2_gamma : Tensor α (.dim outChannels .scalar)        -- BatchNorm gamma for conv2
  /-- bn 2 beta. -/
  bn2_beta : Tensor α (.dim outChannels .scalar)         -- BatchNorm beta for conv2
  /-- shortcut conv. -/
  shortcut_conv : Option (Conv2DSpec inChannels outChannels 1 1 1 0 α h1 (by norm_num) (by
    norm_num))
  -- Shortcut convolution
  /-- shortcut bn gamma. -/
  shortcut_bn_gamma : Option (Tensor α (.dim outChannels .scalar))  -- Shortcut BatchNorm gamma
  /-- shortcut bn beta. -/
  shortcut_bn_beta : Option (Tensor α (.dim outChannels .scalar))   -- Shortcut BatchNorm beta

/-- Forward pass for a basic residual block.

Type-level note: with stride=1 and padding=1, a 3×3 convolution preserves `H×W` (assuming `H,W>0`),
so the block input and output share spatial dimensions.
-/
def ResNetBlockSpec.forward {inChannels outChannels inH inW : Nat}
  (h1 : inChannels ≠ 0) (h2 : outChannels ≠ 0)
  (block : ResNetBlockSpec α inChannels outChannels h1 h2)
  (x : MultiChannelImage inChannels inH inW α)
  (h3 : inH ≠ 0) (h4 : inW ≠ 0) :
  MultiChannelImage outChannels inH inW α :=

  -- First convolution + BatchNorm + ReLU
  let conv1_out := conv2dSpec block.conv1 x
  let conv1_out' : MultiChannelImage outChannels inH inW α :=
    rwMultiChannelImage conv1_out (by rfl)
      (by simpa using conv3x3_outSize_eq inH h3)
      (by simpa using conv3x3_outSize_eq inW h4)
  let bn1_out :=
    batchNorm2d conv1_out' block.bn1_gamma block.bn1_beta
      (Nat.pos_of_ne_zero h2) (Nat.pos_of_ne_zero h3) (Nat.pos_of_ne_zero h4)
  let relu1_out := Activation.reluSpec bn1_out

  -- Second convolution + BatchNorm
  let conv2_out := conv2dSpec block.conv2 relu1_out
  let conv2_out' : MultiChannelImage outChannels inH inW α :=
    rwMultiChannelImage conv2_out (by rfl)
      (by simpa using conv3x3_outSize_eq inH h3)
      (by simpa using conv3x3_outSize_eq inW h4)
  let bn2_out :=
    batchNorm2d conv2_out' block.bn2_gamma block.bn2_beta
      (Nat.pos_of_ne_zero h2) (Nat.pos_of_ne_zero h3) (Nat.pos_of_ne_zero h4)

  -- Shortcut connection (both paths preserve spatial dimensions)
  let shortcut : MultiChannelImage outChannels inH inW α := match block.shortcut_conv with
  | some conv =>
    let shortcut_conv_out := conv2dSpec conv x
    let shortcut_conv_out' : MultiChannelImage outChannels inH inW α :=
      rwMultiChannelImage shortcut_conv_out (by rfl)
        (by simpa using conv1x1_outSize_eq inH h3)
        (by simpa using conv1x1_outSize_eq inW h4)
    match block.shortcut_bn_gamma, block.shortcut_bn_beta with
    | some gamma, some beta =>
      batchNorm2d shortcut_conv_out' gamma beta
        (Nat.pos_of_ne_zero h2) (Nat.pos_of_ne_zero h3) (Nat.pos_of_ne_zero h4)
    | _, _ => shortcut_conv_out'
  | none =>
    if h_channels: inChannels = outChannels then
      -- Identity shortcut (same dimensions)
      rwMultiChannelImage x h_channels (by rfl) (by rfl)
    else if h_lt : inChannels < outChannels then
      -- Expand channels by zero-padding (common in ResNet)
      padChannelsZero (Nat.le_of_lt h_lt) x
    else
      -- Channel reduction case: take the first `outChannels` channels.
      --
      -- In standard ResNet this situation is usually handled by a projection shortcut
      -- (a 1×1 conv). This branch keeps the model total even when `shortcut_conv = none`.
      have h_le : outChannels ≤ inChannels := Nat.le_of_not_gt h_lt
      sliceRange0Spec (α := α) (n := inChannels)
        (s := .dim inH (.dim inW .scalar)) 0 outChannels (by simpa using h_le) x

  -- Residual connection + ReLU
  let residual_out := addSpec bn2_out shortcut
  Activation.reluSpec residual_out

/-- A layer is a "first" block that can change channels, followed by zero or more homogeneous blocks
(same input/output channels).

We keep the `rest` blocks in a list rather than a fixed-length vector so that the definition stays
lightweight and easy to build in examples. The `blockCount` index is documentation: the list length
is the source of truth for "how many blocks are actually present".
-/
structure ResNetLayerSpec (α : Type) (inChannels outChannels blockCount : Nat)
  (h1 : inChannels ≠ 0) (h2 : outChannels ≠ 0) where
  first : ResNetBlockSpec α inChannels outChannels h1 h2
  rest  : List (ResNetBlockSpec α outChannels outChannels h2 h2)

/-- Forward pass for a ResNet layer: run the first block, then fold over the remaining blocks. -/
def ResNetLayerSpec.forward {inChannels outChannels blockCount inH inW : Nat}
  (h1 : inChannels ≠ 0) (h2 : outChannels ≠ 0)
  (layer : ResNetLayerSpec α inChannels outChannels blockCount h1 h2)
  (x : MultiChannelImage inChannels inH inW α)
  (h3 : inH ≠ 0) (h4 : inW ≠ 0) :
  MultiChannelImage outChannels inH inW α :=
  let firstOut := ResNetBlockSpec.forward h1 h2 layer.first x h3 h4
  layer.rest.foldl (fun acc block => ResNetBlockSpec.forward h2 h2 block acc h3 h4) firstOut

/-!
## Gradients (explicit reverse-mode)

ResNet is a good example of why the spec layer carries explicit shape structure:

- Residual connections force us to be precise about shapes and casting, otherwise the definition
  no longer states exactly which tensors are being added.
- The backward pass follows the same discipline: split gradients across the residual branches, run each
  layer's backward rule, then add the contributions back together.

We keep things simple:
- stride is fixed to `1` inside blocks (matching the forward spec above),
- we recompute intermediates locally (no global tape),
- optional projection shortcuts (`shortcut_conv`) are handled when present, and
  "fallback" shortcuts (pad/slice/identity) have explicit adjoints.
-/

/--
Parameter gradients for a basic residual block.

This mirrors the fields of `ResNetBlockSpec`, plus optional gradients for the optional projection
shortcut.
-/
structure ResNetBlockGrads (inChannels outChannels : Nat) (α : Type) where
  /-- d conv 1 kernel. -/
  d_conv1_kernel : Tensor α (.dim outChannels (.dim inChannels (.dim 3 (.dim 3 .scalar))))
  /-- d conv 1 bias. -/
  d_conv1_bias   : Tensor α (.dim outChannels .scalar)
  /-- d conv 2 kernel. -/
  d_conv2_kernel : Tensor α (.dim outChannels (.dim outChannels (.dim 3 (.dim 3 .scalar))))
  /-- d conv 2 bias. -/
  d_conv2_bias   : Tensor α (.dim outChannels .scalar)
  /-- d bn 1 gamma. -/
  d_bn1_gamma    : Tensor α (.dim outChannels .scalar)
  /-- d bn 1 beta. -/
  d_bn1_beta     : Tensor α (.dim outChannels .scalar)
  /-- d bn 2 gamma. -/
  d_bn2_gamma    : Tensor α (.dim outChannels .scalar)
  /-- d bn 2 beta. -/
  d_bn2_beta     : Tensor α (.dim outChannels .scalar)
  /-- d shortcut conv. -/
  d_shortcut_conv :
    Option (Tensor α (.dim outChannels (.dim inChannels (.dim 1 (.dim 1 .scalar)))) × Tensor α (.dim
      outChannels .scalar))
  /-- d shortcut bn gamma. -/
  d_shortcut_bn_gamma : Option (Tensor α (.dim outChannels .scalar))
  /-- d shortcut bn beta. -/
  d_shortcut_bn_beta  : Option (Tensor α (.dim outChannels .scalar))

/-- Backward/VJP for a basic residual block.

High-level math:

- The block output is `relu(main(x) + shortcut(x))`.
- Backprop therefore:
  1. multiplies by `relu'` at the output,
  2. splits the upstream gradient across the `+` into the main path and the shortcut path,
  3. runs BN/conv adjoints on the main path,
  4. runs either the projection-conv adjoint or the identity/pad/slice adjoint on the shortcut,
  5. adds the resulting `dX` contributions.
-/
def ResNetBlockSpec.backward {inChannels outChannels inH inW : Nat}
  (h1 : inChannels ≠ 0) (h2 : outChannels ≠ 0)
  (block : ResNetBlockSpec α inChannels outChannels h1 h2)
  (x : MultiChannelImage inChannels inH inW α)
  (grad_output : MultiChannelImage outChannels inH inW α)
  (h3 : inH ≠ 0) (h4 : inW ≠ 0) :
  (ResNetBlockGrads inChannels outChannels α × MultiChannelImage inChannels inH inW α) :=

  let H1 : Nat := (inH + 2 * 1 - 3) / 1 + 1
  let W1 : Nat := (inW + 2 * 1 - 3) / 1 + 1
  let H2 : Nat := (H1 + 2 * 1 - 3) / 1 + 1
  let W2 : Nat := (W1 + 2 * 1 - 3) / 1 + 1

  -- Helpful shape equalities for "spatial dims are preserved" (stride=1, padding=1 convs).
  have hH1_eq : H1 = inH := by
    dsimp [H1]
    exact conv3x3_outSize_eq inH h3
  have hW1_eq : W1 = inW := by
    dsimp [W1]
    exact conv3x3_outSize_eq inW h4
  have hH1_ne : H1 ≠ 0 := by
    intro h0
    apply h3
    simp [hH1_eq] at h0
    exact h0
  have hW1_ne : W1 ≠ 0 := by
    intro h0
    apply h4
    simp [hW1_eq] at h0
    exact h0
  have hH2_eq_H1 : H2 = H1 := by
    dsimp [H2]
    exact conv3x3_outSize_eq H1 hH1_ne
  have hW2_eq_W1 : W2 = W1 := by
    dsimp [W2]
    exact conv3x3_outSize_eq W1 hW1_ne
  have hH2_eq : H2 = inH := hH2_eq_H1.trans hH1_eq
  have hW2_eq : W2 = inW := hW2_eq_W1.trans hW1_eq

  have hH1_pos : H1 > 0 := by
    rw [hH1_eq]
    exact Nat.pos_of_ne_zero h3
  have hW1_pos : W1 > 0 := by
    rw [hW1_eq]
    exact Nat.pos_of_ne_zero h4
  have hH2_pos : H2 > 0 := by
    rw [hH2_eq]
    exact Nat.pos_of_ne_zero h3
  have hW2_pos : W2 > 0 := by
    rw [hW2_eq]
    exact Nat.pos_of_ne_zero h4

  have hH1x1_eq : ((inH + 2 * 0 - 1) / 1 + 1) = inH := conv1x1_outSize_eq inH h3
  have hW1x1_eq : ((inW + 2 * 0 - 1) / 1 + 1) = inW := conv1x1_outSize_eq inW h4
  have hH1x1_to_H2 : ((inH + 2 * 0 - 1) / 1 + 1) = H2 := hH1x1_eq.trans hH2_eq.symm
  have hW1x1_to_W2 : ((inW + 2 * 0 - 1) / 1 + 1) = W2 := hW1x1_eq.trans hW2_eq.symm

  -- Forward reconstruction.
  let conv1_out := conv2dSpec block.conv1 x
  let bn1_out :=
    batchNorm2d conv1_out block.bn1_gamma block.bn1_beta
      (Nat.pos_of_ne_zero h2) hH1_pos hW1_pos
  let relu1_out := Activation.reluSpec bn1_out

  let conv2_out := conv2dSpec block.conv2 relu1_out
  let bn2_out :=
    batchNorm2d conv2_out block.bn2_gamma block.bn2_beta
      (Nat.pos_of_ne_zero h2) hH2_pos hW2_pos

  let shortcut : MultiChannelImage outChannels H2 W2 α :=
    match block.shortcut_conv with
    | some conv =>
      let shortcut_conv_out := conv2dSpec conv x
      let shortcut_conv_out' : MultiChannelImage outChannels H2 W2 α :=
        rwMultiChannelImage shortcut_conv_out (by rfl) hH1x1_to_H2 hW1x1_to_W2
      match block.shortcut_bn_gamma, block.shortcut_bn_beta with
      | some gamma, some beta =>
        batchNorm2d shortcut_conv_out' gamma beta
          (Nat.pos_of_ne_zero h2) hH2_pos hW2_pos
      | _, _ => shortcut_conv_out'
    | none =>
      if h_channels : inChannels = outChannels then
        let x' : MultiChannelImage outChannels H2 W2 α :=
          rwMultiChannelImage x h_channels hH2_eq.symm hW2_eq.symm
        x'
      else if h_lt : inChannels < outChannels then
        let x' : MultiChannelImage inChannels H2 W2 α :=
          rwMultiChannelImage x (by rfl) hH2_eq.symm hW2_eq.symm
        padChannelsZero (Nat.le_of_lt h_lt) x'
      else
        have h_le : outChannels ≤ inChannels := Nat.le_of_not_gt h_lt
        let x' : MultiChannelImage inChannels H2 W2 α :=
          rwMultiChannelImage x (by rfl) hH2_eq.symm hW2_eq.symm
        sliceRange0Spec (α := α) (n := inChannels) (s := .dim H2 (.dim W2 .scalar))
          0 outChannels (by simpa using h_le) x'

  let residual_out := addSpec bn2_out shortcut
  let residual_out' : MultiChannelImage outChannels inH inW α :=
    rwMultiChannelImage residual_out (by rfl) hH2_eq hW2_eq
  let y := Activation.reluSpec residual_out'

  -- Backprop through final ReLU.
  let d_residual_out' := mulSpec grad_output (Activation.reluDerivSpec residual_out')

  -- Cast gradient back through the explicit shape transport in `residual_out'`.
  let d_residual_out : MultiChannelImage outChannels H2 W2 α :=
    rwMultiChannelImage (α := α) d_residual_out' (by rfl) hH2_eq.symm hW2_eq.symm

  -- Split residual: residual_out' = bn2_out + shortcut.
  let d_bn2_out := d_residual_out
  let d_shortcut := d_residual_out
  let d_shortcut_inH : MultiChannelImage outChannels inH inW α :=
    rwMultiChannelImage (α := α) d_shortcut (by rfl) hH2_eq hW2_eq

  -- Backprop through BN2.
  let (d_conv2_out, d_bn2_gamma, d_bn2_beta) :=
    batchNorm2dBackward (α := α)
      (x := conv2_out) (gamma := block.bn2_gamma) (grad_output := d_bn2_out)
      (_h_c := Nat.pos_of_ne_zero h2) (_h_h := hH2_pos) (_h_w := hW2_pos)

  -- Conv2 backward.
  let (d_conv2_kernel, d_conv2_bias, d_relu1_out) :=
    conv2dBackwardSpec (α := α)
      (inC := outChannels) (outC := outChannels) (kH := 3) (kW := 3) (stride := 1) (padding := 1)
      (inH := H1) (inW := W1)
      (h1 := h2) (h2 := by norm_num) (h3 := by norm_num)
      block.conv2 relu1_out d_conv2_out

  -- ReLU1 backward.
  let d_bn1_out := mulSpec d_relu1_out (Activation.reluDerivSpec bn1_out)

  -- Backprop through BN1.
  let (d_conv1_out, d_bn1_gamma, d_bn1_beta) :=
    batchNorm2dBackward (α := α)
      (x := conv1_out) (gamma := block.bn1_gamma) (grad_output := d_bn1_out)
      (_h_c := Nat.pos_of_ne_zero h2) (_h_h := hH1_pos) (_h_w := hW1_pos)

  -- Conv1 backward.
  let (d_conv1_kernel, d_conv1_bias, d_x_main) :=
    conv2dBackwardSpec (α := α)
      (inC := inChannels) (outC := outChannels) (kH := 3) (kW := 3) (stride := 1) (padding := 1)
      (inH := inH) (inW := inW)
      (h1 := h1) (h2 := by norm_num) (h3 := by norm_num)
      block.conv1 x d_conv1_out

  -- Shortcut backward.
  let (d_x_shortcut, d_shortcut_conv, d_shortcut_bn_gamma, d_shortcut_bn_beta) :
      (MultiChannelImage inChannels inH inW α ×
       Option (Tensor α (.dim outChannels (.dim inChannels (.dim 1 (.dim 1 .scalar)))) × Tensor α
         (.dim outChannels .scalar)) ×
       Option (Tensor α (.dim outChannels .scalar)) ×
       Option (Tensor α (.dim outChannels .scalar))) :=
    match block.shortcut_conv with
    | some conv =>
      let shortcut_conv_out := conv2dSpec conv x
      let shortcut_conv_out' :=
        rwMultiChannelImage shortcut_conv_out (by rfl) hH1x1_to_H2 hW1x1_to_W2
      match block.shortcut_bn_gamma, block.shortcut_bn_beta with
      | some gamma, some beta =>
        let _ :=
          batchNorm2d shortcut_conv_out' gamma beta
            (Nat.pos_of_ne_zero h2) hH2_pos hW2_pos
        let (d_sc_conv_out, d_sc_gamma, d_sc_beta) :=
          batchNorm2dBackward (α := α)
            (x := shortcut_conv_out') (gamma := gamma) (grad_output := d_shortcut)
            (_h_c := Nat.pos_of_ne_zero h2) (_h_h := hH2_pos) (_h_w := hW2_pos)
        let d_sc_conv_out0 : MultiChannelImage outChannels ((inH + 2 * 0 - 1) / 1 + 1) ((inW + 2 * 0
          - 1) / 1 + 1) α :=
          rwMultiChannelImage (α := α) d_sc_conv_out (by rfl) hH1x1_to_H2.symm hW1x1_to_W2.symm
        let (dK, dB, dX) :=
          conv2dBackwardSpec (α := α)
            (inC := inChannels) (outC := outChannels) (kH := 1) (kW := 1) (stride := 1) (padding :=
              0)
            (inH := inH) (inW := inW)
            (h1 := h1) (h2 := by norm_num) (h3 := by norm_num)
            conv x d_sc_conv_out0
        (dX, some (dK, dB), some d_sc_gamma, some d_sc_beta)
      | _, _ =>
        let d_shortcut0 : MultiChannelImage outChannels ((inH + 2 * 0 - 1) / 1 + 1) ((inW + 2 * 0 -
          1) / 1 + 1) α :=
          rwMultiChannelImage (α := α) d_shortcut_inH (by rfl) hH1x1_eq.symm hW1x1_eq.symm
        let (dK, dB, dX) :=
          conv2dBackwardSpec (α := α)
            (inC := inChannels) (outC := outChannels) (kH := 1) (kW := 1) (stride := 1) (padding :=
              0)
            (inH := inH) (inW := inW)
            (h1 := h1) (h2 := by norm_num) (h3 := by norm_num)
            conv x d_shortcut0
        (dX, some (dK, dB), none, none)
    | none =>
      if h_channels : inChannels = outChannels then
        let dX := rwMultiChannelImage d_shortcut_inH h_channels.symm (by rfl) (by rfl)
        (dX, none, none, none)
      else if h_lt : inChannels < outChannels then
        -- pad forward: adjoint slices the first `inChannels` channels.
        have h_le : inChannels ≤ outChannels := Nat.le_of_lt h_lt
        let dX :=
          sliceRange0Spec (α := α) (n := outChannels) (s := .dim inH (.dim inW .scalar))
            0 inChannels (by simpa using h_le) d_shortcut_inH
        (dX, none, none, none)
      else
        -- slice forward: adjoint injects into the first `outChannels` channels and zeros the rest.
        have h_le : outChannels ≤ inChannels := Nat.le_of_not_gt h_lt
        let dX :=
          sliceRange0BackwardSpec (α := α) (n := inChannels) (s := .dim inH (.dim inW .scalar))
            0 outChannels (by simpa using h_le) d_shortcut_inH
        (dX, none, none, none)

  let d_x := addSpec d_x_main d_x_shortcut

  let grads : ResNetBlockGrads inChannels outChannels α :=
    { d_conv1_kernel := d_conv1_kernel
      d_conv1_bias := d_conv1_bias
      d_conv2_kernel := d_conv2_kernel
      d_conv2_bias := d_conv2_bias
      d_bn1_gamma := d_bn1_gamma
      d_bn1_beta := d_bn1_beta
      d_bn2_gamma := d_bn2_gamma
      d_bn2_beta := d_bn2_beta
      d_shortcut_conv := d_shortcut_conv
      d_shortcut_bn_gamma := d_shortcut_bn_gamma
      d_shortcut_bn_beta := d_shortcut_bn_beta }

  (grads, d_x)

/-- Gradients for a `ResNetLayerSpec`: one gradient bundle per block. -/
structure ResNetLayerGrads (inChannels outChannels : Nat) (α : Type) where
  /-- first. -/
  first : ResNetBlockGrads inChannels outChannels α
  /-- rest. -/
  rest  : List (ResNetBlockGrads outChannels outChannels α)

/-- Backward/VJP for a ResNet layer.

Implementation note: the `rest` blocks are a list, so we explicitly reconstruct the intermediate
inputs needed for each block's backward pass. This keeps the spec self-contained (no global tape).
-/
def ResNetLayerSpec.backward {inChannels outChannels blockCount inH inW : Nat}
  (h1 : inChannels ≠ 0) (h2 : outChannels ≠ 0)
  (layer : ResNetLayerSpec α inChannels outChannels blockCount h1 h2)
  (x : MultiChannelImage inChannels inH inW α)
  (grad_output : MultiChannelImage outChannels inH inW α)
  (h3 : inH ≠ 0) (h4 : inW ≠ 0) :
  (ResNetLayerGrads inChannels outChannels α × MultiChannelImage inChannels inH inW α) :=

  let firstOut := ResNetBlockSpec.forward h1 h2 layer.first x h3 h4

  -- Collect inputs for the homogeneous `rest` blocks.
  let rec collect_inputs
    (blocks : List (ResNetBlockSpec α outChannels outChannels h2 h2))
    (cur : MultiChannelImage outChannels inH inW α) :
    List (MultiChannelImage outChannels inH inW α) :=
    match blocks with
    | [] => []
    | b :: bs =>
        let next := ResNetBlockSpec.forward h2 h2 b cur h3 h4
        cur :: collect_inputs bs next

  let inputs := collect_inputs layer.rest firstOut
  let pairs := List.zip layer.rest inputs

  let step :
    (List (ResNetBlockGrads outChannels outChannels α) × MultiChannelImage outChannels inH inW α) →
    (ResNetBlockSpec α outChannels outChannels h2 h2 × MultiChannelImage outChannels inH inW α) →
    (List (ResNetBlockGrads outChannels outChannels α) × MultiChannelImage outChannels inH inW α) :=
    fun (accGrads, grad) (blk, inp) =>
      let (g, dInp) := ResNetBlockSpec.backward h2 h2 blk inp grad h3 h4
      (g :: accGrads, dInp)

  let (revRestGrads, d_firstOut) := (pairs.reverse).foldl step ([], grad_output)
  let restGrads := revRestGrads.reverse

  let (firstGrad, d_x) := ResNetBlockSpec.backward h1 h2 layer.first x d_firstOut h3 h4
  ({ first := firstGrad, rest := restGrads }, d_x)

/-- Full ResNet-18-like specification.

Pipeline (schematic):

`conv7x7/stride2 -> BN -> ReLU -> maxpool/stride2 -> layer1 -> layer2 -> layer3 -> layer4
 -> global_avg_pool -> linear classifier`.

PyTorch analogy: this matches the main stages of `torchvision.models.resnet18`, but recall the
"simplified stride=1 blocks" note from the file header.
-/
structure ResNetSpec (cfg : ResNetConfig) (α : Type) (inputChannels numClasses : Nat)
  (h1 : inputChannels ≠ 0) (h2 : numClasses ≠ 0) (hCfg : cfg.WF) where
  /-- initial conv. -/
  initial_conv :
    Conv2DSpec inputChannels cfg.stemOutChannels cfg.stemKernel cfg.stemKernel cfg.stemStride
      cfg.stemPadding α h1 hCfg.stemK_ne0 hCfg.stemK_ne0
  -- Initial convolution
  /-- initial bn gamma. -/
  initial_bn_gamma : Tensor α (.dim cfg.stemOutChannels .scalar)  -- Initial BatchNorm gamma
  /-- initial bn beta. -/
  initial_bn_beta : Tensor α (.dim cfg.stemOutChannels .scalar)   -- Initial BatchNorm beta
  /-- initial pool. -/
  initial_pool :
    MaxPool2DSpec cfg.poolKernel cfg.poolKernel cfg.poolStride hCfg.poolK_ne0 hCfg.poolK_ne0
      hCfg.poolStride_ne0
  -- Initial max pooling

  /-- layer 1. -/
  layer1 :
    ResNetLayerSpec α cfg.stemOutChannels cfg.stage1OutChannels cfg.stage1Blocks hCfg.stemOut_ne0
      hCfg.c1_ne0
  -- Stage 1
  /-- layer 2. -/
  layer2 :
    ResNetLayerSpec α cfg.stage1OutChannels cfg.stage2OutChannels cfg.stage2Blocks hCfg.c1_ne0
      hCfg.c2_ne0
  -- Stage 2
  /-- layer 3. -/
  layer3 :
    ResNetLayerSpec α cfg.stage2OutChannels cfg.stage3OutChannels cfg.stage3Blocks hCfg.c2_ne0
      hCfg.c3_ne0
  -- Stage 3
  /-- layer 4. -/
  layer4 :
    ResNetLayerSpec α cfg.stage3OutChannels cfg.stage4OutChannels cfg.stage4Blocks hCfg.c3_ne0
      hCfg.c4_ne0
  -- Stage 4

  /-- classifier. -/
  classifier : LinearSpec α cfg.stage4OutChannels numClasses  -- Final classifier

/-- Forward pass for the ResNet spec.

PyTorch analogy:

- `global_avg_pool2d_flat_spec` corresponds to `AdaptiveAvgPool2d((1,1))` followed by flattening.
- `linear_spec` corresponds to the final `nn.Linear(cfg.stage4OutChannels, numClasses)`.
-/
def ResNetSpec.forward {cfg : ResNetConfig} {inputChannels numClasses inH inW : Nat}
  (h1 : inputChannels ≠ 0) (h2 : numClasses ≠ 0) (hCfg : cfg.WF)
  (resnet : ResNetSpec cfg α inputChannels numClasses h1 h2 hCfg)
  (x : MultiChannelImage inputChannels inH inW α)
  (_h3 : inH ≠ 0) (_h4 : inW ≠ 0) :
  Tensor α (.dim numClasses .scalar) :=

    -- Initial convolution + BatchNorm + ReLU + MaxPool
    let conv_out := conv2dSpec resnet.initial_conv x
    let bn_out :=
      batchNorm2d conv_out resnet.initial_bn_gamma resnet.initial_bn_beta
        (h_c := Nat.pos_of_ne_zero hCfg.stemOut_ne0) (h_h := pos_add_one _) (h_w := pos_add_one _)
    let relu_out := Activation.reluSpec bn_out
    let pool_out := maxPool2dMultiSpec resnet.initial_pool relu_out

    -- ResNet layers
    let layer1_out := ResNetLayerSpec.forward hCfg.stemOut_ne0 hCfg.c1_ne0 resnet.layer1 pool_out
      (ne_zero_add_one _) (ne_zero_add_one _)
    let layer2_out := ResNetLayerSpec.forward hCfg.c1_ne0 hCfg.c2_ne0 resnet.layer2 layer1_out
      (ne_zero_add_one _) (ne_zero_add_one _)
    let layer3_out := ResNetLayerSpec.forward hCfg.c2_ne0 hCfg.c3_ne0 resnet.layer3 layer2_out
      (ne_zero_add_one _) (ne_zero_add_one _)
    let layer4_out := ResNetLayerSpec.forward hCfg.c3_ne0 hCfg.c4_ne0 resnet.layer4 layer3_out
      (ne_zero_add_one _) (ne_zero_add_one _)

    -- Global average pooling
    let pooled_out := globalAvgPool2dFlatSpec (ne_zero_add_one _) (ne_zero_add_one _)
      (GlobalAvgPool2DSpec.mk) layer4_out

    -- Final classifier
    linearSpec resnet.classifier pooled_out

/-- Gradients for the full `ResNetSpec` forward pass (explicit reverse-mode). -/
structure ResNetGrads (cfg : ResNetConfig) (inputChannels numClasses : Nat) (α : Type) where
  /-- d initial kernel. -/
  d_initial_kernel :
    Tensor α
      (.dim cfg.stemOutChannels (.dim inputChannels (.dim cfg.stemKernel (.dim cfg.stemKernel .scalar))))
  /-- d initial bias. -/
  d_initial_bias   : Tensor α (.dim cfg.stemOutChannels .scalar)
  /-- d initial bn gamma. -/
  d_initial_bn_gamma : Tensor α (.dim cfg.stemOutChannels .scalar)
  /-- d initial bn beta. -/
  d_initial_bn_beta  : Tensor α (.dim cfg.stemOutChannels .scalar)
  /-- d layer 1. -/
  d_layer1 : ResNetLayerGrads cfg.stemOutChannels cfg.stage1OutChannels α
  /-- d layer 2. -/
  d_layer2 : ResNetLayerGrads cfg.stage1OutChannels cfg.stage2OutChannels α
  /-- d layer 3. -/
  d_layer3 : ResNetLayerGrads cfg.stage2OutChannels cfg.stage3OutChannels α
  /-- d layer 4. -/
  d_layer4 : ResNetLayerGrads cfg.stage3OutChannels cfg.stage4OutChannels α
  /-- d classifier W. -/
  d_classifier_W : Tensor α (.dim numClasses (.dim cfg.stage4OutChannels .scalar))
  /-- d classifier b. -/
  d_classifier_b : Tensor α (.dim numClasses .scalar)

/-- Backward/VJP for the full ResNet spec.

This follows the same structure as the forward pass, but in reverse:

1. classifier backward,
2. global avg-pool backward,
3. `layer4..layer1` backward,
4. max-pool backward,
5. initial ReLU backward,
6. initial BN backward,
7. initial conv backward.
-/
def ResNetSpec.backward {cfg : ResNetConfig} {inputChannels numClasses inH inW : Nat}
  (h1 : inputChannels ≠ 0) (h2 : numClasses ≠ 0) (hCfg : cfg.WF)
  (resnet : ResNetSpec cfg α inputChannels numClasses h1 h2 hCfg)
  (x : MultiChannelImage inputChannels inH inW α)
  (grad_output : Tensor α (.dim numClasses .scalar))
  (_h3 : inH ≠ 0) (_h4 : inW ≠ 0) :
  (ResNetGrads cfg inputChannels numClasses α × MultiChannelImage inputChannels inH inW α) :=

    -- Forward reconstruction.
    let conv_out := conv2dSpec resnet.initial_conv x
    let bn_out :=
      batchNorm2d conv_out resnet.initial_bn_gamma resnet.initial_bn_beta
        (h_c := Nat.pos_of_ne_zero hCfg.stemOut_ne0) (h_h := pos_add_one _) (h_w := pos_add_one _)
    let relu_out := Activation.reluSpec bn_out
    let pool_out := maxPool2dMultiSpec resnet.initial_pool relu_out

    let layer1_out := ResNetLayerSpec.forward hCfg.stemOut_ne0 hCfg.c1_ne0 resnet.layer1 pool_out
      (ne_zero_add_one _) (ne_zero_add_one _)
    let layer2_out := ResNetLayerSpec.forward hCfg.c1_ne0 hCfg.c2_ne0 resnet.layer2 layer1_out
      (ne_zero_add_one _) (ne_zero_add_one _)
    let layer3_out := ResNetLayerSpec.forward hCfg.c2_ne0 hCfg.c3_ne0 resnet.layer3 layer2_out
      (ne_zero_add_one _) (ne_zero_add_one _)
    let layer4_out := ResNetLayerSpec.forward hCfg.c3_ne0 hCfg.c4_ne0 resnet.layer4 layer3_out
      (ne_zero_add_one _) (ne_zero_add_one _)

    let pooled_out := globalAvgPool2dFlatSpec (ne_zero_add_one _) (ne_zero_add_one _)
      (GlobalAvgPool2DSpec.mk) layer4_out
    let _y := linearSpec resnet.classifier pooled_out

  -- Classifier backward.
  let (dW_cls, db_cls, d_pooled) := linearBackwardSpec (α := α) resnet.classifier pooled_out
    grad_output

    -- Global average pool backward.
    let d_layer4_out :=
      globalAvgPool2dFlatBackwardSpec (α := α) (inC := cfg.stage4OutChannels)
        (_h1 := ne_zero_add_one _) (_h2 := ne_zero_add_one _) (_layer := GlobalAvgPool2DSpec.mk)
          d_pooled

    -- ResNet layers backward (reverse order).
    let (d_layer4, d_layer3_out) :=
      ResNetLayerSpec.backward hCfg.c3_ne0 hCfg.c4_ne0 resnet.layer4 layer3_out d_layer4_out
        (ne_zero_add_one _) (ne_zero_add_one _)
    let (d_layer3, d_layer2_out) :=
      ResNetLayerSpec.backward hCfg.c2_ne0 hCfg.c3_ne0 resnet.layer3 layer2_out d_layer3_out
        (ne_zero_add_one _) (ne_zero_add_one _)
    let (d_layer2, d_layer1_out) :=
      ResNetLayerSpec.backward hCfg.c1_ne0 hCfg.c2_ne0 resnet.layer2 layer1_out d_layer2_out
        (ne_zero_add_one _) (ne_zero_add_one _)
    let (d_layer1, d_pool_out) :=
      ResNetLayerSpec.backward hCfg.stemOut_ne0 hCfg.c1_ne0 resnet.layer1 pool_out d_layer1_out
        (ne_zero_add_one _) (ne_zero_add_one _)

  -- Initial max-pool backward.
  let d_relu_out :=
    maxPool2dMultiBackwardSpec (α := α) (layer := resnet.initial_pool) (input := relu_out)
      (grad_output := d_pool_out)

  -- Initial ReLU backward.
  let d_bn_out := mulSpec d_relu_out (Activation.reluDerivSpec bn_out)

    -- Initial BN backward.
    let (d_conv_out, d_bn_gamma, d_bn_beta) :=
      batchNorm2dBackward (α := α)
        (x := conv_out) (gamma := resnet.initial_bn_gamma) (grad_output := d_bn_out)
        (_h_c := Nat.pos_of_ne_zero hCfg.stemOut_ne0) (_h_h := pos_add_one _) (_h_w := pos_add_one _)

    -- Initial conv backward.
    let (d_initial_kernel, d_initial_bias, d_x) :=
      conv2dBackwardSpec (α := α)
        (inC := inputChannels) (outC := cfg.stemOutChannels) (kH := cfg.stemKernel)
        (kW := cfg.stemKernel) (stride := cfg.stemStride) (padding := cfg.stemPadding)
        (inH := inH) (inW := inW)
        (h1 := h1) (h2 := hCfg.stemK_ne0) (h3 := hCfg.stemK_ne0)
        resnet.initial_conv x d_conv_out

    let grads : ResNetGrads cfg inputChannels numClasses α :=
      { d_initial_kernel := d_initial_kernel
        d_initial_bias := d_initial_bias
        d_initial_bn_gamma := d_bn_gamma
        d_initial_bn_beta := d_bn_beta
        d_layer1 := d_layer1
        d_layer2 := d_layer2
        d_layer3 := d_layer3
        d_layer4 := d_layer4
        d_classifier_W := dW_cls
        d_classifier_b := db_cls }

  (grads, d_x)

/--
Construct a simplified ResNet-18 spec (zero/one initialization).

This is primarily a runnable/spec baseline and a shape-checking harness. It does not aim to match
the trained torchvision weights or initialization schemes.
-/
def ResNetSpec.zeroInit (cfg : ResNetConfig) (hCfg : cfg.WF) (α : Type) [Context α]
  (inputChannels numClasses : Nat) (h1 : inputChannels ≠ 0) (h2 : numClasses ≠ 0) :
  ResNetSpec cfg α inputChannels numClasses h1 h2 hCfg :=
by
  -- Small local helpers to avoid repeating deeply-nested `Tensor.dim` constructors.
  let zero4 {a b c d : Nat} : Tensor α (.dim a (.dim b (.dim c (.dim d .scalar)))) :=
    Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar (0 : α)))))
  let zero2 {a b : Nat} : Tensor α (.dim a (.dim b .scalar)) :=
    Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar (0 : α)))
  let zero1 {a : Nat} : Tensor α (.dim a .scalar) :=
    Tensor.dim (fun _ => Tensor.scalar (0 : α))
  let one1 {a : Nat} : Tensor α (.dim a .scalar) :=
    Tensor.dim (fun _ => Tensor.scalar (1 : α))

  let zeroConvStem :
      Conv2DSpec inputChannels cfg.stemOutChannels cfg.stemKernel cfg.stemKernel cfg.stemStride
        cfg.stemPadding α h1 hCfg.stemK_ne0 hCfg.stemK_ne0 :=
    { kernel := zero4, bias := zero1 }

  let mkProj
      {inC outC : Nat} (h_inC : inC ≠ 0) (need : Bool) :
      Option (Conv2DSpec inC outC 1 1 1 0 α h_inC (by decide) (by decide)) :=
    if need then
      some { kernel := zero4, bias := zero1 }
    else
      none

  let mkProjBN (need : Bool) {c : Nat} : Option (Tensor α (.dim c .scalar)) :=
    if need then some (one1) else none

  let mkProjBN0 (need : Bool) {c : Nat} : Option (Tensor α (.dim c .scalar)) :=
    if need then some (zero1) else none

  -- Whether a stage needs a projection shortcut (channels change).
  --
  -- We default to "no learned projection shortcut" (pad/slice fallback) so the Spec remains
  -- total even without extra parameters. Set `cfg.useProjectionShortcuts = true` to opt into
  -- torchvision-style 1×1 projection shortcuts on channel changes.
  let needProj1 : Bool := cfg.useProjectionShortcuts && decide (cfg.stemOutChannels ≠ cfg.stage1OutChannels)
  let needProj2 : Bool := cfg.useProjectionShortcuts && decide (cfg.stage1OutChannels ≠ cfg.stage2OutChannels)
  let needProj3 : Bool := cfg.useProjectionShortcuts && decide (cfg.stage2OutChannels ≠ cfg.stage3OutChannels)
  let needProj4 : Bool := cfg.useProjectionShortcuts && decide (cfg.stage3OutChannels ≠ cfg.stage4OutChannels)

  -- Build layer blocks explicitly (we thread the `outC ≠ 0` proofs from `hCfg`).
  let layer1_first : ResNetBlockSpec α cfg.stemOutChannels cfg.stage1OutChannels hCfg.stemOut_ne0 hCfg.c1_ne0 :=
    { conv1 := { kernel := zero4, bias := zero1 }
      conv2 := { kernel := zero4, bias := zero1 }
      bn1_gamma := one1
      bn1_beta := zero1
      bn2_gamma := one1
      bn2_beta := zero1
      shortcut_conv := mkProj (inC := cfg.stemOutChannels) (outC := cfg.stage1OutChannels) hCfg.stemOut_ne0 needProj1
      shortcut_bn_gamma := mkProjBN needProj1
      shortcut_bn_beta := mkProjBN0 needProj1 }

  let layer2_first : ResNetBlockSpec α cfg.stage1OutChannels cfg.stage2OutChannels hCfg.c1_ne0 hCfg.c2_ne0 :=
    { conv1 := { kernel := zero4, bias := zero1 }
      conv2 := { kernel := zero4, bias := zero1 }
      bn1_gamma := one1
      bn1_beta := zero1
      bn2_gamma := one1
      bn2_beta := zero1
      shortcut_conv := mkProj (inC := cfg.stage1OutChannels) (outC := cfg.stage2OutChannels) hCfg.c1_ne0 needProj2
      shortcut_bn_gamma := mkProjBN needProj2
      shortcut_bn_beta := mkProjBN0 needProj2 }

  let layer3_first : ResNetBlockSpec α cfg.stage2OutChannels cfg.stage3OutChannels hCfg.c2_ne0 hCfg.c3_ne0 :=
    { conv1 := { kernel := zero4, bias := zero1 }
      conv2 := { kernel := zero4, bias := zero1 }
      bn1_gamma := one1
      bn1_beta := zero1
      bn2_gamma := one1
      bn2_beta := zero1
      shortcut_conv := mkProj (inC := cfg.stage2OutChannels) (outC := cfg.stage3OutChannels) hCfg.c2_ne0 needProj3
      shortcut_bn_gamma := mkProjBN needProj3
      shortcut_bn_beta := mkProjBN0 needProj3 }

  let layer4_first : ResNetBlockSpec α cfg.stage3OutChannels cfg.stage4OutChannels hCfg.c3_ne0 hCfg.c4_ne0 :=
    { conv1 := { kernel := zero4, bias := zero1 }
      conv2 := { kernel := zero4, bias := zero1 }
      bn1_gamma := one1
      bn1_beta := zero1
      bn2_gamma := one1
      bn2_beta := zero1
      shortcut_conv := mkProj (inC := cfg.stage3OutChannels) (outC := cfg.stage4OutChannels) hCfg.c3_ne0 needProj4
      shortcut_bn_gamma := mkProjBN needProj4
      shortcut_bn_beta := mkProjBN0 needProj4 }

  let homBlock {c : Nat} (h_c : c ≠ 0) : ResNetBlockSpec α c c h_c h_c :=
    { conv1 := { kernel := zero4, bias := zero1 }
      conv2 := { kernel := zero4, bias := zero1 }
      bn1_gamma := one1
      bn1_beta := zero1
      bn2_gamma := one1
      bn2_beta := zero1
      shortcut_conv := none
      shortcut_bn_gamma := none
      shortcut_bn_beta := none }

  refine
    { initial_conv := zeroConvStem
      initial_bn_gamma := one1
      initial_bn_beta := zero1
      initial_pool := {}
      layer1 :=
        { first := layer1_first
          rest := List.replicate (cfg.stage1Blocks - 1) (homBlock (c := cfg.stage1OutChannels) hCfg.c1_ne0) }
      layer2 :=
        { first := layer2_first
          rest := List.replicate (cfg.stage2Blocks - 1) (homBlock (c := cfg.stage2OutChannels) hCfg.c2_ne0) }
      layer3 :=
        { first := layer3_first
          rest := List.replicate (cfg.stage3Blocks - 1) (homBlock (c := cfg.stage3OutChannels) hCfg.c3_ne0) }
      layer4 :=
        { first := layer4_first
          rest := List.replicate (cfg.stage4Blocks - 1) (homBlock (c := cfg.stage4OutChannels) hCfg.c4_ne0) }
      classifier := { weights := zero2, bias := zero1 } }

/--
ResNet-18 spec constructor (specialization of `ResNetSpec.zeroInit`).

By default this uses `resnet18Config`, whose `useProjectionShortcuts = false`, so any channel
changes are handled via the total fallback shortcut (pad/slice) when `shortcut_conv = none`.

If you want learned `1×1` projection shortcuts on channel changes, use
`ResNet18SpecWithProjections`.
-/
def ResNet18Spec (α : Type) [Context α]
  (inputChannels numClasses : Nat) (h1 : inputChannels ≠ 0) (h2 : numClasses ≠ 0) :
  ResNetSpec resnet18Config α inputChannels numClasses h1 h2 resnet18Config_wf :=
  ResNetSpec.zeroInit (cfg := resnet18Config) (hCfg := resnet18Config_wf) α inputChannels numClasses h1 h2

/-- ResNet-18 spec constructor using learned `1×1` projection shortcuts on channel changes. -/
def ResNet18SpecWithProjections (α : Type) [Context α]
  (inputChannels numClasses : Nat) (h1 : inputChannels ≠ 0) (h2 : numClasses ≠ 0) :
  ResNetSpec resnet18ConfigWithProjections α inputChannels numClasses h1 h2
    resnet18ConfigWithProjections_wf :=
  ResNetSpec.zeroInit (cfg := resnet18ConfigWithProjections) (hCfg := resnet18ConfigWithProjections_wf)
    α inputChannels numClasses h1 h2

-- Bottleneck ResNet block (for ResNet-50+)
/-!
## Bottleneck blocks (forward-only)

This section defines the bottleneck block used in ResNet-50/101/152.

This is a forward-only baseline. The `stride` field is included to match the usual API, but the
convolution specs here all use stride `1`; wiring stride through the type-level conv shape
expressions is outside this spec baseline.
-/
/--
Bottleneck residual block spec (ResNet-50/101/152 style), forward-only.

The bottleneck block uses a `1x1 -> 3x3 -> 1x1` conv stack with BatchNorms and a residual shortcut.
The `stride` field is included for API shape, but this spec keeps stride fixed to `1` in
the conv specs (see module header for scope note).
-/
structure BottleneckResNetBlockSpec (α : Type) (inChannels outChannels : Nat) (h1 : inChannels ≠ 0)
  (h2 : outChannels ≠ 0) (h3 : outChannels / 4 ≠ 0) where
  /-- conv 1. -/
  conv1 : Conv2DSpec inChannels (outChannels / 4) 1 1 1 0 α h1 (by norm_num) (by norm_num)
  -- 1x1 conv for dimension reduction
  /-- conv 2. -/
  conv2 : Conv2DSpec (outChannels / 4) (outChannels / 4) 3 3 1 1 α h3 (by norm_num) (by norm_num)
  -- 3x3 conv
  /-- conv 3. -/
  conv3 : Conv2DSpec (outChannels / 4) outChannels 1 1 1 0 α h3 (by norm_num) (by norm_num)
  -- 1x1 conv for dimension expansion
  /-- bn 1 gamma. -/
  bn1_gamma : Tensor α (.dim (outChannels / 4) .scalar)
  /-- bn 1 beta. -/
  bn1_beta : Tensor α (.dim (outChannels / 4) .scalar)
  /-- bn 2 gamma. -/
  bn2_gamma : Tensor α (.dim (outChannels / 4) .scalar)
  /-- bn 2 beta. -/
  bn2_beta : Tensor α (.dim (outChannels / 4) .scalar)
  /-- bn 3 gamma. -/
  bn3_gamma : Tensor α (.dim outChannels .scalar)
  /-- bn 3 beta. -/
  bn3_beta : Tensor α (.dim outChannels .scalar)
  /-- shortcut conv. -/
  shortcut_conv : Option (Conv2DSpec inChannels outChannels 1 1 1 0 α h1 (by norm_num) (by
    norm_num))
  /-- shortcut bn gamma. -/
  shortcut_bn_gamma : Option (Tensor α (.dim outChannels .scalar))
  /-- shortcut bn beta. -/
  shortcut_bn_beta : Option (Tensor α (.dim outChannels .scalar))
  /-- Stride. -/
  stride : Nat := 1

-- Bottleneck block forward pass
/-- Forward pass for a bottleneck residual block (forward-only baseline). -/
def BottleneckResNetBlockSpec.forward {inChannels outChannels inH inW : Nat}
  (h1 : inChannels ≠ 0) (h2 : outChannels ≠ 0) (h3 : outChannels / 4 ≠ 0)
  (block : BottleneckResNetBlockSpec α inChannels outChannels h1 h2 h3)
  (x : MultiChannelImage inChannels inH inW α)
  (h4 : inH ≠ 0) (h5 : inW ≠ 0) :
  MultiChannelImage outChannels inH inW α :=

  -- First 1x1 convolution + BatchNorm + ReLU (dimension reduction)
  let conv1_out := conv2dSpec block.conv1 x
  let bn1_out := batchNorm2d conv1_out block.bn1_gamma block.bn1_beta
    (Nat.pos_of_ne_zero h3) (pos_add_one _) (pos_add_one _)
  let relu1_out := Activation.reluSpec bn1_out

  -- Second 3x3 convolution + BatchNorm + ReLU
  let conv2_out := conv2dSpec block.conv2 relu1_out
  let bn2_out := batchNorm2d conv2_out block.bn2_gamma block.bn2_beta
    (Nat.pos_of_ne_zero h3) (pos_add_one _) (pos_add_one _)
  let relu2_out := Activation.reluSpec bn2_out

  -- Third 1x1 convolution + BatchNorm (dimension expansion)
  let conv3_out := conv2dSpec block.conv3 relu2_out
  let bn3_out := batchNorm2d conv3_out block.bn3_gamma block.bn3_beta
    (Nat.pos_of_ne_zero h2) (pos_add_one _) (pos_add_one _)

  -- Shortcut connection
  let shortcut := match block.shortcut_conv with
  | some conv =>
    let shortcut_conv_out := conv2dSpec conv x
    -- Type annotation to ensure spatial dimensions match main path
    let shortcut_conv_out' :=
      rwMultiChannelImageExplicit outChannels
        ((((inH + 2 * 0 - 1) / 1 + 1 + 2 * 1 - 3) / 1 + 1 + 2 * 0 - 1) / 1 + 1)
        ((((inW + 2 * 0 - 1) / 1 + 1 + 2 * 1 - 3) / 1 + 1 + 2 * 0 - 1) / 1 + 1)
        shortcut_conv_out (by rfl) (by grind) (by grind)
    match block.shortcut_bn_gamma, block.shortcut_bn_beta with
    | some gamma, some beta =>
      batchNorm2d shortcut_conv_out' gamma beta
        (Nat.pos_of_ne_zero h2) (pos_add_one _) (pos_add_one _)
    | _, _ => shortcut_conv_out'
  | none =>
    -- The main path (conv1/conv2/conv3) preserves spatial dimensions, but its type-level
    -- expression is a nested application of the conv output formula. We keep the shortcut
    -- branch aligned with that expression so the residual `add_spec` is well-typed.
    let Hmain : Nat :=
      ((((inH + 2 * 0 - 1) / 1 + 1 + 2 * 1 - 3) / 1 + 1 + 2 * 0 - 1) / 1 + 1)
    let Wmain : Nat :=
      ((((inW + 2 * 0 - 1) / 1 + 1 + 2 * 1 - 3) / 1 + 1 + 2 * 0 - 1) / 1 + 1)
    if h_eq : inChannels = outChannels then
      -- Channels match, use identity shortcut.
      let x' : MultiChannelImage outChannels Hmain Wmain α :=
        rwMultiChannelImage x h_eq (by grind) (by grind)
      x'
    else if h_lt : inChannels < outChannels then
      -- Expand channels by zero-padding.
      let x' : MultiChannelImage inChannels Hmain Wmain α :=
        rwMultiChannelImage x (by rfl) (by grind) (by grind)
      padChannelsZero (Nat.le_of_lt h_lt) x'
    else
      -- Channel reduction case: take the first `outChannels` channels.
      have h_le : outChannels ≤ inChannels := Nat.le_of_not_gt h_lt
      let x' : MultiChannelImage inChannels Hmain Wmain α :=
        rwMultiChannelImage x (by rfl) (by grind) (by grind)
      sliceRange0Spec (α := α) (n := inChannels) (s := .dim Hmain (.dim Wmain .scalar))
        0 outChannels (by simpa using h_le) x'

  -- Residual connection + ReLU
  let residual_out := addSpec bn3_out shortcut
  let residual_out' := rwMultiChannelImageExplicit outChannels inH inW residual_out (by rfl) (by
    grind) (by grind)
  Activation.reluSpec residual_out'

-- Bottleneck ResNet layer specification (multiple bottleneck blocks)
/-- Bottleneck ResNet layer spec: one "first" block plus a list of homogeneous "rest" blocks. -/
structure BottleneckResNetLayerSpec (α : Type) (inChannels outChannels blockCount : Nat)
  (h1 : inChannels ≠ 0) (h2 : outChannels ≠ 0) (h3 : outChannels / 4 ≠ 0) where
  first : BottleneckResNetBlockSpec α inChannels outChannels h1 h2 h3
  rest  : List (BottleneckResNetBlockSpec α outChannels outChannels h2 h2 h3)

-- Bottleneck layer forward pass
/-- Forward pass for a bottleneck ResNet layer (fold over the `rest` list after the first block). -/
def BottleneckResNetLayerSpec.forward {inChannels outChannels blockCount inH inW : Nat}
  (h1 : inChannels ≠ 0) (h2 : outChannels ≠ 0) (h3 : outChannels / 4 ≠ 0)
  (layer : BottleneckResNetLayerSpec α inChannels outChannels blockCount h1 h2 h3)
  (x : MultiChannelImage inChannels inH inW α)
  (h4 : inH ≠ 0) (h5 : inW ≠ 0) :
  MultiChannelImage outChannels inH inW α :=
  let firstOut := BottleneckResNetBlockSpec.forward h1 h2 h3 layer.first x h4 h5
  layer.rest.foldl (fun acc block => BottleneckResNetBlockSpec.forward h2 h2 h3 block acc h4 h5)
    firstOut

-- Backward pass note:
-- `ResNetBlockSpec.backward`, `ResNetLayerSpec.backward`, and `ResNetSpec.backward` are implemented
-- above (explicit reverse-mode, no global tape).

/-- Compute the model depth metadata used by examples, counting the stem, each block, and the head. -/
def ResNetSpec.depth {cfg : ResNetConfig} {inputChannels numClasses : Nat}
  (h1 : inputChannels ≠ 0) (h2 : numClasses ≠ 0) (hCfg : cfg.WF)
  (resnet : ResNetSpec cfg α inputChannels numClasses h1 h2 hCfg) : Nat :=
  -- Metadata count: stem + blocks in each stage + classifier head.
  1 + -- initial conv
  (1 + resnet.layer1.rest.length) + -- layer1: first + rest
  (1 + resnet.layer2.rest.length) + -- layer2: first + rest
  (1 + resnet.layer3.rest.length) + -- layer3: first + rest
  (1 + resnet.layer4.rest.length) + -- layer4: first + rest
  1   -- classifier

/-- Compute a lightweight parameter-count estimate for model summaries. -/
def ResNetSpec.parameterCount {cfg : ResNetConfig} {inputChannels numClasses : Nat}
  (h1 : inputChannels ≠ 0) (h2 : numClasses ≠ 0) (_hCfg : cfg.WF)
  (_resnet : ResNetSpec cfg α inputChannels numClasses h1 h2 _hCfg) : Nat :=
  -- Summary estimate: stem convolution, stem normalization scale, and classifier parameters.
  cfg.stemOutChannels * cfg.stemKernel * cfg.stemKernel * inputChannels +  -- stem conv
  cfg.stemOutChannels +                                                   -- stem normalization scale
  numClasses * cfg.stage4OutChannels +                                    -- classifier W
  cfg.stage4OutChannels                                                   -- classifier bias

end Spec
