/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.PositionalEncoding
public import NN.Spec.Models.Transformer

/-!
# Vit

Vision Transformer (ViT) model.

This is a compact “ViT-style” specification:
- patch embedding via `Conv2D` (kernel = patch size),
- flatten patches into a token sequence,
- add a learnable positional encoding,
- run a Transformer encoder,
- mean-pool tokens and apply a linear classifier head.

Notes:
- PyTorch mental model: this corresponds to the core dataflow of `torchvision.models.vit_*`,
  but written without batching: tensors are `(C,H,W)` images and `(T,D)` token sequences.
- This file provides both mean-pool (`ViTSpec`) and CLS-token (`ViTClsSpec`) variants. The CLS-token
  variant prepends one learnable token before the encoder and pools by taking token `0`.
- We intentionally keep the patch embedding as a `Conv2d` with `kernel_size=(patchH,patchW)`.
  When `stride=(patchH,patchW)` and `padding=0`, that matches the usual "non-overlapping patches"
  embedding used in many ViT implementations.
-/

@[expose] public section


namespace Models

open Spec
open Tensor
open Shape

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Output height of the patch-embedding convolution in ViT. -/
abbrev ViTPatchOutH (inH patchH stride padding : Nat) : Nat :=
  (inH + 2 * padding - patchH) / stride + 1

/-- Output width of the patch-embedding convolution in ViT. -/
abbrev ViTPatchOutW (inW patchW stride padding : Nat) : Nat :=
  (inW + 2 * padding - patchW) / stride + 1

/-- Number of patch tokens `T = outH*outW` produced by the patch embedding. -/
abbrev ViTPatchCount (inH inW patchH patchW stride padding : Nat) : Nat :=
  ViTPatchOutH inH patchH stride padding * ViTPatchOutW inW patchW stride padding

/-!
## Configuration

We keep ViT architectural hyperparameters in a dedicated config record so the model definition
does not hide numeric choices in its types. This mirrors the usual config-object pattern in
PyTorch/torchvision model-zoo code.
-/

/-- ViT architectural hyperparameters (spec layer). -/
structure ViTConfig where
  /-- Patch height (kernel height for the patch-embedding conv). -/
  patchH : Nat := 16
  /-- Patch width (kernel width for the patch-embedding conv). -/
  patchW : Nat := 16
  /-- Stride for the patch-embedding conv (typical: equal to patch size for non-overlapping patches). -/
  stride : Nat := 16
  /-- Padding for the patch-embedding conv (typical: `0`). -/
  padding : Nat := 0

  /-- Transformer embedding dimension (`d_model`). -/
  embedDim : Nat := 768
  /-- Transformer feedforward hidden dimension (`d_ff`). -/
  hiddenDim : Nat := 3072
  /-- Number of attention heads. -/
  headCount : Nat := 12
  /-- Number of encoder layers. -/
  numLayers : Nat := 12

  /-- Output classes for the classifier head. -/
  numClasses : Nat := 1000

/-- Well-formedness conditions for `ViTConfig` (the nonzero facts needed by some layer specs). -/
structure ViTConfig.WF (cfg : ViTConfig) : Prop where
  patchH_ne0 : cfg.patchH ≠ 0
  patchW_ne0 : cfg.patchW ≠ 0
  embedDim_pos : cfg.embedDim > 0
  hiddenDim_pos : cfg.hiddenDim > 0
  headCount_pos : cfg.headCount > 0
  numClasses_ne0 : cfg.numClasses ≠ 0

/-- Classic ViT-Base/16-ish hyperparameters (mean-pool variant; spec layer). -/
def vitBasePatch16Config : ViTConfig :=
  { patchH := 16
    patchW := 16
    stride := 16
    padding := 0
    embedDim := 768
    hiddenDim := 3072
    headCount := 12
    numLayers := 12
    numClasses := 1000 }

/-- `vitBasePatch16Config` satisfies `ViTConfig.WF`. -/
theorem vitBasePatch16Config_wf : vitBasePatch16Config.WF := by
  refine
    { patchH_ne0 := by decide
      patchW_ne0 := by decide
      embedDim_pos := by decide
      hiddenDim_pos := by decide
      headCount_pos := by decide
      numClasses_ne0 := by decide }

/-- Classic ViT-Large/16-ish hyperparameters (mean-pool variant; spec layer). -/
def vitLargePatch16Config : ViTConfig :=
  { patchH := 16
    patchW := 16
    stride := 16
    padding := 0
    embedDim := 1024
    hiddenDim := 4096
    headCount := 16
    numLayers := 24
    numClasses := 1000 }

/-- `vitLargePatch16Config` satisfies `ViTConfig.WF`. -/
theorem vitLargePatch16Config_wf : vitLargePatch16Config.WF := by
  refine
    { patchH_ne0 := by decide
      patchW_ne0 := by decide
      embedDim_pos := by decide
      hiddenDim_pos := by decide
      headCount_pos := by decide
      numClasses_ne0 := by decide }

/-- ViT parameter bundle (patch embedding + positional encoding + transformer + head). -/
structure ViTSpec
  (cfg : ViTConfig) (inC inH inW : Nat)
  (α : Type)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (h_inC : inC ≠ 0) (hCfg : cfg.WF) where
  patchEmbed :
    Conv2DSpec inC cfg.embedDim cfg.patchH cfg.patchW cfg.stride cfg.padding α h_inC hCfg.patchH_ne0
      hCfg.patchW_ne0

  posEnc :
    PositionalEncodingSpec
      (ViTPatchCount inH inW cfg.patchH cfg.patchW cfg.stride cfg.padding) cfg.embedDim α

  encoder :
    Spec.TransformerEncoder cfg.numLayers cfg.headCount cfg.embedDim cfg.hiddenDim α

  head : LinearSpec α cfg.embedDim cfg.numClasses

/-!
## Forward pass (patches -> tokens -> encoder -> head)

The forward is the standard ViT dataflow, but with explicit shape transforms so it stays obvious
what each axis means:

1. `conv2d_spec` produces a feature map `(embedDim, outH, outW)`.
2. We flatten `(outH, outW)` into a single token axis `tokN = outH*outW`.
3. We swap to token-major layout `(tokN, embedDim)` (this is the usual transformer convention).
4. We add positional embeddings and run the transformer encoder.
5. We mean-pool tokens and apply a final linear classifier.

PyTorch analogy (no batch axis here):
- patch embedding: `Conv2d(inC, embedDim, kernel_size=patch, stride=stride, padding=padding)`
- flatten: `x.flatten(1).transpose(0, 1)` to get `(T,D)` depending on your convention
- encoder: `TransformerEncoder(...)`
- pooling + head: `encoded.mean(dim=0)` then `Linear(embedDim, numClasses)`
-/

/-- Gradients for the compact ViT spec (matching `ViTSpec`). -/
structure ViTGrads (cfg : ViTConfig) (inC inH inW : Nat) (α : Type) where
  d_patch_kernel : Tensor α (.dim cfg.embedDim (.dim inC (.dim cfg.patchH (.dim cfg.patchW .scalar))))
  d_patch_bias   : Tensor α (.dim cfg.embedDim .scalar)
  d_pos          :
    Tensor α
      (.dim (ViTPatchCount inH inW cfg.patchH cfg.patchW cfg.stride cfg.padding) (.dim cfg.embedDim
        .scalar))
  d_encoder      : List (Spec.TransformerEncoderLayerGrads cfg.headCount cfg.embedDim cfg.hiddenDim α)
  d_head_W       : Tensor α (.dim cfg.numClasses (.dim cfg.embedDim .scalar))
  d_head_b       : Tensor α (.dim cfg.numClasses .scalar)

/-- ViT forward pass (patch embedding → tokens → transformer encoder → pool → head). -/
def ViTSpec.forward
  {cfg : ViTConfig} {inC inH inW : Nat}
  {h_inC : inC ≠ 0} {hCfg : cfg.WF}
  (m : ViTSpec (α := α) cfg inC inH inW h_inC hCfg)
  (x : MultiChannelImage inC inH inW α)
  (h_tok : ViTPatchCount inH inW cfg.patchH cfg.patchW cfg.stride cfg.padding > 0) :
  Tensor α (.dim cfg.numClasses .scalar) :=
  let outH := ViTPatchOutH inH cfg.patchH cfg.stride cfg.padding
  let outW := ViTPatchOutW inW cfg.patchW cfg.stride cfg.padding
  let tokN := ViTPatchCount inH inW cfg.patchH cfg.patchW cfg.stride cfg.padding

  let patches : Tensor α (.dim cfg.embedDim (.dim outH (.dim outW .scalar))) :=
    conv2dSpec (α := α) m.patchEmbed x

  -- Flatten spatial `(outH,outW)` into a single token axis `(outH*outW)`.
  -- This is a pure reshape: it does not change values, only the way we index them.
  have h_size :
      (Shape.dim cfg.embedDim (Shape.dim outH (Shape.dim outW Shape.scalar))).size
        =
      (Shape.dim cfg.embedDim (Shape.dim (outH * outW) Shape.scalar)).size := by
    simp [Shape.size]

  let patchesFlat : Tensor α (.dim cfg.embedDim (.dim (outH * outW) .scalar)) :=
    reshapeSpec patches h_size

  -- Swap to the standard token layout `(tokN, embedDim)`.
  -- In transformer code (including PyTorch), tokens are typically indexed first.
  let tokens : Tensor α (.dim tokN (.dim cfg.embedDim .scalar)) :=
    -- `outH * outW` is definitionally `tokN`.
    have ht : outH * outW = tokN := by rfl
    tensorCast (.dim tokN (.dim cfg.embedDim .scalar)) (by simp [ht]) (swapFirstTwoSpec patchesFlat)

  -- Add a learnable positional embedding, then run the encoder.
  let tokensPos : Tensor α (.dim tokN (.dim cfg.embedDim .scalar)) :=
    addPositionalEncodingSpec (α := α) m.posEnc tokens

  let encoded : Tensor α (.dim tokN (.dim cfg.embedDim .scalar)) :=
    Spec.TransformerEncoder.forward (α := α) (seqLen := tokN)
      m.encoder tokensPos h_tok hCfg.embedDim_pos

  -- Mean pool over tokens (PyTorch analogy: `encoded.mean(dim=0)` in our `(T,D)` convention).
  have h_tok_ne0 : tokN ≠ 0 := Nat.ne_of_gt h_tok
  have h_axis0 :
      Shape.valid_axis_inst 0 (Shape.dim tokN (Shape.dim cfg.embedDim Shape.scalar)) :=
    Shape.validAxisInstZeroAlt h_tok_ne0

  let pooled : Tensor α (.dim cfg.embedDim .scalar) :=
    reduceMeanAuto 0 h_axis0 encoded

  linearSpec (α := α) m.head pooled

/-!
## Backward pass

This is a fully explicit reverse-mode spec (no meta-autograd):

- patch embedding: `Conv2D` backward gives `∂kernel`, `∂bias`, and `∂image`,
- positional encoding: addition splits gradient (`∂pos = ∂tokens`),
- transformer encoder: `TransformerEncoder.backward` (in `NN/Spec/Models/Transformer.lean`),
- mean pooling over tokens: broadcast + scale by `1/tokN`,
- classifier head: `linear_backward_spec`.

We recompute intermediates locally; this keeps the spec self-contained and avoids adding a global
"tape" type for every model.
-/

/-- Fully explicit reverse-mode backward pass for `ViTSpec.forward`. -/
def ViTSpec.backward
  {cfg : ViTConfig} {inC inH inW : Nat}
  {h_inC : inC ≠ 0} {hCfg : cfg.WF}
  (m : ViTSpec (α := α) cfg inC inH inW h_inC hCfg)
  (x : MultiChannelImage inC inH inW α)
  (grad_output : Tensor α (.dim cfg.numClasses .scalar))
  (h_tok : ViTPatchCount inH inW cfg.patchH cfg.patchW cfg.stride cfg.padding > 0) :
  (ViTGrads cfg inC inH inW α ×
   MultiChannelImage inC inH inW α) :=

  let outH := ViTPatchOutH inH cfg.patchH cfg.stride cfg.padding
  let outW := ViTPatchOutW inW cfg.patchW cfg.stride cfg.padding
  let tokN := ViTPatchCount inH inW cfg.patchH cfg.patchW cfg.stride cfg.padding

  -- Forward reconstruction.
  let patches : Tensor α (.dim cfg.embedDim (.dim outH (.dim outW .scalar))) :=
    conv2dSpec (α := α) m.patchEmbed x

  have h_size :
      (Shape.dim cfg.embedDim (Shape.dim outH (Shape.dim outW Shape.scalar))).size
        =
      (Shape.dim cfg.embedDim (Shape.dim (outH * outW) Shape.scalar)).size := by
    simp [Shape.size]

  let patchesFlat : Tensor α (.dim cfg.embedDim (.dim (outH * outW) .scalar)) :=
    reshapeSpec patches h_size

  have ht : outH * outW = tokN := by rfl

  let tokens : Tensor α (.dim tokN (.dim cfg.embedDim .scalar)) :=
    tensorCast (.dim tokN (.dim cfg.embedDim .scalar)) (by simp [ht]) (swapFirstTwoSpec patchesFlat)

  let tokensPos : Tensor α (.dim tokN (.dim cfg.embedDim .scalar)) :=
    addPositionalEncodingSpec (α := α) m.posEnc tokens

  let encoded : Tensor α (.dim tokN (.dim cfg.embedDim .scalar)) :=
    Spec.TransformerEncoder.forward (α := α) (seqLen := tokN)
      m.encoder tokensPos h_tok hCfg.embedDim_pos

  have h_tok_ne0 : tokN ≠ 0 := Nat.ne_of_gt h_tok
  let pooled : Tensor α (.dim cfg.embedDim .scalar) :=
    reduceMeanAuto (α := α) (s := Shape.dim tokN (Shape.dim cfg.embedDim Shape.scalar)) 0
      (Shape.validAxisInstZeroAlt h_tok_ne0) encoded

  -- Head backward.
  let (dW_head, db_head, d_pooled) := Spec.linearBackwardSpec (α := α) m.head pooled grad_output

  -- Mean-pool backward: y = (1/tokN) * Σ tokens
  --
  -- So each token receives the same slice of gradient:
  -- `d_encoded[t] = (1/tokN) * d_pooled` for all `t`.
  have hB : Shape.CanBroadcastTo (.dim cfg.embedDim .scalar) (.dim tokN (.dim cfg.embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.scalar_to_any .scalar

  let d_encoded : Tensor α (.dim tokN (.dim cfg.embedDim .scalar)) :=
    scaleSpec (broadcastTo hB d_pooled) (1 / (tokN : α))

  -- Encoder backward.
  let (d_encoder, d_tokensPos) :=
    Spec.TransformerEncoder.backward (α := α) (seqLen := tokN)
      m.encoder tokensPos d_encoded h_tok hCfg.embedDim_pos

  -- Positional encoding backward (addition).
  -- `tokensPos = tokens + pos`, so gradients split as:
  -- `d_tokens = d_tokensPos` and `d_pos = d_tokensPos`.
  let (d_pos, d_tokens) :=
    addPositionalEncodingBackwardSpec (α := α) m.posEnc d_tokensPos

  -- Undo the cast+swap+reshape sequence.
  -- This is the inverse of the forward's `(reshape -> swap -> cast)` chain, applied to gradients.
  let d_tokens' : Tensor α (.dim (outH * outW) (.dim cfg.embedDim .scalar)) :=
    tensorCast (.dim (outH * outW) (.dim cfg.embedDim .scalar)) (by simp [tokN, ht]) d_tokens

  let d_patchesFlat : Tensor α (.dim cfg.embedDim (.dim (outH * outW) .scalar)) :=
    swapFirstTwoSpec d_tokens'

  let d_patches : Tensor α (.dim cfg.embedDim (.dim outH (.dim outW .scalar))) :=
    reshapeSpec d_patchesFlat h_size.symm

  -- Patch embedding Conv2D backward.
  let (d_patch_kernel, d_patch_bias, d_input) :=
    Spec.conv2dBackwardSpec (α := α)
      (inC := inC) (outC := cfg.embedDim) (kH := cfg.patchH) (kW := cfg.patchW)
      (stride := cfg.stride) (padding := cfg.padding) (inH := inH) (inW := inW)
      (h1 := h_inC) (h2 := hCfg.patchH_ne0) (h3 := hCfg.patchW_ne0)
      m.patchEmbed x d_patches

  let grads : ViTGrads cfg inC inH inW α :=
    { d_patch_kernel := d_patch_kernel
      d_patch_bias := d_patch_bias
      d_pos := d_pos
      d_encoder := d_encoder
      d_head_W := dW_head
      d_head_b := db_head }

  (grads, d_input)


/-!
## CLS-token ViT variant (classic pooling)

Many ViT implementations (including the original ViT paper and `torchvision.models.vit_*`)
use a **learnable CLS token**:

- prepend `clsToken` to the patch-token sequence,
- use positional encodings of length `tokN + 1`,
- run the encoder on a sequence of length `tokN + 1`,
- take token `0` after the encoder as the pooled representation, then apply the head.

We keep the existing mean-pool `ViTSpec` unchanged; this is a separate parameter bundle and
explicit backward pass.
-/

/-- ViT parameter bundle with a learnable CLS token (classic ViT variant). -/
structure ViTClsSpec
  (cfg : ViTConfig) (inC inH inW : Nat)
  (α : Type)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (h_inC : inC ≠ 0) (hCfg : cfg.WF) where
  patchEmbed :
    Conv2DSpec inC cfg.embedDim cfg.patchH cfg.patchW cfg.stride cfg.padding α h_inC hCfg.patchH_ne0
      hCfg.patchW_ne0

  /-- Learnable CLS token embedding (prepended as token 0). -/
  clsToken :
    Tensor α (.dim cfg.embedDim .scalar)

  posEnc :
    PositionalEncodingSpec
      (ViTPatchCount inH inW cfg.patchH cfg.patchW cfg.stride cfg.padding + 1) cfg.embedDim α

  encoder :
    Spec.TransformerEncoder cfg.numLayers cfg.headCount cfg.embedDim cfg.hiddenDim α

  head : LinearSpec α cfg.embedDim cfg.numClasses

/-- Gradients for the CLS-token ViT spec (matching `ViTClsSpec`). -/
structure ViTClsGrads (cfg : ViTConfig) (inC inH inW : Nat) (α : Type) where
  d_patch_kernel : Tensor α (.dim cfg.embedDim (.dim inC (.dim cfg.patchH (.dim cfg.patchW .scalar))))
  d_patch_bias   : Tensor α (.dim cfg.embedDim .scalar)
  d_clsToken     : Tensor α (.dim cfg.embedDim .scalar)
  d_pos          :
    Tensor α
      (.dim (ViTPatchCount inH inW cfg.patchH cfg.patchW cfg.stride cfg.padding + 1)
        (.dim cfg.embedDim .scalar))
  d_encoder      : List (Spec.TransformerEncoderLayerGrads cfg.headCount cfg.embedDim cfg.hiddenDim α)
  d_head_W       : Tensor α (.dim cfg.numClasses (.dim cfg.embedDim .scalar))
  d_head_b       : Tensor α (.dim cfg.numClasses .scalar)

/-- CLS-token ViT forward pass (prepend CLS → transformer encoder → take token 0 → head). -/
def ViTClsSpec.forward
  {cfg : ViTConfig} {inC inH inW : Nat}
  {h_inC : inC ≠ 0} {hCfg : cfg.WF}
  (m : ViTClsSpec (α := α) cfg inC inH inW h_inC hCfg)
  (x : MultiChannelImage inC inH inW α) :
  Tensor α (.dim cfg.numClasses .scalar) :=
  let outH := ViTPatchOutH inH cfg.patchH cfg.stride cfg.padding
  let outW := ViTPatchOutW inW cfg.patchW cfg.stride cfg.padding
  let tokN := ViTPatchCount inH inW cfg.patchH cfg.patchW cfg.stride cfg.padding

  let patches : Tensor α (.dim cfg.embedDim (.dim outH (.dim outW .scalar))) :=
    conv2dSpec (α := α) m.patchEmbed x

  have h_size :
      (Shape.dim cfg.embedDim (Shape.dim outH (Shape.dim outW Shape.scalar))).size
        =
      (Shape.dim cfg.embedDim (Shape.dim (outH * outW) Shape.scalar)).size := by
    simp [Shape.size]

  let patchesFlat : Tensor α (.dim cfg.embedDim (.dim (outH * outW) .scalar)) :=
    reshapeSpec patches h_size

  let tokens : Tensor α (.dim tokN (.dim cfg.embedDim .scalar)) :=
    -- `outH * outW` is definitionally `tokN`.
    have ht : outH * outW = tokN := by rfl
    tensorCast (.dim tokN (.dim cfg.embedDim .scalar)) (by simp [ht])
      (swapFirstTwoSpec patchesFlat)

  -- Prepend CLS token: `tokensWithCls = [clsToken] ++ tokens`.
  let clsSeq : Tensor α (.dim 1 (.dim cfg.embedDim .scalar)) :=
    Tensor.dim (fun _ : Fin 1 => m.clsToken)

  let tokensWithCls0 : Tensor α (.dim (1 + tokN) (.dim cfg.embedDim .scalar)) :=
    concatDim0Spec clsSeq tokens

  let tokensWithCls : Tensor α (.dim (tokN + 1) (.dim cfg.embedDim .scalar)) :=
    tensorCast (.dim (tokN + 1) (.dim cfg.embedDim .scalar)) (by
      simp [Nat.add_comm 1 tokN]) tokensWithCls0

  let tokensPos : Tensor α (.dim (tokN + 1) (.dim cfg.embedDim .scalar)) :=
    addPositionalEncodingSpec (α := α) m.posEnc tokensWithCls

  let encoded : Tensor α (.dim (tokN + 1) (.dim cfg.embedDim .scalar)) :=
    Spec.TransformerEncoder.forward (α := α) (seqLen := tokN + 1)
      m.encoder tokensPos (Nat.succ_pos tokN) hCfg.embedDim_pos

  -- Classic ViT pooling: take token 0 (CLS) after the encoder.
  let clsOut : Tensor α (.dim cfg.embedDim .scalar) :=
    _root_.Spec.get encoded ⟨0, Nat.succ_pos tokN⟩

  linearSpec (α := α) m.head clsOut

/-- Fully explicit reverse-mode backward pass for `ViTClsSpec.forward`. -/
def ViTClsSpec.backward
  {cfg : ViTConfig} {inC inH inW : Nat}
  {h_inC : inC ≠ 0} {hCfg : cfg.WF}
  (m : ViTClsSpec (α := α) cfg inC inH inW h_inC hCfg)
  (x : MultiChannelImage inC inH inW α)
  (grad_output : Tensor α (.dim cfg.numClasses .scalar)) :
  (ViTClsGrads cfg inC inH inW α ×
   MultiChannelImage inC inH inW α) :=

  let outH := ViTPatchOutH inH cfg.patchH cfg.stride cfg.padding
  let outW := ViTPatchOutW inW cfg.patchW cfg.stride cfg.padding
  let tokN := ViTPatchCount inH inW cfg.patchH cfg.patchW cfg.stride cfg.padding

  -- Forward reconstruction.
  let patches : Tensor α (.dim cfg.embedDim (.dim outH (.dim outW .scalar))) :=
    conv2dSpec (α := α) m.patchEmbed x

  have h_size :
      (Shape.dim cfg.embedDim (Shape.dim outH (Shape.dim outW Shape.scalar))).size
        =
      (Shape.dim cfg.embedDim (Shape.dim (outH * outW) Shape.scalar)).size := by
    simp [Shape.size]

  let patchesFlat : Tensor α (.dim cfg.embedDim (.dim (outH * outW) .scalar)) :=
    reshapeSpec patches h_size

  have ht : outH * outW = tokN := by rfl

  let tokens : Tensor α (.dim tokN (.dim cfg.embedDim .scalar)) :=
    tensorCast (.dim tokN (.dim cfg.embedDim .scalar)) (by simp [ht])
      (swapFirstTwoSpec patchesFlat)

  let clsSeq : Tensor α (.dim 1 (.dim cfg.embedDim .scalar)) :=
    Tensor.dim (fun _ : Fin 1 => m.clsToken)

  let tokensWithCls0 : Tensor α (.dim (1 + tokN) (.dim cfg.embedDim .scalar)) :=
    concatDim0Spec clsSeq tokens

  let tokensWithCls : Tensor α (.dim (tokN + 1) (.dim cfg.embedDim .scalar)) :=
    tensorCast (.dim (tokN + 1) (.dim cfg.embedDim .scalar)) (by
      simp [Nat.add_comm 1 tokN]) tokensWithCls0

  let tokensPos : Tensor α (.dim (tokN + 1) (.dim cfg.embedDim .scalar)) :=
    addPositionalEncodingSpec (α := α) m.posEnc tokensWithCls

  let encoded : Tensor α (.dim (tokN + 1) (.dim cfg.embedDim .scalar)) :=
    Spec.TransformerEncoder.forward (α := α) (seqLen := tokN + 1)
      m.encoder tokensPos (Nat.succ_pos tokN) hCfg.embedDim_pos

  let clsOut : Tensor α (.dim cfg.embedDim .scalar) :=
    _root_.Spec.get encoded ⟨0, Nat.succ_pos tokN⟩

  -- Head backward.
  let (dW_head, db_head, d_clsOut) := Spec.linearBackwardSpec (α := α) m.head clsOut grad_output

  -- CLS pooling backward: only token 0 receives gradient, all other tokens get 0.
  let d_encoded : Tensor α (.dim (tokN + 1) (.dim cfg.embedDim .scalar)) :=
    Tensor.dim (fun i =>
      if h0 : i.val = 0 then
        d_clsOut
      else
        fill (0 : α) (.dim cfg.embedDim .scalar))

  -- Encoder backward.
  let (d_encoder, d_tokensPos) :=
    Spec.TransformerEncoder.backward (α := α) (seqLen := tokN + 1)
      m.encoder tokensPos d_encoded (Nat.succ_pos tokN) hCfg.embedDim_pos

  -- Positional encoding backward (addition).
  let (d_pos, d_tokensWithCls) :=
    addPositionalEncodingBackwardSpec (α := α) m.posEnc d_tokensPos

  -- Split CLS token vs patch tokens.
  let d_clsSeq : Tensor α (.dim 1 (.dim cfg.embedDim .scalar)) :=
    sliceRange0Spec (α := α) (n := tokN + 1) (s := .dim cfg.embedDim .scalar)
      0 1 (by simp) d_tokensWithCls

  let d_clsToken : Tensor α (.dim cfg.embedDim .scalar) :=
    _root_.Spec.get d_clsSeq ⟨0, by decide⟩

  let d_tokens : Tensor α (.dim tokN (.dim cfg.embedDim .scalar)) :=
    sliceRange0Spec (α := α) (n := tokN + 1) (s := .dim cfg.embedDim .scalar)
      1 tokN (by simp) d_tokensWithCls

  -- Undo the cast+swap+reshape sequence (gradient w.r.t. patch-embedding output).
  let d_tokens' : Tensor α (.dim (outH * outW) (.dim cfg.embedDim .scalar)) :=
    tensorCast (.dim (outH * outW) (.dim cfg.embedDim .scalar)) (by simp [tokN, ht]) d_tokens

  let d_patchesFlat : Tensor α (.dim cfg.embedDim (.dim (outH * outW) .scalar)) :=
    swapFirstTwoSpec d_tokens'

  let d_patches : Tensor α (.dim cfg.embedDim (.dim outH (.dim outW .scalar))) :=
    reshapeSpec d_patchesFlat h_size.symm

  -- Patch embedding Conv2D backward.
  let (d_patch_kernel, d_patch_bias, d_input) :=
    Spec.conv2dBackwardSpec (α := α)
      (inC := inC) (outC := cfg.embedDim) (kH := cfg.patchH) (kW := cfg.patchW)
      (stride := cfg.stride) (padding := cfg.padding) (inH := inH) (inW := inW)
      (h1 := h_inC) (h2 := hCfg.patchH_ne0) (h3 := hCfg.patchW_ne0)
      m.patchEmbed x d_patches

  let grads : ViTClsGrads cfg inC inH inW α :=
    { d_patch_kernel := d_patch_kernel
      d_patch_bias := d_patch_bias
      d_clsToken := d_clsToken
      d_pos := d_pos
      d_encoder := d_encoder
      d_head_W := dW_head
      d_head_b := db_head }

  (grads, d_input)

end Models
