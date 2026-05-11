/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Normalization

/-!
# Transformer (spec model)

This file defines a small Transformer-style model in a way that matches the usual PyTorch mental
model:

- encoder layers (self-attention + FFN, each wrapped in residual + LayerNorm),
- decoder layers (masked self-attention, cross-attention, FFN, each wrapped in residual +
  LayerNorm),
- an encoder-decoder wrapper (`Transformer`),
- spec-level backward passes for the encoder stack,
- small utilities like sinusoidal positional encodings and causal masks.

Shapes follow the common convention:
- sequence tensors are `(seqLen × embedDim)`,
- attention is "last-axis softmax" over the key dimension.

PyTorch analogy:
- `TransformerEncoderLayer.forward` corresponds to the core of `torch.nn.TransformerEncoderLayer`
  (ignoring dropout and some configuration knobs),
- `TransformerEncoder.forward` corresponds to `torch.nn.TransformerEncoder`.
- `TransformerDecoderLayer.forward` corresponds to the core of `torch.nn.TransformerDecoderLayer`,
- `TransformerDecoder.forward` corresponds to `torch.nn.TransformerDecoder`,
- `Transformer.forward` is similar in spirit to `torch.nn.Transformer` (but simplified).

References:
- Vaswani et al., "Attention Is All You Need" (2017).
- Ba et al., "Layer Normalization" (2016).
- He et al., "Deep Residual Learning for Image Recognition" (2015) for the residual/skip-connection
  pattern.

PyTorch docs (for API shape intuition, not semantics):
- `torch.nn.TransformerEncoderLayer`:
  https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
- `torch.nn.TransformerEncoder`:
  https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
- `torch.nn.TransformerDecoderLayer`:
  https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html
- `torch.nn.TransformerDecoder`:
  https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
- `torch.nn.Transformer`: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
-/

@[expose] public section


namespace Spec
open Tensor
open Activation

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
## Configuration helpers

This file mostly defines reusable transformer *building blocks* (encoder/decoder layers, attention,
layer-norm wrappers, etc.). To make "model-zoo style" instantiations easier, we also provide a
small config record for the common hyperparameters together with a couple of canonical configs
(Base/Big).

The core definitions below still expose the hyperparameters as Nat parameters so the spec remains
flexible; the config layer is a convenience wrapper, not a new abstraction boundary.
-/

/-- Common transformer layer hyperparameters. -/
structure TransformerLayerConfig where
  /-- Number of attention heads. -/
  headCount : Nat := 8
  /-- Embedding dimension (`d_model`). -/
  embedDim : Nat := 512
  /-- Feedforward hidden dimension (`d_ff`). -/
  hiddenDim : Nat := 2048

/-- Stack hyperparameters for an encoder/decoder: common layer config plus a layer count. -/
structure TransformerStackConfig extends TransformerLayerConfig where
  /-- Number of layers in the stack. -/
  numLayers : Nat := 6

/--
Well-formedness conditions for `TransformerLayerConfig`.

The divisibility condition keeps the per-head width exact: `embedDim / headCount` should partition
the model dimension without silently dropping a tail through `Nat` floor division.
-/
structure TransformerLayerConfig.WF (cfg : TransformerLayerConfig) : Prop where
  headCount_pos : cfg.headCount > 0
  embedDim_pos : cfg.embedDim > 0
  hiddenDim_pos : cfg.hiddenDim > 0
  headCount_dvd_embedDim : cfg.headCount ∣ cfg.embedDim

/-- Well-formedness conditions for `TransformerStackConfig`. -/
structure TransformerStackConfig.WF (cfg : TransformerStackConfig) : Prop where
  layer : cfg.toTransformerLayerConfig.WF

/-- Canonical Transformer "base" hyperparameters (Vaswani et al. 2017). -/
def transformerBaseConfig : TransformerStackConfig :=
  { headCount := 8
    embedDim := 512
    hiddenDim := 2048
    numLayers := 6 }

/-- `transformerBaseConfig` is well-formed. -/
theorem transformerBaseConfig_wf : transformerBaseConfig.WF := by
  refine { layer := ?_ }
  refine
    { headCount_pos := by decide
      embedDim_pos := by decide
      hiddenDim_pos := by decide
      headCount_dvd_embedDim := by decide }

/-- Canonical Transformer "big" hyperparameters (Vaswani et al. 2017). -/
def transformerBigConfig : TransformerStackConfig :=
  { headCount := 16
    embedDim := 1024
    hiddenDim := 4096
    numLayers := 6 }

/-- `transformerBigConfig` is well-formed. -/
theorem transformerBigConfig_wf : transformerBigConfig.WF := by
  refine { layer := ?_ }
  refine
    { headCount_pos := by decide
      embedDim_pos := by decide
      hiddenDim_pos := by decide
      headCount_dvd_embedDim := by decide }

/-!
## Gradient containers

To keep the backward pass readable (and easy to reuse from downstream models like ViT/Seq2Seq),
we bundle parameter gradients into records that mirror the parameter records.
-/

/--
Gradients for a `FeedForward` block (field-for-field).

This is a lightweight container used by downstream models that want a readable backward pass.
-/
structure FeedForwardGrads (embedDim hiddenDim : Nat) (α : Type) where
  /-- Gradient of `W1`. -/
  dW1 : Tensor α (.dim embedDim (.dim hiddenDim .scalar))
  /-- Gradient of `W2`. -/
  dW2 : Tensor α (.dim hiddenDim (.dim embedDim .scalar))
  /-- Gradient of `b1`. -/
  db1 : Tensor α (.dim hiddenDim .scalar)
  /-- Gradient of `b2`. -/
  db2 : Tensor α (.dim embedDim .scalar)

/--
Gradients for `MultiHeadAttention` parameters (field-for-field).

This mirrors the `MultiHeadAttention` record defined in `NN.Spec.Module.Attention`.
-/
structure MultiHeadAttentionGrads (numHeads dModel headDim : Nat) (α : Type) where
  /-- Gradient of the query projection matrix `Wq`. -/
  dWq : Tensor α (.dim dModel (.dim (numHeads * headDim) .scalar))
  /-- Gradient of the key projection matrix `Wk`. -/
  dWk : Tensor α (.dim dModel (.dim (numHeads * headDim) .scalar))
  /-- Gradient of the value projection matrix `Wv`. -/
  dWv : Tensor α (.dim dModel (.dim (numHeads * headDim) .scalar))
  /-- Gradient of the output projection matrix `Wo`. -/
  dWo : Tensor α (.dim (numHeads * headDim) (.dim dModel .scalar))

/--
Gradients for a `TransformerEncoderLayer` (field-for-field).

This container is intended to keep the backward pass readable by mirroring the parameter layout.
-/
structure TransformerEncoderLayerGrads (headCount embedDim hiddenDim : Nat) (α : Type) where
  /-- Gradients for the self-attention block. -/
  mha : MultiHeadAttentionGrads headCount embedDim (embedDim / headCount) α
  /-- Gradients for the feedforward block. -/
  ffn : FeedForwardGrads embedDim hiddenDim α
  /-- Gradient of LayerNorm 1 gamma (attention "Add & Norm"). -/
  d_norm1_gamma : Tensor α (.dim embedDim .scalar)
  /-- Gradient of LayerNorm 1 beta (attention "Add & Norm"). -/
  d_norm1_beta  : Tensor α (.dim embedDim .scalar)
  /-- Gradient of LayerNorm 2 gamma (FFN "Add & Norm"). -/
  d_norm2_gamma : Tensor α (.dim embedDim .scalar)
  /-- Gradient of LayerNorm 2 beta (FFN "Add & Norm"). -/
  d_norm2_beta  : Tensor α (.dim embedDim .scalar)

/--
2-layer position-wise feedforward network used inside Transformer layers.

Semantics (per token):
`ffn(x) = (relu(x * W1 + b1) * W2) + b2`.

PyTorch analogue: the `linear1` / `linear2` submodule in `torch.nn.TransformerEncoderLayer`.
-/
structure FeedForward (embedDim hiddenDim : Nat) (α : Type) [Context α] [DecidableRel ((· > ·) : α →
  α → Prop)] where
  /-- First linear layer weights (`embedDim -> hiddenDim`). -/
  W1 : Tensor α (.dim embedDim (.dim hiddenDim .scalar))
  /-- Second linear layer weights (`hiddenDim -> embedDim`). -/
  W2 : Tensor α (.dim hiddenDim (.dim embedDim .scalar))
  /-- First layer bias (length `hiddenDim`). -/
  b1 : Tensor α (.dim hiddenDim .scalar)
  /-- Second layer bias (length `embedDim`). -/
  b2 : Tensor α (.dim embedDim .scalar)

/--
Forward pass for `FeedForward`.

Shape convention: inputs and outputs are `(seqLen × embedDim)`; the feedforward operates
independently on each sequence position.
-/
def FeedForward.forward {embedDim hiddenDim seqLen : Nat}
  (ffn : FeedForward embedDim hiddenDim α)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  : Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  let preact := matMulSpec x ffn.W1
  let bc_b1 : Shape.CanBroadcastTo (.dim hiddenDim .scalar) (.dim seqLen (.dim hiddenDim .scalar))
    := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.scalar_to_any .scalar

  let preact_reshaped := broadcastTo bc_b1 ffn.b1
  let preact_added := addSpec preact preact_reshaped
  let hidden := reluSpec preact_added

  let bc_b2 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) :=
    by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.scalar_to_any .scalar
  let hidden_reshaped := broadcastTo bc_b2 ffn.b2
  addSpec (matMulSpec hidden ffn.W2) hidden_reshaped

/--
Transformer encoder layer (post-norm).

This follows the common "Add & Norm" structure:
1. Self-attention, residual add, LayerNorm
2. Feedforward, residual add, LayerNorm

PyTorch analogue: `torch.nn.TransformerEncoderLayer` with `norm_first=False` (post-norm),
ignoring dropout and other configuration knobs.
-/
structure TransformerEncoderLayer (headCount embedDim hiddenDim : Nat) (α : Type)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- Multi-head self-attention block. -/
  mha : MultiHeadAttention α headCount embedDim (embedDim / headCount)
  /-- Position-wise feedforward block. -/
  ffn : FeedForward embedDim hiddenDim α

  /-- LayerNorm 1 gamma (attention "Add & Norm"). -/
  norm1_gamma : Tensor α (.dim embedDim .scalar)
  /-- LayerNorm 1 beta (attention "Add & Norm"). -/
  norm1_beta  : Tensor α (.dim embedDim .scalar)

  /-- LayerNorm 2 gamma (FFN "Add & Norm"). -/
  norm2_gamma : Tensor α (.dim embedDim .scalar)
  /-- LayerNorm 2 beta (FFN "Add & Norm"). -/
  norm2_beta  : Tensor α (.dim embedDim .scalar)

/--
Forward pass for a post-norm `TransformerEncoderLayer`.

Input/output shape: `(seqLen × embedDim)`.
The proofs `h1`/`h2` are used by `layerNorm` to justify nondegenerate normalization.
-/
def TransformerEncoderLayer.forward
  {headCount embedDim hiddenDim seqLen : Nat}
  (layer : TransformerEncoderLayer headCount embedDim hiddenDim α)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar))) (h1 : seqLen > 0) (h2 : embedDim > 0)
  : Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  have h3 : seqLen ≠ 0 := by apply Shape.gt_pos_to_ne_zero h1
  let attnOut := MultiHeadAttention.forward seqLen h3 layer.mha x none
  let attnAdded := addSpec x attnOut
  let normAttn := layerNorm attnAdded layer.norm1_gamma layer.norm1_beta h1 h2
  let ffnOut := FeedForward.forward layer.ffn normAttn
  let ffnAdded := addSpec normAttn ffnOut
  layerNorm ffnAdded layer.norm2_gamma layer.norm2_beta h1 h2

/--
Transformer encoder: a stack of `TransformerEncoderLayer`s.

PyTorch analogue: `torch.nn.TransformerEncoder` (a list of layers composed sequentially).
-/
structure TransformerEncoder (numLayers headCount embedDim hiddenDim : Nat) (α : Type) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- Layer list; typically has length `numLayers`, but the spec does not enforce that invariant. -/
  layers : List (TransformerEncoderLayer headCount embedDim hiddenDim α)

/--
Forward pass for `TransformerEncoder` (left-fold over layers).

Input/output shape: `(seqLen × embedDim)`.
-/
def TransformerEncoder.forward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (encoder : TransformerEncoder numLayers headCount embedDim hiddenDim α)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar))) (h1 : seqLen > 0) (h2 : embedDim > 0)
  : Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  encoder.layers.foldl (fun acc layer => TransformerEncoderLayer.forward layer acc h1 h2) x

/-!
### Config-indexed aliases

These are small convenience abbreviations so downstream models can index transformer components
by a config record rather than repeating the Nat parameters.
-/

/-- Encoder-layer gradients indexed by a `TransformerLayerConfig`. -/
abbrev TransformerEncoderLayerGradsCfg (cfg : TransformerLayerConfig) (α : Type) :=
  TransformerEncoderLayerGrads cfg.headCount cfg.embedDim cfg.hiddenDim α

/-- Encoder stack indexed by a `TransformerStackConfig`. -/
abbrev TransformerEncoderCfg (cfg : TransformerStackConfig) (α : Type) [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] :=
  TransformerEncoder cfg.numLayers cfg.headCount cfg.embedDim cfg.hiddenDim α

/-!
## Decoder notes

We include a small Transformer-style decoder layer for completeness:

- self-attention over the decoder sequence,
- cross-attention where queries come from the decoder and keys/values come from the encoder,
- then the same feedforward block.

PyTorch analogy: this corresponds to the core of `torch.nn.TransformerDecoderLayer` (ignoring
dropout and a few configuration knobs).
-/

/-!
### Cross-attention helper

The attention layer provides `MultiHeadAttention.forward` for the common self-attention case
(`Q=K=V=x`). A decoder block also needs cross-attention, where `Q` comes from the decoder stream
and `K,V` come from the encoder stream.

We keep the helper here small and explicit by following the same structure as the self-attention
definition: project, split into heads, run scaled dot-product attention per head, combine heads,
then project with `Wo`.
-/
/--
Cross-attention forward pass using a `MultiHeadAttention` parameter record.

This is the decoder-specific variant of `MultiHeadAttention.forward`:
- queries come from `qInput` (decoder stream),
- keys/values come from `kvInput` (encoder stream),
- an optional boolean mask of shape `(nQ × nK)` can be applied.

Shape conventions:
- `qInput : (nQ × embedDim)`,
- `kvInput : (nK × embedDim)`,
- output : `(nQ × embedDim)`.

PyTorch analogue: the cross-attention inside `torch.nn.TransformerDecoderLayer`, typically
implemented via `torch.nn.MultiheadAttention` with separate `query` and `key/value` inputs.
-/
def multiHeadCrossAttention
  {headCount embedDim nQ nK : Nat} (hQ : nQ ≠ 0) (hK : nK ≠ 0)
  (mha : MultiHeadAttention α headCount embedDim (embedDim / headCount))
  (qInput : Tensor α (.dim nQ (.dim embedDim .scalar)))
  (kvInput : Tensor α (.dim nK (.dim embedDim .scalar)))
  (mask : Option (Tensor Bool (.dim nQ (.dim nK .scalar)))) :
  Tensor α (.dim nQ (.dim embedDim .scalar)) :=
  let Q := matMulSpec qInput mha.Wq
  let K := matMulSpec kvInput mha.Wk
  let V := matMulSpec kvInput mha.Wv
  let h : headCount * (embedDim / headCount) = headCount * (embedDim / headCount) := by rfl
  let QHeads := splitHeadsSpec Q headCount (embedDim / headCount) h
  let KHeads := splitHeadsSpec K headCount (embedDim / headCount) h
  let VHeads := splitHeadsSpec V headCount (embedDim / headCount) h
  let attentionHeads : Tensor α (.dim headCount (.dim nQ (.dim (embedDim / headCount) .scalar))) :=
    match QHeads, KHeads, VHeads with
    | Tensor.dim qF, Tensor.dim kF, Tensor.dim vF =>
        Tensor.dim (fun headIdx =>
          let bc : Shape.BroadcastTo (.dim nQ .scalar) (.dim nQ (.dim nK .scalar)) :=
            by infer_instance
          let ctx : AttentionContext α nQ nK (embedDim / headCount) hQ hK :=
            { Q := qF headIdx
              K := kF headIdx
              V := vF headIdx
              bc_sum_to_target := bc
              mask := mask }
          scaledDotProductAttention ctx)
  let concatenated :=
    combineHeadsSpec (α := α) (n := nQ) (numHeads := headCount) (headDim := (embedDim /
      headCount)) attentionHeads
  matMulSpec concatenated mha.Wo

/--
Transformer decoder layer (post-norm).

This mirrors the standard structure:
1. Self-attention (decoder stream), residual add, LayerNorm
2. Cross-attention (queries from decoder, keys/values from encoder), residual add, LayerNorm
3. Feedforward, residual add, LayerNorm

PyTorch analogue: `torch.nn.TransformerDecoderLayer` with `norm_first=False` (post-norm),
ignoring dropout and a few configuration knobs.
-/
structure TransformerDecoderLayer (headCount embedDim hiddenDim : Nat) (α : Type) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- Self-attention block over the decoder sequence. -/
  selfAttn : MultiHeadAttention α headCount embedDim (embedDim / headCount)
  /-- Cross-attention block (decoder queries, encoder keys/values). -/
  crossAttn : MultiHeadAttention α headCount embedDim (embedDim / headCount)
  /-- Position-wise feedforward block. -/
  ffn : FeedForward embedDim hiddenDim α
  /-- LayerNorm 1 gamma (self-attention "Add & Norm"). -/
  norm1_gamma : Tensor α (.dim embedDim .scalar)
  /-- LayerNorm 1 beta (self-attention "Add & Norm"). -/
  norm1_beta  : Tensor α (.dim embedDim .scalar)

  /-- LayerNorm 2 gamma (cross-attention "Add & Norm"). -/
  norm2_gamma : Tensor α (.dim embedDim .scalar)
  /-- LayerNorm 2 beta (cross-attention "Add & Norm"). -/
  norm2_beta  : Tensor α (.dim embedDim .scalar)

  /-- LayerNorm 3 gamma (FFN "Add & Norm"). -/
  norm3_gamma : Tensor α (.dim embedDim .scalar)
  /-- LayerNorm 3 beta (FFN "Add & Norm"). -/
  norm3_beta  : Tensor α (.dim embedDim .scalar)

/--
Forward pass for a post-norm `TransformerDecoderLayer`.

Input/output shape: `(seqLen × embedDim)`. This spec uses the same `seqLen` for encoder and decoder
streams for simplicity (cross-attention uses `nQ = nK = seqLen`).
-/
def TransformerDecoderLayer.forward
  {headCount embedDim hiddenDim seqLen : Nat}
  (layer : TransformerDecoderLayer headCount embedDim hiddenDim α)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (encoderOutput : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h1 : seqLen > 0) (h2 : embedDim > 0)
  :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  have h3 : seqLen ≠ 0 := by apply Shape.gt_pos_to_ne_zero h1
  -- Self-attention with residual connection
  let selfAttnOut := MultiHeadAttention.forward seqLen h3 layer.selfAttn x none
  let selfAttnAdded := addSpec x selfAttnOut
  let normSelfAttn := layerNorm selfAttnAdded layer.norm1_gamma layer.norm1_beta h1 h2

  -- Cross-attention with residual connection
  let crossAttnOut :=
    multiHeadCrossAttention (α := α)
      (headCount := headCount) (embedDim := embedDim) (nQ := seqLen) (nK := seqLen)
      h3 h3 layer.crossAttn normSelfAttn encoderOutput none
  let crossAttnAdded := addSpec normSelfAttn crossAttnOut
  let normCrossAttn := layerNorm crossAttnAdded layer.norm2_gamma layer.norm2_beta h1 h2

  -- Feedforward with residual connection
  let ffnOut := FeedForward.forward layer.ffn normCrossAttn
  let ffnAdded := addSpec normCrossAttn ffnOut
  layerNorm ffnAdded layer.norm3_gamma layer.norm3_beta h1 h2

/--
Transformer decoder: a stack of `TransformerDecoderLayer`s.

PyTorch analogue: `torch.nn.TransformerDecoder` (a list of decoder layers composed sequentially).
-/
structure TransformerDecoder (numLayers headCount embedDim hiddenDim : Nat) (α : Type) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- Layer list; typically has length `numLayers`, but the spec does not enforce that invariant. -/
  layers : List (TransformerDecoderLayer headCount embedDim hiddenDim α)

/--
Forward pass for `TransformerDecoder` (left-fold over layers).

Input/output shape: `(seqLen × embedDim)`.
-/
def TransformerDecoder.forward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (decoder : TransformerDecoder numLayers headCount embedDim hiddenDim α)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (encoderOutput : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  decoder.layers.foldl (fun acc layer => TransformerDecoderLayer.forward layer acc encoderOutput h1
    h2) x

/--
End-to-end encoder-decoder Transformer (spec model).

This is a seq2seq Transformer wrapper built out of the encoder and decoder stacks above.
Compared to `torch.nn.Transformer`, it is intentionally simplified:
- embeddings are modeled as explicit linear projections,
- sequence length is shared between source and target streams,
- we omit dropout, caching, and most configuration knobs.

Shape convention: all activations in this file use `(seqLen × embedDim)`.
In a full implementation, `outputProjection` would usually map to a vocabulary size; here it is
kept as an `embedDim -> embedDim` projection to stay in the "core tensor algebra" setting.
-/
structure Transformer (numLayers headCount embedDim hiddenDim : Nat) (α : Type) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- Encoder stack. -/
  encoder : TransformerEncoder numLayers headCount embedDim hiddenDim α
  /-- Decoder stack. -/
  decoder : TransformerDecoder numLayers headCount embedDim hiddenDim α
  /-- Source/input embedding projection matrix. -/
  inputEmbedding : Tensor α (.dim embedDim (.dim embedDim .scalar))
  /-- Target embedding projection matrix. -/
  outputEmbedding : Tensor α (.dim embedDim (.dim embedDim .scalar))
  /-- Final output projection matrix (here `embedDim -> embedDim`). -/
  outputProjection : Tensor α (.dim embedDim (.dim embedDim .scalar))

/--
Forward pass for `Transformer`.

Runs:
1. source embedding projection,
2. encoder stack,
3. target embedding projection,
4. decoder stack (with cross-attention to the encoder output),
5. output projection.

All tensors in this simplified spec have shape `(seqLen × embedDim)`.
-/
def Transformer.forward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (transformer : Transformer numLayers headCount embedDim hiddenDim α)
  (input : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (target : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  -- Input embedding
  let embeddedInput := matMulSpec input transformer.inputEmbedding

  -- Encoder
  let encoderOutput := TransformerEncoder.forward transformer.encoder embeddedInput h1 h2

  -- Target embedding
  let embeddedTarget := matMulSpec target transformer.outputEmbedding

  -- Decoder
  let decoderOutput := TransformerDecoder.forward transformer.decoder embeddedTarget encoderOutput
    h1 h2

  -- Output projection
  matMulSpec decoderOutput transformer.outputProjection


/--
Backward pass for `FeedForward.forward`.

Given the input `x` and an upstream gradient `outputGrad = dL/dy` (w.r.t. the FFN output),
returns:
- parameter gradients (as `FeedForwardGrads`),
- the gradient w.r.t. the input `x`.

This is a spec-level backward that reconstructs the forward intermediates (pre-activations and
ReLU mask) instead of relying on a mutable tape, similar to the math underlying PyTorch autograd.
-/
def FeedForward.backward {embedDim hiddenDim seqLen : Nat}
  (ffn : FeedForward embedDim hiddenDim α)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))  -- input to FFN
  (outputGrad : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h_seq : seqLen > 0) (_h_embed : embedDim > 0) :
  (FeedForwardGrads embedDim hiddenDim α ×
   Tensor α (.dim seqLen (.dim embedDim .scalar))) :=

  -- Forward pass reconstruction
  let preact := matMulSpec x ffn.W1
  let h1 : Shape.CanBroadcastTo (.dim hiddenDim .scalar) (.dim seqLen (.dim hiddenDim .scalar)) :=
    by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.scalar_to_any .scalar

  let z1 := addSpec preact (broadcastTo h1 ffn.b1)
  let a1 := reluSpec z1

  let h1 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.scalar_to_any .scalar
  let z2 := addSpec (matMulSpec a1 ffn.W2) (broadcastTo h1 ffn.b2)

  -- Backward pass
  let dz2 := outputGrad

  -- dW2 = a1^T ⋅ dz2
  let dW2 := matMulSpec (matrixTransposeSpec a1) dz2

  let h3 : seqLen ≠ 0 := by apply Shape.gt_pos_to_ne_zero h_seq
  have : Shape.valid_axis_inst 0 (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) :=
    Shape.validAxisInstZeroAlt h3

  -- db2 = sum across seqLen
  let db2 := reduceSumAuto 0 dz2

  -- da1 = dz2 ⋅ W2^T
  let da1 := matMulSpec dz2 (matrixTransposeSpec ffn.W2)

  -- dz1 = da1 ⊙ ReLU'(z1)
  let drelu := mulSpec da1 (reluDerivSpec z1)

  -- dW1 = x^T ⋅ dz1
  let dW1 := matMulSpec (matrixTransposeSpec x) drelu

  -- db1 = sum across seqLen
  have : Shape.valid_axis_inst 0 (Shape.dim seqLen (Shape.dim hiddenDim Shape.scalar)) :=
    Shape.validAxisInstZeroAlt h3
  let db1 := reduceSumAuto 0 drelu

  -- dInput = dz1 ⋅ W1^T
  let dInput := matMulSpec drelu (matrixTransposeSpec ffn.W1)

  ({ dW1 := dW1, dW2 := dW2, db1 := db1, db2 := db2 }, dInput)


/--
Backward pass for `TransformerEncoderLayer.forward`.

Inputs:
- `x`: the layer input `(seqLen × embedDim)`,
- `outputGrad`: upstream gradient w.r.t. the layer output.

Outputs:
- parameter gradients (`TransformerEncoderLayerGrads`),
- gradient w.r.t. `x`.

The implementation mirrors the forward pass structure (residuals + LayerNorm) and uses
`layerNorm_backward` and `MultiHeadAttention_backward` as its core primitives.
-/
def TransformerEncoderLayer.backward
  {headCount embedDim hiddenDim seqLen : Nat}
  (layer : TransformerEncoderLayer headCount embedDim hiddenDim α)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))  -- input
  (outputGrad : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  (TransformerEncoderLayerGrads headCount embedDim hiddenDim α ×
   Tensor α (.dim seqLen (.dim embedDim .scalar))) :=

  let h3 : seqLen ≠ 0 := by apply Shape.gt_pos_to_ne_zero h1

  -- Reconstruct forward intermediates (this keeps the backward spec local and compositional).
  let mhaOut := MultiHeadAttention.forward seqLen h3 layer.mha x none
  let res1 := addSpec x mhaOut
  let norm1 := layerNorm res1 layer.norm1_gamma layer.norm1_beta h1 h2
  let ffnOut := FeedForward.forward layer.ffn norm1
  let res2 := addSpec norm1 ffnOut
  let _y := layerNorm res2 layer.norm2_gamma layer.norm2_beta h1 h2

  -- Backprop through final LayerNorm.
  let (dRes2, dGamma2, dBeta2) :=
    layerNormBackward h1 h2 res2 layer.norm2_gamma layer.norm2_beta outputGrad

  -- Residual: res2 = norm1 + ffnOut
  let dNorm1_from_residual := dRes2
  let dFfnOut := dRes2

  -- FFN backward: returns (parameter grads, input grad).
  let (dFfnParams, dNorm1_from_ffn) := FeedForward.backward layer.ffn norm1 dFfnOut h1 h2
  let dNorm1 := addSpec dNorm1_from_residual dNorm1_from_ffn

  -- Backprop through first LayerNorm.
  let (dRes1, dGamma1, dBeta1) :=
    layerNormBackward h1 h2 res1 layer.norm1_gamma layer.norm1_beta dNorm1

  -- Residual: res1 = x + mhaOut
  let dX_from_residual := dRes1
  let dMhaOut := dRes1

  -- MHA backward: returns (dX_from_mha, dWq, dWk, dWv, dWo)
  let (dX_from_mha, dWq, dWk, dWv, dWo) :=
    MultiHeadAttentionBackward (α := α) (n := seqLen) (dModel := embedDim)
      (numHeads := headCount) (headDim := (embedDim / headCount))
      h3 layer.mha x none dMhaOut

  let dX := addSpec dX_from_residual dX_from_mha

  let grads : TransformerEncoderLayerGrads headCount embedDim hiddenDim α :=
    { mha := { dWq := dWq, dWk := dWk, dWv := dWv, dWo := dWo }
      ffn := dFfnParams
      d_norm1_gamma := dGamma1
      d_norm1_beta := dBeta1
      d_norm2_gamma := dGamma2
      d_norm2_beta := dBeta2 }

  (grads, dX)

/-!
## Backward pass for an encoder stack

The encoder is a list of layers applied sequentially. To compute gradients we:
1. re-run the forward pass to collect each layer's input (a small "cache"),
2. traverse layers in reverse, applying `TransformerEncoderLayer.backward`,
3. return per-layer parameter gradients plus the gradient w.r.t. the encoder input.

This is purely a spec (no mutation, no state), so we do the simplest thing: recompute.
-/

/--
Backward pass for `TransformerEncoder.forward` (a sequential stack of layers).

Returns:
- a list of per-layer parameter gradients (in the same order as `encoder.layers`),
- the gradient w.r.t. the encoder input `x`.

Because this is a pure spec, we recompute forward intermediates (each layer input) instead of
storing a mutable cache.
-/
def TransformerEncoder.backward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (encoder : TransformerEncoder numLayers headCount embedDim hiddenDim α)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (outputGrad : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  (List (TransformerEncoderLayerGrads headCount embedDim hiddenDim α) ×
   Tensor α (.dim seqLen (.dim embedDim .scalar))) :=

  -- Collect the input seen by each layer during the forward pass.
  let rec collect_inputs
    (layers : List (TransformerEncoderLayer headCount embedDim hiddenDim α))
    (cur : Tensor α (.dim seqLen (.dim embedDim .scalar))) :
    List (Tensor α (.dim seqLen (.dim embedDim .scalar))) :=
    match layers with
    | [] => []
    | l :: ls =>
        let next := TransformerEncoderLayer.forward l cur h1 h2
        cur :: collect_inputs ls next

  let inputs := collect_inputs encoder.layers x
  let pairs := List.zip encoder.layers inputs

  -- Reverse-mode: fold from the end.
  let step :
    (List (TransformerEncoderLayerGrads headCount embedDim hiddenDim α) ×
      Tensor α (.dim seqLen (.dim embedDim .scalar))) →
    (TransformerEncoderLayer headCount embedDim hiddenDim α × Tensor α (.dim seqLen (.dim embedDim
      .scalar))) →
    (List (TransformerEncoderLayerGrads headCount embedDim hiddenDim α) ×
      Tensor α (.dim seqLen (.dim embedDim .scalar))) :=
    fun (accGrads, grad) (layer, inp) =>
      let (g, dInp) := TransformerEncoderLayer.backward layer inp grad h1 h2
      (g :: accGrads, dInp)

  let (revGrads, dInput) := (pairs.reverse).foldl step ([], outputGrad)
  (revGrads.reverse, dInput)

/--
Sinusoidal positional encoding (Vaswani et al., 2017), added to the input sequence.

Given `x : (seqLen × embedDim)`, returns `x + pe` where `pe[pos, i]` alternates `sin`/`cos`
features with geometrically-spaced frequencies.

PyTorch analogue: positional encodings are often applied externally in PyTorch examples; the
high-level `torch.nn.Transformer` module does not force a particular encoding.
-/
def positionalEncoding {seqLen embedDim : Nat}
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar))) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  let pe := Tensor.dim (fun pos =>
    Tensor.dim (fun i =>
      let pos' : α := pos.val
      let i' : α := (i.val : α) / Numbers.two
      let denom : α := (embedDim : α)
      let angle := pos' * MathFunctions.exp (-i' * Numbers.log10000 / denom)
      Tensor.scalar (if i.val % 2 == 0 then MathFunctions.sin angle else MathFunctions.cos angle)
    )
  )
  addSpec x pe

-- Causal (autoregressive) attention masks are defined in `NN.Spec.Layers.Attention` as
-- `Spec.causalMask`.
/--
Multi-head self-attention with an optional boolean mask.

This is a thin wrapper around `MultiHeadAttention.forward` that:
- derives the required `seqLen ≠ 0` proof from `h1 : seqLen > 0`,
- forwards the provided `mask` (typically a causal mask for autoregressive decoding).

PyTorch analogue: masked self-attention in `torch.nn.TransformerDecoderLayer` implemented via
`torch.nn.MultiheadAttention(..., attn_mask=...)`.
-/
def maskedMultiHeadAttention {headCount embedDim seqLen : Nat}
  (mha : MultiHeadAttention α headCount embedDim (embedDim / headCount))
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (mask : Option (Tensor Bool (.dim seqLen (.dim seqLen .scalar))))
  (h1 : seqLen > 0) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  let h3 : seqLen ≠ 0 := by apply Shape.gt_pos_to_ne_zero h1
  MultiHeadAttention.forward seqLen h3 mha x mask


end Spec
