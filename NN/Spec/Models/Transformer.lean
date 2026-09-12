/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Normalization.Core
public import NN.Spec.Core.Sequence
public import NN.Tensor.Conversion

/-!
# Transformer (spec model)

This file defines Transformer-style spec components in a way that matches the usual PyTorch mental
model:

- encoder layers (self-attention + FFN, each wrapped in residual + LayerNorm),
- decoder layers (masked self-attention, cross-attention, FFN, each wrapped in residual +
  LayerNorm),
- an encoder-decoder wrapper (`Transformer`),
- spec-level backward passes for the encoder stack,
- a helper for masked multi-head self-attention.

Sinusoidal positional encodings live in `NN.Spec.Layers.PositionalEncoding` as
`Spec.sinusoidalPositionalEncodingSpec`; causal masks are `Spec.causalMask` in
`NN.Spec.Layers.Attention`.

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


open TorchLean

namespace Spec
open TorchLean TorchLean.Tensor
open Activation

variable {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-!
## Configuration helpers

This file mostly defines reusable transformer *building blocks* (encoder/decoder layers, attention,
layer-norm wrappers, etc.). To make compact model instantiations easier, we also provide a
small config record for the common hyperparameters together with a couple of canonical configs
(Base/Big).

The core definitions below still expose the hyperparameters as Nat parameters. The config layer is
only a named packaging of those parameters, so the mathematical specification remains the
parameterized transformer definition.
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
structure TransformerLayerConfig.WF (config : TransformerLayerConfig) : Prop where
  headCount_pos : config.headCount > 0
  embedDim_pos : config.embedDim > 0
  hiddenDim_pos : config.hiddenDim > 0
  headCount_dvd_embedDim : config.headCount ∣ config.embedDim

/-- Well-formedness conditions for `TransformerStackConfig`. -/
structure TransformerStackConfig.WF (config : TransformerStackConfig) : Prop where
  layer : config.toTransformerLayerConfig.WF

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

This container is used by downstream models that want a readable backward pass.
-/
structure FeedForwardGrads (embedDim hiddenDim : Nat) (α : Type)
    [TorchLean.Storage α] where
  /-- Gradient of the input projection weight. -/
  inputWeight : Tensor α [embedDim, hiddenDim]
  /-- Gradient of the output projection weight. -/
  outputWeight : Tensor α [hiddenDim, embedDim]
  /-- Gradient of the input projection bias. -/
  inputBias : Tensor α [hiddenDim]
  /-- Gradient of the output projection bias. -/
  outputBias : Tensor α [embedDim]

/--
Gradients for a `TransformerEncoderLayer` (field-for-field).

This container is intended to keep the backward pass readable by mirroring the parameter layout.
-/
structure TransformerEncoderLayerGrads (headCount embedDim hiddenDim : Nat) (α : Type)
    [TorchLean.Storage α] where
  /-- Gradients for the self-attention block. -/
  mha : MultiHeadAttentionParameterGradients headCount embedDim (embedDim / headCount) α
  /-- Gradients for the feedforward block. -/
  ffn : FeedForwardGrads embedDim hiddenDim α
  /-- Gradient of LayerNorm 1 gamma (attention "Add & Norm"). -/
  norm1Scale : Tensor α [embedDim]
  /-- Gradient of LayerNorm 1 beta (attention "Add & Norm"). -/
  norm1Bias : Tensor α [embedDim]
  /-- Gradient of LayerNorm 2 gamma (FFN "Add & Norm"). -/
  norm2Scale : Tensor α [embedDim]
  /-- Gradient of LayerNorm 2 beta (FFN "Add & Norm"). -/
  norm2Bias : Tensor α [embedDim]

/--
2-layer position-wise feedforward network used inside Transformer layers.

Semantics (per token):
`ffn(x) = (relu(x * W1 + b1) * W2) + b2`.

PyTorch analogue: the `linear1` / `linear2` submodule in `torch.nn.TransformerEncoderLayer`.
-/
structure FeedForward (embedDim hiddenDim : Nat) (α : Type)
    [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- First linear layer weights (`embedDim -> hiddenDim`). -/
  inputWeight : Tensor α [embedDim, hiddenDim]
  /-- Second linear layer weights (`hiddenDim -> embedDim`). -/
  outputWeight : Tensor α [hiddenDim, embedDim]
  /-- First layer bias (length `hiddenDim`). -/
  inputBias : Tensor α [hiddenDim]
  /-- Second layer bias (length `embedDim`). -/
  outputBias : Tensor α [embedDim]

/--
Forward pass for `FeedForward`.

Shape convention: inputs and outputs are `(seqLen × embedDim)`; the feedforward operates
independently on each sequence position.
-/
def FeedForward.forward {embedDim hiddenDim seqLen : Nat}
  (ffn : FeedForward embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  : Tensor α [seqLen, embedDim] :=
  let preact := matMulSpec x ffn.inputWeight
  let bc_b1 : Shape.CanBroadcastTo (.dim hiddenDim .scalar) (.dim seqLen (.dim hiddenDim .scalar))
    := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  let preact_reshaped := broadcastTo bc_b1 ffn.inputBias
  let preact_added := addSpec preact preact_reshaped
  let hidden := reluSpec preact_added

  let bc_b2 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) :=
    by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar
  let hidden_reshaped := broadcastTo bc_b2 ffn.outputBias
  addSpec (matMulSpec hidden ffn.outputWeight) hidden_reshaped

/--
Transformer encoder layer (post-norm).

This follows the common "Add & Norm" structure:
1. Self-attention, residual add, LayerNorm
2. Feedforward, residual add, LayerNorm

PyTorch analogue: `torch.nn.TransformerEncoderLayer` with `norm_first=False` (post-norm),
ignoring dropout and other configuration knobs.
-/
structure TransformerEncoderLayer (headCount embedDim hiddenDim : Nat) (α : Type)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- Multi-head self-attention block. -/
  mha : MultiHeadAttention α headCount embedDim (embedDim / headCount)
  /-- Position-wise feedforward block. -/
  ffn : FeedForward embedDim hiddenDim α

  /-- LayerNorm 1 gamma (attention "Add & Norm"). -/
  norm1Scale : Tensor α [embedDim]
  /-- LayerNorm 1 beta (attention "Add & Norm"). -/
  norm1Bias : Tensor α [embedDim]

  /-- LayerNorm 2 gamma (FFN "Add & Norm"). -/
  norm2Scale : Tensor α [embedDim]
  /-- LayerNorm 2 beta (FFN "Add & Norm"). -/
  norm2Bias : Tensor α [embedDim]

/--
Forward pass for a post-norm `TransformerEncoderLayer`.

Input/output shape: `(seqLen × embedDim)`.
The proofs `h1`/`h2` are used by `layerNorm` to justify nondegenerate normalization.
-/
def TransformerEncoderLayer.forward
  {headCount embedDim hiddenDim seqLen : Nat}
  (layer : TransformerEncoderLayer headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim]) (h1 : seqLen > 0) (h2 : embedDim > 0)
  : Tensor α [seqLen, embedDim] :=
  have h3 : seqLen ≠ 0 := Nat.ne_of_gt h1
  let attnOut := MultiHeadAttention.forward seqLen h3 layer.mha x none
  let attnAdded := addSpec x attnOut
  let normAttn := layerNorm attnAdded layer.norm1Scale layer.norm1Bias h1 h2
  let ffnOut := FeedForward.forward layer.ffn normAttn
  let ffnAdded := addSpec normAttn ffnOut
  layerNorm ffnAdded layer.norm2Scale layer.norm2Bias h1 h2

/--
Transformer encoder: a stack of `TransformerEncoderLayer`s.

PyTorch analogue: `torch.nn.TransformerEncoder` (a list of layers composed sequentially).
-/
structure TransformerEncoder (numLayers headCount embedDim hiddenDim : Nat) (α : Type)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- Encoder layers, with the stack depth recorded in the type. -/
  layers : Tensor (TransformerEncoderLayer headCount embedDim hiddenDim α)
    [numLayers]

/--
Forward pass for `TransformerEncoder` (left-fold over layers).

Input/output shape: `(seqLen × embedDim)`.
-/
def TransformerEncoder.forward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (encoder : TransformerEncoder numLayers headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim]) (h1 : seqLen > 0) (h2 : embedDim > 0)
  : Tensor α [seqLen, embedDim] :=
  (Tensor.to encoder.layers
    (Array (TransformerEncoderLayer headCount embedDim hiddenDim α))).foldl
      (fun acc layer => TransformerEncoderLayer.forward layer acc h1 h2) x

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
  (qInput : Tensor α [nQ, embedDim])
  (kvInput : Tensor α [nK, embedDim])
  (mask : Option (Tensor Bool [nQ, nK])) :
  Tensor α [nQ, embedDim] :=
  let Q := matMulSpec qInput mha.queryWeight
  let K := matMulSpec kvInput mha.keyWeight
  let V := matMulSpec kvInput mha.valueWeight
  let h : headCount * (embedDim / headCount) = headCount * (embedDim / headCount) := by rfl
  let QHeads := splitHeadsSpec Q headCount (embedDim / headCount) h
  let KHeads := splitHeadsSpec K headCount (embedDim / headCount) h
  let VHeads := splitHeadsSpec V headCount (embedDim / headCount) h
  let attentionHeads : Tensor α [headCount, nQ, embedDim / headCount] :=
    Tensor.dim (fun headIdx =>
      let ctx : AttentionContext α nQ nK (embedDim / headCount) hQ hK :=
        { Q := Tensor.unstack QHeads headIdx
          K := Tensor.unstack KHeads headIdx
          V := Tensor.unstack VHeads headIdx
          mask := mask }
      scaledDotProductAttention ctx)
  let concatenated :=
    combineHeadsSpec (α := α) (n := nQ) (numHeads := headCount) (headDim := (embedDim /
      headCount)) attentionHeads
  matMulSpec concatenated mha.outputWeight

/--
Transformer decoder layer (post-norm).

This mirrors the standard structure:
1. Self-attention (decoder stream), residual add, LayerNorm
2. Cross-attention (queries from decoder, keys/values from encoder), residual add, LayerNorm
3. Feedforward, residual add, LayerNorm

PyTorch analogue: `torch.nn.TransformerDecoderLayer` with `norm_first=False` (post-norm),
ignoring dropout and a few configuration knobs.
-/
structure TransformerDecoderLayer (headCount embedDim hiddenDim : Nat) (α : Type)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- Self-attention block over the decoder sequence. -/
  selfAttn : MultiHeadAttention α headCount embedDim (embedDim / headCount)
  /-- Cross-attention block (decoder queries, encoder keys/values). -/
  crossAttn : MultiHeadAttention α headCount embedDim (embedDim / headCount)
  /-- Position-wise feedforward block. -/
  ffn : FeedForward embedDim hiddenDim α
  /-- Scale of the layer normalization after self-attention. -/
  norm1Scale : Tensor α [embedDim]
  /-- Bias of the layer normalization after self-attention. -/
  norm1Bias : Tensor α [embedDim]

  /-- Scale of the layer normalization after cross-attention. -/
  norm2Scale : Tensor α [embedDim]
  /-- Bias of the layer normalization after cross-attention. -/
  norm2Bias : Tensor α [embedDim]

  /-- Scale of the layer normalization after the feed-forward block. -/
  norm3Scale : Tensor α [embedDim]
  /-- Bias of the layer normalization after the feed-forward block. -/
  norm3Bias : Tensor α [embedDim]

/--
Forward pass for a post-norm `TransformerDecoderLayer`.

Input/output shape: `(seqLen × embedDim)`. This spec uses the same `seqLen` for encoder and decoder
streams for simplicity (cross-attention uses `nQ = nK = seqLen`).

`targetMask` controls decoder self-attention, and `memoryMask` controls attention to the encoder
output. In both masks, `true` allows an attention edge and `false` blocks it. Pass
`some (causalMask seqLen)` as `targetMask` for autoregressive decoding: position `i` then uses only
target positions `0, ..., i`. The source sequence remains available through cross-attention unless
`memoryMask` restricts it separately. Both arguments default to `none` for unmasked attention.
-/
def TransformerDecoderLayer.forward
  {headCount embedDim hiddenDim seqLen : Nat}
  (layer : TransformerDecoderLayer headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  (encoderOutput : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0) (h2 : embedDim > 0)
  (targetMask : Option (Tensor Bool [seqLen, seqLen]) := none)
  (memoryMask : Option (Tensor Bool [seqLen, seqLen]) := none) :
  Tensor α [seqLen, embedDim] :=
  have h3 : seqLen ≠ 0 := Nat.ne_of_gt h1
  -- Self-attention with residual connection
  let selfAttnOut := MultiHeadAttention.forward seqLen h3 layer.selfAttn x targetMask
  let selfAttnAdded := addSpec x selfAttnOut
  let normSelfAttn := layerNorm selfAttnAdded layer.norm1Scale layer.norm1Bias h1 h2

  -- Cross-attention with residual connection
  let crossAttnOut :=
    multiHeadCrossAttention (α := α)
      (headCount := headCount) (embedDim := embedDim) (nQ := seqLen) (nK := seqLen)
      h3 h3 layer.crossAttn normSelfAttn encoderOutput memoryMask
  let crossAttnAdded := addSpec normSelfAttn crossAttnOut
  let normCrossAttn := layerNorm crossAttnAdded layer.norm2Scale layer.norm2Bias h1 h2

  -- Feedforward with residual connection
  let ffnOut := FeedForward.forward layer.ffn normCrossAttn
  let ffnAdded := addSpec normCrossAttn ffnOut
  layerNorm ffnAdded layer.norm3Scale layer.norm3Bias h1 h2

/--
Transformer decoder: a stack of `TransformerDecoderLayer`s.

PyTorch analogue: `torch.nn.TransformerDecoder` (a list of decoder layers composed sequentially).
-/
structure TransformerDecoder (numLayers headCount embedDim hiddenDim : Nat) (α : Type)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- Decoder layers, with the stack depth recorded in the type. -/
  layers : Tensor (TransformerDecoderLayer headCount embedDim hiddenDim α)
    [numLayers]

/--
Forward pass for `TransformerDecoder` (left-fold over layers).

Input/output shape: `(seqLen × embedDim)`. Every layer receives the same target and memory masks.
A causal target mask must reach the whole stack: unmasked attention in any layer would let future
target positions enter the earlier residual streams.
-/
def TransformerDecoder.forward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (decoder : TransformerDecoder numLayers headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  (encoderOutput : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0) (h2 : embedDim > 0)
  (targetMask : Option (Tensor Bool [seqLen, seqLen]) := none)
  (memoryMask : Option (Tensor Bool [seqLen, seqLen]) := none) :
  Tensor α [seqLen, embedDim] :=
  (Tensor.to decoder.layers
    (Array (TransformerDecoderLayer headCount embedDim hiddenDim α))).foldl
    (fun acc layer =>
      TransformerDecoderLayer.forward layer acc encoderOutput h1 h2 targetMask memoryMask) x

/--
End-to-end encoder-decoder Transformer (spec model).

This is a seq2seq Transformer wrapper built out of the encoder and decoder stacks above.
It models the core tensor algebra of `torch.nn.Transformer` while making the proof-relevant choices
explicit:
- embeddings are modeled as explicit linear projections,
- sequence length is shared between source and target streams,
- we omit dropout, caching, and most configuration knobs.

Shape convention: all activations in this file use `(seqLen × embedDim)`.
In a full implementation, `outputProjection` would usually map to a vocabulary size; here it is
kept as an `embedDim -> embedDim` projection to stay in the "core tensor algebra" setting.
-/
structure Transformer (numLayers headCount embedDim hiddenDim : Nat) (α : Type)
  [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] where
  /-- Encoder stack. -/
  encoder : TransformerEncoder numLayers headCount embedDim hiddenDim α
  /-- Decoder stack. -/
  decoder : TransformerDecoder numLayers headCount embedDim hiddenDim α
  /-- Source/input embedding projection matrix. -/
  inputEmbedding : Tensor α [embedDim, embedDim]
  /-- Target embedding projection matrix. -/
  outputEmbedding : Tensor α [embedDim, embedDim]
  /-- Final output projection matrix (here `embedDim -> embedDim`). -/
  outputProjection : Tensor α [embedDim, embedDim]

/--
Forward pass for `Transformer`.

Runs:
1. source embedding projection,
2. encoder stack,
3. target embedding projection,
4. decoder stack (with cross-attention to the encoder output),
5. output projection.

All tensors in this simplified spec have shape `(seqLen × embedDim)`. `targetMask` and `memoryMask`
are passed to every decoder layer. For next-token prediction, supply shifted target inputs and
`targetMask := some (causalMask seqLen)`; the mask permits the current input token while blocking
later inputs. Source encoding stays bidirectional.
-/
def Transformer.forward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (transformer : Transformer numLayers headCount embedDim hiddenDim α)
  (input : Tensor α [seqLen, embedDim])
  (target : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0) (h2 : embedDim > 0)
  (targetMask : Option (Tensor Bool [seqLen, seqLen]) := none)
  (memoryMask : Option (Tensor Bool [seqLen, seqLen]) := none) :
  Tensor α [seqLen, embedDim] :=
  -- Input embedding
  let embeddedInput := matMulSpec input transformer.inputEmbedding

  -- Encoder
  let encoderOutput := TransformerEncoder.forward transformer.encoder embeddedInput h1 h2

  -- Target embedding
  let embeddedTarget := matMulSpec target transformer.outputEmbedding

  -- Decoder
  let decoderOutput := TransformerDecoder.forward transformer.decoder embeddedTarget encoderOutput
    h1 h2 targetMask memoryMask

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
  (x : Tensor α [seqLen, embedDim])
  (outputGrad : Tensor α [seqLen, embedDim])
  (h_seq : seqLen > 0) (_h_embed : embedDim > 0) :
  (FeedForwardGrads embedDim hiddenDim α ×
   Tensor α [seqLen, embedDim]) :=

  -- Forward pass reconstruction
  let preact := matMulSpec x ffn.inputWeight
  let h1 : Shape.CanBroadcastTo (.dim hiddenDim .scalar) (.dim seqLen (.dim hiddenDim .scalar)) :=
    by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  let z1 := addSpec preact (broadcastTo h1 ffn.inputBias)
  let a1 := reluSpec z1

  let h1 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar
  let z2 := addSpec (matMulSpec a1 ffn.outputWeight) (broadcastTo h1 ffn.outputBias)

  -- Backward pass
  let dz2 := outputGrad

  -- dW2 = a1^T ⋅ dz2
  let outputWeight := matMulSpec (swapAdjacentAxes a1 0) dz2

  let h3 : seqLen ≠ 0 := Nat.ne_of_gt h_seq
  have : Shape.HasNonemptyAxis 0 (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) :=
    Shape.hasNonemptyAxisZeroOfNe h3

  -- db2 = sum across seqLen
  let outputBias := reduceSum 0 dz2 this.proof

  -- da1 = dz2 ⋅ W2^T
  let da1 := matMulSpec dz2 (swapAdjacentAxes ffn.outputWeight 0)

  -- dz1 = da1 ⊙ ReLU'(z1)
  let drelu := mulSpec da1 (reluDerivSpec z1)

  -- dW1 = x^T ⋅ dz1
  let inputWeight := matMulSpec (swapAdjacentAxes x 0) drelu

  -- db1 = sum across seqLen
  have : Shape.HasNonemptyAxis 0 (Shape.dim seqLen (Shape.dim hiddenDim Shape.scalar)) :=
    Shape.hasNonemptyAxisZeroOfNe h3
  let inputBias := reduceSum 0 drelu this.proof

  -- dInput = dz1 ⋅ W1^T
  let dInput := matMulSpec drelu (swapAdjacentAxes ffn.inputWeight 0)

  ({ inputWeight, outputWeight, inputBias, outputBias }, dInput)


/--
Backward pass for `TransformerEncoderLayer.forward`.

Inputs:
- `x`: the layer input `(seqLen × embedDim)`,
- `outputGrad`: upstream gradient w.r.t. the layer output.

Outputs:
- parameter gradients (`TransformerEncoderLayerGrads`),
- gradient w.r.t. `x`.

The implementation mirrors the forward pass structure (residuals + LayerNorm) and uses
`layerNormBackward` and `multiHeadAttentionBackward` as its core primitives.
-/
def TransformerEncoderLayer.backward
  {headCount embedDim hiddenDim seqLen : Nat}
  (layer : TransformerEncoderLayer headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  (outputGrad : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  (TransformerEncoderLayerGrads headCount embedDim hiddenDim α ×
   Tensor α [seqLen, embedDim]) :=

  let h3 : seqLen ≠ 0 := Nat.ne_of_gt h1

  -- Reconstruct forward intermediates locally for this backward spec.
  let mhaOut := MultiHeadAttention.forward seqLen h3 layer.mha x none
  let res1 := addSpec x mhaOut
  let norm1 := layerNorm res1 layer.norm1Scale layer.norm1Bias h1 h2
  let ffnOut := FeedForward.forward layer.ffn norm1
  let res2 := addSpec norm1 ffnOut
  let _y := layerNorm res2 layer.norm2Scale layer.norm2Bias h1 h2

  -- Backprop through final LayerNorm.
  let norm2Gradients :=
    layerNormBackward h1 h2 res2 layer.norm2Scale outputGrad
  let dRes2 := norm2Gradients.inputGradient
  let dGamma2 := norm2Gradients.scaleGradient
  let dBeta2 := norm2Gradients.biasGradient

  -- Residual: res2 = norm1 + ffnOut
  let dNorm1_from_residual := dRes2
  let dFfnOut := dRes2

  -- FFN backward: returns (parameter grads, input grad).
  let (dFfnParams, dNorm1_from_ffn) := FeedForward.backward layer.ffn norm1 dFfnOut h1 h2
  let dNorm1 := addSpec dNorm1_from_residual dNorm1_from_ffn

  -- Backprop through first LayerNorm.
  let norm1Gradients :=
    layerNormBackward h1 h2 res1 layer.norm1Scale dNorm1
  let dRes1 := norm1Gradients.inputGradient
  let dGamma1 := norm1Gradients.scaleGradient
  let dBeta1 := norm1Gradients.biasGradient

  -- Residual: res1 = x + mhaOut
  let dX_from_residual := dRes1
  let dMhaOut := dRes1

  let mhaGradients :=
    multiHeadAttentionBackward (α := α) (n := seqLen) (dModel := embedDim)
      (numHeads := headCount) (headDim := (embedDim / headCount))
      h3 layer.mha x none dMhaOut

  let dX := addSpec dX_from_residual mhaGradients.input

  let grads : TransformerEncoderLayerGrads headCount embedDim hiddenDim α :=
    { mha := mhaGradients.parameters
      ffn := dFfnParams
      norm1Scale := dGamma1
      norm1Bias := dBeta1
      norm2Scale := dGamma2
      norm2Bias := dBeta2 }

  (grads, dX)

/-!
## Backward pass for an encoder stack

The encoder is a fixed-length vector of layers applied sequentially. To compute gradients we:
1. re-run the forward pass to collect each layer's input (a small "cache"),
2. traverse layers in reverse, applying `TransformerEncoderLayer.backward`,
3. return per-layer parameter gradients plus the gradient w.r.t. the encoder input.

This is purely a spec (no mutation, no state), so we do the simplest thing: recompute.
-/

/--
Backward pass for `TransformerEncoder.forward` (a sequential stack of layers).

Returns:
- one parameter-gradient record per layer, in the same order as `encoder.layers`,
- the gradient w.r.t. the encoder input `x`.

Because this is a pure spec, we recompute forward intermediates (each layer input) instead of
storing a mutable cache.
-/
def TransformerEncoder.backward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (encoder : TransformerEncoder numLayers headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  (outputGrad : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  (Tensor (TransformerEncoderLayerGrads headCount embedDim hiddenDim α)
      [numLayers] ×
   Tensor α [seqLen, embedDim]) :=
  let (_, inputs) := Sequence.mapAccum numLayers x fun i current =>
    let next := TransformerEncoderLayer.forward (encoder.layers.getScalar i) current h1 h2
    (next, current)
  let (inputGrad, layerGrads) := Sequence.mapAccumRight numLayers outputGrad fun i grad =>
    let (paramsGrad, previousGrad) :=
      TransformerEncoderLayer.backward (encoder.layers.getScalar i) (inputs.getScalar i) grad h1 h2
    (previousGrad, paramsGrad)
  (layerGrads, inputGrad)

-- Causal (autoregressive) attention masks are defined in `NN.Spec.Layers.Attention` as
-- `Spec.causalMask`.
/--
Multi-head self-attention with an optional boolean mask.

This helper prepares the proof obligations required by `MultiHeadAttention.forward`:
- derives the required `seqLen ≠ 0` proof from `h1 : seqLen > 0`,
- forwards the provided `mask` (typically a causal mask for autoregressive decoding).

PyTorch analogue: masked self-attention in `torch.nn.TransformerDecoderLayer` implemented via
`torch.nn.MultiheadAttention(..., attn_mask=...)`.
-/
def maskedMultiHeadAttention {headCount embedDim seqLen : Nat}
  (mha : MultiHeadAttention α headCount embedDim (embedDim / headCount))
  (x : Tensor α [seqLen, embedDim])
  (mask : Option (Tensor Bool [seqLen, seqLen]))
  (h1 : seqLen > 0) :
  Tensor α [seqLen, embedDim] :=
  let h3 : seqLen ≠ 0 := Nat.ne_of_gt h1
  MultiHeadAttention.forward seqLen h3 mha x mask


end Spec
