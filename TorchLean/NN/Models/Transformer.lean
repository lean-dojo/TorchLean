/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Attention
public import TorchLean.Tensor.Layers.Normalization.LayerNorm
public import TorchLean.Tensor.Sequence
public import TorchLean.Tensor.Reduce.Fold
public import TorchLean.Tensor.Layers.Normalization

/-!
# Transformer (spec model)

The spec Transformer follows the structure of the usual PyTorch modules. An encoder layer wraps
self-attention and a feedforward network (FFN) each in a residual add and LayerNorm. A decoder
layer does the same for self-attention, cross-attention and an FFN, with an optional mask on each
attention, and `Transformer` joins an encoder stack to a decoder stack. Backward passes cover the
FFN, the encoder layer and the encoder stack, and each returns its parameter gradients in the
block's own parameter record (`Transformer.FeedForward`, `Transformer.EncoderLayer`).

Sinusoidal positional encodings are `Tensor.sinusoidalPositionalEncoding` in
`TorchLean.Tensor.Layers.PositionalEncoding`, and causal masks are `TorchLean.Attention.causalMask`
in `TorchLean.Tensor.Layers.Attention`.

Sequence tensors have shape `(seqLen × embedDim)`, and attention takes its softmax over the last
axis, the key axis.

We wrote the forward passes by hand from Vaswani et al. Each sublayer ends in
`LayerNorm(x + Sublayer(x))`, the post-norm of their Section 3.1, and the FFN is their eq. (2)
with ReLU. Those are the defaults of `torch.nn.TransformerEncoderLayer` in PyTorch 2.8
(`norm_first=False`, `activation=relu`), and our LayerNorm `epsilon` of `1e-5` is its
`layer_norm_eps`. Attention is where we differ from torch. `MultiHeadAttention` holds four
bias-free matrices, while `torch.nn.MultiheadAttention` defaults to `bias=True` on its input and
output projections. There's also no positional encoding (Section 3.5) and no `√d_model` embedding
factor (Section 3.4). The backward passes are chain rules written by hand that recompute the forward
intermediates. No theorem relates them to a derivative, and no test runs them against torch.
`TorchLean.Test.PyTorchRoundtrip.run` compares a one-token encoder block of width 2 with PyTorch
in float64, but it builds the block with `nn.transformerBlock` and never calls
`Transformer.EncoderLayer.forward`.

PyTorch analogy:
- `Transformer.EncoderLayer.forward` corresponds to the core of `torch.nn.TransformerEncoderLayer`,
  ignoring dropout and some configuration options,
- `Transformer.EncoderStack.forward` corresponds to `torch.nn.TransformerEncoder`,
- `Transformer.DecoderLayer.forward` corresponds to the core of `torch.nn.TransformerDecoderLayer`,
- `Transformer.DecoderStack.forward` corresponds to `torch.nn.TransformerDecoder`,
- `Transformer.forward` is a simplified `torch.nn.Transformer`.

References:
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
  Łukasz Kaiser, and Illia Polosukhin (2017), "Attention Is All You Need", NIPS 2017,
  arXiv:1706.03762, Section 3.1 and eq. (2).
- Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton (2016), "Layer Normalization",
  arXiv:1607.06450, Section 3, eq. (3).
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016), "Deep Residual Learning for Image
  Recognition", CVPR 2016, arXiv:1512.03385, for the residual/skip-connection pattern.

PyTorch 2.8 docs:
- `torch.nn.TransformerEncoderLayer`:
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.TransformerEncoderLayer.html
- `torch.nn.TransformerEncoder`:
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.TransformerEncoder.html
- `torch.nn.TransformerDecoderLayer`:
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.TransformerDecoderLayer.html
- `torch.nn.TransformerDecoder`:
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.TransformerDecoder.html
- `torch.nn.Transformer`: https://docs.pytorch.org/docs/2.8/generated/torch.nn.Transformer.html
- `torch.nn.MultiheadAttention`:
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html
-/

@[expose] public section


namespace TorchLean
open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/--
Two-layer position-wise feedforward network of a Transformer layer: each token `x` maps to
`relu(x W1 + b1) W2 + b2`.

PyTorch analogue: the `linear1` and `linear2` submodules of `torch.nn.TransformerEncoderLayer`
with its default ReLU activation.
-/
structure Transformer.FeedForward (embedDim hiddenDim : Nat) (α : Type)
    [Storage α] where
  /-- First linear layer weights (`embedDim -> hiddenDim`). -/
  inputWeight : Tensor α [embedDim, hiddenDim]
  /-- Second linear layer weights (`hiddenDim -> embedDim`). -/
  outputWeight : Tensor α [hiddenDim, embedDim]
  /-- First layer bias (length `hiddenDim`). -/
  inputBias : Tensor α [hiddenDim]
  /-- Second layer bias (length `embedDim`). -/
  outputBias : Tensor α [embedDim]

/--
Forward pass of `FeedForward` on a `(seqLen × embedDim)` sequence, applied to each position
independently.
-/
def Transformer.FeedForward.forward {embedDim hiddenDim seqLen : Nat}
  (ffn : FeedForward embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  : Tensor α [seqLen, embedDim] :=
  let preact := Tensor.matmul x ffn.inputWeight
  let bc_b1 : Shape.CanBroadcastTo (.dim hiddenDim .scalar) (.dim seqLen (.dim hiddenDim .scalar))
    := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  let preact_reshaped := Tensor.broadcast ffn.inputBias _ bc_b1.dims bc_b1.broadcastDims
  let preact_added := preact + preact_reshaped
  let hidden := Tensor.relu preact_added

  let bc_b2 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) :=
    by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar
  let hidden_reshaped := Tensor.broadcast ffn.outputBias _ bc_b2.dims bc_b2.broadcastDims
  Tensor.matmul hidden ffn.outputWeight + hidden_reshaped

/--
Post-norm Transformer encoder layer: self-attention and then the feedforward block, each followed
by a residual add and LayerNorm ("Add & Norm").

PyTorch analogue: `torch.nn.TransformerEncoderLayer` with `norm_first=False`, ignoring dropout and
other configuration options.
-/
structure Transformer.EncoderLayer (headCount embedDim hiddenDim : Nat) (α : Type)
  [Storage α] where
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
Forward pass of a post-norm `EncoderLayer` on a `(seqLen × embedDim)` sequence, with
unmasked self-attention. `h1` makes the attention sequence nonempty.
-/
def Transformer.EncoderLayer.forward
  {headCount embedDim hiddenDim seqLen : Nat}
  (layer : EncoderLayer headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim]) (h1 : seqLen > 0)
  : Tensor α [seqLen, embedDim] :=
  have h3 : seqLen ≠ 0 := Nat.ne_of_gt h1
  let attnOut := MultiHeadAttention.forward h3 h3 layer.mha x x none
  let attnAdded := x + attnOut
  let normAttn := Tensor.layerNorm attnAdded [⟨1, Nat.one_lt_two⟩] layer.norm1Scale layer.norm1Bias
  let ffnOut := FeedForward.forward layer.ffn normAttn
  let ffnAdded := normAttn + ffnOut
  Tensor.layerNorm ffnAdded [⟨1, Nat.one_lt_two⟩] layer.norm2Scale layer.norm2Bias

/--
Transformer encoder: a stack of `EncoderLayer`s.

PyTorch analogue: `torch.nn.TransformerEncoder` (a list of layers composed sequentially).
-/
structure Transformer.EncoderStack (numLayers headCount embedDim hiddenDim : Nat) (α : Type)
  [Storage α] where
  /-- Encoder layers, with the stack depth recorded in the type. -/
  layers : Tensor (EncoderLayer headCount embedDim hiddenDim α)
    [numLayers]

/--
Forward pass for `EncoderStack` (left-fold over layers).

Input/output shape: `(seqLen × embedDim)`.
-/
def Transformer.EncoderStack.forward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (encoder : EncoderStack numLayers headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim]) (h1 : seqLen > 0)
  : Tensor α [seqLen, embedDim] :=
  Tensor.foldl (fun acc layer => EncoderLayer.forward layer acc h1) x encoder.layers

/-!
## Decoder notes

The decoder layer has the blocks of the encoder layer, with cross-attention inserted between
self-attention and the feedforward block. Its queries come from the decoder stream and its keys and
values from the encoder output. Both attentions call `MultiHeadAttention.forward`, which takes the
query rows and the key and value rows as separate inputs.
-/

/--
Post-norm Transformer decoder layer: self-attention over the decoder stream, cross-attention to the
encoder output and the feedforward block, each followed by a residual add and LayerNorm.

PyTorch analogue: `torch.nn.TransformerDecoderLayer` with `norm_first=False`, ignoring dropout and
a few configuration options.
-/
structure Transformer.DecoderLayer (headCount embedDim hiddenDim : Nat) (α : Type)
  [Storage α] where
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
Forward pass of a post-norm `DecoderLayer`. The decoder stream `x` and the encoder
output share the shape `(seqLen × embedDim)`, so cross-attention has `nQ = nK = seqLen`.

`targetMask` masks decoder self-attention and `memoryMask` masks attention to the encoder output.
In both, `true` allows an attention edge and `false` blocks it. With `some (causalMask seqLen)` as
`targetMask`, position `i` uses only target positions `0, ..., i`, as autoregressive decoding
needs, while cross-attention sees the whole source unless `memoryMask` restricts it. Both masks
default to `none`, which leaves attention unmasked.
-/
def Transformer.DecoderLayer.forward
  {headCount embedDim hiddenDim seqLen : Nat}
  (layer : DecoderLayer headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  (encoderOutput : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0)
  (targetMask : Option (Tensor Bool [seqLen, seqLen]) := none)
  (memoryMask : Option (Tensor Bool [seqLen, seqLen]) := none) :
  Tensor α [seqLen, embedDim] :=
  have h3 : seqLen ≠ 0 := Nat.ne_of_gt h1
  -- Self-attention with residual connection
  let selfAttnOut := MultiHeadAttention.forward h3 h3 layer.selfAttn x x targetMask
  let selfAttnAdded := x + selfAttnOut
  let normSelfAttn :=
    Tensor.layerNorm selfAttnAdded [⟨1, Nat.one_lt_two⟩] layer.norm1Scale layer.norm1Bias

  let crossAttnOut :=
    MultiHeadAttention.forward h3 h3 layer.crossAttn normSelfAttn encoderOutput memoryMask
  let crossAttnAdded := normSelfAttn + crossAttnOut
  let normCrossAttn :=
    Tensor.layerNorm crossAttnAdded [⟨1, Nat.one_lt_two⟩] layer.norm2Scale layer.norm2Bias

  let ffnOut := FeedForward.forward layer.ffn normCrossAttn
  let ffnAdded := normCrossAttn + ffnOut
  Tensor.layerNorm ffnAdded [⟨1, Nat.one_lt_two⟩] layer.norm3Scale layer.norm3Bias

/--
Transformer decoder: a stack of `DecoderLayer`s.

PyTorch analogue: `torch.nn.TransformerDecoder` (a list of decoder layers composed sequentially).
-/
structure Transformer.DecoderStack (numLayers headCount embedDim hiddenDim : Nat) (α : Type)
  [Storage α] where
  /-- Decoder layers, with the stack depth recorded in the type. -/
  layers : Tensor (DecoderLayer headCount embedDim hiddenDim α)
    [numLayers]

/--
Forward pass for `DecoderStack` (left-fold over layers).

Input/output shape: `(seqLen × embedDim)`. Every layer receives the same target and memory masks.
A causal target mask must reach the whole stack: unmasked attention in any layer would let future
target positions enter the earlier residual streams.
-/
def Transformer.DecoderStack.forward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (decoder : DecoderStack numLayers headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  (encoderOutput : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0)
  (targetMask : Option (Tensor Bool [seqLen, seqLen]) := none)
  (memoryMask : Option (Tensor Bool [seqLen, seqLen]) := none) :
  Tensor α [seqLen, embedDim] :=
  Tensor.foldl
    (fun acc layer =>
      DecoderLayer.forward layer acc encoderOutput h1 targetMask memoryMask)
    x decoder.layers

/--
End-to-end encoder-decoder Transformer built from the encoder and decoder stacks.

It models the core tensor algebra of `torch.nn.Transformer` with some simplifications. The source
and target embeddings are `embedDim × embedDim` linear projections, both streams share the
sequence length `seqLen`, and dropout, caching, the final LayerNorm after each stack and most
configuration options are left out. Every activation, the output included, has shape
`(seqLen × embedDim)`, so `outputProjection` maps `embedDim` to `embedDim` where a full model would
map to a vocabulary.
-/
structure Transformer (numLayers headCount embedDim hiddenDim : Nat) (α : Type)
  [Storage α] where
  /-- Encoder stack. -/
  encoder : Transformer.EncoderStack numLayers headCount embedDim hiddenDim α
  /-- Decoder stack. -/
  decoder : Transformer.DecoderStack numLayers headCount embedDim hiddenDim α
  /-- Source/input embedding projection matrix. -/
  inputEmbedding : Tensor α [embedDim, embedDim]
  /-- Target embedding projection matrix. -/
  outputEmbedding : Tensor α [embedDim, embedDim]
  /-- Final output projection matrix (here `embedDim -> embedDim`). -/
  outputProjection : Tensor α [embedDim, embedDim]

/--
Forward pass of `Transformer`: project the source and run the encoder stack, project the target and
run the decoder stack with cross-attention to the encoder output, then apply `outputProjection`.

Every decoder layer receives `targetMask` and `memoryMask`, while the encoder runs unmasked, so
source encoding is bidirectional. For next-token prediction, supply shifted target inputs and
`targetMask := some (causalMask seqLen)`, which lets each position attend to its own input token
and blocks the later ones.
-/
def Transformer.forward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (transformer : Transformer numLayers headCount embedDim hiddenDim α)
  (input : Tensor α [seqLen, embedDim])
  (target : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0)
  (targetMask : Option (Tensor Bool [seqLen, seqLen]) := none)
  (memoryMask : Option (Tensor Bool [seqLen, seqLen]) := none) :
  Tensor α [seqLen, embedDim] :=
  let embeddedInput := Tensor.matmul input transformer.inputEmbedding

  let encoderOutput := EncoderStack.forward transformer.encoder embeddedInput h1

  let embeddedTarget := Tensor.matmul target transformer.outputEmbedding

  let decoderOutput := DecoderStack.forward transformer.decoder embeddedTarget encoderOutput
    h1 targetMask memoryMask

  Tensor.matmul decoderOutput transformer.outputProjection


/--
Backward pass of `FeedForward.forward` at input `x` and upstream gradient `outputGrad`, returning
the parameter gradients as a `FeedForward`, each field the gradient of that field, and the gradient
with respect to `x`. The pre-activation and the ReLU mask are recomputed from `x`, so the pass
needs no tape.
-/
def Transformer.FeedForward.backward {embedDim hiddenDim seqLen : Nat}
  (ffn : FeedForward embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  (outputGrad : Tensor α [seqLen, embedDim]) :
  (FeedForward embedDim hiddenDim α ×
   Tensor α [seqLen, embedDim]) :=

  let preact := Tensor.matmul x ffn.inputWeight
  let h1 : Shape.CanBroadcastTo (.dim hiddenDim .scalar) (.dim seqLen (.dim hiddenDim .scalar)) :=
    by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  let z1 := preact + Tensor.broadcast ffn.inputBias _ h1.dims h1.broadcastDims
  let a1 := Tensor.relu z1

  let dz2 := outputGrad

  -- dW2 = a1^T ⋅ dz2
  let outputWeight := Tensor.matmul (Tensor.permute a1
      (Shape.adjacentSwap [seqLen, hiddenDim] 0) [hiddenDim, seqLen]) dz2

  -- db2 = sum across seqLen
  let outputBias := castShape (reduce .sum dz2 [⟨0, by simp⟩] false)
      (Shape.reduce_singleton _ _)

  -- da1 = dz2 ⋅ W2^T
  let da1 := Tensor.matmul dz2 (Tensor.permute ffn.outputWeight
      (Shape.adjacentSwap [hiddenDim, embedDim] 0) [embedDim, hiddenDim])

  -- dz1 = da1 ⊙ ReLU'(z1)
  let drelu := da1 * Tensor.map Activation.reluDeriv z1

  -- dW1 = x^T ⋅ dz1
  let inputWeight := Tensor.matmul (Tensor.permute x
      (Shape.adjacentSwap [seqLen, embedDim] 0) [embedDim, seqLen]) drelu

  -- db1 = sum across seqLen
  let inputBias := castShape (reduce .sum drelu [⟨0, by simp⟩] false)
      (Shape.reduce_singleton _ _)

  -- dInput = dz1 ⋅ W1^T
  let dInput := Tensor.matmul drelu (Tensor.permute ffn.inputWeight
      (Shape.adjacentSwap [embedDim, hiddenDim] 0) [hiddenDim, embedDim])

  ({ inputWeight, outputWeight, inputBias, outputBias }, dInput)


/--
Backward pass of `EncoderLayer.forward` at the layer input `x : (seqLen × embedDim)` and
the upstream gradient `outputGrad`, returning the parameter gradients as an `EncoderLayer` and the
gradient with respect to `x`.

It recomputes the forward intermediates and runs `LayerNorm.backward`,
`Transformer.FeedForward.backward` and `MultiHeadAttention.backward` back through the two residual
blocks. `h1` and `h2` make the sequence and the embedding nonempty, as `LayerNorm.backward`
requires.
-/
def Transformer.EncoderLayer.backward
  {headCount embedDim hiddenDim seqLen : Nat}
  (layer : EncoderLayer headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  (outputGrad : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  (EncoderLayer headCount embedDim hiddenDim α ×
   Tensor α [seqLen, embedDim]) :=

  let h3 : seqLen ≠ 0 := Nat.ne_of_gt h1

  -- Reconstruct forward intermediates locally for this backward spec.
  let mhaOut := MultiHeadAttention.forward h3 h3 layer.mha x x none
  let res1 := x + mhaOut
  let norm1 := Tensor.layerNorm res1 [⟨1, Nat.one_lt_two⟩] layer.norm1Scale layer.norm1Bias
  let ffnOut := FeedForward.forward layer.ffn norm1
  let res2 := norm1 + ffnOut

  -- Backprop through final LayerNorm.
  let norm2Gradients :=
    LayerNorm.backward h1 h2 res2 layer.norm2Scale outputGrad
  let dRes2 := norm2Gradients.inputGradient
  let dGamma2 := norm2Gradients.scaleGradient
  let dBeta2 := norm2Gradients.biasGradient

  -- `res2 = norm1 + ffnOut` passes `dRes2` unchanged to both summands.
  let dNorm1_from_residual := dRes2
  let dFfnOut := dRes2

  -- `norm1` reaches `res2` directly and through the FFN, so its gradient sums both paths.
  let (dFfnParams, dNorm1_from_ffn) := FeedForward.backward layer.ffn norm1 dFfnOut
  let dNorm1 := dNorm1_from_residual + dNorm1_from_ffn

  -- Backprop through first LayerNorm.
  let norm1Gradients :=
    LayerNorm.backward h1 h2 res1 layer.norm1Scale dNorm1
  let dRes1 := norm1Gradients.inputGradient
  let dGamma1 := norm1Gradients.scaleGradient
  let dBeta1 := norm1Gradients.biasGradient

  -- `res1 = x + mhaOut` passes `dRes1` to both summands, so `dX` adds the attention input
  -- gradient to the residual one.
  let dX_from_residual := dRes1
  let dMhaOut := dRes1

  let mhaGradients :=
    MultiHeadAttention.backward (α := α) (n := seqLen) (dModel := embedDim)
      (numHeads := headCount) (headDim := (embedDim / headCount))
      h3 layer.mha x none dMhaOut

  let dX := dX_from_residual + mhaGradients.input

  let grads : EncoderLayer headCount embedDim hiddenDim α :=
    { mha := mhaGradients.parameters
      ffn := dFfnParams
      norm1Scale := dGamma1
      norm1Bias := dBeta1
      norm2Scale := dGamma2
      norm2Bias := dBeta2 }

  (grads, dX)

/-!
## Backward pass for an encoder stack

The encoder applies a fixed-length vector of layers in sequence. Its backward pass re-runs the
forward pass to collect each layer's input, then walks the layers in reverse with
`Transformer.EncoderLayer.backward`, returning the per-layer parameter gradients and the gradient
with respect to the encoder input. A pure specification has no mutable cache, and recomputation is
the simplest way to recover the layer inputs.
-/

/--
Backward pass of `EncoderStack.forward`, returning one parameter-gradient record per layer,
in the order of `encoder.layers`, and the gradient with respect to the encoder input `x`.
-/
def Transformer.EncoderStack.backward {numLayers headCount embedDim hiddenDim seqLen : Nat}
  (encoder : EncoderStack numLayers headCount embedDim hiddenDim α)
  (x : Tensor α [seqLen, embedDim])
  (outputGrad : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  (Tensor (EncoderLayer headCount embedDim hiddenDim α)
      [numLayers] ×
   Tensor α [seqLen, embedDim]) :=
  let (_, inputs) := Sequence.mapAccum numLayers x fun i current =>
    let next := EncoderLayer.forward (encoder.layers.getScalar i) current h1
    (next, current)
  let (inputGrad, layerGrads) := Sequence.mapAccumRight numLayers outputGrad fun i grad =>
    let (paramsGrad, previousGrad) :=
      EncoderLayer.backward (encoder.layers.getScalar i) (inputs.getScalar i) grad h1 h2
    (previousGrad, paramsGrad)
  (layerGrads, inputGrad)

end TorchLean
