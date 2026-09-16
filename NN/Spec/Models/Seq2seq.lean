/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Loss
public import NN.Spec.Layers.Lstm
public import NN.Spec.Models.Transformer
public import NN.Spec.Layers.Rnn

/-!
# Seq2Seq (spec model)

Encoder-decoder models for sequence generation.

This file supports both bounded token indices and differentiable token distributions. A bounded
index has type `Fin vocabularySize`, so embedding lookup cannot silently substitute a value for an
invalid token. A token distribution uses a vector of length `vocabularySize` and realizes embedding
as a matrix multiplication.

PyTorch analogue:

- encoder: `nn.RNN` / `nn.LSTM` (or `nn.TransformerEncoder`) over source token embeddings
- decoder: `nn.RNN` over target embeddings (teacher forcing in training), then a final `nn.linear`
  to vocabulary logits

Scope of this baseline:

- the optional attention in `Seq2SeqDecoderSpec` is causal self-attention over decoder inputs.
  Training and inference use the same attention parameters and RNN recurrence. The baseline
  receives the encoder's final hidden state; it does not attend to the encoder's output sequence.
- for cross-attention style mechanisms, we include a small additive/Bahdanau-style attention at the
  bottom of the file (`computeAttentionWeightsSpec` / `applyAttentionSpec`).

The transformer encoder blocks used by the transformer variant come from
`NN/Spec/Models/Transformer.lean`.

References:
- Sutskever et al., "Sequence to Sequence Learning with Neural Networks" (NeurIPS 2014).
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015).
- Hochreiter and Schmidhuber, "Long Short-Term Memory" (1997).
- Cho et al.,
  "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
  (2014).
- Vaswani et al., "Attention Is All You Need" (2017) for the transformer encoder variant.

PyTorch docs (for API intuition, not semantics):
- `torch.nn.Embedding`: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- `torch.nn.RNN`: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
- `torch.nn.LSTM`: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- `torch.nn.Linear`: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- `torch.nn.MultiheadAttention`:
  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- `torch.nn.TransformerEncoderLayer`:
  https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html

## Implementation status

No API builder implements an encoder-decoder model. `nn.rnn`, `nn.lstm`, and `nn.gru` build single
recurrent layers and `NN/API/Models/Recurrent.lean` builds sequence models with a per-step linear
head, neither of which is this architecture. `NN/Spec/Module/Seq2seq.lean` wraps this file as a
`Spec.Module`. No theorem is proved about it.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-!
## Training + gradients (one-hot inputs)

Most of this file focuses on *architecture variants* and *forward passes* (teacher-forcing,
inference-time decoding, optional self-attention in the decoder, etc.).

To make Seq2Seq usable as a first-class baseline, we also provide an explicit training objective
and reverse-mode gradients for the differentiable path:

- inputs are **one-hot / token distributions** (so embedding lookup is a matrix multiply),
- teacher forcing is used in the decoder,
- the loss is per-timestep cross-entropy between `softmax(logits)` and the target distribution,
- gradients flow through embeddings, encoder RNN, decoder RNN, output projection, and (optionally)
  the decoder self-attention block.

Bounded token indices are intentionally treated as non-differentiable.
-/

/-! ### Small gradient records -/

/--
Gradients for a token embedding table `E : (vocabularySize × embedDim)`.

PyTorch analogue: `nn.Embedding.weight.grad`.
-/
structure Seq2SeqEmbeddingGrads (α : Type) [TorchLean.Storage α]
    (vocabularySize embedDim : Nat) where
  /-- Gradient of the embedding matrix. -/
  embedding : Tensor α [vocabularySize, embedDim]

/--
End-to-end gradient record for the differentiable Seq2Seq baseline.

This bundles gradients for:
- source/target embeddings,
- encoder RNN,
- decoder RNN,
- decoder output projection,
- optional decoder self-attention (if enabled in the decoder spec).
-/
structure Seq2SeqGrads (α : Type) [TorchLean.Storage α]
    (srcVocabSize tgtVocabSize embedDim hiddenDim : Nat) where
  /-- Gradients for the source embedding table. -/
  sourceEmbedding : Seq2SeqEmbeddingGrads α srcVocabSize embedDim
  /-- Gradients for the target embedding table. -/
  targetEmbedding : Seq2SeqEmbeddingGrads α tgtVocabSize embedDim
  /-- Gradients for the encoder RNN parameters. -/
  encoder : RNNParameterGradients α embedDim hiddenDim
  /-- Gradients for the decoder RNN parameters. -/
  decoderRnn : RNNParameterGradients α embedDim hiddenDim
  /-- Gradients for the decoder output projection (`hiddenDim -> tgtVocabSize`). -/
  outputProjection : LinearParameterGradients α hiddenDim tgtVocabSize
  /-- Gradients for optional decoder self-attention parameters. -/
  decoderAttention :
    Option (Σ numHeads : Nat,
      MultiHeadAttentionParameterGradients numHeads embedDim (embedDim / numHeads) α) :=
    none

/--
Seq2Seq token embedding specification.

Parameters:
- `embedding`: a lookup table `E : (vocabularySize × embedDim)`.

PyTorch analogue: `nn.Embedding(vocabularySize, embedDim)`.
-/
structure Seq2SeqEmbeddingSpec (α : Type) [TorchLean.Storage α]
    (vocabularySize embedDim : Nat) where
  /-- Embedding table `E : (vocabularySize × embedDim)`. -/
  embedding : Tensor α [vocabularySize, embedDim]

/--
Embedding forward pass for discrete token ids.

Inputs:
- `tokenIds : (seqLen)`, a tensor of indices bounded by the vocabulary size.

Output:
- `y : (seqLen × embedDim)`, where each timestep selects a row of the embedding table.

PyTorch analogue: `nn.Embedding` on an integer tensor. The `Fin vocabularySize` element type
expresses the lookup precondition directly, rather than assigning an arbitrary meaning to an invalid
token.
-/
def Seq2SeqEmbeddingSpec.forward {vocabularySize embedDim seqLen : Nat}
  (embedding : Seq2SeqEmbeddingSpec α vocabularySize embedDim)
  (tokenIds : Tensor (Fin vocabularySize) [seqLen]):
  Tensor α [seqLen, embedDim] :=
  Tensor.dim (fun i =>
    get embedding.embedding (Tensor.getScalar tokenIds i))

/-- Seq2Seq embedding forward pass for one-hot / token distributions.

This is the usual "embedding lookup as a matrix multiply":

- if `E : (vocabularySize × embedDim)` is the embedding table,
- and `x_t : (vocabularySize)` is a one-hot / probability vector for time step `t`,
- then the embedded vector is `y_t = x_tᵀ · E : (embedDim)`.

PyTorch analogy: `y = x @ E` where `x` is one-hot / a distribution; this matches `nn.Embedding`
when the input is exactly one-hot.
-/
def Seq2SeqEmbeddingSpec.forwardOneHot {vocabularySize embedDim seqLen : Nat}
  (embedding : Seq2SeqEmbeddingSpec α vocabularySize embedDim)
  (tokenOneHot : Tensor α [seqLen, vocabularySize]) :
  Tensor α [seqLen, embedDim] :=
  Tensor.dim (fun i => vecMatMulSpec (get tokenOneHot i) embedding.embedding)

/--
Backward pass for `Seq2SeqEmbeddingSpec.forwardOneHot`.

This is just a time-distributed linear layer:

`y_t = token_tᵀ · E`

So:
- `dE = Σ_t token_t ⊗ dY_t`
- `dToken_t = E · dY_t` (not usually needed, but included for completeness)
-/
def Seq2SeqEmbeddingSpec.backwardOneHot {vocabularySize embedDim seqLen : Nat}
  (embedding : Seq2SeqEmbeddingSpec α vocabularySize embedDim)
  (tokenOneHot : Tensor α [seqLen, vocabularySize])
  (gradOutput : Tensor α [seqLen, embedDim]) :
  (Seq2SeqEmbeddingGrads α vocabularySize embedDim × Tensor α [seqLen, vocabularySize]) :=
  let step (i : Fin seqLen) (acc : Seq2SeqEmbeddingGrads α vocabularySize embedDim) :=
    let token_t := get tokenOneHot i
    let dY_t := get gradOutput i
    let dE_t := outerProductSpec token_t dY_t
    let dToken_t := matVecMulSpec embedding.embedding dY_t
    ({ embedding := addSpec acc.embedding dE_t }, dToken_t)
  let init : Seq2SeqEmbeddingGrads α vocabularySize embedDim :=
    { embedding := Tensor.full (.dim vocabularySize (.dim embedDim .scalar)) 0 }
  let (dE, dX) := Sequence.mapAccum seqLen init step
  (dE, Tensor.dim dX.getScalar)

/--
RNN-based encoder specification for Seq2Seq.

This models an `nn.RNN`-style encoder over embedded tokens:
- input is a sequence of embeddings `(seqLen × embedDim)`,
- output is the full hidden-state sequence plus the final hidden state.

PyTorch analogue: `nn.RNN(..., batch_first=True)` (ignoring the batch axis), returning `(output,
  h_n)`.
-/
structure Seq2SeqRNNEncoderSpec (α : Type) [TorchLean.Storage α]
    (embedDim hiddenDim : Nat) where
  /-- RNN cell parameters. -/
  rnn : RNNSpec α embedDim hiddenDim

/--
Forward pass for `Seq2SeqRNNEncoderSpec`.

Inputs:
- `x : (seqLen × embedDim)`, embedded source tokens,
- `h0`, optional initial hidden state (`hiddenDim`).

Returns:
- `(outputs, final_h)` where `outputs : (seqLen × hiddenDim)` is the per-timestep hidden sequence.
-/
def Seq2SeqRNNEncoderSpec.forward {α : Type} [TorchLean.Storage α] [Context α]
  {embedDim hiddenDim seqLen : Nat}
  (encoder : Seq2SeqRNNEncoderSpec α embedDim hiddenDim)
  (x : Tensor α [seqLen, embedDim])
  (h0 : Option (Tensor α [hiddenDim])):
  (Tensor α [seqLen, hiddenDim] × Tensor α [hiddenDim]) :=
  let initialHidden := match h0 with
    | some h => h
    | none => Tensor.full (.dim hiddenDim .scalar) 0
  let (finalHidden, outputs) := Sequence.mapAccum seqLen initialHidden fun i previous =>
    let hidden := rnnCellSpec encoder.rnn (get x i) previous
    (hidden, hidden)
  (Tensor.dim outputs.getScalar, finalHidden)


/--
LSTM-based encoder specification for Seq2Seq.

This models an `nn.LSTM`-style encoder over embedded tokens, returning the full hidden sequence,
final hidden state, and final cell state.

PyTorch analogue: `nn.LSTM(..., batch_first=True)` (ignoring the batch axis), returning
`(output, (h_n, c_n))`.
-/
structure Seq2SeqLSTMEncoderSpec (α : Type) [TorchLean.Storage α]
    (embedDim hiddenDim : Nat) where
  /-- LSTM cell parameters. -/
  lstm : LSTMSpec α embedDim hiddenDim

/--
Forward pass for `Seq2SeqLSTMEncoderSpec`.

Inputs:
- `x : (seqLen × embedDim)`, embedded source tokens,
- `h0`, optional initial hidden state (`hiddenDim`),
- `c0`, optional initial cell state (`hiddenDim`).

Returns:
- `(outputs, final_h, final_c)` where `outputs : (seqLen × hiddenDim)` is the per-timestep hidden
  sequence.
-/
def Seq2SeqLSTMEncoderSpec.forward {embedDim hiddenDim seqLen : Nat}
  (encoder : Seq2SeqLSTMEncoderSpec α embedDim hiddenDim)
  (x : Tensor α [seqLen, embedDim])
  (h0 : Option (Tensor α [hiddenDim]))
  (c0 : Option (Tensor α [hiddenDim])):
  (Tensor α [seqLen, hiddenDim] ×
   Tensor α [hiddenDim] ×
   Tensor α [hiddenDim]) :=
  let initialHidden := match h0 with
  | some h => h
  | none => Tensor.full (.dim hiddenDim .scalar) 0
  let initialCell := match c0 with
  | some c => c
  | none => Tensor.full (.dim hiddenDim .scalar) 0
  let (outputs, finalState) :=
    lstmSequenceSpec encoder.lstm x { hidden := initialHidden, cell := initialCell }
  (outputs, finalState.hidden, finalState.cell)

/--
Transformer-based encoder specification for Seq2Seq.

This wrapper applies exactly `numLayers` `TransformerEncoderLayer`s from
`NN.Spec.Models.Transformer` as a left fold.

PyTorch analogue: `nn.TransformerEncoder(nn.TransformerEncoderLayer(...), num_layers=...)`
(ignoring dropout and most configuration knobs).
-/
structure Seq2SeqTransformerEncoderSpec (α : Type) [TorchLean.Storage α] [Context α]
  (embedDim numHeads numLayers : Nat) where
  /-- Encoder layer stack. Its length is part of the type. -/
  layers : Tensor (TransformerEncoderLayer numHeads embedDim (embedDim * 4) α)
    [numLayers]

/--
Forward pass for `Seq2SeqTransformerEncoderSpec`.

Input/output shape: `(seqLen × embedDim)`.

This uses post-norm transformer layers from `NN.Spec.Models.Transformer` and does not model
dropout; it is meant as a clean semantic reference rather than a full training-ready implementation.
-/
def Seq2SeqTransformerEncoderSpec.forward {embedDim numHeads numLayers seqLen : Nat}
  (encoder : Seq2SeqTransformerEncoderSpec α embedDim numHeads numLayers)
  (x : Tensor α [seqLen, embedDim])
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  Tensor α [seqLen, embedDim] :=
  (Tensor.to encoder.layers
    (Array (TransformerEncoderLayer numHeads embedDim (embedDim * 4) α))).foldl
      (fun acc layer => TransformerEncoderLayer.forward layer acc h1 h2) x

/--
RNN decoder specification for Seq2Seq.

This decoder consumes a sequence of target-side embeddings and produces vocabulary logits:
- an `RNNSpec` cell updates the hidden state per timestep,
- a time-distributed `LinearSpec` maps hidden states to logits,
- optionally, causal self-attention transforms the decoder input embeddings before the RNN.
  Position `i` attends to inputs `0, ..., i`, in both teacher forcing and greedy decoding.

PyTorch analogue: a hand-rolled decoder using `nn.RNN` and `nn.linear`, optionally preceded by
`nn.MultiheadAttention` over the target embeddings (note: this is not encoder-decoder
  cross-attention).
-/
structure Seq2SeqDecoderSpec (α : Type) [TorchLean.Storage α]
    (embedDim hiddenDim vocabularySize : Nat) where
  /-- Decoder RNN cell parameters. -/
  rnn : RNNSpec α embedDim hiddenDim
  /-- Optional causal self-attention over decoder inputs, shared by training and inference. -/
  attention :
    Option (Σ numHeads : Nat, MultiHeadAttention α numHeads embedDim (embedDim / numHeads)) := none
  /-- Output projection (`hiddenDim -> vocabularySize`) producing per-timestep logits. -/
  outputProjection : LinearSpec α hiddenDim vocabularySize

/--
Prepare the RNN inputs from a nonempty sequence of decoder embeddings.

With attention enabled, row `i` uses only rows `0, ..., i`. The hard mask gives later positions
zero weight, including in the attention backward pass. Without attention, the embeddings pass
through unchanged. Teacher forcing and single-step decoding both use this function, so attention
has the same parameters, projection order, and mask convention in both paths.
-/
def Seq2SeqDecoderSpec.attendInputs {embedDim hiddenDim vocabularySize seqLen : Nat}
    (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabularySize)
    (embeddings : Tensor α [seqLen, embedDim]) (hLen : seqLen ≠ 0) :
    Tensor α [seqLen, embedDim] :=
  match decoder.attention with
  | some ⟨_numHeads, attn⟩ =>
      MultiHeadAttention.forward seqLen hLen attn embeddings (some (causalMask seqLen))
  | none => embeddings

/--
Teacher-forcing logits for a sequence of decoder inputs.

`targetEmbeddings` has shape `(tgtSeqLen × embedDim)` and contains the tokens fed to the decoder.
For next-token prediction, the caller supplies the start token followed by the preceding target
tokens; the labels are one position ahead of these inputs. Causal self-attention prepares each RNN
input, then the recurrence starts at `h0` and the output projection produces vocabulary logits.
Changing a later decoder input cannot create an attention edge into an earlier position.
-/
def Seq2SeqDecoderSpec.forwardTeacherForcing {embedDim hiddenDim vocabularySize tgtSeqLen : Nat}
  (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabularySize)
  (targetEmbeddings : Tensor α [tgtSeqLen, embedDim])
  (h0 : Tensor α [hiddenDim])
  (h_len_nonzero : tgtSeqLen ≠ 0) :
  Tensor α [tgtSeqLen, vocabularySize] :=

  let attendedEmbeddings := decoder.attendInputs targetEmbeddings h_len_nonzero
  let hiddens := rnnSequenceSpec decoder.rnn attendedEmbeddings h0
  Tensor.dim (fun i => linearSpec decoder.outputProjection (get hiddens i))

/-!
### Decoder backward (teacher forcing)

The decoder is: (optional self-attention) → RNN → time-distributed linear projection.

We compute gradients by:
1) recomputing the attended embeddings (if any),
2) recomputing the decoder hidden sequence,
3) backpropagating through the output projection per timestep,
4) backpropagating through the RNN sequence,
5) optionally backpropagating through self-attention.
-/

/--
Gradients for the decoder parameters, target embeddings, and initial hidden state.

The attention field is present exactly when the decoder has an attention block. The final two
fields keep the input sequence gradient separate from the gradient passed back to the encoder
through its final hidden state.
-/
structure Seq2SeqDecoderGradients (α : Type) [TorchLean.Storage α]
    (embedDim hiddenDim vocabularySize tgtSeqLen : Nat) where
  /-- Gradients for the decoder RNN parameters. -/
  rnn : RNNParameterGradients α embedDim hiddenDim
  /-- Gradients for the time-distributed output projection. -/
  outputProjection : LinearParameterGradients α hiddenDim vocabularySize
  /-- Gradients for the optional decoder self-attention parameters. -/
  attention :
    Option (Σ numHeads : Nat,
      MultiHeadAttentionParameterGradients numHeads embedDim (embedDim / numHeads) α)
  /-- Gradient with respect to the target embedding sequence. -/
  targetEmbeddings : Tensor α [tgtSeqLen, embedDim]
  /-- Gradient with respect to the initial hidden state `h0`. -/
  initialHidden : Tensor α [hiddenDim]

/--
Backward pass for `Seq2SeqDecoderSpec.forwardTeacherForcing`.

Returns a `Seq2SeqDecoderGradients` record.

The attended embeddings and hidden sequence are recomputed with the forward pass's causal mask.
The same mask is passed to the attention VJP. An upstream gradient supported on an initial target
prefix therefore cannot flow through an attention edge to a later decoder input.
-/
def Seq2SeqDecoderSpec.backwardTeacherForcing
  {embedDim hiddenDim vocabularySize tgtSeqLen : Nat}
  (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabularySize)
  (targetEmbeddings : Tensor α [tgtSeqLen, embedDim])
  (h0 : Tensor α [hiddenDim])
  (h_len_nonzero : tgtSeqLen ≠ 0)
  (gradLogits : Tensor α [tgtSeqLen, vocabularySize]) :
  Seq2SeqDecoderGradients α embedDim hiddenDim vocabularySize tgtSeqLen :=

  let attendedEmbeddings := decoder.attendInputs targetEmbeddings h_len_nonzero
  let hiddens := rnnSequenceSpec decoder.rnn attendedEmbeddings h0
  let projectionGrads :=
    timeDistributedLinearBackward decoder.outputProjection hiddens gradLogits
  let projGrads := projectionGrads.parameters
  let dH := projectionGrads.inputGradient

  let rnnBackward := rnnSequenceBackwardSpec decoder.rnn attendedEmbeddings h0 hiddens dH
  let dAttended0 := rnnBackward.inputs

  match decoder.attention with
  | none =>
      { rnn := rnnBackward.parameters
        outputProjection := projGrads
        attention := none
        targetEmbeddings := dAttended0
        initialHidden := rnnBackward.initialHidden }
  | some ⟨numHeads, attn⟩ =>
      let attentionGrads :=
        multiHeadAttentionBackward (α := α) (n := tgtSeqLen) (dModel := embedDim)
          h_len_nonzero attn targetEmbeddings (some (causalMask tgtSeqLen)) dAttended0
      { rnn := rnnBackward.parameters
        outputProjection := projGrads
        attention := some ⟨numHeads, attentionGrads.parameters⟩
        targetEmbeddings := attentionGrads.input
        initialHidden := rnnBackward.initialHidden }

/--
Advance the decoder once using a nonempty prefix of input embeddings.

`previousHidden` is the RNN state after processing all but the last prefix token. Attention sees the
whole prefix, and its last row supplies the current RNN input. The earlier RNN steps are not
replayed. The result contains the new hidden state and this step's vocabulary logits. Keeping the
prefix explicit also lets a caller compare a teacher-forced prefix with a single inference step.
-/
def Seq2SeqDecoderSpec.forwardStep {embedDim hiddenDim vocabularySize prefixLen : Nat}
    (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabularySize)
    (inputPrefix : Tensor α [prefixLen + 1, embedDim])
    (previousHidden : Tensor α [hiddenDim]) :
    Tensor α [hiddenDim] × Tensor α [vocabularySize] :=
  let attended := decoder.attendInputs inputPrefix (Nat.succ_ne_zero prefixLen)
  let input := get attended ⟨prefixLen, Nat.lt_succ_self prefixLen⟩
  let hidden := rnnCellSpec decoder.rnn input previousHidden
  (hidden, linearSpec decoder.outputProjection hidden)

/--
Greedy autoregressive decoding from `startToken` and the initial hidden state `h0`.

Each step appends the current token embedding to the input prefix, applies `forwardStep`, and
feeds the argmax token back as the next input. Optional self-attention therefore sees the same
prefix as teacher forcing with those input tokens. The RNN state advances once per emitted token.
Attention projections are recomputed from the stored prefix; this specification has no key/value
cache.

The result contains logits of shape `(maxLen × vocabularySize)` and `maxLen` predicted token ids.
When `maxLen = 0`, both outputs are empty and no decoder step runs.
-/
def Seq2SeqDecoderSpec.forwardInference {embedDim hiddenDim vocabularySize : Nat}
  (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabularySize)
  (h0 : Tensor α [hiddenDim])
  (targetEmbedding : Tensor α [vocabularySize, embedDim])
  (startToken : Fin vocabularySize) (maxLen : Nat) :
  (Tensor α [maxLen, vocabularySize] ×
    Tensor (Fin vocabularySize) [maxLen]) :=
  let initialInput := get targetEmbedding startToken
  let hVocab : 0 < vocabularySize := lt_of_le_of_lt (Nat.zero_le startToken.val) startToken.isLt
  let initialHistory : Array (Tensor α [embedDim]) := #[]
  let (_, results) := Sequence.mapAccum maxLen (h0, initialInput, initialHistory) fun _ state =>
    let (hidden, input, history) := state
    -- A decoder without attention only needs the current embedding. Discarding its history keeps
    -- the RNN-only loop linear in `maxLen`, while the attention path retains the full input prefix.
    let history := if decoder.attention.isSome then history else #[]
    let nextHistory := history.push input
    let inputPrefix : Tensor α [history.size + 1, embedDim] :=
      Tensor.dim (fun i =>
        nextHistory[i.val]'(by simpa only [nextHistory, Array.size_push] using i.isLt))
    let (nextHidden, logits) := decoder.forwardStep inputPrefix hidden
    let token := Fin.cast (by simp [Shape.size])
      (argmax (s := [vocabularySize]) (by simpa [Shape.size] using hVocab) logits)
    let nextInput := get targetEmbedding token
    ((nextHidden, nextInput, nextHistory), (logits, token))
  (Tensor.dim (fun i => (results.getScalar i).1),
    Tensor.dim (fun i => Tensor.scalar (results.getScalar i).2))

/--
Complete Seq2Seq model specification (baseline).

This bundles:
- source and target embedding tables,
- an RNN encoder,
- an RNN decoder with output projection (and optional decoder self-attention).

PyTorch analogue: a small encoder-decoder model built from `nn.Embedding`, `nn.RNN`, and
  `nn.linear`.
-/
structure Seq2SeqSpec (α : Type) [TorchLean.Storage α]
    (srcVocabSize tgtVocabSize embedDim hiddenDim : Nat)
  where
  /-- Source embedding table. -/
  sourceEmbedding : Seq2SeqEmbeddingSpec α srcVocabSize embedDim
  /-- Target embedding table. -/
  targetEmbedding : Seq2SeqEmbeddingSpec α tgtVocabSize embedDim
  /-- Encoder RNN parameters. -/
  encoder : Seq2SeqRNNEncoderSpec α embedDim hiddenDim
  /-- Decoder parameters (RNN + output projection + optional self-attention). -/
  decoder : Seq2SeqDecoderSpec α embedDim hiddenDim tgtVocabSize

/--
Teacher-forcing logits from discrete source and decoder input tokens.

`sourceTokens` supplies the encoder sequence. `targetTokens` supplies the decoder inputs: for
next-token prediction, these are the start token followed by the preceding target tokens. The
caller pairs the returned `(tgtSeqLen × tgtVocabSize)` logits with labels one position ahead.
The function embeds these inputs as given; it does not insert a start token or shift the sequence.

The encoder's final hidden state initializes the decoder. Optional decoder self-attention is causal,
and embedding lookup treats the bounded token ids as discrete inputs without token-id gradients.
-/
def Seq2SeqSpec.forwardTraining {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen :
  Nat}
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (sourceTokens : Tensor (Fin srcVocabSize) [srcSeqLen])
  (targetTokens : Tensor (Fin tgtVocabSize) [tgtSeqLen])
  (hTarget : tgtSeqLen ≠ 0) :
  Tensor α [tgtSeqLen, tgtVocabSize] :=
  let sourceEmbeddings := Seq2SeqEmbeddingSpec.forward model.sourceEmbedding sourceTokens
  let (_encoderOutputs, encoderHidden) :=
    Seq2SeqRNNEncoderSpec.forward model.encoder sourceEmbeddings
    none
  let targetEmbeddings := Seq2SeqEmbeddingSpec.forward model.targetEmbedding targetTokens
  Seq2SeqDecoderSpec.forwardTeacherForcing model.decoder targetEmbeddings encoderHidden hTarget

/--
Encode the source once and generate `maxTgtLen` target tokens greedily.

The encoder's final hidden state initializes the decoder, and `startToken` supplies its first input.
Each predicted token becomes the next decoder input. Optional causal self-attention uses that
growing input prefix, with the same attention parameters as teacher forcing. The returned pair
contains `(maxTgtLen × tgtVocabSize)` logits and `maxTgtLen` bounded token ids.
-/
def Seq2SeqSpec.forwardInference {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen : Nat}
  (maxTgtLen : Nat)
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (sourceTokens : Tensor (Fin srcVocabSize) [srcSeqLen])
  (startToken : Fin tgtVocabSize) :
  (Tensor α [maxTgtLen, tgtVocabSize] ×
    Tensor (Fin tgtVocabSize) [maxTgtLen]) :=
  let sourceEmbeddings := Seq2SeqEmbeddingSpec.forward model.sourceEmbedding sourceTokens
  let (_encoderOutputs, encoderHidden) :=
    Seq2SeqRNNEncoderSpec.forward model.encoder sourceEmbeddings none
  Seq2SeqDecoderSpec.forwardInference model.decoder encoderHidden model.targetEmbedding.embedding
    startToken maxTgtLen

/-!
### Differentiable training + backward (one-hot inputs)

This is the “full” training interface for the Seq2Seq baseline.
-/

/--
Differentiable forward pass for training (teacher forcing) using one-hot/token-distribution inputs.

This is the same computation as `Seq2SeqSpec.forwardTraining`, except that embedding lookup is
expressed as a matrix multiplication (`forwardOneHot`), so gradients can flow into the embedding
tables and back into upstream token distributions. `tgtOneHot` contains the decoder inputs, in the
same start-token/preceding-token order as the discrete path. For next-token prediction, the labels
must be supplied separately to the loss, one position ahead of these inputs.
-/
def Seq2SeqSpec.forwardTrainingOneHot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (srcOneHot : Tensor α [srcSeqLen, srcVocabSize])
  (tgtOneHot : Tensor α [tgtSeqLen, tgtVocabSize])
  (hTgt : tgtSeqLen ≠ 0) :
  Tensor α [tgtSeqLen, tgtVocabSize] :=
  let srcEmbeds := Seq2SeqEmbeddingSpec.forwardOneHot model.sourceEmbedding srcOneHot
  let (_encOut, encHidden) := Seq2SeqRNNEncoderSpec.forward model.encoder srcEmbeds none
  let tgtEmbeds := Seq2SeqEmbeddingSpec.forwardOneHot model.targetEmbedding tgtOneHot
  Seq2SeqDecoderSpec.forwardTeacherForcing model.decoder tgtEmbeds encHidden hTgt

/--
Per-timestep cross-entropy loss for the differentiable Seq2Seq baseline.

Computes:
1. logits via `Seq2SeqSpec.forwardTrainingOneHot`,
2. probabilities via `softmax`,
3. cross-entropy against the target token distribution at each timestep.

PyTorch analogue: `nn.CrossEntropyLoss` applied per timestep (with probabilities represented as
  one-hot).
-/
def Seq2SeqSpec.crossEntropyLossOneHot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  [Shape.HasNonemptyAxis 1 (.dim tgtSeqLen (.dim tgtVocabSize .scalar))]
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (srcOneHot : Tensor α [srcSeqLen, srcVocabSize])
  (tgtOneHot : Tensor α [tgtSeqLen, tgtVocabSize])
  (hTgt : tgtSeqLen ≠ 0) : α :=
  let logits := Seq2SeqSpec.forwardTrainingOneHot (α := α) model srcOneHot tgtOneHot hTgt
  let probs := Activation.softmaxSpec 1 logits
  crossEntropySpec 1 probs tgtOneHot

/--
Compute `(loss, grads)` for the Seq2Seq baseline under per-timestep cross-entropy.

This returns gradients for:
- both embedding tables,
- the encoder RNN,
- the decoder RNN,
- the decoder output projection,
- and decoder self-attention (if present).
-/
def Seq2SeqSpec.crossEntropyGradOneHot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  [Shape.HasNonemptyAxis 1 (.dim tgtSeqLen (.dim tgtVocabSize .scalar))]
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (srcOneHot : Tensor α [srcSeqLen, srcVocabSize])
  (tgtOneHot : Tensor α [tgtSeqLen, tgtVocabSize])
  (hTgt : tgtSeqLen ≠ 0) :
  (α × Seq2SeqGrads α srcVocabSize tgtVocabSize embedDim hiddenDim) :=

  let srcEmbeds := Seq2SeqEmbeddingSpec.forwardOneHot model.sourceEmbedding srcOneHot
  let (encHiddens, encHidden) := Seq2SeqRNNEncoderSpec.forward model.encoder srcEmbeds none
  let tgtEmbeds := Seq2SeqEmbeddingSpec.forwardOneHot model.targetEmbedding tgtOneHot

  let logits := Seq2SeqDecoderSpec.forwardTeacherForcing model.decoder tgtEmbeds encHidden hTgt
  let probs := Activation.softmaxSpec 1 logits
  let loss := crossEntropySpec 1 probs tgtOneHot

  let dProbs := crossEntropyDerivSpec 1 probs tgtOneHot
  let dLogits := Activation.softmaxBackwardSpec 1 logits dProbs

  let decoderGrads :=
    Seq2SeqDecoderSpec.backwardTeacherForcing (α := α)
      (embedDim := embedDim) (hiddenDim := hiddenDim) (vocabularySize := tgtVocabSize) (tgtSeqLen :=
        tgtSeqLen)
      model.decoder tgtEmbeds encHidden hTgt dLogits
  let dTgtEmbeds := decoderGrads.targetEmbeddings
  let dEncHidden := decoderGrads.initialHidden

  let (dTgtEmbTable, _dTgtOneHot) :=
    Seq2SeqEmbeddingSpec.backwardOneHot (α := α)
      (vocabularySize := tgtVocabSize) (embedDim := embedDim) (seqLen := tgtSeqLen)
      model.targetEmbedding tgtOneHot dTgtEmbeds

  -- Encoder only feeds the decoder through the final hidden state.
  let dEncHiddens :=
    if _h0 : srcSeqLen = 0 then
      Tensor.full (.dim srcSeqLen (.dim hiddenDim .scalar)) 0
    else
      Tensor.dim (fun i =>
        if _ : i.val = srcSeqLen - 1 then dEncHidden else Tensor.full (.dim hiddenDim .scalar) 0)

  let encoderBackward :=
    rnnSequenceBackwardSpec model.encoder.rnn srcEmbeds (Tensor.full ([hiddenDim]) 0)
      encHiddens dEncHiddens
  let dSrcEmbeds := encoderBackward.inputs

  let (dSrcEmbTable, _dSrcOneHot) :=
    Seq2SeqEmbeddingSpec.backwardOneHot (α := α)
      (vocabularySize := srcVocabSize) (embedDim := embedDim) (seqLen := srcSeqLen)
      model.sourceEmbedding srcOneHot dSrcEmbeds

  let grads : Seq2SeqGrads α srcVocabSize tgtVocabSize embedDim hiddenDim :=
    { sourceEmbedding := dSrcEmbTable
      targetEmbedding := dTgtEmbTable
      encoder := encoderBackward.parameters
      decoderRnn := decoderGrads.rnn
      outputProjection := decoderGrads.outputProjection
      decoderAttention := decoderGrads.attention }

  (loss, grads)

/--
Attention-augmented Seq2Seq specification (simple encoder-output attention).

This record extends the baseline with an additional projection matrix used by the helper
attention functions below (`computeAttentionWeightsSpec` / `applyAttentionSpec`).

Note: this file includes these attention helpers as a building block; the main baseline forward
passes above do not integrate encoder-decoder cross-attention by default.
-/
structure AttentionSeq2SeqSpec (α : Type) [TorchLean.Storage α]
    (srcVocabSize tgtVocabSize embedDim hiddenDim
  : Nat) where
  /-- Source embedding table. -/
  sourceEmbedding : Seq2SeqEmbeddingSpec α srcVocabSize embedDim
  /-- Target embedding table. -/
  targetEmbedding : Seq2SeqEmbeddingSpec α tgtVocabSize embedDim
  /-- Encoder RNN parameters. -/
  encoder : Seq2SeqRNNEncoderSpec α embedDim hiddenDim
  /-- Decoder parameters (RNN + output projection + optional self-attention). -/
  decoder : Seq2SeqDecoderSpec α embedDim hiddenDim tgtVocabSize
  /-- Attention projection matrix used to score encoder outputs against the decoder hidden state. -/
  attentionWeights : Tensor α [hiddenDim, hiddenDim]

/--
Compute attention weights over encoder outputs for a single decoder hidden state.

This is a simple dot-product style attention:
1. project the decoder hidden state (`attention_weights · decoder_hidden`),
2. score each encoder hidden vector by an elementwise product + sum,
3. normalize scores with `softmax` over the sequence axis.

It is inspired by classic encoder-decoder attention mechanisms (Bahdanau-style), and this spec keeps
the scoring rule compact.
-/
def computeAttentionWeightsSpec {α : Type} [TorchLean.Storage α] [Context α]
  {hiddenDim seqLen : Nat}
  (attentionWeights : Tensor α [hiddenDim, hiddenDim])
  (decoderHidden : Tensor α [hiddenDim])
  (encoderOutputs : Tensor α [seqLen, hiddenDim])
  (h1 : hiddenDim ≠ 0) (_h2 : seqLen ≠ 0) :
  Tensor α [seqLen] :=
  -- Compute attention scores
  let projectedHidden := matVecMulSpec attentionWeights decoderHidden
  let scores := Tensor.dim (fun i =>
    let encoderVec := get encoderOutputs i
    let mulVec := mulSpec projectedHidden encoderVec
    reduceSum 0 mulVec (Shape.hasNonemptyAxisZeroOfNe h1).proof
  )
  -- Apply softmax to get attention weights
  Activation.softmaxSpec 0 scores

/--
Apply attention weights to encoder outputs (weighted sum / context vector).

Given attention weights `a : (seqLen)` and encoder outputs `H : (seqLen × hiddenDim)`, returns the
context vector `c = Σ_i a_i · H_i : (hiddenDim)`.
-/
def applyAttentionSpec {hiddenDim seqLen : Nat}
  (attentionWeights : Tensor α [seqLen])
  (encoderOutputs : Tensor α [seqLen, hiddenDim])
  (h1 : seqLen ≠ 0) (_h2 : hiddenDim ≠ 0) :
  Tensor α [hiddenDim] :=
  -- Weighted sum of encoder outputs
  let weightedOutputs := Tensor.dim (fun i =>
    scaleSpec (get encoderOutputs i) (Tensor.getScalar attentionWeights i)
  )
  -- Sum across sequence dimension
  reduceSum 0 weightedOutputs (Shape.hasNonemptyAxisZeroOfNe h1).proof

end Spec
