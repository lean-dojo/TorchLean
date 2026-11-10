/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.Tensor.Layers.Attention
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.RNN
public import TorchLean.Tensor.Contract

/-!
# Seq2Seq (spec model)

Encoder-decoder models for sequence generation. Tokens enter in two forms. A bounded index has
type `Fin vocabularySize`, so embedding lookup cannot silently substitute a value for an invalid
token. A token distribution is a vector of length `vocabularySize`, and its embedding is a matrix
multiplication with the table, which makes it differentiable.

PyTorch analogue:

- encoder: `nn.RNN` over source token embeddings
- decoder: `nn.RNN` over target embeddings (teacher forcing in training), then a final `nn.Linear`
  to vocabulary logits

The optional attention in `Seq2Seq.AttentionDecoder` is causal self-attention over the decoder
inputs, and training and inference share its parameters and the RNN recurrence. The decoder receives
the encoder's final hidden state and does not attend to the encoder's output sequence.

The model follows Sutskever et al. (2014), written by hand. In their eq. (1) the decoder starts
from `v`, the encoder's last hidden state, and ours does the same in `forwardTraining` and
`forwardInference`. Like them we keep two separate cells, but each side is one tanh `RNN.cell`
where they stack four LSTM layers, and we read the source in order where they reverse it
(Section 3.3). Greedy argmax is their beam search with a beam of 1 (Section 3.2). It always runs
`maxLen` steps, with no stop at an end-of-sentence token. The decoder of Cho et al. (2014) also
reads the summary `c` at every step (Section 2.2), which ours leaves out.

References:
- Ilya Sutskever, Oriol Vinyals, and Quoc V. Le (2014), "Sequence to Sequence Learning with Neural
  Networks", NIPS 2014, arXiv:1409.3215, eq. (1), Sections 3.2 and 3.3.
- Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger
  Schwenk, and Yoshua Bengio (2014), "Learning Phrase Representations using RNN Encoder-Decoder
  for Statistical Machine Translation", EMNLP 2014, pages 1724-1734, doi:10.3115/v1/D14-1179,
  Section 2.2.

PyTorch 2.8 docs (for API intuition, not semantics):
- `torch.nn.Embedding`: https://docs.pytorch.org/docs/2.8/generated/torch.nn.Embedding.html
- `torch.nn.RNN`: https://docs.pytorch.org/docs/2.8/generated/torch.nn.RNN.html
- `torch.nn.Linear`: https://docs.pytorch.org/docs/2.8/generated/torch.nn.Linear.html
- `torch.nn.MultiheadAttention`:
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html

## Implementation status

No API builder implements an encoder-decoder model. `nn.rnn`, `nn.lstm`, and `nn.gru` build single
recurrent layers and `TorchLean/NN/Models/Recurrent.lean` builds sequence models with a per-step
linear head, neither of which is this architecture. No theorem is proved about it, and no
TorchLeanTest test runs it.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/--
RNN encoder of the Seq2Seq model. It reads a sequence of embeddings `(seqLen × embedDim)` and
returns the full hidden-state sequence together with the final hidden state.

PyTorch analogue: `nn.RNN(..., batch_first=True)` without the batch axis, returning
`(output, h_n)`.
-/
structure Seq2Seq.RNNEncoder (α : Type) [Storage α]
    (embedDim hiddenDim : Nat) where
  /-- RNN cell parameters. -/
  rnn : Linear α (embedDim + hiddenDim) hiddenDim

/--
Run the encoder over the embedded source tokens `x : (seqLen × embedDim)`, starting from `h0`, or
from zero when `h0` is `none`. The result is `RNN.sequence` of the encoder cell: the hidden
state at every timestep `(seqLen × hiddenDim)` and the final hidden state.
-/
def Seq2Seq.RNNEncoder.forward {α : Type} [Storage α] [Context α]
  {embedDim hiddenDim seqLen : Nat}
  (encoder : RNNEncoder α embedDim hiddenDim)
  (x : Tensor α [seqLen, embedDim])
  (h0 : Option (Tensor α [hiddenDim])):
  (Tensor α [seqLen, hiddenDim] × Tensor α [hiddenDim]) :=
  RNN.sequence encoder.rnn x (h0.getD (Tensor.full (.dim hiddenDim .scalar) 0))


/--
RNN decoder of the Seq2Seq model, from target-side embeddings to vocabulary logits. An optional
causal self-attention block first transforms the input embeddings, with position `i` attending to
inputs `0, ..., i` in both teacher forcing and greedy decoding. An RNN cell (`RNN.cell`) then
updates the hidden state at each timestep, and a time-distributed `Linear` maps the hidden
states to logits.

PyTorch analogue: a hand-rolled decoder built from `nn.RNN` and `nn.Linear`, optionally preceded
by `nn.MultiheadAttention` over the target embeddings. That attention is self-attention, with no
encoder-decoder cross-attention.
-/
structure Seq2Seq.AttentionDecoder (α : Type) [Storage α]
    (embedDim hiddenDim vocabularySize : Nat) where
  /-- Decoder RNN cell parameters. -/
  rnn : Linear α (embedDim + hiddenDim) hiddenDim
  /-- Optional causal self-attention over decoder inputs, shared by training and inference. -/
  attention :
    Option (Σ numHeads : Nat, MultiHeadAttention α numHeads embedDim (embedDim /
numHeads)) := none
  /-- Output projection (`hiddenDim -> vocabularySize`) producing per-timestep logits. -/
  outputProjection : Linear α hiddenDim vocabularySize

/--
Prepare the RNN inputs from a nonempty sequence of decoder embeddings.

With attention enabled, row `i` uses only rows `0, ..., i`. The hard mask gives later positions
zero weight, including in the attention backward pass. Without attention, the embeddings pass
through unchanged. Teacher forcing and single-step decoding both use this function, so attention
has the same parameters, projection order, and mask convention in both paths.
-/
def Seq2Seq.AttentionDecoder.attendInputs {embedDim hiddenDim vocabularySize seqLen : Nat}
    (decoder : AttentionDecoder α embedDim hiddenDim vocabularySize)
    (embeddings : Tensor α [seqLen, embedDim]) (hLen : seqLen ≠ 0) :
    Tensor α [seqLen, embedDim] :=
  match decoder.attention with
  | some ⟨_numHeads, attn⟩ =>
      MultiHeadAttention.forward hLen hLen attn embeddings embeddings
        (some (Attention.causalMask seqLen))
  | none => embeddings

/--
Teacher-forcing logits for a sequence of decoder inputs.

`targetEmbeddings` has shape `(tgtSeqLen × embedDim)` and holds the tokens fed to the decoder. For
next-token prediction the caller supplies the start token followed by the preceding target tokens,
so the labels are one position ahead of these inputs. Causal self-attention prepares each RNN
input, then the recurrence starts at `h0` and the output projection produces vocabulary logits.
Changing a later decoder input cannot create an attention edge into an earlier position.
-/
def Seq2Seq.AttentionDecoder.forwardTeacherForcing
    {embedDim hiddenDim vocabularySize tgtSeqLen : Nat}
  (decoder : AttentionDecoder α embedDim hiddenDim vocabularySize)
  (targetEmbeddings : Tensor α [tgtSeqLen, embedDim])
  (h0 : Tensor α [hiddenDim])
  (h_len_nonzero : tgtSeqLen ≠ 0) :
  Tensor α [tgtSeqLen, vocabularySize] :=

  let attendedEmbeddings := decoder.attendInputs targetEmbeddings h_len_nonzero
  let hiddens := (RNN.sequence decoder.rnn attendedEmbeddings h0).1
  Linear.forwardLeading [tgtSeqLen] decoder.outputProjection hiddens

/--
Advance the decoder once using a nonempty prefix of input embeddings.

`previousHidden` is the RNN state after processing all but the last prefix token. Attention sees the
whole prefix, and its last row supplies the current RNN input. The earlier RNN steps are not
replayed. The result contains the updated hidden state and this step's vocabulary logits. Passing
the whole prefix also lets a caller compare a teacher-forced prefix with a single inference step.
-/
def Seq2Seq.AttentionDecoder.forwardStep {embedDim hiddenDim vocabularySize prefixLen : Nat}
    (decoder : AttentionDecoder α embedDim hiddenDim vocabularySize)
    (inputPrefix : Tensor α [prefixLen + 1, embedDim])
    (previousHidden : Tensor α [hiddenDim]) :
    Tensor α [hiddenDim] × Tensor α [vocabularySize] :=
  let attended := decoder.attendInputs inputPrefix (Nat.succ_ne_zero prefixLen)
  let input := Tensor.unstack attended ⟨prefixLen, Nat.lt_succ_self prefixLen⟩
  let hidden := RNN.cell decoder.rnn input previousHidden
  (hidden, Linear.forward decoder.outputProjection hidden)

/--
Greedy autoregressive decoding from `startToken` and the initial hidden state `h0`.

Each step appends the current token embedding to the input prefix, applies `forwardStep`, and
feeds the argmax token back as the next input. Optional self-attention therefore sees the same
prefix as teacher forcing with those input tokens. The RNN state advances once per emitted token.
Attention projections are recomputed from the stored prefix at every step, since this
specification has no key/value cache.

The result contains logits of shape `(maxLen × vocabularySize)` and `maxLen` predicted token ids.
When `maxLen = 0`, both outputs are empty and no decoder step runs.
-/
def Seq2Seq.AttentionDecoder.forwardInference {embedDim hiddenDim vocabularySize : Nat}
  (decoder : AttentionDecoder α embedDim hiddenDim vocabularySize)
  (h0 : Tensor α [hiddenDim])
  (targetEmbedding : Tensor α [vocabularySize, embedDim])
  (startToken : Fin vocabularySize) (maxLen : Nat) :
  (Tensor α [maxLen, vocabularySize] ×
    Tensor (Fin vocabularySize) [maxLen]) :=
  let initialInput := Tensor.unstack targetEmbedding startToken
  let hVocab : 0 < vocabularySize := lt_of_le_of_lt (Nat.zero_le startToken.val) startToken.isLt
  let initialHistory : Array (Tensor α [embedDim]) := #[]
  let (_, results) := Sequence.mapAccum maxLen (h0, initialInput, initialHistory) fun _ state =>
    let (previous, input, history) := state
    -- A decoder without attention only needs the current embedding. Discarding its history keeps
    -- the RNN-only loop linear in `maxLen`, while the attention path retains the full input prefix.
    let history := if decoder.attention.isSome then history else #[]
    let nextHistory := history.push input
    let inputPrefix : Tensor α [history.size + 1, embedDim] :=
      Tensor.dim (fun i =>
        nextHistory[i.val]'(by simpa only [nextHistory, Array.size_push] using i.isLt))
    let (nextHidden, logits) := decoder.forwardStep inputPrefix previous
    let token : Fin vocabularySize := (argmax logits ⟨0, Nat.one_pos⟩ false hVocab).item
    let nextInput := Tensor.unstack targetEmbedding token
    ((nextHidden, nextInput, nextHistory), (logits, token))
  (Tensor.dim (fun i => (results.getScalar i).1),
    Tensor.dim (fun i => Tensor.scalar (results.getScalar i).2))

/--
The complete Seq2Seq baseline: source and target embedding tables, an RNN encoder, and an RNN
decoder with its output projection and optional self-attention.

PyTorch analogue: a small encoder-decoder model built from `nn.Embedding`, `nn.RNN` and
`nn.Linear`.
-/
structure Seq2Seq (α : Type) [Storage α]
    (srcVocabSize tgtVocabSize embedDim hiddenDim : Nat)
  where
  /-- Source embedding table, one row of width `embedDim` per source token. -/
  sourceEmbedding : Tensor α [srcVocabSize, embedDim]
  /-- Target embedding table, one row of width `embedDim` per target token. -/
  targetEmbedding : Tensor α [tgtVocabSize, embedDim]
  /-- Encoder RNN parameters. -/
  encoder : Seq2Seq.RNNEncoder α embedDim hiddenDim
  /-- Decoder parameters (RNN + output projection + optional self-attention). -/
  decoder : Seq2Seq.AttentionDecoder α embedDim hiddenDim tgtVocabSize

/--
Teacher-forcing logits from discrete source and decoder input tokens.

`sourceTokens` supplies the encoder sequence. `targetTokens` supplies the decoder inputs: for
next-token prediction, these are the start token followed by the preceding target tokens. The
caller pairs the returned `(tgtSeqLen × tgtVocabSize)` logits with labels one position ahead.
The inputs are embedded as given, with no start token inserted and no shift.

The encoder's final hidden state initializes the decoder. Optional decoder self-attention is causal,
and embedding lookup treats the bounded token ids as discrete inputs without token-id gradients.
-/
def Seq2Seq.forwardTraining {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen :
  Nat}
  (model : Seq2Seq α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (sourceTokens : Tensor (Fin srcVocabSize) [srcSeqLen])
  (targetTokens : Tensor (Fin tgtVocabSize) [tgtSeqLen])
  (hTarget : tgtSeqLen ≠ 0) :
  Tensor α [tgtSeqLen, tgtVocabSize] :=
  let sourceEmbeddings : Tensor α [srcSeqLen, embedDim] :=
    Tensor.embedding model.sourceEmbedding sourceTokens
  let (_encoderOutputs, encoderHidden) :=
    RNNEncoder.forward model.encoder sourceEmbeddings
    none
  let targetEmbeddings : Tensor α [tgtSeqLen, embedDim] :=
    Tensor.embedding model.targetEmbedding targetTokens
  AttentionDecoder.forwardTeacherForcing model.decoder targetEmbeddings encoderHidden hTarget

/--
Encode the source once and generate `maxTgtLen` target tokens greedily.

The encoder's final hidden state initializes the decoder, and `startToken` supplies its first input.
Each predicted token becomes the next decoder input. Optional causal self-attention uses that
growing input prefix, with the same attention parameters as teacher forcing. The returned pair
contains `(maxTgtLen × tgtVocabSize)` logits and `maxTgtLen` bounded token ids.
-/
def Seq2Seq.forwardInference {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen : Nat}
  (maxTgtLen : Nat)
  (model : Seq2Seq α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (sourceTokens : Tensor (Fin srcVocabSize) [srcSeqLen])
  (startToken : Fin tgtVocabSize) :
  (Tensor α [maxTgtLen, tgtVocabSize] ×
    Tensor (Fin tgtVocabSize) [maxTgtLen]) :=
  let sourceEmbeddings : Tensor α [srcSeqLen, embedDim] :=
    Tensor.embedding model.sourceEmbedding sourceTokens
  let (_encoderOutputs, encoderHidden) :=
    RNNEncoder.forward model.encoder sourceEmbeddings none
  AttentionDecoder.forwardInference model.decoder encoderHidden model.targetEmbedding
    startToken maxTgtLen

/-!
### Training on token distributions

The teacher-forcing pass `Seq2Seq.forwardTrainingOneHot` and its loss
`Seq2Seq.crossEntropyLossOneHot` take one-hot vectors or token distributions in place of `Fin`
token indices.
-/

/--
Differentiable teacher-forcing forward pass over one-hot or token-distribution inputs.

The pass is `Seq2Seq.forwardTraining` with each embedding computed as the `Tensor.matmul` of
the token distributions with the table, so gradients can flow into the embedding tables and back
into upstream token distributions. Over a semiring, exactly one-hot inputs embed to the rows that
`forwardTraining` looks up. In a binary format an embedding entry is NaN when another row of its
table column holds `±∞` or NaN, and `+0` when the selected entry is `-0`.

`tgtOneHot` holds the decoder inputs, in the same start-token/preceding-token order as the
discrete path. For next-token prediction, supply the labels to the loss separately, one position
ahead of these inputs.
-/
def Seq2Seq.forwardTrainingOneHot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  (model : Seq2Seq α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (srcOneHot : Tensor α [srcSeqLen, srcVocabSize])
  (tgtOneHot : Tensor α [tgtSeqLen, tgtVocabSize])
  (hTgt : tgtSeqLen ≠ 0) :
  Tensor α [tgtSeqLen, tgtVocabSize] :=
  let srcEmbeds := Tensor.matmul srcOneHot model.sourceEmbedding
  let (_encOut, encHidden) := RNNEncoder.forward model.encoder srcEmbeds none
  let tgtEmbeds := Tensor.matmul tgtOneHot model.targetEmbedding
  AttentionDecoder.forwardTeacherForcing model.decoder tgtEmbeds encHidden hTgt

/--
Same-position reconstruction loss for the differentiable Seq2Seq baseline: cross-entropy of the
`softmax` of the `Seq2Seq.forwardTrainingOneHot` logits against `tgtOneHot`, which also
supplies the decoder inputs.

For shifted next-token supervision, call `forwardTrainingOneHot`, apply softmax to its logits and
pass separate labels to `Tensor.crossEntropy (.probabilities epsilon)`, whose reduction and
clamping this loss uses.
-/
def Seq2Seq.crossEntropyLossOneHot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  [Shape.HasNonemptyAxis 1 (.dim tgtSeqLen (.dim tgtVocabSize .scalar))]
  (model : Seq2Seq α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (srcOneHot : Tensor α [srcSeqLen, srcVocabSize])
  (tgtOneHot : Tensor α [tgtSeqLen, tgtVocabSize])
  (hTgt : tgtSeqLen ≠ 0) : α :=
  let logits := Seq2Seq.forwardTrainingOneHot (α := α) model srcOneHot tgtOneHot hTgt
  let probs := Tensor.softmax logits ⟨1, Shape.AxisInBounds.proof⟩
  Tensor.crossEntropy (.probabilities Context.defaultEpsilon) probs tgtOneHot
    ⟨1, Shape.AxisInBounds.proof⟩

end TorchLean
