/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Loss
public import NN.Spec.Layers.Lstm
public import NN.Spec.Models.Transformer

/-!
# Seq2Seq (spec model)

Encoder-decoder models for sequence generation.

This file supports both:
- discrete token id inputs (non‑differentiable lookup, good for runtime demos), and
- one‑hot / token‑distribution inputs (differentiable embedding via a matrix multiply).

PyTorch mental model:

- encoder: `nn.RNN` / `nn.LSTM` (or `nn.TransformerEncoder`) over source token embeddings
- decoder: `nn.RNN` over target embeddings (teacher forcing in training), then a final `nn.Linear`
  to vocabulary logits

What we deliberately keep simple:

- the optional attention in `Seq2SeqDecoderSpec` is *self-attention over the decoder inputs* (a
  small variant you can toggle on/off); this file does not model encoder-decoder cross-attention
  in the main baseline.
- for cross-attention style mechanisms, we include a small additive/Bahdanau-style attention at the
  bottom of the file (`compute_attention_weights_spec` / `apply_attention_spec`).

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
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (JMLR
  2014).

PyTorch docs (for API intuition, not semantics):
- `torch.nn.Embedding`: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- `torch.nn.RNN`: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
- `torch.nn.LSTM`: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- `torch.nn.Linear`: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- `torch.nn.MultiheadAttention`:
  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- `torch.nn.TransformerEncoderLayer`:
  https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

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

Token-id based training (`Tensor Nat`) is still useful for demos, but it is intentionally treated as
non-differentiable.
-/

/-! ### Small gradient records -/

/--
Gradients for a time-distributed affine map `y = x·Wᵀ + b`.

This mirrors the parameters in `LinearSpec` and is used for the decoder output projection.
PyTorch analogue: the gradient pair for `nn.Linear`.
-/
structure Seq2SeqLinearGrads (α : Type) (inDim outDim : Nat) where
  /-- Gradient of the weight matrix `W`. -/
  dW : Tensor α (.dim outDim (.dim inDim .scalar))
  /-- Gradient of the bias vector `b`. -/
  db : Tensor α (.dim outDim .scalar)

/--
Gradients for an `RNNSpec` cell.

PyTorch analogue: the gradients for `nn.RNN` parameters (weight and bias).
-/
structure Seq2SeqRNNGrads (α : Type) (inputSize hiddenSize : Nat) where
  /-- Gradient of the concatenated input+hidden weight matrix. -/
  dW : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- Gradient of the bias term. -/
  db : HiddenVector α hiddenSize

/--
Gradients for a token embedding table `E : (vocabSize × embedDim)`.

PyTorch analogue: `nn.Embedding.weight.grad`.
-/
structure Seq2SeqEmbeddingGrads (α : Type) (vocabSize embedDim : Nat) where
  /-- Gradient of the embedding matrix. -/
  d_embedding : Tensor α (.dim vocabSize (.dim embedDim .scalar))

/--
End-to-end gradient record for the differentiable Seq2Seq baseline.

This bundles gradients for:
- source/target embeddings,
- encoder RNN,
- decoder RNN,
- decoder output projection,
- optional decoder self-attention (if enabled in the decoder spec).
-/
structure Seq2SeqGrads (α : Type)
    (srcVocabSize tgtVocabSize embedDim hiddenDim : Nat) where
  /-- Gradients for the source embedding table. -/
  d_src_embedding : Seq2SeqEmbeddingGrads α srcVocabSize embedDim
  /-- Gradients for the target embedding table. -/
  d_tgt_embedding : Seq2SeqEmbeddingGrads α tgtVocabSize embedDim
  /-- Gradients for the encoder RNN parameters. -/
  d_encoder : Seq2SeqRNNGrads α embedDim hiddenDim
  /-- Gradients for the decoder RNN parameters. -/
  d_decoder_rnn : Seq2SeqRNNGrads α embedDim hiddenDim
  /-- Gradients for the decoder output projection (`hiddenDim -> tgtVocabSize`). -/
  d_output_projection : Seq2SeqLinearGrads α hiddenDim tgtVocabSize
  /-- Gradients for optional decoder self-attention parameters. -/
  d_decoder_attention :
    Option (Σ numHeads : Nat, MultiHeadAttentionGrads numHeads embedDim (embedDim / numHeads) α) :=
    none

/--
Seq2Seq token embedding specification.

Parameters:
- `embedding`: a lookup table `E : (vocabSize × embedDim)`,
- `dropout_rate`: a scalar `p` used by `dropout_inference_spec` (deterministic scaling).

PyTorch analogue: `nn.Embedding(vocabSize, embedDim)` plus a `nn.Dropout(p)` applied to the
sequence of embeddings.
-/
structure Seq2SeqEmbeddingSpec (α : Type) [Numbers α] (vocabSize embedDim : Nat) where
  /-- Embedding table `E : (vocabSize × embedDim)`. -/
  embedding : Tensor α (.dim vocabSize (.dim embedDim .scalar))
  /-- Dropout probability `p` (used in a deterministic inference-style way). -/
  dropout_rate : α := Numbers.pointone

/--
Embedding forward pass for discrete token ids.

Inputs:
- `token_ids : (seqLen)`, a tensor of natural-number token ids.

Output:
- `y : (seqLen × embedDim)`, where each timestep selects a row of the embedding table.

Out-of-range token ids map to a zero vector in this spec. We then apply
`dropout_inference_spec` deterministically (no RNG), which is useful for runtime demos.

PyTorch analogue: `nn.Embedding` on an integer tensor, followed by `nn.Dropout(p)` (but with
randomness replaced by deterministic scaling; see `NN.Spec.Layers.Dropout`).
-/
def Seq2SeqEmbeddingSpec.forward {vocabSize embedDim seqLen : Nat}
  (embedding : Seq2SeqEmbeddingSpec α vocabSize embedDim)
  (token_ids : Tensor Nat (.dim seqLen .scalar)):
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=

  -- Token embeddings
  let token_embeds := Tensor.dim (fun i =>
    match get token_ids i with
    | Tensor.scalar token_id =>
      if h : token_id < vocabSize then
        match get embedding.embedding ⟨token_id, h⟩ with
        | Tensor.dim embed_vals => Tensor.dim embed_vals
      else
        Tensor.dim (fun _ => Tensor.scalar (0 : α))
  )

  -- Deterministic dropout (inference-style scaling).
  dropoutInferenceSpec (p := embedding.dropout_rate) token_embeds

/-- Seq2Seq embedding forward pass for one-hot / token distributions.

This is the usual "embedding lookup as a matrix multiply":

- if `E : (vocabSize × embedDim)` is the embedding table,
- and `x_t : (vocabSize)` is a one-hot / probability vector for time step `t`,
- then the embedded vector is `y_t = x_tᵀ · E : (embedDim)`.

PyTorch analogy: `y = x @ E` where `x` is one-hot / a distribution; this matches `nn.Embedding`
when the input is exactly one-hot.
-/
def Seq2SeqEmbeddingSpec.forwardOnehot {vocabSize embedDim seqLen : Nat}
  (embedding : Seq2SeqEmbeddingSpec α vocabSize embedDim)
  (token_onehot : Tensor α (.dim seqLen (.dim vocabSize .scalar))) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  match token_onehot with
  | Tensor.dim f =>
    let token_embeds := Tensor.dim (fun i => vecMatMulSpec (f i) embedding.embedding)
    dropoutInferenceSpec (p := embedding.dropout_rate) token_embeds

/--
Backward pass for `Seq2SeqEmbeddingSpec.forwardOnehot`.

This is just a time-distributed linear layer:

`y_t = token_tᵀ · E`

So:
- `dE = Σ_t token_t ⊗ dY_t`
- `dToken_t = E · dY_t` (not usually needed, but included for completeness)
-/
def Seq2SeqEmbeddingSpec.backwardOnehot {vocabSize embedDim seqLen : Nat}
  (embedding : Seq2SeqEmbeddingSpec α vocabSize embedDim)
  (token_onehot : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (grad_output : Tensor α (.dim seqLen (.dim embedDim .scalar))) :
  (Seq2SeqEmbeddingGrads α vocabSize embedDim × Tensor α (.dim seqLen (.dim vocabSize .scalar))) :=
  let grad_output0 :=
    dropoutInferenceBackwardSpec (p := embedding.dropout_rate) grad_output
  let step (acc : Seq2SeqEmbeddingGrads α vocabSize embedDim × List (Tensor α (.dim vocabSize
    .scalar))) (i : Fin seqLen) :=
    let (accE, accX) := acc
    let token_t := get token_onehot i
    let dY_t := get grad_output0 i
    let dE_t := outerProductSpec token_t dY_t
    let dToken_t := matVecMulSpec embedding.embedding dY_t
    ({ d_embedding := addSpec accE.d_embedding dE_t }, dToken_t :: accX)
  let init : Seq2SeqEmbeddingGrads α vocabSize embedDim :=
    { d_embedding := fill 0 (.dim vocabSize (.dim embedDim .scalar)) }
  let (dE, dX_rev) := (List.finRange seqLen).foldl step (init, [])
  let dX_list := dX_rev.reverse
  let dX :=
    match dX_list with
    | [] => fill 0 (.dim seqLen (.dim vocabSize .scalar))
    | h :: _ => Tensor.dim (fun i => dX_list.getD i.val h)
  (dE, dX)

/--
RNN-based encoder specification for Seq2Seq.

This models an `nn.RNN`-style encoder over embedded tokens:
- input is a sequence of embeddings `(seqLen × embedDim)`,
- output is the full hidden-state sequence plus the final hidden state.

PyTorch analogue: `nn.RNN(..., batch_first=True)` (ignoring the batch axis), returning `(output,
  h_n)`.
-/
structure Seq2SeqRNNEncoderSpec (α : Type) [Numbers α] (embedDim hiddenDim : Nat) where
  /-- RNN cell parameters. -/
  rnn : RNNSpec α embedDim hiddenDim
  /-- Dropout probability `p` (applied as `dropout_inference_spec` to the input sequence). -/
  dropout_rate : α := Numbers.pointone

/--
Forward pass for `Seq2SeqRNNEncoderSpec`.

Inputs:
- `x : (seqLen × embedDim)`, embedded source tokens,
- `h0`, optional initial hidden state (`hiddenDim`).

Returns:
- `(outputs, final_h)` where `outputs : (seqLen × hiddenDim)` is the per-timestep hidden sequence.
-/
def Seq2SeqRNNEncoderSpec.forward {α : Type} [Context α] {embedDim hiddenDim seqLen : Nat}
  (encoder : Seq2SeqRNNEncoderSpec α embedDim hiddenDim)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h0 : Option (Tensor α (.dim hiddenDim .scalar))):
  (Tensor α (.dim seqLen (.dim hiddenDim .scalar)) × Tensor α (.dim hiddenDim .scalar)) :=

  let x := dropoutInferenceSpec (p := encoder.dropout_rate) x

  -- Initial hidden state
  let initial_h := match h0 with
    | some h => h
    | none => Tensor.dim (fun _ => Tensor.scalar (0 : α))

  -- Recursive sequence processor
  let rec process_sequence (i : Nat) (h : Tensor α (.dim hiddenDim .scalar))
    (acc : List (Tensor α (.dim hiddenDim .scalar))) (h1 : i <= seqLen) :
    (Tensor α (.dim hiddenDim .scalar) × List (Tensor α (.dim hiddenDim .scalar))) :=
    if h_eq : i = seqLen then
      (h, acc.reverse)
    else
      match x with
      | Tensor.dim tokens =>
        have hi : i < seqLen := Nat.lt_of_le_of_ne h1 h_eq
        let input_token := tokens ⟨i, hi⟩
        let new_h := rnnCellSpec encoder.rnn input_token h
        let h2 : i + 1 <= seqLen := Nat.succ_le_of_lt hi
        process_sequence (i + 1) new_h (new_h :: acc) h2
  termination_by seqLen - i
  decreasing_by
    all_goals
      simpa using Nat.sub_succ_lt_self (a := seqLen) (i := i) hi

  let h_zero : 0 <= seqLen := Nat.zero_le seqLen
  let (final_h, outputs) := process_sequence 0 initial_h [] h_zero

  -- Convert list to tensor
  let output_tensor : Tensor α (.dim seqLen (.dim hiddenDim .scalar)) :=
    Tensor.dim (fun i =>
      outputs.getD i.val (Tensor.dim (fun _ => Tensor.scalar (0 : α))))

  (output_tensor, final_h)


/--
LSTM-based encoder specification for Seq2Seq.

This models an `nn.LSTM`-style encoder over embedded tokens, returning the full hidden sequence,
final hidden state, and final cell state.

PyTorch analogue: `nn.LSTM(..., batch_first=True)` (ignoring the batch axis), returning
`(output, (h_n, c_n))`.
-/
structure Seq2SeqLSTMEncoderSpec (α : Type) [Numbers α] (embedDim hiddenDim : Nat) where
  /-- LSTM cell parameters. -/
  lstm : LSTMSpec α embedDim hiddenDim
  /-- Dropout probability `p` (applied as `dropout_inference_spec` to the input sequence). -/
  dropout_rate : α := Numbers.pointone

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
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h0 : Option (Tensor α (.dim hiddenDim .scalar)))
  (c0 : Option (Tensor α (.dim hiddenDim .scalar))):
  (Tensor α (.dim seqLen (.dim hiddenDim .scalar)) ×
   Tensor α (.dim hiddenDim .scalar) ×
   Tensor α (.dim hiddenDim .scalar)) :=

  let x := dropoutInferenceSpec (p := encoder.dropout_rate) x

  -- Initialize hidden and cell states
  let initial_h := match h0 with
  | some h => h
  | none => Tensor.dim (fun _ => Tensor.scalar (0 : α))

  let initial_c := match c0 with
  | some c => c
  | none => Tensor.dim (fun _ => Tensor.scalar (0 : α))

  -- Process sequence step by step
  let rec process_sequence (i : Nat) (h : Tensor α (.dim hiddenDim .scalar))
    (c : Tensor α (.dim hiddenDim .scalar))
    (acc : List (Tensor α (.dim hiddenDim .scalar))) (h1 : i <= seqLen):
    (Tensor α (.dim hiddenDim .scalar) × Tensor α (.dim hiddenDim .scalar) × List (Tensor α (.dim
      hiddenDim .scalar))) :=
    if h4 : i = seqLen then (h, c, acc.reverse)
    else
      let input_token := match x with
      | Tensor.dim tokens =>
        have hi : i < seqLen := Nat.lt_of_le_of_ne h1 h4
        tokens ⟨i, hi⟩
      let lstm_state : LSTMState α hiddenDim := ⟨h, c⟩
      let new_state := lstmCellSpec encoder.lstm input_token lstm_state
      let new_h := new_state.hidden
      let new_c := new_state.cell
      have hi : i < seqLen := Nat.lt_of_le_of_ne h1 h4
      let h2 : i + 1 <= seqLen := Nat.succ_le_of_lt hi
      process_sequence (i + 1) new_h new_c (new_h :: acc) h2
    termination_by seqLen - i
    decreasing_by
      all_goals
        simpa using Nat.sub_succ_lt_self (a := seqLen) (i := i) hi

  let h_zero : 0 <= seqLen := Nat.zero_le seqLen
  let (final_h, final_c, outputs) := process_sequence 0 initial_h initial_c [] h_zero

  -- Convert list to tensor
  let output_tensor := Tensor.dim (fun i =>
    outputs.getD i.val (Tensor.dim (fun _ => Tensor.scalar (0 : α))))

  (output_tensor, final_h, final_c)

/--
Transformer-based encoder specification for Seq2Seq.

This is a lightweight wrapper around a list of `TransformerEncoderLayer`s from
`NN.Spec.Models.Transformer`, applied as a left-fold.

PyTorch analogue: `nn.TransformerEncoder(nn.TransformerEncoderLayer(...), num_layers=...)`
(ignoring dropout and most configuration knobs).
-/
structure Seq2SeqTransformerEncoderSpec (α : Type) [Context α] [Numbers α] (embedDim numHeads
  numLayers : Nat) where
  /-- Encoder layer stack; typically length `numLayers`, but not enforced by the spec. -/
  layers : List (TransformerEncoderLayer numHeads embedDim (embedDim * 4) α)
  /-- Dropout probability `p` (applied as `dropout_inference_spec` to the input sequence). -/
  dropout_rate : α := Numbers.pointone

/--
Forward pass for `Seq2SeqTransformerEncoderSpec`.

Input/output shape: `(seqLen × embedDim)`.

This uses post-norm transformer layers from `NN.Spec.Models.Transformer` and does not model
dropout; it is meant as a clean semantic reference rather than a full training-ready implementation.
-/
def Seq2SeqTransformerEncoderSpec.forward {embedDim numHeads numLayers seqLen : Nat}
  (encoder : Seq2SeqTransformerEncoderSpec α embedDim numHeads numLayers)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (h1 : seqLen > 0) (h2 : embedDim > 0) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  let x := dropoutInferenceSpec (p := encoder.dropout_rate) x
  encoder.layers.foldl (fun acc layer => TransformerEncoderLayer.forward layer acc h1 h2) x

/--
RNN decoder specification for Seq2Seq.

This decoder consumes a sequence of target-side embeddings and produces vocabulary logits:
- an `RNNSpec` cell updates the hidden state per timestep,
- a time-distributed `LinearSpec` maps hidden states to logits,
- optionally, a self-attention block can be applied over the *decoder input embeddings* before the
  RNN.

PyTorch analogue: a hand-rolled decoder using `nn.RNN` and `nn.Linear`, optionally preceded by
`nn.MultiheadAttention` over the target embeddings (note: this is not encoder-decoder
  cross-attention).
-/
structure Seq2SeqDecoderSpec (α : Type) [Numbers α] (embedDim hiddenDim vocabSize : Nat) where
  /-- Decoder RNN cell parameters. -/
  rnn : RNNSpec α embedDim hiddenDim
  /-- Optional self-attention block over decoder input embeddings. -/
  attention :
    Option (Σ numHeads : Nat, MultiHeadAttention α numHeads embedDim (embedDim / numHeads)) := none
  /-- Output projection (`hiddenDim -> vocabSize`) producing per-timestep logits. -/
  output_projection : LinearSpec α hiddenDim vocabSize
  /-- Dropout probability `p` (applied as `dropout_inference_spec` to decoder input embeddings). -/
  dropout_rate : α := Numbers.pointone

/--
Seq2Seq decoder forward pass (teacher forcing)

- `target_embeddings` : Tensor of shape (tgtSeqLen × embedDim)
- `h0` : initial hidden state (hiddenDim)
- Returns: Tensor of shape (tgtSeqLen × vocabSize)
- If `decoder.attention` is `some`, this runs self-attention over `target_embeddings` and feeds the
  attended embedding at each timestep.
- Note: this spec does not model cross-attention over encoder outputs.
-/
def Seq2SeqDecoderSpec.forwardTeacherForcing {embedDim hiddenDim vocabSize tgtSeqLen : Nat}
  (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabSize)
  (target_embeddings : Tensor α (.dim tgtSeqLen (.dim embedDim .scalar)))
  (h0 : Tensor α (.dim hiddenDim .scalar))
  (h_len_nonzero : tgtSeqLen ≠ 0) :
  Tensor α (.dim tgtSeqLen (.dim vocabSize .scalar)) :=

  -- Optional self-attention over the full target embedding sequence.
  let attended_embeddings :=
    match decoder.attention with
    | some ⟨_numHeads, attn⟩ =>
        MultiHeadAttention.forward tgtSeqLen h_len_nonzero attn target_embeddings none
    | none => target_embeddings

  let attended_embeddings :=
    dropoutInferenceSpec (p := decoder.dropout_rate) attended_embeddings

  -- Recursive function to process sequence step by step
  let rec process_sequence
    (i : Nat)                             -- current time step
    (h : Tensor α (.dim hiddenDim .scalar)) -- current hidden state
    (acc : List (Tensor α (.dim vocabSize .scalar))) -- accumulated logits
    (h_bound : i <= tgtSeqLen)            -- proof that i <= tgtSeqLen
    : List (Tensor α (.dim vocabSize .scalar)) :=
    if h_end : i = tgtSeqLen then
      acc.reverse  -- return accumulated logits
    else
      -- Safe access to target embedding at step i
      have h_idx : i < tgtSeqLen := Nat.lt_of_le_of_ne h_bound h_end
      let idx : Fin tgtSeqLen := ⟨i, h_idx⟩

      let attended_input :=
        match attended_embeddings with
        | Tensor.dim tokens => tokens idx

      -- Update hidden state via RNN cell
      let new_h := rnnCellSpec decoder.rnn attended_input h

      -- Project hidden state to logits
      let output_logits := linearSpec decoder.output_projection new_h

      -- Increment recursion
      let h_next : i + 1 <= tgtSeqLen := Nat.succ_le_of_lt h_idx
      process_sequence (i + 1) new_h (output_logits :: acc) h_next

  termination_by tgtSeqLen - i
  decreasing_by
    all_goals
      simpa using Nat.sub_succ_lt_self (a := tgtSeqLen) (i := i) h_idx

  -- Start recursion
  let h_zero : 0 <= tgtSeqLen := Nat.zero_le tgtSeqLen
  let outputs := process_sequence 0 h0 [] h_zero

  -- Convert list of logits to tensor safely using Fin indexing
  Tensor.dim (fun i : Fin tgtSeqLen =>
    outputs.getD i.val (Tensor.dim (fun _ => Tensor.scalar (0 : α))))

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
Backward pass for a time-distributed `LinearSpec`.

Given a hidden-state sequence `hiddens : (tgtSeqLen × hiddenDim)` and upstream gradients
`grad_logits : (tgtSeqLen × vocabSize)`, computes:
- accumulated parameter gradients for the shared `LinearSpec`,
- gradients w.r.t. each hidden state `(tgtSeqLen × hiddenDim)`.

PyTorch analogue: backprop through `nn.Linear` applied at each timestep.
-/
def timeDistributedLinearBackward
  {tgtSeqLen hiddenDim vocabSize : Nat}
  (layer : LinearSpec α hiddenDim vocabSize)
  (hiddens : Tensor α (.dim tgtSeqLen (.dim hiddenDim .scalar)))
  (grad_logits : Tensor α (.dim tgtSeqLen (.dim vocabSize .scalar))) :
  (Seq2SeqLinearGrads α hiddenDim vocabSize × Tensor α (.dim tgtSeqLen (.dim hiddenDim .scalar))) :=
  let step (acc : Seq2SeqLinearGrads α hiddenDim vocabSize × List (Tensor α (.dim hiddenDim
    .scalar))) (i : Fin tgtSeqLen) :=
    let (accLin, accDH) := acc
    let hi := get hiddens i
    let dYi := get grad_logits i
    let (dW, db, dH) := linearBackwardSpec layer hi dYi
    ({ dW := addSpec accLin.dW dW, db := addSpec accLin.db db }, dH :: accDH)
  let init : Seq2SeqLinearGrads α hiddenDim vocabSize := {
    dW := fill 0 (.dim vocabSize (.dim hiddenDim .scalar)),
    db := fill 0 (.dim vocabSize .scalar)
  }
  let (linGrads, dH_rev) := (List.finRange tgtSeqLen).foldl step (init, [])
  let dH_list := dH_rev.reverse
  let dH :=
    match dH_list with
    | [] => fill 0 (.dim tgtSeqLen (.dim hiddenDim .scalar))
    | h :: _ => Tensor.dim (fun i => dH_list.getD i.val h)
  (linGrads, dH)

/--
Backward pass for `Seq2SeqDecoderSpec.forwardTeacherForcing`.

Returns:
- RNN parameter gradients,
- output projection gradients,
- optional self-attention parameter gradients,
- gradient w.r.t. the target embeddings sequence,
- gradient w.r.t. the initial hidden state `h0`.

Implementation note: this spec recomputes the attended embeddings and hidden sequence to keep the
backward pass self-contained (no mutable tape).
-/
def Seq2SeqDecoderSpec.backwardTeacherForcing
  {embedDim hiddenDim vocabSize tgtSeqLen : Nat}
  (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabSize)
  (target_embeddings : Tensor α (.dim tgtSeqLen (.dim embedDim .scalar)))
  (h0 : Tensor α (.dim hiddenDim .scalar))
  (h_len_nonzero : tgtSeqLen ≠ 0)
  (grad_logits : Tensor α (.dim tgtSeqLen (.dim vocabSize .scalar))) :
  (Seq2SeqRNNGrads α embedDim hiddenDim ×
    Seq2SeqLinearGrads α hiddenDim vocabSize ×
    Option (Σ numHeads : Nat, MultiHeadAttentionGrads numHeads embedDim (embedDim / numHeads) α) ×
    Tensor α (.dim tgtSeqLen (.dim embedDim .scalar)) ×
    Tensor α (.dim hiddenDim .scalar)) :=

  let attended_embeddings0 :=
    match decoder.attention with
    | some ⟨_numHeads, attn⟩ =>
        MultiHeadAttention.forward tgtSeqLen h_len_nonzero attn target_embeddings none
    | none => target_embeddings

  let attended_embeddings :=
    dropoutInferenceSpec (p := decoder.dropout_rate) attended_embeddings0

  let hiddens := rnnSequenceSpec decoder.rnn attended_embeddings h0
  let (projGrads, dH) := timeDistributedLinearBackward (α := α)
    (tgtSeqLen := tgtSeqLen) (hiddenDim := hiddenDim) (vocabSize := vocabSize)
    decoder.output_projection hiddens grad_logits

  let (dW_rnn, db_rnn, dAttended0, dH0) :=
    rnnSequenceBackwardSpec decoder.rnn attended_embeddings h0 hiddens dH

  let dAttended :=
    dropoutInferenceBackwardSpec (p := decoder.dropout_rate) dAttended0

  let rnnGrads : Seq2SeqRNNGrads α embedDim hiddenDim := { dW := dW_rnn, db := db_rnn }

  match decoder.attention with
  | none =>
      (rnnGrads, projGrads, none, dAttended, dH0)
  | some ⟨numHeads, attn⟩ =>
      let (dTargetEmb, dWq, dWk, dWv, dWo) :=
        MultiHeadAttentionBackward (α := α) (n := tgtSeqLen) (dModel := embedDim)
          h_len_nonzero attn target_embeddings none dAttended
      let attnGrads : MultiHeadAttentionGrads numHeads embedDim (embedDim / numHeads) α :=
        { dWq := dWq, dWk := dWk, dWv := dWv, dWo := dWo }
      (rnnGrads, projGrads, some ⟨numHeads, attnGrads⟩, dTargetEmb, dH0)

/--
Seq2Seq decoder forward pass (inference-time autoregressive decoding).

This runs a greedy decoding loop for `maxLen` steps, starting from:
- an initial hidden state `h0`,
- a starting input embedding vector (`start_token` already embedded),
- and a target embedding table `tgt_embedding` used to embed the predicted next token id.

Returns:
- the per-step logits `(maxLen × vocabSize)`,
- the greedy-decoded token ids `(maxLen)`.

PyTorch analogue: a manual decoding loop using `nn.RNNCell`/`nn.RNN` + `nn.Linear`, with
`argmax` sampling and embedding lookup each step.

Note: `decoder.attention` is only modeled in the teacher-forcing forward/backward in this file; the
greedy decoding loop below does not implement autoregressive self-attention.
-/
def Seq2SeqDecoderSpec.forwardInference {embedDim hiddenDim vocabSize : Nat}
  (decoder : Seq2SeqDecoderSpec α embedDim hiddenDim vocabSize)
  (h0 : Tensor α (.dim hiddenDim .scalar))
  (start_token : Tensor α (.dim embedDim .scalar))
  (tgt_embedding : Tensor α (.dim vocabSize (.dim embedDim .scalar)))
  (maxLen : Nat) (h_len : maxLen ≠ 0)
  (h3 : vocabSize ≠ 0):
  (Tensor α (.dim maxLen (.dim vocabSize .scalar)) × Tensor Nat (.dim maxLen .scalar)) :=

  let rec process_sequence (i : Nat) (h : Tensor α (.dim hiddenDim .scalar))
    (current_input : Tensor α (.dim embedDim .scalar))
    (acc_logits : List (Tensor α (.dim vocabSize .scalar)))
    (acc_tokens : List Nat) (h_bound : i <= maxLen) (h_len : maxLen ≠ 0) (h_vocab : vocabSize > 0):
    (List (Tensor α (.dim vocabSize .scalar)) × List Nat) :=
    if h_max : i = maxLen then (acc_logits.reverse, acc_tokens.reverse)
    else
      -- NOTE: we do not model `decoder.attention` in this greedy RNN decoding loop.
      -- The self-attention variant in this file is intended for teacher-forcing over a known target
      -- sequence. Autoregressive attention (prefix attention + causal masking) belongs in the
      -- transformer-style decoders.
      let attended_input :=
        dropoutInferenceSpec (p := decoder.dropout_rate) current_input

      let new_h := rnnCellSpec decoder.rnn attended_input h
      let output_logits := linearSpec decoder.output_projection new_h

      -- Greedy decoding
      let predicted_token :=
        match output_logits with
        | Tensor.dim logits =>
          -- Recursive greedy argmax over Fin vocabSize → Tensor α Shape.scalar
          let rec find_max (j : Nat) (max_val : α) (max_idx : Nat) : Nat :=
            if hlt : j < vocabSize then
              let val_tensor := logits ⟨j, hlt⟩
              match val_tensor with
              | Tensor.scalar v =>
                if v > max_val then
                  find_max (j + 1) v j
                else
                  find_max (j + 1) max_val max_idx
            else max_idx
          termination_by vocabSize - j
          decreasing_by
            all_goals
              simpa using Nat.sub_succ_lt_self (a := vocabSize) (i := j) hlt

          -- Initialize recursion with first element
          let first_tensor := logits ⟨0, h_vocab⟩
          match first_tensor with
          | Tensor.scalar first_val => find_max 1 first_val 0


      -- Next input embedding: lookup the predicted token in the target embedding table.
      let next_input :=
        if h : predicted_token < vocabSize then
          match get tgt_embedding ⟨predicted_token, h⟩ with
          | Tensor.dim embed_vals => Tensor.dim embed_vals
        else
          Tensor.dim (fun _ => Tensor.scalar (0 : α))

      have hi : i < maxLen := Nat.lt_of_le_of_ne h_bound h_max
      let h_next : i + 1 <= maxLen := Nat.succ_le_of_lt hi
      process_sequence (i + 1) new_h next_input (output_logits :: acc_logits) (predicted_token ::
        acc_tokens) h_next h_len h_vocab

  termination_by maxLen - i
  decreasing_by
    all_goals
      simpa using Nat.sub_succ_lt_self (a := maxLen) (i := i) hi

  let h_zero : 0 <= maxLen := Nat.zero_le maxLen
  let (outputs, tokens) :=
    process_sequence 0 h0 start_token [] [] h_zero h_len (Nat.pos_of_ne_zero h3)

  -- Convert lists to tensors
  let output_tensor := Tensor.dim (fun i =>
    outputs.getD i.val (Tensor.dim (fun _ => Tensor.scalar (0 : α))))

  let token_tensor := Tensor.dim (fun i =>
    Tensor.scalar (tokens.getD i.val 0))

  (output_tensor, token_tensor)

/--
Complete Seq2Seq model specification (baseline).

This bundles:
- source and target embedding tables,
- an RNN encoder,
- an RNN decoder with output projection (and optional decoder self-attention).

PyTorch analogue: a small encoder-decoder model built from `nn.Embedding`, `nn.RNN`, and
  `nn.Linear`.
-/
structure Seq2SeqSpec (α : Type) [Numbers α] (srcVocabSize tgtVocabSize embedDim hiddenDim : Nat)
  where
  /-- Source embedding table + dropout configuration. -/
  src_embedding : Seq2SeqEmbeddingSpec α srcVocabSize embedDim
  /-- Target embedding table + dropout configuration. -/
  tgt_embedding : Seq2SeqEmbeddingSpec α tgtVocabSize embedDim
  /-- Encoder RNN parameters. -/
  encoder : Seq2SeqRNNEncoderSpec α embedDim hiddenDim
  /-- Decoder parameters (RNN + output projection + optional self-attention). -/
  decoder : Seq2SeqDecoderSpec α embedDim hiddenDim tgtVocabSize

/--
Seq2Seq forward pass for training (teacher forcing) using discrete token ids.

Inputs:
- `src_tokens : (srcSeqLen)` and `tgt_tokens : (tgtSeqLen)` are token id tensors.

Output:
- logits of shape `(tgtSeqLen × tgtVocabSize)`.

This path is convenient for runtime demos but intentionally treated as non-differentiable
(token-id embedding lookup is not modeled as a differentiable op).
-/
def Seq2SeqSpec.forwardTraining {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen :
  Nat}
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (src_tokens : Tensor Nat (.dim srcSeqLen .scalar))
  (tgt_tokens : Tensor Nat (.dim tgtSeqLen .scalar))
  (_h1 : srcVocabSize ≠ 0) (_h2 : tgtVocabSize ≠ 0) (_h3 : embedDim ≠ 0) (_h4 : hiddenDim ≠ 0)
  (_h5 : srcSeqLen ≠ 0) (h6 : tgtSeqLen ≠ 0) :
  Tensor α (.dim tgtSeqLen (.dim tgtVocabSize .scalar)) :=

  -- Source embeddings
  let src_embeds := Seq2SeqEmbeddingSpec.forward model.src_embedding src_tokens

  -- Encode source sequence
  let (_encoder_outputs, encoder_hidden) := Seq2SeqRNNEncoderSpec.forward model.encoder src_embeds
    none

  -- Target embeddings
  let tgt_embeds := Seq2SeqEmbeddingSpec.forward model.tgt_embedding tgt_tokens

  -- Decode with teacher forcing
  Seq2SeqDecoderSpec.forwardTeacherForcing model.decoder tgt_embeds encoder_hidden h6

/--
Seq2Seq forward pass for inference-time decoding using discrete token ids.

This embeds the source token ids, encodes them to get an initial decoder hidden state, then runs
greedy decoding for `maxTgtLen` steps starting from the given `start_token`.

Returns:
- logits `(maxTgtLen × tgtVocabSize)`,
- greedy-decoded token ids `(maxTgtLen)`.
-/
def Seq2SeqSpec.forwardInference {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen : Nat}
  (maxTgtLen : Nat)
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (src_tokens : Tensor Nat (.dim srcSeqLen .scalar))
  (start_token : Nat)
  (_h1 : srcVocabSize ≠ 0) (h2 : tgtVocabSize ≠ 0) (_h3 : embedDim ≠ 0) (_h4 : hiddenDim ≠ 0)
  (_h5 : srcSeqLen ≠ 0) (h6 : maxTgtLen ≠ 0) :
  (Tensor α (.dim maxTgtLen (.dim tgtVocabSize .scalar)) × Tensor Nat (.dim maxTgtLen .scalar)) :=

  -- Source embeddings
  let src_embeds := Seq2SeqEmbeddingSpec.forward model.src_embedding src_tokens

  -- Encode source sequence
  let (_encoder_outputs, encoder_hidden) := Seq2SeqRNNEncoderSpec.forward model.encoder src_embeds
    none

  -- Start token embedding
  let start_embed := if h : start_token < tgtVocabSize then
    match get model.tgt_embedding.embedding ⟨start_token, h⟩ with
    | Tensor.dim embed_vals => Tensor.dim embed_vals
  else
    Tensor.dim (fun _ => Tensor.scalar (0 : α))

  -- Decode with inference
  Seq2SeqDecoderSpec.forwardInference model.decoder encoder_hidden start_embed
    model.tgt_embedding.embedding maxTgtLen h6 h2

/-!
### Differentiable training + backward (one-hot inputs)

This is the “full” training interface for the Seq2Seq baseline.
-/

/--
Differentiable forward pass for training (teacher forcing) using one-hot/token-distribution inputs.

This is the same computation as `Seq2SeqSpec.forwardTraining`, except that embedding lookup is
expressed as a matrix multiplication (`forwardOnehot`), so gradients can flow into the embedding
tables and back into upstream token distributions (if desired).
-/
def Seq2SeqSpec.forwardTrainingOnehot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (src_onehot : Tensor α (.dim srcSeqLen (.dim srcVocabSize .scalar)))
  (tgt_onehot : Tensor α (.dim tgtSeqLen (.dim tgtVocabSize .scalar)))
  (_hSrc : srcSeqLen ≠ 0) (hTgt : tgtSeqLen ≠ 0) :
  Tensor α (.dim tgtSeqLen (.dim tgtVocabSize .scalar)) :=
  let src_embeds := Seq2SeqEmbeddingSpec.forwardOnehot model.src_embedding src_onehot
  let (_encOut, encHidden) := Seq2SeqRNNEncoderSpec.forward model.encoder src_embeds none
  let tgt_embeds := Seq2SeqEmbeddingSpec.forwardOnehot model.tgt_embedding tgt_onehot
  Seq2SeqDecoderSpec.forwardTeacherForcing model.decoder tgt_embeds encHidden hTgt

/--
Per-timestep cross-entropy loss for the differentiable Seq2Seq baseline.

Computes:
1. logits via `Seq2SeqSpec.forwardTrainingOnehot`,
2. probabilities via `softmax`,
3. cross-entropy against the target token distribution at each timestep.

PyTorch analogue: `nn.CrossEntropyLoss` applied per timestep (with probabilities represented as
  one-hot).
-/
def Seq2SeqSpec.crossEntropyLossOnehot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (src_onehot : Tensor α (.dim srcSeqLen (.dim srcVocabSize .scalar)))
  (tgt_onehot : Tensor α (.dim tgtSeqLen (.dim tgtVocabSize .scalar)))
  (hSrc : srcSeqLen ≠ 0) (hTgt : tgtSeqLen ≠ 0) : α :=
  let logits := Seq2SeqSpec.forwardTrainingOnehot (α := α) model src_onehot tgt_onehot hSrc hTgt
  let probs := Activation.softmaxSpec logits
  crossEntropySpec probs tgt_onehot

/--
Compute `(loss, grads)` for the Seq2Seq baseline under per-timestep cross-entropy.

This returns gradients for:
- both embedding tables,
- the encoder RNN,
- the decoder RNN,
- the decoder output projection,
- and decoder self-attention (if present).
-/
def Seq2SeqSpec.crossEntropyGradOnehot
  {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  (model : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (src_onehot : Tensor α (.dim srcSeqLen (.dim srcVocabSize .scalar)))
  (tgt_onehot : Tensor α (.dim tgtSeqLen (.dim tgtVocabSize .scalar)))
  (_hSrc : srcSeqLen ≠ 0) (hTgt : tgtSeqLen ≠ 0) :
  (α × Seq2SeqGrads α srcVocabSize tgtVocabSize embedDim hiddenDim) :=

  let src_embeds := Seq2SeqEmbeddingSpec.forwardOnehot model.src_embedding src_onehot
  let (encHiddens, encHidden) := Seq2SeqRNNEncoderSpec.forward model.encoder src_embeds none
  let tgt_embeds := Seq2SeqEmbeddingSpec.forwardOnehot model.tgt_embedding tgt_onehot

  let logits := Seq2SeqDecoderSpec.forwardTeacherForcing model.decoder tgt_embeds encHidden hTgt
  let probs := Activation.softmaxSpec logits
  let loss := crossEntropySpec probs tgt_onehot

  let dProbs := crossEntropyDerivSpec probs tgt_onehot
  let dLogits := Activation.softmaxBackwardSpec logits dProbs

  let (decRnnGrads, outProjGrads, attnGradsOpt, dTgtEmbeds, dEncHidden) :=
    Seq2SeqDecoderSpec.backwardTeacherForcing (α := α)
      (embedDim := embedDim) (hiddenDim := hiddenDim) (vocabSize := tgtVocabSize) (tgtSeqLen :=
        tgtSeqLen)
      model.decoder tgt_embeds encHidden hTgt dLogits

  let (dTgtEmbTable, _dTgtOnehot) :=
    Seq2SeqEmbeddingSpec.backwardOnehot (α := α)
      (vocabSize := tgtVocabSize) (embedDim := embedDim) (seqLen := tgtSeqLen)
      model.tgt_embedding tgt_onehot dTgtEmbeds

  -- Encoder only feeds the decoder through the final hidden state.
  let dEncHiddens :=
    if _h0 : srcSeqLen = 0 then
      fill 0 (.dim srcSeqLen (.dim hiddenDim .scalar))
    else
      Tensor.dim (fun i =>
        if _ : i.val = srcSeqLen - 1 then dEncHidden else fill 0 (.dim hiddenDim .scalar))

  let src_embeds_dropped :=
    dropoutInferenceSpec (p := model.encoder.dropout_rate) src_embeds

  let (dW_enc, db_enc, dSrcEmbeds0, _dH0_enc) :=
    rnnSequenceBackwardSpec model.encoder.rnn src_embeds_dropped (fill 0 (.dim hiddenDim .scalar))
      encHiddens dEncHiddens

  let dSrcEmbeds :=
    dropoutInferenceBackwardSpec (p := model.encoder.dropout_rate) dSrcEmbeds0

  let encGrads : Seq2SeqRNNGrads α embedDim hiddenDim := { dW := dW_enc, db := db_enc }

  let (dSrcEmbTable, _dSrcOnehot) :=
    Seq2SeqEmbeddingSpec.backwardOnehot (α := α)
      (vocabSize := srcVocabSize) (embedDim := embedDim) (seqLen := srcSeqLen)
      model.src_embedding src_onehot dSrcEmbeds

  let grads : Seq2SeqGrads α srcVocabSize tgtVocabSize embedDim hiddenDim :=
    { d_src_embedding := dSrcEmbTable
      d_tgt_embedding := dTgtEmbTable
      d_encoder := encGrads
      d_decoder_rnn := decRnnGrads
      d_output_projection := outProjGrads
      d_decoder_attention := attnGradsOpt }

  (loss, grads)

/--
Attention-augmented Seq2Seq specification (simple encoder-output attention).

This record extends the baseline with an additional projection matrix used by the helper
attention functions below (`compute_attention_weights_spec` / `apply_attention_spec`).

Note: this file includes these attention helpers as a building block; the main baseline forward
passes above do not integrate encoder-decoder cross-attention by default.
-/
structure AttentionSeq2SeqSpec (α : Type) [Numbers α] (srcVocabSize tgtVocabSize embedDim hiddenDim
  : Nat) where
  /-- Source embedding table + dropout configuration. -/
  src_embedding : Seq2SeqEmbeddingSpec α srcVocabSize embedDim
  /-- Target embedding table + dropout configuration. -/
  tgt_embedding : Seq2SeqEmbeddingSpec α tgtVocabSize embedDim
  /-- Encoder RNN parameters. -/
  encoder : Seq2SeqRNNEncoderSpec α embedDim hiddenDim
  /-- Decoder parameters (RNN + output projection + optional self-attention). -/
  decoder : Seq2SeqDecoderSpec α embedDim hiddenDim tgtVocabSize
  /-- Attention projection matrix used to score encoder outputs against the decoder hidden state. -/
  attention_weights : Tensor α (.dim hiddenDim (.dim hiddenDim .scalar))

/--
Compute attention weights over encoder outputs for a single decoder hidden state.

This is a simple dot-product style attention:
1. project the decoder hidden state (`attention_weights · decoder_hidden`),
2. score each encoder hidden vector by an elementwise product + sum,
3. normalize scores with `softmax` over the sequence axis.

It is inspired by classic encoder-decoder attention mechanisms (Bahdanau-style), but is
  intentionally
kept compact in this spec.
-/
def computeAttentionWeightsSpec {α : Type} [Context α] {hiddenDim seqLen : Nat}
  (attention_weights : Tensor α (.dim hiddenDim (.dim hiddenDim .scalar)))
  (decoder_hidden : Tensor α (.dim hiddenDim .scalar))
  (encoder_outputs : Tensor α (.dim seqLen (.dim hiddenDim .scalar)))
  (h1 : hiddenDim ≠ 0) (_h2 : seqLen ≠ 0) :
  Tensor α (.dim seqLen .scalar) :=
  -- Compute attention scores
  let projected_hidden := matVecMulSpec attention_weights decoder_hidden
  let scores := Tensor.dim (fun i =>
    match get encoder_outputs i with
    | Tensor.dim encoder_hidden =>
      let encoder_vec := Tensor.dim encoder_hidden
      let mul_vec := mulSpec projected_hidden encoder_vec
      have _inst : Shape.valid_axis_inst 0 (Shape.dim hiddenDim Shape.scalar) :=
        Shape.validAxisInstZeroAlt h1
      reduceSumAuto 0 mul_vec
  )
  -- Apply softmax to get attention weights
  Activation.softmaxSpec scores

/--
Apply attention weights to encoder outputs (weighted sum / context vector).

Given attention weights `a : (seqLen)` and encoder outputs `H : (seqLen × hiddenDim)`, returns the
context vector `c = Σ_i a_i · H_i : (hiddenDim)`.
-/
def applyAttentionSpec {hiddenDim seqLen : Nat}
  (attention_weights : Tensor α (.dim seqLen .scalar))
  (encoder_outputs : Tensor α (.dim seqLen (.dim hiddenDim .scalar)))
  (h1 : seqLen ≠ 0) (_h2 : hiddenDim ≠ 0) :
  Tensor α (.dim hiddenDim .scalar) :=
  -- Weighted sum of encoder outputs
  let weighted_outputs := Tensor.dim (fun i =>
    match get attention_weights i, get encoder_outputs i with
    | Tensor.scalar weight, Tensor.dim encoder_hidden =>
      let encoder_vec := Tensor.dim encoder_hidden
      scaleSpec encoder_vec weight
  )
  -- Sum across sequence dimension
  have _inst : Shape.valid_axis_inst 0 (Shape.dim seqLen (Shape.dim hiddenDim Shape.scalar)) :=
    Shape.validAxisInstZeroAlt h1
  reduceSumAuto 0 weighted_outputs

end Spec
