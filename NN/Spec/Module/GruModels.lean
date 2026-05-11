/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Rnn

/-!
# GRU models (spec)

TorchLean provides GRU layers/cells in `NN.Spec.Layers.Gru`. This file builds *models* on top of
that layer API: common compositions, heads, and a couple of end-to-end forward/backward routines.

Higher‑level GRU architectures built from module specs (`SpecChain`):

- sequence‑to‑sequence outputs,
- classifier heads (many‑to‑one),
- multi‑layer compositions.

GRU cell equations are in `NN/Spec/Layers/Gru.lean`; this file is primarily “wiring”.

References:

- Cho et al. (2014), "Learning Phrase Representations using RNN Encoder–Decoder for Statistical
  Machine Translation" (introduces GRU): https://arxiv.org/abs/1406.1078
- Chung et al. (2014), "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
  Modeling" (GRU variants/ablation): https://arxiv.org/abs/1412.3555
- PyTorch `nn.GRUCell` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
- PyTorch `nn.GRU` docs: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

PyTorch analogy: this corresponds to wiring `torch.nn.GRU` with linear heads and pooling over time
(e.g. last hidden state for classification).
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- A sequence-to-sequence GRU model, written as a `SpecChain`.

Pipeline:
`GRU(seqLen, inputSize → hiddenSize)` then `Linear` applied at each timestep.

PyTorch analogy: `nn.GRU(..., batch_first=False)` followed by an `nn.Linear` on the output sequence.
-/
def simpleGruModelSpec
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (gru_spec : GRUSpec α inputSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  SpecChain α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
  let gru_module := GRUModuleSpec gru_spec
  let linear_module := LinearSeqModuleSpec linearSpec
  SpecChain.single gru_module
    |>.composeRight linear_module

/-- A many-to-one GRU classifier (use the last hidden state, then a linear head).

PyTorch analogy: run `nn.GRU` over the sequence and feed the last output/hidden state into
`nn.Linear(hiddenSize, numClasses)`.
-/
def gruClassifierModelSpec
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize numClasses : Nat}
  (gru_spec : GRUSpec α inputSize hiddenSize)
  (classifier_spec : LinearSpec α hiddenSize numClasses)
  (h : seqLen ≠ 0) :
  SpecChain α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
  let gru_module := GRUModuleSpec gru_spec
  let classifier_module := LinearClassifierModuleSpec classifier_spec h
  SpecChain.single gru_module
    |>.composeRight classifier_module

/-- A 2-layer GRU stack (sequence-to-sequence), followed by a per-timestep linear head. -/
def multilayerGruSpec
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (gru1_spec : GRUSpec α inputSize hiddenSize)
  (gru2_spec : GRUSpec α hiddenSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  SpecChain α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
  let gru1_module := GRUModuleSpec gru1_spec
  let gru2_module := GRUModuleSpec gru2_spec
  let linear_module := LinearSeqModuleSpec linearSpec
  SpecChain.single gru1_module
    |>.composeRight gru2_module
    |>.composeRight linear_module

/-- A simple GRU language-model style pipeline:

`Linear` as a lightweight embedding/projection, then GRU, then a per-timestep projection back to
`vocabSize`.

PyTorch analogy: embedding (often `nn.Embedding`), `nn.GRU`, and `nn.Linear(hiddenSize, vocabSize)`.
We use `LinearSpec` here as a spec-friendly stand-in for a one-hot embedding matrix.
-/
def gruLanguageModelSpec
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen vocabSize hiddenSize : Nat}
  (embedding_spec : LinearSpec α vocabSize hiddenSize)
  (gru_spec : GRUSpec α hiddenSize hiddenSize)
  (output_spec : LinearSpec α hiddenSize vocabSize) :
  SpecChain α (.dim seqLen (.dim vocabSize .scalar)) (.dim seqLen (.dim vocabSize .scalar)) :=
  let embedding_module := LinearSeqModuleSpec embedding_spec
  let gru_module := GRUModuleSpec gru_spec
  let output_module := LinearSeqModuleSpec output_spec
  SpecChain.single embedding_module
    |>.composeRight gru_module
    |>.composeRight output_module

/-!
## Record-style model specs

The `SpecChain` builders above are the most uniform way to assemble models in TorchLean.

This section uses small record types with explicit forward functions. It is useful when you want
to talk about a particular architecture directly (e.g. encoder-decoder), or when you need to carry
extra per-model parameters (e.g. a dropout rate) without building a full module stack.
-/

-- Basic GRU model with a single GRU cell + a linear output head.
/--
Bundle of parameters for a single-layer GRU model with a linear output head.

This is a direct record representation (as opposed to the `SpecChain` representation above).
-/
structure SimpleGRUModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- gru. -/
  gru : GRUSpec α inputSize hiddenSize
  /-- output layer. -/
  output_layer : LinearSpec α hiddenSize outputSize

-- Multi-layer GRU model
/--
Bundle of parameters for a multi-layer GRU model.

The first layer consumes `inputSize`, and all subsequent layers consume `hiddenSize`.
-/
structure MultiLayerGRUModel (α : Type) (inputSize hiddenSize outputSize numLayers : Nat) where
  /-- first layer. -/
  first_layer : GRUSpec α inputSize hiddenSize
  /-- hidden layers. -/
  hidden_layers : (i : Fin (numLayers - 1)) → GRUSpec α hiddenSize hiddenSize
  /-- output layer. -/
  output_layer : LinearSpec α hiddenSize outputSize

-- GRU model for classification (many-to-one)
/--
Bundle of parameters for a many-to-one GRU classifier.

The classifier head is applied to the final hidden state.
-/
structure GRUClassifier (α : Type) (inputSize hiddenSize numClasses : Nat) where
  /-- gru. -/
  gru : GRUSpec α inputSize hiddenSize
  /-- classifier. -/
  classifier : LinearSpec α hiddenSize numClasses

-- GRU model for sequence generation (many-to-many)
/--
Bundle of parameters for a many-to-many GRU generator (language-model style).

This includes an (embedding) linear map, recurrent core, and output projection back to vocabulary.
-/
structure GRUGenerator (α : Type) (vocabSize hiddenSize : Nat) where
  /-- embedding. -/
  embedding : LinearSpec α vocabSize hiddenSize  -- Simple embedding layer
  /-- gru. -/
  gru : GRUSpec α hiddenSize hiddenSize
  /-- output projection. -/
  output_projection : LinearSpec α hiddenSize vocabSize

-- Bidirectional GRU model
/--
Bundle of parameters for a bidirectional GRU model with an output head.

The head consumes the concatenation of forward and backward hidden states.
PyTorch analogue: `nn.GRU(..., bidirectional=true)` plus a linear projection.
-/
structure BiGRUModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- forward gru. -/
  forward_gru : GRUSpec α inputSize hiddenSize
  /-- backward gru. -/
  backward_gru : GRUSpec α inputSize hiddenSize
  /-- output layer. -/
  output_layer : LinearSpec α (hiddenSize + hiddenSize) outputSize

-- GRU Language Model
/--
Bundle of parameters for a stacked GRU language model with deterministic dropout.

This model uses a list of GRU layers (all with `hiddenSize` input/output) and applies
`dropout_inference_spec` scaling between the GRU stack and the output projection.
-/
structure GRULanguageModel (α : Type) (vocabSize hiddenSize : Nat) where
  /-- embedding. -/
  embedding : LinearSpec α vocabSize hiddenSize
  /-- gru layers. -/
  gru_layers : List (GRUSpec α hiddenSize hiddenSize)
  /-- output projection. -/
  output_projection : LinearSpec α hiddenSize vocabSize
  /-- dropout rate. -/
  dropout_rate : α

-- GRU Encoder-Decoder Model
/--
Bundle of parameters for a GRU encoder-decoder model (seq2seq).

This uses separate embeddings and GRU cores for encoder and decoder, plus an output projection.
PyTorch analogue: an encoder `nn.GRU` and a decoder `nn.GRU` with teacher forcing.
-/
structure GRUEncoderDecoder (α : Type) (inputVocabSize hiddenSize outputVocabSize : Nat) where
  /-- encoder embedding. -/
  encoder_embedding : LinearSpec α inputVocabSize hiddenSize
  /-- encoder gru. -/
  encoder_gru : GRUSpec α hiddenSize hiddenSize
  /-- decoder embedding. -/
  decoder_embedding : LinearSpec α outputVocabSize hiddenSize
  /-- decoder gru. -/
  decoder_gru : GRUSpec α hiddenSize hiddenSize
  /-- output projection. -/
  output_projection : LinearSpec α hiddenSize outputVocabSize

/-- One-step forward for `SimpleGRUModel`.

Input: `(x_t, h_{t-1})`. Output: `(y_t, h_t)`.
-/
def simpleGruForward {inputSize hiddenSize outputSize : Nat}
  (model : SimpleGRUModel α inputSize hiddenSize outputSize)
  (input : Tensor α (.dim inputSize .scalar))
  (hidden : Tensor α (.dim hiddenSize .scalar)) :
  (Tensor α (.dim outputSize .scalar) × Tensor α (.dim hiddenSize .scalar)) :=
  let new_hidden := gruCellSpec model.gru input hidden
  let output := linearSpec model.output_layer new_hidden
  (output, new_hidden)

/-- Sequence forward for `SimpleGRUModel` (time-major).

Returns `(outputs, final_hidden)`.

PyTorch analogy: run `nn.GRU` over the sequence, then apply `nn.Linear` at each timestep.
-/
def simpleGruSequenceForward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleGRUModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim outputSize .scalar)) × Tensor α (.dim hiddenSize .scalar)) :=
  let hidden_states := gruSequenceSpec model.gru inputs initial_hidden
  -- Apply output layer to each hidden state
  let outputs := mapSequenceSpec (linearSpec model.output_layer) hidden_states
  have h_last : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let final_hidden := getAtSpec hidden_states ⟨seqLen - 1, h_last⟩
  (outputs, final_hidden)

-- Forward pass for GRU classifier (many-to-one)
/--
Forward pass for a `GRUClassifier` (many-to-one).

This runs the GRU over the input sequence and applies the classifier head to the final hidden
state.
-/
def gruClassifierForward {seqLen inputSize hiddenSize numClasses : Nat}
  (model : GRUClassifier α inputSize hiddenSize numClasses)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  Tensor α (.dim numClasses .scalar) :=
  let hidden_states := gruSequenceSpec model.gru inputs initial_hidden
  -- Use final hidden state for classification
  have h_last : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let final_hidden := getAtSpec hidden_states ⟨seqLen - 1, h_last⟩
  linearSpec model.classifier final_hidden

-- Forward pass for GRU generator (many-to-many)
/--
Forward pass for a `GRUGenerator` (many-to-many).

This applies an embedding linear map to each token vector, runs the GRU, and projects each hidden
state back into vocabulary space.
-/
def gruGeneratorForward {seqLen vocabSize hiddenSize : Nat}
  (model : GRUGenerator α vocabSize hiddenSize)
  (input_tokens : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim vocabSize .scalar)) × Tensor α (.dim hiddenSize .scalar)) :=
  -- Apply embedding to input tokens
  let embedded := mapSequenceSpec (linearSpec model.embedding) input_tokens
  -- Process through GRU
  let hidden_states := gruSequenceSpec model.gru embedded initial_hidden
  -- Project back to vocabulary space
  let outputs := mapSequenceSpec (linearSpec model.output_projection) hidden_states
  have h_last : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let final_hidden := getAtSpec hidden_states ⟨seqLen - 1, h_last⟩
  (outputs, final_hidden)

-- Forward pass for bidirectional GRU
/--
Forward pass for a bidirectional GRU model (time-major).

This runs a forward GRU on the sequence, a backward GRU on the reversed sequence, concatenates the
two hidden streams per timestep, and applies an output head.
-/
def bigruForward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BiGRUModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (forward_hidden : Tensor α (.dim hiddenSize .scalar))
  (backward_hidden : Tensor α (.dim hiddenSize .scalar)) :
  Tensor α (.dim seqLen (.dim outputSize .scalar)) :=
  -- Forward pass
  let forward_states := gruSequenceSpec model.forward_gru inputs forward_hidden
  -- Backward pass (reverse inputs)
  let reversed_inputs := reverseSequenceSpec inputs
  let backward_states_rev := gruSequenceSpec model.backward_gru reversed_inputs backward_hidden
  let backward_states := reverseSequenceSpec backward_states_rev
  -- Concatenate forward and backward states
  let combined_states := concatSequenceSpec forward_states backward_states
  -- Apply output layer
  mapSequenceSpec (linearSpec model.output_layer) combined_states

-- Multi-layer GRU forward pass (stack multiple GRU layers)
/--
Forward pass for a `MultiLayerGRUModel` (stacked GRU layers).

This runs the first layer on the input sequence, then threads the resulting hidden stream through
each additional hidden layer, and finally applies the output head per timestep.
-/
def multilayerGruForward {seqLen inputSize hiddenSize outputSize numLayers : Nat}
  (model : MultiLayerGRUModel α inputSize hiddenSize outputSize numLayers)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_hiddens : Fin numLayers → Tensor α (.dim hiddenSize .scalar)) (h : numLayers > 0) (h2 : 0
    < seqLen):
  (Tensor α (.dim seqLen (.dim outputSize .scalar)) × (Fin numLayers → Tensor α (.dim hiddenSize
    .scalar))) :=
  -- Process through each layer sequentially with explicit case splitting
  let rec process_hidden_layers (layer : Nat)
    (layer_input : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
    (hiddens : Fin numLayers → Tensor α (.dim hiddenSize .scalar)) :
    (Tensor α (.dim seqLen (.dim hiddenSize .scalar)) × (Fin numLayers → Tensor α (.dim hiddenSize
      .scalar))) :=
    if hLayer : layer < numLayers - 1 then
      let layer_idx : Fin (numLayers - 1) := ⟨layer, hLayer⟩
      have h_state : layer + 1 < numLayers := by
        have h_state' : layer + 1 ≤ numLayers - 1 := Nat.succ_le_of_lt hLayer
        exact lt_of_le_of_lt h_state' (Nat.sub_one_lt (Nat.ne_of_gt h))
      let state_idx : Fin numLayers := ⟨layer + 1, h_state⟩
      let layer_hidden := hiddens state_idx
      let layer_output := gruSequenceSpec (model.hidden_layers layer_idx) layer_input layer_hidden
      have h_last : seqLen - 1 < seqLen := by
        simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h2)
      let final_layer_hidden := getAtSpec layer_output ⟨seqLen - 1, h_last⟩
      let updated_hiddens := Function.update hiddens state_idx final_layer_hidden
      process_hidden_layers (layer + 1) layer_output updated_hiddens
    else
      (layer_input, hiddens)

  -- Handle the first layer (layer 0) with explicit inputSize type
  let first_layer_idx : Fin numLayers := ⟨0, h⟩
  let first_hidden := initial_hiddens first_layer_idx
  let first_output := gruSequenceSpec model.first_layer inputs first_hidden
  have h_last : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h2)
  let first_final_hidden := getAtSpec first_output ⟨seqLen - 1, h_last⟩
  let updated_initial_hiddens := Function.update initial_hiddens first_layer_idx first_final_hidden

  let (final_hidden_states, final_hiddens) := process_hidden_layers 0 first_output
    updated_initial_hiddens
  let outputs := mapSequenceSpec (linearSpec model.output_layer) final_hidden_states
  (outputs, final_hiddens)

-- GRU Language Model forward pass
/--
Forward pass for `GRULanguageModel` (teacher forcing, time-major).

This runs the embedding, then a stack of GRU layers with provided initial hiddens, applies
deterministic dropout scaling (`dropout_inference_spec`), and projects to vocabulary logits.
-/
def gruLmForward {seqLen vocabSize hiddenSize : Nat}
  (model : GRULanguageModel α vocabSize hiddenSize)
  (input_tokens : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (initial_hiddens : List (Tensor α (.dim hiddenSize .scalar))) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim vocabSize .scalar)) × List (Tensor α (.dim hiddenSize .scalar))) :=
  -- Apply embedding
  let embedded := mapSequenceSpec (linearSpec model.embedding) input_tokens

  -- Process through GRU layers sequentially
  let rec process_gru_layers (layers : List (GRUSpec α hiddenSize hiddenSize))
    (hiddens : List (Tensor α (.dim hiddenSize .scalar)))
    (layer_input : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) (h : 0 < seqLen) :
    (Tensor α (.dim seqLen (.dim hiddenSize .scalar)) × List (Tensor α (.dim hiddenSize .scalar)))
      :=
    match layers, hiddens with
    | [], [] => (layer_input, [])
    | layer :: rest_layers, hidden :: rest_hiddens =>
      let layer_output := gruSequenceSpec layer layer_input hidden
      have h_last : seqLen - 1 < seqLen := by
        simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
      let final_hidden := getAtSpec layer_output ⟨seqLen - 1, h_last⟩
      let (final_output, final_hiddens) := process_gru_layers rest_layers rest_hiddens layer_output
        h
      (final_output, final_hidden :: final_hiddens)
    | _, _ => (layer_input, hiddens) -- Mismatched lists

  let (gru_output, final_hiddens) := process_gru_layers model.gru_layers initial_hiddens embedded h

  -- Deterministic dropout (inference-style scaling).
  -- For a training-style variant, see `Spec.dropout_masked_spec` in `NN/Spec/Layers/Dropout.lean`.
  let dropped_output := dropoutInferenceSpec (p := model.dropout_rate) gru_output

  -- Project to vocabulary
  let logits := mapSequenceSpec (linearSpec model.output_projection) dropped_output

  (logits, final_hiddens)

-- GRU Encoder-Decoder forward pass
/-- Encoder-decoder forward pass (GRU encoder + GRU decoder).

This is a small reference architecture:

- encode `src_tokens` into a final hidden state,
- decode `tgt_tokens` starting from that hidden state (teacher forcing),
- project decoder states into output-vocabulary logits.

PyTorch analogy: `nn.GRU` encoder + `nn.GRU` decoder with a linear output projection.
-/
def gruEncoderDecoderForward {srcSeqLen tgtSeqLen inputVocabSize hiddenSize outputVocabSize :
  Nat}
  (model : GRUEncoderDecoder α inputVocabSize hiddenSize outputVocabSize)
  (src_tokens : Tensor α (.dim srcSeqLen (.dim inputVocabSize .scalar)))
  (tgt_tokens : Tensor α (.dim tgtSeqLen (.dim outputVocabSize .scalar)))
  (encoder_hidden : Tensor α (.dim hiddenSize .scalar))
  (_decoder_hidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < srcSeqLen) (h2 : 0 < tgtSeqLen) :
  (Tensor α (.dim tgtSeqLen (.dim outputVocabSize .scalar)) ×
   Tensor α (.dim hiddenSize .scalar) × Tensor α (.dim hiddenSize .scalar)) :=
  -- Encoder pass
  let src_embedded := mapSequenceSpec (linearSpec model.encoder_embedding) src_tokens
  let encoder_states := gruSequenceSpec model.encoder_gru src_embedded encoder_hidden
  have h_src_last : srcSeqLen - 1 < srcSeqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let encoder_final := getAtSpec encoder_states ⟨srcSeqLen - 1, h_src_last⟩

  -- Decoder pass (using encoder final state as initial decoder state)
  let tgt_embedded := mapSequenceSpec (linearSpec model.decoder_embedding) tgt_tokens
  let decoder_states := gruSequenceSpec model.decoder_gru tgt_embedded encoder_final
  have h_tgt_last : tgtSeqLen - 1 < tgtSeqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h2)
  let decoder_final := getAtSpec decoder_states ⟨tgtSeqLen - 1, h_tgt_last⟩

  -- Project to output vocabulary
  let outputs := mapSequenceSpec (linearSpec model.output_projection) decoder_states

  (outputs, encoder_final, decoder_final)

-- Backward pass for simple GRU model with full BPTT
/-- Backward pass for `SimpleGRUModel` (full BPTT, gate-aware).

This assumes you already ran a forward pass that saved:
- `hidden_states`,
- the GRU intermediates (`reset_gates`, `update_gates`, `new_candidates`, `reset_hiddens`).

Those intermediates can be produced using `Spec.gru_extract_intermediate_values` from
`NN.Spec.Layers.Gru`.

Return values:
- gradients for GRU parameters (reset/update/new weights + biases),
- gradients for the output linear layer (weights + bias),
- gradient for each timestep input.
-/
def simpleGruBackward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleGRUModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (hidden_states : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (grad_outputs : Tensor α (.dim seqLen (.dim outputSize .scalar)))
  -- Intermediate values from forward pass (needed for proper BPTT)
  (reset_gates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (update_gates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (new_candidates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (_reset_hiddens : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) (h : seqLen ≠ 0):
  ( Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))  -- grad_reset_weights
  × Tensor α (.dim hiddenSize .scalar)                                  -- grad_reset_bias
  × Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))  -- grad_update_weights
  × Tensor α (.dim hiddenSize .scalar)                                  -- grad_update_bias
  × Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))  -- grad_new_weights
  × Tensor α (.dim hiddenSize .scalar)                                  -- grad_new_bias
  × Tensor α (.dim outputSize (.dim hiddenSize .scalar))                -- grad_output_layer_weights
  × Tensor α (.dim outputSize .scalar)                                  -- grad_output_layer_bias
  × Tensor α (.dim seqLen (.dim inputSize .scalar)) ) :=                -- grad_inputs

  -- Backward through output layers
  let grad_hidden_from_output := mapSequenceSpec
    (fun grad_out => linearInputDerivSpec model.output_layer.weights grad_out) grad_outputs
  let grad_output_weights := reduceSumSequenceSpec2
    (map2SequenceSpec2 (Shape.dim outputSize (Shape.dim hiddenSize Shape.scalar))
      (linearWeightsDerivSpec) hidden_states grad_outputs) h
  let grad_output_bias := reduceSumSequenceSpec grad_outputs h

  -- Full BPTT backward through GRU (gate-aware, includes sigmoid/tanh chain rules).
  let initial_hidden := fill 0 (.dim hiddenSize .scalar)
  let (grad_reset_weights, grad_reset_bias,
       grad_update_weights, grad_update_bias,
       grad_new_weights, grad_new_bias,
       grad_inputs, _grad_initial_hidden) :=
    gruSequenceBackwardFullSpec model.gru inputs hidden_states grad_hidden_from_output
      reset_gates update_gates new_candidates initial_hidden

  (grad_reset_weights, grad_reset_bias, grad_update_weights, grad_update_bias,
   grad_new_weights, grad_new_bias, grad_output_weights, grad_output_bias, grad_inputs)


-- Advanced GRU with attention mechanism
/--
Attention-style GRU model bundle.

This record defines the parameters for an encoder/decoder GRU with learned attention scores.
Forward passes can choose additive, dot-product, or domain-specific attention semantics while
sharing this typed parameter bundle.
-/
structure AttentionGRUModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- encoder gru. -/
  encoder_gru : GRUSpec α inputSize hiddenSize
  /-- decoder gru. -/
  decoder_gru : GRUSpec α (inputSize + hiddenSize) hiddenSize  -- Input + context
  /-- attention weights. -/
  attention_weights : LinearSpec α (hiddenSize + hiddenSize) 1   -- For attention scoring
  /-- output layer. -/
  output_layer : LinearSpec α hiddenSize outputSize

-- GRU with residual connections
/--
Bundle of parameters for a residual GRU model.

This includes a projection from input space to hidden space so the input can be added as a residual
to the GRU hidden stream.
-/
structure ResidualGRUModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- gru. -/
  gru : GRUSpec α inputSize hiddenSize
  /-- residual projection. -/
  residual_projection : LinearSpec α inputSize hiddenSize
  -- Project input to hidden size for residual
  /-- output layer. -/
  output_layer : LinearSpec α hiddenSize outputSize

-- Residual GRU forward pass
/--
Forward pass for `ResidualGRUModel`.

This runs the GRU, adds a projected version of the input as a residual connection, and applies the
output head per timestep.
-/
def residualGruForward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : ResidualGRUModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim outputSize .scalar)) × Tensor α (.dim hiddenSize .scalar)) :=
  -- Project inputs to hidden size for residual connection
  let projected_inputs := mapSequenceSpec (linearSpec model.residual_projection) inputs
  -- Standard GRU forward pass
  let hidden_states := gruSequenceSpec model.gru inputs initial_hidden
  -- Add residual connections
  let residual_states := map2SequenceSpec2 (.dim hiddenSize .scalar) addSpec hidden_states
    projected_inputs
  -- Apply output layer
  let outputs := mapSequenceSpec (linearSpec model.output_layer) residual_states
  have h_last : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let final_hidden := getAtSpec residual_states ⟨seqLen - 1, h_last⟩
  (outputs, final_hidden)

-- Module specifications for integration with NNModuleSpec system

-- Convert SimpleGRUModel to NNModuleSpec
/--
Package `SimpleGRUModel` as an `NNModuleSpec`.

This is used to plug the spec model into the common module pipeline. The `export_func.toPyTorch`
field is documentation-oriented and indicates the intended PyTorch analogue.
-/
def simpleGRUToModuleSpec {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleGRUModel α inputSize hiddenSize outputSize) (h : 0 < seqLen) :
  NNModuleSpec α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
{
  forward := fun inputs =>
    let initial_hidden := fill 0 (.dim hiddenSize .scalar)
    (simpleGruSequenceForward model inputs initial_hidden h).1,
  kind := "SimpleGRU",
  export_func := {
    toPyTorch :=
      s!"SimpleGRU(input_size={inputSize}, hidden_size={hiddenSize}, output_size={outputSize})",
    dimensions := (inputSize, outputSize)
  }
}

-- Convert GRUClassifier to NNModuleSpec
/--
Package `GRUClassifier` as an `NNModuleSpec`.

PyTorch analogue: `nn.GRU` feeding a `nn.Linear` classifier head.
-/
def gruClassifierToModuleSpec {seqLen inputSize hiddenSize numClasses : Nat}
  (model : GRUClassifier α inputSize hiddenSize numClasses) (h : 0 < seqLen) :
  NNModuleSpec α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
{
  forward := fun inputs =>
    let initial_hidden := fill 0 (.dim hiddenSize .scalar)
    gruClassifierForward model inputs initial_hidden h,
  kind := "GRUClassifier",
  export_func := {
    toPyTorch :=
      s!"GRUClassifier(input_size={inputSize}, hidden_size={hiddenSize}, num_classes={numClasses})",
    dimensions := (inputSize, numClasses)
  }
}

-- Convert BiGRUModel to NNModuleSpec
/--
Package `BiGRUModel` as an `NNModuleSpec`.

PyTorch analogue: `nn.GRU(..., bidirectional=true)` feeding a per-timestep linear head.
-/
def biGRUToModuleSpec {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BiGRUModel α inputSize hiddenSize outputSize) :
  NNModuleSpec α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
{
  forward := fun inputs =>
    let initial_hidden := fill 0 (.dim hiddenSize .scalar)
    bigruForward model inputs initial_hidden initial_hidden,
  kind := "BiGRU",
  export_func := {
    toPyTorch :=
      s!"SimpleGRU(input_size={inputSize}, hidden_size={hiddenSize}, output_size={outputSize}, " ++
        s!"bidirectional=True)",
    dimensions := (inputSize, outputSize)
  }
}

-- Convert GRUGenerator to NNModuleSpec for language modeling
/--
Package `GRUGenerator` as an `NNModuleSpec`.

PyTorch analogue: GRU language model (`nn.GRU` + vocabulary projection) producing a sequence of
logits.
-/
def gruGeneratorToModuleSpec {seqLen vocabSize hiddenSize : Nat}
  (model : GRUGenerator α vocabSize hiddenSize) (h : 0 < seqLen) :
  NNModuleSpec α (.dim seqLen (.dim vocabSize .scalar)) (.dim seqLen (.dim vocabSize .scalar)) :=
{
  forward := fun inputs =>
    let initial_hidden := fill 0 (.dim hiddenSize .scalar)
    (gruGeneratorForward model inputs initial_hidden h).1,
  kind := "GRUGenerator",
  export_func := {
    toPyTorch := s!"GRULanguageModel(vocab_size={vocabSize}, hidden_size={hiddenSize})",
    dimensions := (vocabSize, vocabSize)
  }
}

end Spec
