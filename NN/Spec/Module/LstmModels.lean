/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import NN.Spec.Layers.Loss
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Rnn

/-!
# LSTM models (spec)

Higher‑level LSTM architectures built from module specs (`SpecChain`), including:

- sequence‑to‑sequence outputs,
- classifier heads (many‑to‑one),
- multi‑layer compositions.

Cell equations are in `NN/Spec/Layers/Lstm.lean`; this file focuses on composing modules.

References (math + PyTorch behavior):

- Hochreiter and Schmidhuber (1997), "Long Short-Term Memory" (original LSTM):
  https://www.bioinf.jku.at/publications/older/2604.pdf
- PyTorch `nn.LSTM` docs:
  https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- PyTorch `nn.LSTMCell` docs:
  https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-!
## “Fully implemented” LSTM models

The LSTM model layer exposes first-class model objects with:

- a forward pass,
- a standard training objective, and
- an explicit reverse-mode / BPTT backward pass producing parameter gradients.

This section provides those “full” APIs for the simple LSTM models in this file by reusing the
gate-aware BPTT implementation in `NN/Spec/Layers/Lstm.lean`.
-/

/-! ### Gradient records -/

/--
Gradients for a linear layer `y = W x + b`.

This is the natural gradient bundle for `Spec.LinearSpec` (PyTorch analogue: `torch.nn.Linear`),
with `dW` matching the weight shape `[outDim, inDim]` and `db` matching `[outDim]`.
-/
structure LinearGrads (α : Type) (inDim outDim : Nat) where
  /-- d W. -/
  dW : Tensor α (.dim outDim (.dim inDim .scalar))
  /-- db. -/
  db : Tensor α (.dim outDim .scalar)

/--
Gate-wise gradients for an LSTM cell.

This matches the parameterization used by `Spec.LSTMSpec` (see `NN/Spec/Layers/Lstm.lean`): each
gate has a weight matrix of shape `[hiddenSize, inputSize + hiddenSize]` applied to a concatenated
vector, plus a bias of shape `[hiddenSize]`.
-/
structure LSTMGrads (α : Type) (inputSize hiddenSize : Nat) where
  /-- d forget weights. -/
  d_forget_weights : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- d forget bias. -/
  d_forget_bias    : HiddenVector α hiddenSize
  /-- d input weights. -/
  d_input_weights  : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- d input bias. -/
  d_input_bias     : HiddenVector α hiddenSize
  /-- d candidate weights. -/
  d_candidate_weights : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- d candidate bias. -/
  d_candidate_bias    : HiddenVector α hiddenSize
  /-- d output weights. -/
  d_output_weights : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- d output bias. -/
  d_output_bias    : HiddenVector α hiddenSize

/--
Parameter gradients for `SimpleLSTMModel`.

This bundles the LSTM cell gradients and the time-distributed linear head gradients.
-/
structure SimpleLSTMModelGrads (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- lstm. -/
  lstm : LSTMGrads α inputSize hiddenSize
  /-- output layer. -/
  output_layer : LinearGrads α hiddenSize outputSize

-- Simple LSTM model using SpecChain (following MLP pattern)
/--
Sequence-to-sequence LSTM model as a `SpecChain`: LSTM over time, then a per-timestep linear head.

PyTorch analogue: `nn.LSTM` producing an output sequence, followed by `nn.Linear` applied at each
time step.
-/
def simpleLstmModelSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (lstm_spec : LSTMSpec α inputSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  SpecChain α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
  let lstm_module := LSTMModuleSpec lstm_spec
  let linear_module := LinearSeqModuleSpec linearSpec
  SpecChain.single lstm_module
    |>.composeRight linear_module

-- LSTM classifier (many-to-one) using SpecChain
/--
Many-to-one LSTM classifier as a `SpecChain`.

This runs an LSTM over the sequence and applies a linear classifier head to the final hidden state.
PyTorch analogue: `nn.LSTM` + `nn.Linear`, taking the last output/hidden.
-/
def lstmClassifierModelSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize numClasses : Nat}
  (lstm_spec : LSTMSpec α inputSize hiddenSize)
  (classifier_spec : LinearSpec α hiddenSize numClasses)
  (h : seqLen ≠ 0) :
  SpecChain α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
  let lstm_module := LSTMModuleSpec lstm_spec
  let classifier_module := LinearClassifierModuleSpec classifier_spec h
  SpecChain.single lstm_module
    |>.composeRight classifier_module

-- Multi-layer LSTM using SpecChain composition
/--
Two-layer LSTM stack (sequence-to-sequence), followed by a per-timestep linear head.

The second LSTM consumes the hidden stream produced by the first.
-/
def multilayerLstmSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (lstm1_spec : LSTMSpec α inputSize hiddenSize)
  (lstm2_spec : LSTMSpec α hiddenSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  SpecChain α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
  let lstm1_module := LSTMModuleSpec lstm1_spec
  let lstm2_module := LSTMModuleSpec lstm2_spec
  let linear_module := LinearSeqModuleSpec linearSpec
  SpecChain.single lstm1_module
    |>.composeRight lstm2_module
    |>.composeRight linear_module

-- LSTM language model using SpecChain
/--
Simple LSTM language-model pipeline as a `SpecChain`: embedding, LSTM core, and output projection.

In this spec layer we represent the embedding/projection as `LinearSpec`s (often used with one-hot
token vectors). PyTorch analogue: `nn.Embedding` (conceptually) + `nn.LSTM` + `nn.Linear`.
-/
def lstmLanguageModelSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen vocabSize hiddenSize : Nat}
  (embedding_spec : LinearSpec α vocabSize hiddenSize)
  (lstm_spec : LSTMSpec α hiddenSize hiddenSize)
  (output_spec : LinearSpec α hiddenSize vocabSize) :
  SpecChain α (.dim seqLen (.dim vocabSize .scalar)) (.dim seqLen (.dim vocabSize .scalar)) :=
  let embedding_module := LinearSeqModuleSpec embedding_spec
  let lstm_module := LSTMModuleSpec lstm_spec
  let output_module := LinearSeqModuleSpec output_spec
  SpecChain.single embedding_module
    |>.composeRight lstm_module
    |>.composeRight output_module

-- Basic LSTM model with single layer
/--
Bundle of parameters for a single-layer LSTM model with a linear output head.

This is a direct record representation (as opposed to the `SpecChain` representation above).
-/
structure SimpleLSTMModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- lstm. -/
  lstm : LSTMSpec α inputSize hiddenSize
  /-- output layer. -/
  output_layer : LinearSpec α hiddenSize outputSize

-- Multi-layer LSTM model
/--
Bundle of parameters for a multi-layer LSTM model with a linear output head.

The first layer consumes `inputSize`, and all subsequent layers consume `hiddenSize`.
-/
structure MultiLayerLSTMModel (α : Type) (inputSize hiddenSize outputSize numLayers : Nat) where
  /-- first layer. -/
  first_layer : LSTMSpec α inputSize hiddenSize
  /-- hidden layers. -/
  hidden_layers : (i : Fin (numLayers - 1)) → LSTMSpec α hiddenSize hiddenSize
  /-- output layer. -/
  output_layer : LinearSpec α hiddenSize outputSize

-- LSTM model for classification (many-to-one)
/--
Bundle of parameters for a many-to-one LSTM classifier.

The classifier head is applied to the final hidden state.
-/
structure LSTMClassifier (α : Type) (inputSize hiddenSize numClasses : Nat) where
  /-- lstm. -/
  lstm : LSTMSpec α inputSize hiddenSize
  /-- classifier. -/
  classifier : LinearSpec α hiddenSize numClasses

-- LSTM model for sequence generation (many-to-many)
/--
Bundle of parameters for a many-to-many LSTM generator (language-model style).

This includes an (embedding) linear map, recurrent core, and output projection back to vocabulary.
-/
structure LSTMGenerator (α : Type) (vocabSize hiddenSize : Nat) where
  /-- embedding. -/
  embedding : LinearSpec α vocabSize hiddenSize  -- Simple embedding layer
  /-- lstm. -/
  lstm : LSTMSpec α hiddenSize hiddenSize
  /-- output projection. -/
  output_projection : LinearSpec α hiddenSize vocabSize

-- Bidirectional LSTM model
/--
Bundle of parameters for a bidirectional LSTM model with an output head.

The head consumes the concatenation of forward and backward hidden states.
PyTorch analogue: `nn.LSTM(..., bidirectional=true)` plus a linear projection.
-/
structure BiLSTMModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- forward lstm. -/
  forward_lstm : LSTMSpec α inputSize hiddenSize
  /-- backward lstm. -/
  backward_lstm : LSTMSpec α inputSize hiddenSize
  /-- output layer. -/
  output_layer : LinearSpec α (hiddenSize + hiddenSize) outputSize

-- LSTM Language Model
/--
Bundle of parameters for a stacked LSTM language model with deterministic dropout.

This model uses a list of LSTM layers (all with `hiddenSize` input/output) and applies a
`dropout_inference_spec` scaling step between the recurrent stack and the output projection.
-/
structure LSTMLanguageModel (α : Type) (vocabSize hiddenSize : Nat) where
  /-- embedding. -/
  embedding : LinearSpec α vocabSize hiddenSize
  /-- lstm layers. -/
  lstm_layers : List (LSTMSpec α hiddenSize hiddenSize)
  /-- output projection. -/
  output_projection : LinearSpec α hiddenSize vocabSize
  /-- dropout rate. -/
  dropout_rate : α

-- Forward pass for simple LSTM model
/--
One-step forward pass for `SimpleLSTMModel`.

Given an input vector and the previous LSTM state `(hidden, cell)`, compute `(output, new_state)`.
PyTorch analogue: `nn.LSTMCell` step followed by a `nn.Linear` head.
-/
def simpleLstmForward {inputSize hiddenSize outputSize : Nat}
  (model : SimpleLSTMModel α inputSize hiddenSize outputSize)
  (input : Tensor α (.dim inputSize .scalar))
  (state : LSTMState α hiddenSize) :
  (Tensor α (.dim outputSize .scalar) × LSTMState α hiddenSize) :=
  let new_state := lstmCellSpec model.lstm input state
  let output := linearSpec model.output_layer new_state.hidden
  (output, new_state)

-- Forward pass for simple LSTM model on sequences
/--
Sequence forward pass for `SimpleLSTMModel`.

Runs the LSTM over all timesteps (time-major), applies the output head to each hidden state, and
returns `(outputs, final_state)`.
-/
def simpleLstmSequenceForward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleLSTMModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_state : LSTMState α hiddenSize) :
  (Tensor α (.dim seqLen (.dim outputSize .scalar)) × LSTMState α hiddenSize) :=
  let (hidden_states, final_state) := lstmSequenceSpec model.lstm inputs initial_state
  -- Apply output layer to each hidden state
  let outputs := mapSequenceSpec (linearSpec model.output_layer) hidden_states
  (outputs, final_state)

/-!
### Backward pass (BPTT) for the simple LSTM sequence model

This is the model-level analogue of `Spec.lstm_sequence_backward_spec`. The only extra work we do
here is to backprop through the per-timestep output projection and feed its gradient into the LSTM
sequence backward pass.
-/

namespace LSTMModels.Internal

/-- Backprop through the time-distributed linear head and produce hidden-state gradients. -/
def timeDistributedLinearBackward
  {seqLen hiddenSize outputSize : Nat}
  (layer : LinearSpec α hiddenSize outputSize)
  (hiddens : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (grad_outputs : Tensor α (.dim seqLen (.dim outputSize .scalar))) :
  (LinearGrads α hiddenSize outputSize × Tensor α (.dim seqLen (.dim hiddenSize .scalar))) :=
  let step (acc : LinearGrads α hiddenSize outputSize × List (Tensor α (.dim hiddenSize .scalar)))
    (i : Fin seqLen) :=
    let (accLin, accDH) := acc
    let hi := get hiddens i
    let dYi := get grad_outputs i
    let (dW, db, dH) := linearBackwardSpec layer hi dYi
    ({ dW := addSpec accLin.dW dW, db := addSpec accLin.db db }, dH :: accDH)
  let init : LinearGrads α hiddenSize outputSize := {
    dW := fill 0 (.dim outputSize (.dim hiddenSize .scalar)),
    db := fill 0 (.dim outputSize .scalar)
  }
  let (linGrads, dH_rev) := (List.finRange seqLen).foldl step (init, [])
  let dH_list := dH_rev.reverse
  let dH :=
    match dH_list with
    | [] => fill 0 (.dim seqLen (.dim hiddenSize .scalar))
    | h :: _ => Tensor.dim (fun i => dH_list.getD i.val h)
  (linGrads, dH)

end LSTMModels.Internal

open LSTMModels.Internal

/--
Backward pass for `simple_lstm_sequence_forward`.

Returns:
- parameter gradients (`SimpleLSTMModelGrads`)
- gradient w.r.t. input sequence (`dInputs`)
- gradient w.r.t. initial recurrent state (`dInitialState`)
-/
def simpleLstmBackward
  {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleLSTMModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_state : LSTMState α hiddenSize)
  (grad_outputs : Tensor α (.dim seqLen (.dim outputSize .scalar))) :
  (SimpleLSTMModelGrads α inputSize hiddenSize outputSize ×
    Tensor α (.dim seqLen (.dim inputSize .scalar)) ×
    LSTMState α hiddenSize) :=
  let (hiddens, _final) := lstmSequenceSpec model.lstm inputs initial_state
  let (headGrads, dH) := timeDistributedLinearBackward (α := α) (seqLen := seqLen)
    (hiddenSize := hiddenSize) (outputSize := outputSize) model.output_layer hiddens grad_outputs
  let (dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dInputs, dInit) :=
    lstmSequenceBackwardSpec model.lstm inputs initial_state dH
  let lstmGrads : LSTMGrads α inputSize hiddenSize :=
    { d_forget_weights := dWf, d_forget_bias := dbf
      d_input_weights := dWi, d_input_bias := dbi
      d_candidate_weights := dWc, d_candidate_bias := dbc
      d_output_weights := dWo, d_output_bias := dbo }
  ({ lstm := lstmGrads, output_layer := headGrads }, dInputs, dInit)

/--
MSE loss for the simple LSTM sequence model.

This runs `simple_lstm_sequence_forward` and compares the predicted output sequence against
`targets` using `mse_spec`.
-/
def simpleLstmMseLoss
  {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleLSTMModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (targets : Tensor α (.dim seqLen (.dim outputSize .scalar)))
  (initial_state : LSTMState α hiddenSize) : α :=
  let (pred, _st) := simpleLstmSequenceForward model inputs initial_state
  mseSpec pred targets

/--
Compute `(loss, grads)` for the simple LSTM sequence model under MSE.

This is the “full training API” building block: an optimizer (SGD/Adam) can consume these grads.
-/
def simpleLstmMseGrad
  {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleLSTMModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (targets : Tensor α (.dim seqLen (.dim outputSize .scalar)))
  (initial_state : LSTMState α hiddenSize) :
  (α × SimpleLSTMModelGrads α inputSize hiddenSize outputSize) :=
  let (pred, _st) := simpleLstmSequenceForward model inputs initial_state
  let loss := mseSpec pred targets
  let dPred := mseDerivSpec pred targets
  let (grads, _dInputs, _dInit) := simpleLstmBackward (α := α) model inputs initial_state dPred
  (loss, grads)

-- Forward pass for LSTM classifier (many-to-one)
/--
Forward pass for an `LSTMClassifier` (many-to-one).

This uses the final hidden state of the LSTM sequence as the classifier input.
-/
def lstmClassifierForward {seqLen inputSize hiddenSize numClasses : Nat}
  (model : LSTMClassifier α inputSize hiddenSize numClasses)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_state : LSTMState α hiddenSize) :
  Tensor α (.dim numClasses .scalar) :=
  let (_, final_state) := lstmSequenceSpec model.lstm inputs initial_state
  -- Use final hidden state for classification
  linearSpec model.classifier final_state.hidden

/-!
### Backward for the classifier head (many-to-one)

The classifier only consumes the final hidden state. We express that by feeding a gradient sequence
that is zero everywhere except the last timestep.
-/

/--
Backward pass for an `LSTMClassifier` (many-to-one).

This backprops through the classifier head, then runs an LSTM sequence backward pass where the
hidden-state gradient is zero at all timesteps except the last.
-/
def lstmClassifierBackward
  {seqLen inputSize hiddenSize numClasses : Nat}
  (model : LSTMClassifier α inputSize hiddenSize numClasses)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_state : LSTMState α hiddenSize)
  (grad_logits : Tensor α (.dim numClasses .scalar)) :
  (LinearGrads α hiddenSize numClasses ×
    LSTMGrads α inputSize hiddenSize ×
    Tensor α (.dim seqLen (.dim inputSize .scalar))) :=
  let (hiddens, _final) := lstmSequenceSpec model.lstm inputs initial_state
  -- final hidden = last hidden in the sequence (or 0 if empty)
  let final_hidden :=
    if h0 : seqLen = 0 then
      fill 0 (.dim hiddenSize .scalar)
    else
      get hiddens ⟨seqLen - 1, by
        have : seqLen - 1 < seqLen := Nat.sub_lt (Nat.pos_of_ne_zero h0) (by decide : 0 < 1)
        simpa using this⟩
  let (dWc, dbc, dFinalHidden) := linearBackwardSpec model.classifier final_hidden grad_logits
  let linGrads : LinearGrads α hiddenSize numClasses := { dW := dWc, db := dbc }
  let dH :=
    if h0 : seqLen = 0 then
      fill 0 (.dim seqLen (.dim hiddenSize .scalar))
    else
      Tensor.dim (fun i =>
        if _ : i.val = seqLen - 1 then dFinalHidden else fill 0 (.dim hiddenSize .scalar))
  let (dWf, dbf, dWi, dbi, dWg, dbg, dWo, dbo, dInputs, _dInit) :=
    lstmSequenceBackwardSpec model.lstm inputs initial_state dH
  let lstmGrads : LSTMGrads α inputSize hiddenSize :=
    { d_forget_weights := dWf, d_forget_bias := dbf
      d_input_weights := dWi, d_input_bias := dbi
      d_candidate_weights := dWg, d_candidate_bias := dbg
      d_output_weights := dWo, d_output_bias := dbo }
  (linGrads, lstmGrads, dInputs)

-- Forward pass for LSTM generator (many-to-many)
/--
Forward pass for an `LSTMGenerator` (many-to-many).

This applies an embedding linear map to each token vector, runs the LSTM, and projects each hidden
state back into vocabulary space.
-/
def lstmGeneratorForward {seqLen vocabSize hiddenSize : Nat}
  (model : LSTMGenerator α vocabSize hiddenSize)
  (input_tokens : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (initial_state : LSTMState α hiddenSize) :
  (Tensor α (.dim seqLen (.dim vocabSize .scalar)) × LSTMState α hiddenSize) :=
  -- Apply embedding to input tokens
  let embedded := mapSequenceSpec (linearSpec model.embedding) input_tokens
  -- Process through LSTM
  let (hidden_states, final_state) := lstmSequenceSpec model.lstm embedded initial_state
  -- Project back to vocabulary space
  let outputs := mapSequenceSpec (linearSpec model.output_projection) hidden_states
  (outputs, final_state)

-- Forward pass for bidirectional LSTM
/--
Forward pass for a bidirectional LSTM model (time-major).

This runs a forward LSTM on the sequence, a backward LSTM on the reversed sequence, concatenates
the two hidden streams per timestep, and applies an output head.
-/
def bilstmForward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BiLSTMModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (forward_state : LSTMState α hiddenSize)
  (backward_state : LSTMState α hiddenSize) :
  Tensor α (.dim seqLen (.dim outputSize .scalar)) :=
  -- Forward pass
  let (forward_states, _) := lstmSequenceSpec model.forward_lstm inputs forward_state
  -- Backward pass (reverse inputs)
  let reversed_inputs := reverseSequenceSpec inputs
  let (backward_states_rev, _) := lstmSequenceSpec model.backward_lstm reversed_inputs
    backward_state
  let backward_states := reverseSequenceSpec backward_states_rev
  -- Concatenate forward and backward states
  let combined_states := concatSequenceSpec forward_states backward_states
  -- Apply output layer
  mapSequenceSpec (linearSpec model.output_layer) combined_states

-- Multi-layer LSTM forward pass (stack multiple LSTM layers)
/--
Forward pass for a `MultiLayerLSTMModel` (stacked LSTM layers).

This runs the first layer on the input sequence, then threads the resulting hidden stream through
each additional hidden layer, and finally applies the output head per timestep.
-/
def multilayerLstmForward {seqLen inputSize hiddenSize outputSize numLayers : Nat}
  (model : MultiLayerLSTMModel α inputSize hiddenSize outputSize numLayers)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_states : Fin numLayers → LSTMState α hiddenSize) (h : numLayers > 0) :
  (Tensor α (.dim seqLen (.dim outputSize .scalar)) × (Fin numLayers → LSTMState α hiddenSize)) :=
  -- Process through each layer sequentially with explicit case splitting
  let rec process_hidden_layers (layer : Nat)
    (layer_input : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
    (states : Fin numLayers → LSTMState α hiddenSize) :
    (Tensor α (.dim seqLen (.dim hiddenSize .scalar)) × (Fin numLayers → LSTMState α hiddenSize)) :=
    if hLayer : layer < numLayers - 1 then
      let layer_idx : Fin (numLayers - 1) := ⟨layer, hLayer⟩
      have h_state : layer + 1 < numLayers := by
        have h_state' : layer + 1 ≤ numLayers - 1 := Nat.succ_le_of_lt hLayer
        exact lt_of_le_of_lt h_state' (Nat.sub_one_lt (Nat.ne_of_gt h))
      let state_idx : Fin numLayers := ⟨layer + 1, h_state⟩
      let (layer_output, new_state) :=
        -- For hidden layers, we know the input size is hiddenSize
        lstmSequenceSpec (model.hidden_layers layer_idx) layer_input (states state_idx)
      let updated_states := Function.update states state_idx new_state
      process_hidden_layers (layer + 1) layer_output updated_states
    else
      (layer_input, states)

  -- Handle the first layer (layer 0) with explicit inputSize type
  let first_layer_idx : Fin numLayers := ⟨0, h⟩
  let (first_output, first_new_state) :=
    -- For layer 0, we know the input size is inputSize
    lstmSequenceSpec model.first_layer inputs (initial_states first_layer_idx)
  let updated_initial_states := Function.update initial_states first_layer_idx first_new_state

  let (final_hidden, final_states) := process_hidden_layers 0 first_output updated_initial_states
  let outputs := mapSequenceSpec (linearSpec model.output_layer) final_hidden
  (outputs, final_states)

-- LSTM Language Model forward pass with teacher forcing
/--
Forward pass for `LSTMLanguageModel` (teacher forcing, time-major).

This runs the embedding, then a stack of LSTM layers with provided initial states, applies
deterministic dropout scaling (`dropout_inference_spec`), and projects to vocabulary logits.
-/
def lstmLmForward {seqLen vocabSize hiddenSize : Nat}
  (model : LSTMLanguageModel α vocabSize hiddenSize)
  (input_tokens : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (initial_states : List (LSTMState α hiddenSize)) :
  (Tensor α (.dim seqLen (.dim vocabSize .scalar)) × List (LSTMState α hiddenSize)) :=
  -- Apply embedding
  let embedded := mapSequenceSpec (linearSpec model.embedding) input_tokens

  -- Process through LSTM layers sequentially
  let rec process_lstm_layers (layers : List (LSTMSpec α hiddenSize hiddenSize))
    (states : List (LSTMState α hiddenSize))
    (layer_input : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) :
    (Tensor α (.dim seqLen (.dim hiddenSize .scalar)) × List (LSTMState α hiddenSize)) :=
    match layers, states with
    | [], [] => (layer_input, [])
    | layer :: rest_layers, state :: rest_states =>
      let (layer_output, new_state) := lstmSequenceSpec layer layer_input state
      let (final_output, final_states) := process_lstm_layers rest_layers rest_states layer_output
      (final_output, new_state :: final_states)
    | _, _ => (layer_input, states) -- Mismatched lists

  let (lstm_output, final_states) := process_lstm_layers model.lstm_layers initial_states embedded

  -- Deterministic dropout (inference-style scaling).
  let dropped_output := dropoutInferenceSpec (p := model.dropout_rate) lstm_output

  -- Project to vocabulary
  let logits := mapSequenceSpec (linearSpec model.output_projection) dropped_output

  (logits, final_states)

-- Attention-based LSTM (simplified attention mechanism)
/--
Attention-style LSTM model bundle.

This record defines the parameters for an encoder/decoder LSTM with learned attention scores.
Forward passes can choose additive, dot-product, or domain-specific attention semantics while
sharing this typed parameter bundle.
-/
structure AttentionLSTMModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- encoder lstm. -/
  encoder_lstm : LSTMSpec α inputSize hiddenSize
  /-- decoder lstm. -/
  decoder_lstm : LSTMSpec α (inputSize + hiddenSize) hiddenSize  -- Input + context
  /-- attention weights. -/
  attention_weights : LinearSpec α (hiddenSize + hiddenSize) 1   -- For attention scoring
  /-- output layer. -/
  output_layer : LinearSpec α hiddenSize outputSize

-- Module specifications for integration with NNModuleSpec system

-- Convert SimpleLSTMModel to NNModuleSpec
/--
Package `SimpleLSTMModel` as an `NNModuleSpec`.

This is used to plug the spec model into the common module pipeline. The `export_func.toPyTorch`
field is documentation-oriented and indicates the intended PyTorch analogue.
-/
def simpleLSTMToModuleSpec {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleLSTMModel α inputSize hiddenSize outputSize) :
  NNModuleSpec α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
{
  forward := fun inputs =>
    let initial_state : LSTMState α hiddenSize := {
      hidden := fill 0 (.dim hiddenSize .scalar),
      cell := fill 0 (.dim hiddenSize .scalar)
    }
    (simpleLstmSequenceForward model inputs initial_state).1,
  kind := "SimpleLSTM",
  export_func := {
    toPyTorch :=
      s!"SimpleLSTM(input_size={inputSize}, hidden_size={hiddenSize}, output_size={outputSize})",
    dimensions := (inputSize, outputSize)
  }
}

-- Convert LSTMClassifier to NNModuleSpec
/--
Package `LSTMClassifier` as an `NNModuleSpec`.

PyTorch analogue: `nn.LSTM` feeding a `nn.Linear` classifier head.
-/
def lstmClassifierToModuleSpec {seqLen inputSize hiddenSize numClasses : Nat}
  (model : LSTMClassifier α inputSize hiddenSize numClasses) :
  NNModuleSpec α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
{
  forward := fun inputs =>
    let initial_state : LSTMState α hiddenSize := {
      hidden := fill 0 (.dim hiddenSize .scalar),
      cell := fill 0 (.dim hiddenSize .scalar)
    }
    lstmClassifierForward model inputs initial_state,
  kind := "LSTMClassifier",
  export_func := {
    toPyTorch :=
      s!"LSTMClassifier(input_size={inputSize}, hidden_size={hiddenSize}, " ++
        s!"num_classes={numClasses})",
    dimensions := (inputSize, numClasses)
  }
}

-- Convert BiLSTMModel to NNModuleSpec
/--
Package `BiLSTMModel` as an `NNModuleSpec`.

PyTorch analogue: `nn.LSTM(..., bidirectional=true)` feeding a per-timestep linear head.
-/
def biLSTMToModuleSpec {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BiLSTMModel α inputSize hiddenSize outputSize) :
  NNModuleSpec α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
{
  forward := fun inputs =>
    let initial_state : LSTMState α hiddenSize := {
      hidden := fill 0 (.dim hiddenSize .scalar),
      cell := fill 0 (.dim hiddenSize .scalar)
    }
    bilstmForward model inputs initial_state initial_state,
  kind := "BiLSTM",
  export_func := {
    toPyTorch :=
      s!"SimpleLSTM(input_size={inputSize}, hidden_size={hiddenSize}, output_size={outputSize}, " ++
        s!"bidirectional=True)",
    dimensions := (inputSize, outputSize)
  }
}

end Spec
