/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Loss
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Rnn

/-!
# RNN models (spec)

This file builds higher‑level RNN architectures by composing module specs (`SpecChain`), e.g.:

- sequence‑to‑sequence: RNN over inputs + per‑step linear projection,
- many‑to‑one classification: RNN + classifier head on the final hidden state,
- bidirectional variants (where supported by module specs).

The actual cell dynamics live in `NN/Spec/Layers/Rnn.lean`; this file is "model wiring".
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/--
A simple sequence-to-sequence RNN "model wiring" expressed as a `SpecChain`.

This composes an `RNNModuleSpec` with a per-time-step linear projection (`LinearSeqModuleSpec`),
so the overall model maps:

- input shape:  `[seqLen, inputSize]`
- output shape: `[seqLen, outputSize]`

PyTorch analogue: applying `nn.RNN` (or a custom recurrent cell) followed by a `nn.Linear` at each
time step.
-/
def simpleRnnModelSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (rnn_spec : RNNSpec α inputSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  SpecChain α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
  let rnn_module := RNNModuleSpec rnn_spec
  let linear_module := LinearSeqModuleSpec linearSpec
  SpecChain.single rnn_module
    |>.composeRight linear_module

/--
A many-to-one RNN classifier expressed as a `SpecChain`.

This runs an RNN over the input sequence and then applies a linear classifier head to the final
hidden state.

PyTorch analogue: `nn.RNN` (or `nn.GRU`/`nn.LSTM`) feeding a `nn.Linear` head, taking the last
hidden/output.
-/
def rnnClassifierModelSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize numClasses : Nat}
  (rnn_spec : RNNSpec α inputSize hiddenSize)
  (classifier_spec : LinearSpec α hiddenSize numClasses)
  (h : seqLen ≠ 0) :
  SpecChain α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
  let rnn_module := RNNModuleSpec rnn_spec
  let classifier_module := LinearClassifierModuleSpec classifier_spec h
  SpecChain.single rnn_module
    |>.composeRight classifier_module


/--
A bidirectional LSTM classifier expressed as a `SpecChain`.

This uses a `BiLSTMModuleSpec` to combine forward/backward LSTM passes, concatenates the two
hidden-state streams, and applies a linear classifier head.

PyTorch analogue: `nn.LSTM(bidirectional=true)` followed by a `nn.Linear` classifier.
-/
def bilstmClassifierSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize numClasses : Nat}
  (forward_lstm : LSTMSpec α inputSize hiddenSize)
  (backward_lstm : LSTMSpec α inputSize hiddenSize)
  (classifier_spec : LinearSpec α (hiddenSize + hiddenSize) numClasses)
  (h : seqLen ≠ 0) :
  SpecChain α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
  -- Bidirectional LSTM + classifier head.
  -- The actual BiLSTM module wrapper is provided by `BiLSTMModuleSpec`.
  let lstm_module := BiLSTMModuleSpec forward_lstm backward_lstm
  let linear_module := LinearClassifierModuleSpec classifier_spec h
  SpecChain.single lstm_module
    |>.composeRight linear_module

/--
A two-layer RNN encoder with a per-step linear projection, expressed as a `SpecChain`.

The second recurrent layer consumes the hidden stream of the first.
-/
def multilayerRnnSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (rnn1_spec : RNNSpec α inputSize hiddenSize)
  (rnn2_spec : RNNSpec α hiddenSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  SpecChain α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
  let rnn1_module := RNNModuleSpec rnn1_spec
  let rnn2_module := RNNModuleSpec rnn2_spec
  let linear_module := LinearSeqModuleSpec linearSpec
  SpecChain.single rnn1_module
    |>.composeRight rnn2_module
    |>.composeRight linear_module

/--
A simple RNN language model spec: "embedding" linear map, RNN core, and output projection.

This file treats embedding/projection as `LinearSpec`s. A common spec-level usage is that tokens
are one-hot vectors of length `vocabSize`, so the embedding is just a matrix multiply.

PyTorch analogue: `nn.Embedding` (conceptually) + `nn.RNN` + `nn.Linear` vocabulary projection.
-/
def rnnLanguageModelSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen vocabSize hiddenSize : Nat}
  (embedding_spec : LinearSpec α vocabSize hiddenSize)
  (rnn_spec : RNNSpec α hiddenSize hiddenSize)
  (output_spec : LinearSpec α hiddenSize vocabSize) :
  SpecChain α (.dim seqLen (.dim vocabSize .scalar)) (.dim seqLen (.dim vocabSize .scalar)) :=
  let embedding_module := LinearSeqModuleSpec embedding_spec
  let rnn_module := RNNModuleSpec rnn_spec
  let output_module := LinearSeqModuleSpec output_spec
  SpecChain.single embedding_module
    |>.composeRight rnn_module
    |>.composeRight output_module

/--
Bundle of parameters for a simple single-layer RNN model with a linear output head.

This is a "record of specs" representation, as opposed to the `SpecChain` representation used
above.
-/
structure SimpleRNNModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- rnn. -/
  rnn : RNNSpec α inputSize hiddenSize
  /-- output layer. -/
  output_layer : LinearSpec α hiddenSize outputSize

/--
Bundle of parameters for a multi-layer RNN model.

`layers` is indexed by `Fin numLayers` and selects the appropriate input size for the first layer
versus subsequent layers.
-/
structure MultiLayerRNNModel (α : Type) (inputSize hiddenSize outputSize numLayers : Nat) where
  /-- Layer stack. -/
  layers :
    (i : Fin numLayers) →
      RNNSpec α (if i.val = 0 then inputSize else hiddenSize) hiddenSize
  /-- output layer. -/
  output_layer : LinearSpec α hiddenSize outputSize

/--
Bundle of parameters for a many-to-one RNN classifier.

The classifier head is a linear layer applied to the final hidden state.
-/
structure RNNClassifier (α : Type) (inputSize hiddenSize numClasses : Nat) where
  /-- rnn. -/
  rnn : RNNSpec α inputSize hiddenSize
  /-- classifier. -/
  classifier : LinearSpec α hiddenSize numClasses

/--
Bundle of parameters for a many-to-many RNN generator (language-model style).

This includes a (linear) embedding, recurrent core, and output projection back to vocabulary.
-/
structure RNNGenerator (α : Type) (vocabSize hiddenSize : Nat) where
  /-- embedding. -/
  embedding : LinearSpec α vocabSize hiddenSize  -- Simple embedding layer
  /-- rnn. -/
  rnn : RNNSpec α hiddenSize hiddenSize
  /-- output projection. -/
  output_projection : LinearSpec α hiddenSize vocabSize

/--
Bundle of parameters for a bidirectional RNN model with an output head.

The output head consumes the concatenation of forward and backward hidden states.
-/
structure BiRNNModel (α : Type) (inputSize hiddenSize outputSize : Nat) where
  /-- forward rnn. -/
  forward_rnn : RNNSpec α inputSize hiddenSize
  /-- backward rnn. -/
  backward_rnn : RNNSpec α inputSize hiddenSize
  /-- output layer. -/
  output_layer : LinearSpec α (hiddenSize + hiddenSize) outputSize

/--
One-step forward pass for `SimpleRNNModel`.

Given an input vector and current hidden state, compute `(output, new_hidden)` using the RNN cell
and the linear head.
-/
def simpleRnnForward {inputSize hiddenSize outputSize : Nat}
  (model : SimpleRNNModel α inputSize hiddenSize outputSize)
  (input : Tensor α (.dim inputSize .scalar))
  (hidden : Tensor α (.dim hiddenSize .scalar)) :
  (Tensor α (.dim outputSize .scalar) × Tensor α (.dim hiddenSize .scalar)) :=
  let new_hidden := rnnCellSpec model.rnn input hidden
  let output := linearSpec model.output_layer new_hidden
  (output, new_hidden)

/--
Sequence forward pass for `SimpleRNNModel`.

Runs `rnn_sequence_spec` over the full sequence, applies the output layer at each time step, and
returns both the per-step outputs and the final hidden state.
-/
def simpleRnnSequenceForward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleRNNModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim outputSize .scalar)) × Tensor α (.dim hiddenSize .scalar))  :=
  let hidden_states := rnnSequenceSpec model.rnn inputs initial_hidden
  -- Apply output layer to each hidden state
  let outputs := mapSequenceSpec (linearSpec model.output_layer) hidden_states
  have h_last : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let final_hidden := getAtSpec hidden_states ⟨seqLen - 1, h_last⟩
  (outputs, final_hidden)

/--
Forward pass for an `RNNClassifier` (many-to-one).

This runs the recurrent core over the input sequence and feeds the last hidden state to the
classifier head.
-/
def rnnClassifierForward {seqLen inputSize hiddenSize numClasses : Nat}
  (model : RNNClassifier α inputSize hiddenSize numClasses)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  Tensor α (.dim numClasses .scalar) :=
  let hidden_states := rnnSequenceSpec model.rnn inputs initial_hidden
  -- Use final hidden state for classification
  have h_last : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let final_hidden := getAtSpec hidden_states ⟨seqLen - 1, h_last⟩
  linearSpec model.classifier final_hidden

/--
Forward pass for an `RNNGenerator` (many-to-many).

This applies an "embedding" linear map to each token, runs the RNN, and projects each hidden state
back into vocabulary space.
-/
def rnnGeneratorForward {seqLen vocabSize hiddenSize : Nat}
  (model : RNNGenerator α vocabSize hiddenSize)
  (input_tokens : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar)) (h : 0 < seqLen) :
  (Tensor α (.dim seqLen (.dim vocabSize .scalar)) × Tensor α (.dim hiddenSize .scalar)) :=
  -- Apply embedding to input tokens
  let embedded := mapSequenceSpec (linearSpec model.embedding) input_tokens
  -- Process through RNN
  let hidden_states := rnnSequenceSpec model.rnn embedded initial_hidden
  -- Project back to vocabulary space
  let outputs := mapSequenceSpec (linearSpec model.output_projection) hidden_states
  have h_last : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let final_hidden := getAtSpec hidden_states ⟨seqLen - 1, h_last⟩
  (outputs, final_hidden)

/--
Forward pass for a bidirectional RNN model.

This runs a forward RNN on the sequence, a backward RNN on the reversed sequence, concatenates the
two state streams per time step, and applies the output head.
-/
def birnnForward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BiRNNModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (forward_hidden : Tensor α (.dim hiddenSize .scalar))
  (backward_hidden : Tensor α (.dim hiddenSize .scalar)) :
  Tensor α (.dim seqLen (.dim outputSize .scalar)) :=
  -- Forward pass
  let forward_states := rnnSequenceSpec model.forward_rnn inputs forward_hidden
  -- Backward pass (reverse inputs)
  let reversed_inputs := reverseSequenceSpec inputs
  let backward_states_rev := rnnSequenceSpec model.backward_rnn reversed_inputs backward_hidden
  let backward_states := reverseSequenceSpec backward_states_rev
  -- Concatenate forward and backward states
  let combined_states := concatSequenceSpec forward_states backward_states
  -- Apply output layer
  mapSequenceSpec (linearSpec model.output_layer) combined_states

-- One-step helper used by some small examples (single cell update + output projection).
/--
One-step helper: run a single RNN cell update and apply an output projection.

This is used by some small examples that do not build a full `SpecChain` or multi-layer bundle.
-/
def multilayerRnnForwardSingle {inputSize hiddenSize outputSize : Nat}
  (layers : RNNSpec α inputSize hiddenSize)
  (output_layer : LinearSpec α hiddenSize outputSize)
  (inputs : Tensor α (.dim inputSize .scalar))
  (hidden : Tensor α (.dim hiddenSize .scalar)) :
  (Tensor α (.dim outputSize .scalar) × Tensor α (.dim hiddenSize .scalar)) :=
  let new_hidden := rnnCellSpec layers inputs hidden
  let output := linearSpec output_layer new_hidden
  (output, new_hidden)

-- Backward pass for simple RNN model
/--
Backward pass for `SimpleRNNModel` over a full sequence.

Returns gradients for:
- the RNN cell parameters,
- the output head parameters, and
- the input sequence.

This is a spec-level reference implementation; performance is not a goal here.
-/
def simpleRnnBackward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleRNNModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (hidden_states : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (grad_outputs : Tensor α (.dim seqLen (.dim outputSize .scalar))) (h : 0 < seqLen) :
  ( Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))  -- grad_rnn_weights
  × Tensor α (.dim hiddenSize .scalar)                                  -- grad_rnn_bias
  × Tensor α (.dim outputSize (.dim hiddenSize .scalar))                -- grad_output_weights
  × Tensor α (.dim outputSize .scalar)                                  -- grad_output_bias
  × Tensor α (.dim seqLen (.dim inputSize .scalar)) ) :=                -- grad_inputs

  -- Backward through output layers
  let grad_hidden_from_output := mapSequenceSpec
    (fun grad_out => linearInputDerivSpec model.output_layer.weights grad_out) grad_outputs
  let grad_output_weights := reduceSumSequenceSpec2
    (map2SequenceSpec2 (Shape.dim outputSize (Shape.dim hiddenSize Shape.scalar))
      (linearWeightsDerivSpec) hidden_states grad_outputs) h.ne'
  let grad_output_bias := reduceSumSequenceSpec grad_outputs h.ne'

  -- Backward through RNN
  let initial_hidden := fill 0 (.dim hiddenSize .scalar)
  let (grad_rnn_weights, grad_rnn_bias, grad_inputs, _grad_initial_hidden) :=
    rnnSequenceBackwardSpec model.rnn inputs initial_hidden hidden_states grad_hidden_from_output

  (grad_rnn_weights, grad_rnn_bias, grad_output_weights, grad_output_bias, grad_inputs)

-- Training utilities
-- Apply a binary function returning a scalar to corresponding elements in two sequences
/--
Map a scalar-valued function over two aligned sequences, producing a sequence of scalars.

This is a lightweight helper used in the example loss definitions below.
-/
def map2SequenceSpec {seqLen dim1 dim2 : Nat}
  (f : Tensor α (.dim dim1 .scalar) → Tensor α (.dim dim2 .scalar) → α)
  (seq1 : Tensor α (.dim seqLen (.dim dim1 .scalar)))
  (seq2 : Tensor α (.dim seqLen (.dim dim2 .scalar))) :
  Tensor α (.dim seqLen .scalar) :=
  match seq1, seq2 with
  | Tensor.dim fn1, Tensor.dim fn2 =>
    Tensor.dim (fun i => Tensor.scalar (f (fn1 i) (fn2 i)))

-- Loss function for sequence classification
/--
Mean cross-entropy loss over a sequence of class-probability predictions.

This is the spec-level analogue of a per-time-step classification loss, averaged across steps.
PyTorch analogue: `torch.nn.CrossEntropyLoss` applied per step and then averaged.
-/
def sequenceClassificationLoss {seqLen numClasses : Nat}
  (predictions : Tensor α (.dim seqLen (.dim numClasses .scalar)))
  (targets : Tensor α (.dim seqLen (.dim numClasses .scalar))) :
  α :=
  let losses := map2SequenceSpec crossEntropySpec predictions targets
  meanSpec losses

-- Loss function for language modeling
/--
Language-modeling loss (alias of `sequence_classification_loss`).

This treats each time step as a vocabulary classification task.
-/
def languageModelingLoss {seqLen vocabSize : Nat}
  (predictions : Tensor α (.dim seqLen (.dim vocabSize .scalar)))
  (targets : Tensor α (.dim seqLen (.dim vocabSize .scalar))) :
  α :=
  sequenceClassificationLoss predictions targets

-- Module specifications for integration with NNModuleSpec system

-- Convert SimpleRNNModel to NNModuleSpec
/--
Package a `SimpleRNNModel` as an `NNModuleSpec` for use with the module-spec tooling.

The `export_func.toPyTorch` string is documentation-oriented: it indicates the intended PyTorch
analogue, but is not itself an executable exporter.
-/
def simpleRNNToModuleSpec {seqLen inputSize hiddenSize outputSize : Nat}
  (model : SimpleRNNModel α inputSize hiddenSize outputSize) (h : 0 < seqLen) :
  NNModuleSpec α (.dim seqLen (.dim inputSize .scalar)) (.dim seqLen (.dim outputSize .scalar)) :=
{
  forward := fun inputs =>
    let initial_hidden := fill 0 (.dim hiddenSize .scalar)
    (simpleRnnSequenceForward model inputs initial_hidden h).1,
  kind := "SimpleRNN",
  export_func := {
    toPyTorch :=
      s!"SimpleRNN(input_size={inputSize}, hidden_size={hiddenSize}, output_size={outputSize})",
    dimensions := (inputSize, outputSize)
  }
}

-- Convert RNNClassifier to NNModuleSpec
/--
Package an `RNNClassifier` as an `NNModuleSpec`.

As with `simpleRNNToModuleSpec`, this is primarily used to plug the spec model into the common
module pipeline (and to generate a PyTorch-analogue string for documentation).
-/
def rnnClassifierToModuleSpec {seqLen inputSize hiddenSize numClasses : Nat}
  (model : RNNClassifier α inputSize hiddenSize numClasses) (h : 0 < seqLen) :
  NNModuleSpec α (.dim seqLen (.dim inputSize .scalar)) (.dim numClasses .scalar) :=
{
  forward := fun inputs =>
    let initial_hidden := fill 0 (.dim hiddenSize .scalar)
    rnnClassifierForward model inputs initial_hidden h,
  kind := "RNNClassifier",
  export_func := {
    toPyTorch :=
      s!"RNNClassifier(input_size={inputSize}, hidden_size={hiddenSize}, num_classes={numClasses})",
    dimensions := (inputSize, numClasses)
  }
}

end Spec
