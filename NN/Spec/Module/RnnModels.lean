/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Loss
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Rnn
public import NN.Spec.Module.RecurrentStack

/-!
# Recurrent Models

This file builds higher-level recurrent architectures by composing module specifications:

- sequence‑to‑sequence: RNN over inputs + per‑step linear projection,
- many‑to‑one classification: RNN + classifier head on the final hidden state,
- bidirectional variants (where supported by module specs).

The cell dynamics live in `NN.Spec.Layers.Rnn`; this module defines model-level compositions and
reference forward and backward functions.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open Spec.Module

variable {α : Type} [TorchLean.Storage α] [Context α]

namespace Rnn

/--
A simple sequence-to-sequence RNN "model wiring" expressed as a `Spec.Module.Chain`.

This composes `Spec.Module.rnn` with a linear projection applied independently at each timestep,
so the overall model maps:

- input shape:  `[seqLen, inputSize]`
- output shape: `[seqLen, outputSize]`

PyTorch analogue: applying `nn.RNN` (or a custom recurrent cell) followed by a `nn.linear` at each
time step.
-/
def sequence
  {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (rnnSpec : RNNSpec α inputSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  Spec.Module.Chain α ([seqLen, inputSize]) ([seqLen, outputSize]) :=
  let rnnModule := Spec.Module.rnn rnnSpec
  let linearModule := Spec.Module.liftLeading (Spec.Module.linear linearSpec)
  Spec.Module.Chain.single rnnModule
    |>.append linearModule

/--
A many-to-one RNN classifier expressed as a `Spec.Module.Chain`.

This runs an RNN over the input sequence and then applies a linear classifier head to the final
hidden state.

PyTorch analogue: `nn.RNN` (or `nn.GRU`/`nn.LSTM`) feeding a `nn.linear` head, taking the last
hidden/output.
-/
def classifier
  {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize numClasses : Nat}
  (rnnSpec : RNNSpec α inputSize hiddenSize)
  (classifierHead : LinearSpec α hiddenSize numClasses)
  (h : seqLen ≠ 0) :
  Spec.Module.Chain α ([seqLen, inputSize]) ([numClasses]) :=
  let rnnModule := Spec.Module.rnn rnnSpec
  let lastOutput := Spec.Module.select (shape := [seqLen, hiddenSize]) 0
    (⟨Nat.pred seqLen, Nat.pred_lt h⟩)
  let classifierModule := Spec.Module.linear classifierHead
  Spec.Module.Chain.single rnnModule
    |>.append lastOutput
    |>.append classifierModule

/-- A recurrent stack of arbitrary depth and widths, followed by a per-timestep linear head. -/
def stacked
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (layers : RecurrentStack (RNNSpec α) inputSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  Spec.Module.Chain α [seqLen, inputSize] [seqLen, outputSize] :=
  layers.toChain (fun cell => Spec.Module.rnn cell)
    (Spec.Module.liftLeading (Spec.Module.linear linearSpec))

/--
A simple RNN language model spec: "embedding" linear map, RNN core, and output projection.

This file treats embedding/projection as `LinearSpec`s. A common spec-level usage is that tokens
are one-hot vectors of length `vocabularySize`, so the embedding is just a matrix multiply.

PyTorch analogue: `nn.Embedding` (conceptually) + `nn.RNN` + `nn.linear` vocabulary projection.
-/
def languageModel
  {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen vocabularySize hiddenSize : Nat}
  (embeddingSpec : LinearSpec α vocabularySize hiddenSize)
  (rnnSpec : RNNSpec α hiddenSize hiddenSize)
  (outputSpec : LinearSpec α hiddenSize vocabularySize) :
  Spec.Module.Chain α ([seqLen, vocabularySize]) ([seqLen, vocabularySize]) :=
  let embeddingModule := Spec.Module.liftLeading (Spec.Module.linear embeddingSpec)
  let rnnModule := Spec.Module.rnn rnnSpec
  let outputModule := Spec.Module.liftLeading (Spec.Module.linear outputSpec)
  Spec.Module.Chain.single embeddingModule
    |>.append rnnModule
    |>.append outputModule

/--
Bundle of parameters for a simple single-layer RNN model with a linear output head.

This is a "record of specs" representation, as opposed to the `Spec.Module.Chain` representation
used above.
-/
structure Model (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell parameters. -/
  rnn : RNNSpec α inputSize hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

/-- Parameter gradients for a recurrent model and its linear output head.

Both halves reuse the gradient records of the layers they belong to, so a caller that already knows
how to read `Spec.RNNParameterGradients` needs to learn nothing new here. -/
structure Grads (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize outputSize : Nat) where
  /-- Gradients of the recurrent cell parameters. -/
  rnn : RNNParameterGradients α inputSize hiddenSize
  /-- Gradients of the linear output projection. -/
  output : LinearParameterGradients α hiddenSize outputSize

/-- Recurrent cells with independently chosen widths and a linear output head.

The endpoint `hiddenSize` is the width of the last cell.
An empty stack has `hiddenSize = inputSize`.
-/
structure StackedModel (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize outputSize : Nat) where
  /-- Layer dimensions determine the type of every intermediate hidden stream and state. -/
  layers : RecurrentStack (RNNSpec α) inputSize hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

/-- Run every layer with its own initial state, then project the final hidden stream.

The result retains all final states. Empty sequences leave each initial state unchanged, and an
empty stack applies the head directly to the input sequence.
-/
def StackedModel.forward {seqLen inputSize hiddenSize outputSize : Nat}
    (model : StackedModel α inputSize hiddenSize outputSize)
    (inputs : Tensor α [seqLen, inputSize])
    (initialHiddens : model.layers.States (fun width => Tensor α [width])) :
    Tensor α [seqLen, outputSize] × model.layers.States (fun width => Tensor α [width]) :=
  let (hidden, finalStates) := model.layers.run
    (fun cell input state =>
      let outputs := rnnSequenceSpec cell input state
      let finalState := if h : seqLen = 0 then state else
        get outputs ⟨seqLen - 1, Nat.sub_one_lt h⟩
      (outputs, finalState)) inputs initialHiddens
  (Tensor.mapLeading [seqLen] (linearSpec model.outputLayer) hidden, finalStates)

/--
Bundle of parameters for a many-to-one RNN classifier.

The classifier head is a linear layer applied to the final hidden state.
-/
structure Classifier (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize numClasses : Nat) where
  /-- Recurrent cell parameters. -/
  rnn : RNNSpec α inputSize hiddenSize
  /-- Linear classifier head. -/
  classifier : LinearSpec α hiddenSize numClasses

/--
Bundle of parameters for a many-to-many RNN generator (language-model style).

This includes a (linear) embedding, recurrent core, and output projection back to vocabulary.
-/
structure Generator (α : Type) [TorchLean.Storage α]
    (vocabularySize hiddenSize : Nat) where
  /-- Token projection used by this one-hot specification. -/
  embedding : LinearSpec α vocabularySize hiddenSize
  /-- Recurrent cell parameters. -/
  rnn : RNNSpec α hiddenSize hiddenSize
  /-- Projection from hidden states to vocabulary logits. -/
  outputProjection : LinearSpec α hiddenSize vocabularySize

/--
Bundle of parameters for a bidirectional RNN model with an output head.

The output head consumes the concatenation of forward and backward hidden states.
-/
structure BidirectionalModel (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell for the original sequence order. -/
  forwardRnn : RNNSpec α inputSize hiddenSize
  /-- Recurrent cell for the reversed sequence order. -/
  backwardRnn : RNNSpec α inputSize hiddenSize
  /-- Projection from concatenated forward and backward states. -/
  outputLayer : LinearSpec α (hiddenSize + hiddenSize) outputSize

/--
One-step forward pass for `Rnn.Model`.

Given an input vector and current hidden state, compute the output and next state using the RNN cell
and the linear head.
-/
def Model.forward {inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (input : Tensor α [inputSize])
  (hidden : Tensor α [hiddenSize]) :
  (Tensor α [outputSize] × Tensor α [hiddenSize]) :=
  let nextHidden := rnnCellSpec model.rnn input hidden
  let output := linearSpec model.outputLayer nextHidden
  (output, nextHidden)

/--
Sequence forward pass for `Rnn.Model`.

Runs the recurrent cell over the full sequence, applies the output layer at each time step, and
returns both the per-step outputs and the final hidden state. Empty sequences preserve the
initial hidden state.
-/
def Model.forwardSequence {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize]) :
  (Tensor α [seqLen, outputSize] × Tensor α [hiddenSize]) :=
  let stack : StackedModel α inputSize hiddenSize outputSize :=
    { layers := .cons model.rnn .nil, outputLayer := model.outputLayer }
  let (outputs, finalStates) := stack.forward inputs (initialHidden, ())
  (outputs, finalStates.1)

/--
Forward pass for an `Rnn.Classifier` (many-to-one).

This runs the recurrent core over the input sequence and feeds the last hidden state to the
classifier head.
-/
def Classifier.forward {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize]) (h : 0 < seqLen) :
  Tensor α [numClasses] :=
  let hiddenStates := rnnSequenceSpec model.rnn inputs initialHidden
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := get hiddenStates ⟨seqLen - 1, hLast⟩
  linearSpec model.classifier finalHidden

/--
Forward pass for an `Rnn.Generator` (many-to-many).

This applies an "embedding" linear map to each token, runs the RNN, and projects each hidden state
back into vocabulary space.
-/
def Generator.forward {seqLen vocabularySize hiddenSize : Nat}
  (model : Generator α vocabularySize hiddenSize)
  (inputTokens : Tensor α [seqLen, vocabularySize])
  (initialHidden : Tensor α [hiddenSize]) (h : 0 < seqLen) :
  (Tensor α [seqLen, vocabularySize] × Tensor α [hiddenSize]) :=
  let embedded := Tensor.mapLeading ([seqLen]) (linearSpec model.embedding) inputTokens
  let hiddenStates := rnnSequenceSpec model.rnn embedded initialHidden
  let outputs := Tensor.mapLeading ([seqLen])
    (linearSpec model.outputProjection) hiddenStates
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := get hiddenStates ⟨seqLen - 1, hLast⟩
  (outputs, finalHidden)

/--
Forward pass for a bidirectional RNN model.

This runs a forward RNN on the sequence, a backward RNN on the reversed sequence, concatenates the
two state streams per time step, and applies the output head.
-/
def BidirectionalModel.forward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BidirectionalModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (forwardHidden : Tensor α [hiddenSize])
  (backwardHidden : Tensor α [hiddenSize]) :
  Tensor α [seqLen, outputSize] :=
  let forwardStates := rnnSequenceSpec model.forwardRnn inputs forwardHidden
  let reversedInputs := Tensor.reverseAxis 0 inputs
  let reversedBackwardStates := rnnSequenceSpec model.backwardRnn reversedInputs backwardHidden
  let backwardStates := Tensor.reverseAxis 0 reversedBackwardStates
  let combinedStates := Tensor.zipEach ([seqLen])
    ([(hiddenSize + hiddenSize)])
    (Tensor.concatAxisSpec .scalar) forwardStates backwardStates
  Tensor.mapLeading ([seqLen]) (linearSpec model.outputLayer) combinedStates

-- One-step helper used by some compact examples (single cell update + output projection).
/--
One-step helper: run a single RNN cell update and apply an output projection.

This is used by some compact examples that do not build a full `Spec.Module.Chain` or multi-layer
bundle.
-/
def cellWithHead {inputSize hiddenSize outputSize : Nat}
  (cell : RNNSpec α inputSize hiddenSize)
  (outputLayer : LinearSpec α hiddenSize outputSize)
  (inputs : Tensor α [inputSize])
  (hidden : Tensor α [hiddenSize]) :
  (Tensor α [outputSize] × Tensor α [hiddenSize]) :=
  let nextHidden := rnnCellSpec cell inputs hidden
  let output := linearSpec outputLayer nextHidden
  (output, nextHidden)

-- Backward pass for simple RNN model
/--
Backward pass for `Rnn.Model` over a full sequence.

Returns parameter gradients together with the gradient of the input sequence.

This is a spec-level reference implementation; performance is not a goal here.
-/
def Model.backward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (hiddenStates : Tensor α [seqLen, hiddenSize])
  (gradOutputs : Tensor α [seqLen, outputSize]) (h : 0 < seqLen) :
  Grads α inputSize hiddenSize outputSize ×
    Tensor α [seqLen, inputSize] :=

  let gradHiddenFromOutput := Tensor.mapLeading ([seqLen])
    (fun gradOutput => linearInputDerivSpec model.outputLayer.weights gradOutput) gradOutputs
  let gradOutputWeights := Tensor.reduceSum 0
    (Tensor.zipEach ([seqLen])
      [outputSize, hiddenSize]
      linearWeightsDerivSpec hiddenStates gradOutputs)
    (Shape.hasNonemptyAxisZeroOfNe h.ne').proof
  let gradOutputBias := Tensor.reduceSum 0 gradOutputs
    (Shape.hasNonemptyAxisZeroOfNe h.ne').proof

  let initialHidden := Tensor.full ([hiddenSize]) 0
  let rnnBackward :=
    rnnSequenceBackwardSpec model.rnn inputs initialHidden hiddenStates gradHiddenFromOutput

  ( { rnn := rnnBackward.parameters
      output :=
        { weightGradient := gradOutputWeights
          biasGradient := gradOutputBias } },
    rnnBackward.inputs )

namespace Internal

/--
Map a scalar-valued function over two aligned sequences, producing a sequence of scalars.

This helper lifts a scalar comparison over aligned sequence elements.
-/
def mapSequence2 {seqLen dim1 dim2 : Nat}
  (f : Tensor α [dim1] → Tensor α [dim2] → α)
  (leftSeq : Tensor α [seqLen, dim1])
  (rightSeq : Tensor α [seqLen, dim2]) :
  Tensor α [seqLen] :=
  Tensor.dim (fun i =>
    Tensor.scalar (f (Tensor.unstack leftSeq i) (Tensor.unstack rightSeq i)))

end Internal

-- Loss function for sequence classification
/--
Mean cross-entropy loss over a sequence of class-probability predictions.

This is the spec-level analogue of a per-time-step classification loss, averaged across steps.
PyTorch analogue: `torch.nn.CrossEntropyLoss` applied per step and then averaged.
-/
def classificationLoss {seqLen numClasses : Nat}
  [Shape.HasNonemptyAxis 0 ([numClasses])]
  (predictions : Tensor α [seqLen, numClasses])
  (targets : Tensor α [seqLen, numClasses]) :
  α :=
  let losses := Internal.mapSequence2 (crossEntropySpec 0) predictions targets
  meanSpec losses

/--
Package an `Rnn.Model` as a shape-indexed module.

The Python expression records the intended runtime analogue; `forward` remains the mathematical
meaning of the module.
-/
def Model.toModule {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize) :
  Spec.Module α ([seqLen, inputSize]) ([seqLen, outputSize]) :=
{
  forward := fun inputs =>
    let initialHidden := Tensor.full ([hiddenSize]) 0
    (model.forwardSequence inputs initialHidden).1,
  kind := "SimpleRNN",
  pythonExpr :=
    s!"SimpleRNN(input_size={inputSize}, hidden_size={hiddenSize}, " ++
      s!"output_size={outputSize})"
}

/--
Package an `Rnn.Classifier` as an `Spec.Module`.

This plugs the classifier into the common module pipeline and records a PyTorch-oriented summary.
-/
def Classifier.toModule {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses) (h : 0 < seqLen) :
  Spec.Module α ([seqLen, inputSize]) ([numClasses]) :=
{
  forward := fun inputs =>
    let initialHidden := Tensor.full ([hiddenSize]) 0
    model.forward inputs initialHidden h,
  kind := "RNNClassifier",
  pythonExpr :=
    s!"RNNClassifier(input_size={inputSize}, hidden_size={hiddenSize}, " ++
      s!"num_classes={numClasses})"
}

end Rnn

end Spec
