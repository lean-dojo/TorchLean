/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Rnn
public import NN.Spec.Module.RecurrentStack

/-!
# Gated Recurrent Models

TorchLean provides GRU layers/cells in `NN.Spec.Layers.Gru`. This file builds *models* on top of
that layer API: common compositions, heads, and a couple of end-to-end forward/backward routines.

Higher‑level GRU architectures built from module specs (`Spec.Module.Chain`):

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
(e.g. last hidden state for classification). This is an architectural comparison only. The
recurrent core is the original Cho reset-before GRU from `NN.Spec.Layers.Gru`; PyTorch uses a
reset-after candidate-state equation, so its checkpoints are not equation-compatible with these
models without an explicit conversion.
-/

@[expose] public section

open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open Spec.Module

variable {α : Type} [TorchLean.Storage α] [Context α]

namespace Gru

/-- A sequence-to-sequence GRU model, written as a `Spec.Module.Chain`.

Pipeline:
`GRU(seqLen, inputSize → hiddenSize)` then `Linear` applied at each timestep.

PyTorch analogy: `nn.GRU(..., batch_first=False)` followed by an `nn.linear` on the output sequence.
-/
def sequence
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (gruSpec : GRUSpec α inputSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  Spec.Module.Chain α ([seqLen, inputSize]) ([seqLen, outputSize]) :=
  let gruModule := Spec.Module.gru gruSpec
  let linearModule := Spec.Module.liftLeading (Spec.Module.linear linearSpec)
  Spec.Module.Chain.single gruModule
    |>.append linearModule

/-- A many-to-one GRU classifier (use the last hidden state, then a linear head).

PyTorch analogy: run `nn.GRU` over the sequence and feed the last output/hidden state into
`nn.Linear(hiddenSize, numClasses)`.
-/
def classifier
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize numClasses : Nat}
  (gruSpec : GRUSpec α inputSize hiddenSize)
  (classifierHead : LinearSpec α hiddenSize numClasses)
  (h : seqLen ≠ 0) :
  Spec.Module.Chain α ([seqLen, inputSize]) ([numClasses]) :=
  let gruModule := Spec.Module.gru gruSpec
  let lastOutput := Spec.Module.select (shape := [seqLen, hiddenSize]) 0
    (⟨Nat.pred seqLen, Nat.pred_lt h⟩)
  let classifierModule := Spec.Module.linear classifierHead
  Spec.Module.Chain.single gruModule
    |>.append lastOutput
    |>.append classifierModule

/-- A recurrent stack of arbitrary depth and widths, followed by a per-timestep linear head. -/
def stacked
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen inputSize hiddenSize outputSize : Nat}
  (layers : RecurrentStack (GRUSpec α) inputSize hiddenSize)
  (linearSpec : LinearSpec α hiddenSize outputSize) :
  Spec.Module.Chain α [seqLen, inputSize] [seqLen, outputSize] :=
  layers.toChain (fun cell => Spec.Module.gru cell)
    (Spec.Module.liftLeading (Spec.Module.linear linearSpec))

/-- A simple GRU language-model style pipeline:

`Linear` as the embedding/projection map, then GRU, then a per-timestep projection back to
`vocabularySize`.

PyTorch analogy: embedding (often `nn.Embedding`), `nn.GRU`, and `nn.Linear(hiddenSize,
vocabularySize)`. We use `LinearSpec` here as a spec-friendly stand-in for a one-hot embedding
matrix.
-/
def languageModel
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen vocabularySize hiddenSize : Nat}
  (embeddingSpec : LinearSpec α vocabularySize hiddenSize)
  (gruSpec : GRUSpec α hiddenSize hiddenSize)
  (outputSpec : LinearSpec α hiddenSize vocabularySize) :
  Spec.Module.Chain α ([seqLen, vocabularySize]) ([seqLen, vocabularySize]) :=
  let embeddingModule := Spec.Module.liftLeading (Spec.Module.linear embeddingSpec)
  let gruModule := Spec.Module.gru gruSpec
  let outputModule := Spec.Module.liftLeading (Spec.Module.linear outputSpec)
  Spec.Module.Chain.single embeddingModule
    |>.append gruModule
    |>.append outputModule

/-!
## Record-style model specs

The `Spec.Module.Chain` builders above are the most uniform way to assemble models in TorchLean.

The declarations below use small record types with explicit forward functions. This is useful when
you want to talk about a particular architecture directly (e.g. encoder-decoder), or when you need
to carry extra per-model parameters (e.g. a dropout rate) without building a full module stack.
-/

-- Basic GRU model with a single GRU cell + a linear output head.
/--
Bundle of parameters for a single-layer GRU model with a linear output head.

This is a direct record representation (as opposed to the `Spec.Module.Chain` representation above).
-/
structure Model (α : Type) [TorchLean.Storage α] (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell parameters. -/
  gru : GRUSpec α inputSize hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

/-- Gradients of the three GRU gates. -/
structure CellGrads (α : Type) [TorchLean.Storage α] (inputSize hiddenSize : Nat) where
  /-- Gradient of the reset-gate weight matrix. -/
  resetWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the reset-gate bias. -/
  resetBias : Tensor α [hiddenSize]
  /-- Gradient of the update-gate weight matrix. -/
  updateWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the update-gate bias. -/
  updateBias : Tensor α [hiddenSize]
  /-- Gradient of the candidate-state weight matrix. -/
  candidateWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the candidate-state bias. -/
  candidateBias : Tensor α [hiddenSize]

/-- Parameter gradients for a GRU model and its linear output head. -/
structure Grads (α : Type) [TorchLean.Storage α] (inputSize hiddenSize outputSize : Nat) where
  /-- Gradients of the recurrent cell. -/
  cell : CellGrads α inputSize hiddenSize
  /-- Gradient of the output projection weight. -/
  outputWeight : Tensor α [outputSize, hiddenSize]
  /-- Gradient of the output projection bias. -/
  outputBias : Tensor α [outputSize]

-- Multi-layer GRU model
/-- Recurrent cells with independently chosen widths and a linear output head.

The endpoint `hiddenSize` is the width of the last cell.
An empty stack has `hiddenSize = inputSize`.
-/
structure StackedModel (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize outputSize : Nat) where
  /-- Layer dimensions determine the type of every intermediate hidden stream and state. -/
  layers : RecurrentStack (GRUSpec α) inputSize hiddenSize
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
      let outputs := gruSequenceSpec cell input state
      let finalState := if h : seqLen = 0 then state else
        get outputs ⟨seqLen - 1, Nat.sub_one_lt h⟩
      (outputs, finalState)) inputs initialHiddens
  (Tensor.mapLeading [seqLen] (linearSpec model.outputLayer) hidden, finalStates)

/--
Bundle of parameters for a many-to-one GRU classifier.

The classifier head is applied to the final hidden state.
-/
structure Classifier (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize numClasses : Nat) where
  /-- Recurrent cell parameters. -/
  gru : GRUSpec α inputSize hiddenSize
  /-- Linear classifier head. -/
  classifier : LinearSpec α hiddenSize numClasses

-- GRU model for sequence generation (many-to-many)
/--
Bundle of parameters for a many-to-many GRU generator (language-model style).

This includes an (embedding) linear map, recurrent core, and output projection back to vocabulary.
-/
structure Generator (α : Type) [TorchLean.Storage α] (vocabularySize hiddenSize : Nat) where
  /-- Token projection used by this one-hot specification. -/
  embedding : LinearSpec α vocabularySize hiddenSize
  /-- Recurrent cell parameters. -/
  gru : GRUSpec α hiddenSize hiddenSize
  /-- Projection from hidden states to vocabulary logits. -/
  outputProjection : LinearSpec α hiddenSize vocabularySize

/--
Bundle of parameters for a bidirectional GRU model with an output head.

The head consumes the concatenation of forward and backward hidden states.
PyTorch analogue: `nn.GRU(..., bidirectional=true)` plus a linear projection.
-/
structure BidirectionalModel (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell for the original sequence order. -/
  forwardGru : GRUSpec α inputSize hiddenSize
  /-- Recurrent cell for the reversed sequence order. -/
  backwardGru : GRUSpec α inputSize hiddenSize
  /-- Projection from concatenated forward and backward states. -/
  outputLayer : LinearSpec α (hiddenSize + hiddenSize) outputSize

/--
Bundle of parameters for a stacked GRU language model with deterministic dropout.

This model uses an array of GRU layers (all with `hiddenSize` input/output) and applies
evaluation-mode dropout between the GRU stack and the output projection.
-/
structure LanguageModel (α : Type) [TorchLean.Storage α] (vocabularySize hiddenSize : Nat) where
  /-- Token projection used by this one-hot specification. -/
  embedding : LinearSpec α vocabularySize hiddenSize
  /-- Recurrent layers, ordered from input to output. -/
  layers : Array (GRUSpec α hiddenSize hiddenSize)
  /-- Projection from hidden states to vocabulary logits. -/
  outputProjection : LinearSpec α hiddenSize vocabularySize
  /-- Dropout probability used between the recurrent stack and output projection. -/
  dropoutRate : α

-- GRU Encoder-Decoder Model
/--
Bundle of parameters for a GRU encoder-decoder model (seq2seq).

This uses separate embeddings and GRU cores for encoder and decoder, plus an output projection.
PyTorch analogue: an encoder `nn.GRU` and a decoder `nn.GRU` with teacher forcing.
-/
structure EncoderDecoder (α : Type) [TorchLean.Storage α]
    (inputVocabSize hiddenSize outputVocabSize : Nat) where
  /-- Source-token projection. -/
  encoderEmbedding : LinearSpec α inputVocabSize hiddenSize
  /-- Encoder recurrent cell. -/
  encoderGru : GRUSpec α hiddenSize hiddenSize
  /-- Target-token projection. -/
  decoderEmbedding : LinearSpec α outputVocabSize hiddenSize
  /-- Decoder recurrent cell. -/
  decoderGru : GRUSpec α hiddenSize hiddenSize
  /-- Projection from decoder states to target-vocabulary logits. -/
  outputProjection : LinearSpec α hiddenSize outputVocabSize

/-- One-step forward for `Gru.Model`.

Input: `(x_t, h_{t-1})`. Output: `(y_t, h_t)`.
-/
def Model.forward {inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (input : Tensor α [inputSize])
  (hidden : Tensor α [hiddenSize]) :
  (Tensor α [outputSize] × Tensor α [hiddenSize]) :=
  let nextHidden := gruCellSpec model.gru input hidden
  let output := linearSpec model.outputLayer nextHidden
  (output, nextHidden)

/-- Sequence forward for `Gru.Model` (time-major).

Returns `(outputs, final_hidden)`. Empty sequences preserve the initial hidden state.

PyTorch analogy: run `nn.GRU` over the sequence, then apply `nn.linear` at each timestep.
-/
def Model.forwardSequence {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize]) :
  (Tensor α [seqLen, outputSize] × Tensor α [hiddenSize]) :=
  let stack : StackedModel α inputSize hiddenSize outputSize :=
    { layers := .cons model.gru .nil, outputLayer := model.outputLayer }
  let (outputs, finalStates) := stack.forward inputs (initialHidden, ())
  (outputs, finalStates.1)

-- Forward pass for GRU classifier (many-to-one)
/--
Forward pass for a `Gru.Classifier` (many-to-one).

This runs the GRU over the input sequence and applies the classifier head to the final hidden
state.
-/
def Classifier.forward {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize]) (h : 0 < seqLen) :
  Tensor α [numClasses] :=
  let hiddenStates := gruSequenceSpec model.gru inputs initialHidden
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := get hiddenStates ⟨seqLen - 1, hLast⟩
  linearSpec model.classifier finalHidden

-- Forward pass for GRU generator (many-to-many)
/--
Forward pass for a `Gru.Generator` (many-to-many).

This applies an embedding linear map to each token vector, runs the GRU, and projects each hidden
state back into vocabulary space.
-/
def Generator.forward {seqLen vocabularySize hiddenSize : Nat}
  (model : Generator α vocabularySize hiddenSize)
  (inputTokens : Tensor α [seqLen, vocabularySize])
  (initialHidden : Tensor α [hiddenSize]) (h : 0 < seqLen) :
  (Tensor α [seqLen, vocabularySize] × Tensor α [hiddenSize]) :=
  let embedded := Tensor.mapLeading ([seqLen]) (linearSpec model.embedding) inputTokens
  let hiddenStates := gruSequenceSpec model.gru embedded initialHidden
  let outputs := Tensor.mapLeading ([seqLen])
    (linearSpec model.outputProjection) hiddenStates
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := get hiddenStates ⟨seqLen - 1, hLast⟩
  (outputs, finalHidden)

/--
Forward pass for a bidirectional GRU model (time-major).

This runs a forward GRU on the sequence, a backward GRU on the reversed sequence, concatenates the
two hidden streams per timestep, and applies an output head.
-/
def BidirectionalModel.forward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BidirectionalModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (forwardHidden : Tensor α [hiddenSize])
  (backwardHidden : Tensor α [hiddenSize]) :
  Tensor α [seqLen, outputSize] :=
  let forwardStates := gruSequenceSpec model.forwardGru inputs forwardHidden
  let reversedInputs := Tensor.reverseAxis 0 inputs
  let reversedBackwardStates :=
    gruSequenceSpec model.backwardGru reversedInputs backwardHidden
  let backwardStates := Tensor.reverseAxis 0 reversedBackwardStates
  let combinedStates := Tensor.zipEach ([seqLen])
    ([(hiddenSize + hiddenSize)])
    (Tensor.concatAxisSpec .scalar) forwardStates backwardStates
  Tensor.mapLeading ([seqLen]) (linearSpec model.outputLayer) combinedStates

-- GRU Language Model forward pass
/--
Forward pass for `Gru.LanguageModel` (teacher forcing, time-major).

This runs the embedding, then a stack of GRU layers with provided initial hiddens, applies
evaluation-mode dropout (`dropoutInferenceSpec`), and projects to vocabulary logits.
-/
def LanguageModel.forward {seqLen vocabularySize hiddenSize : Nat}
  (model : LanguageModel α vocabularySize hiddenSize)
  (inputTokens : Tensor α [seqLen, vocabularySize])
  (initialHiddens : Array (Tensor α [hiddenSize])) (h : 0 < seqLen) :
  Option
    (Tensor α [seqLen, vocabularySize] ×
      Array (Tensor α [hiddenSize])) := do
  let embedded :=
    Tensor.mapLeading ([seqLen]) (linearSpec model.embedding) inputTokens
  let rec processLayers (layers : List (GRUSpec α hiddenSize hiddenSize))
    (index : Nat)
    (layerInput : Tensor α [seqLen, hiddenSize]) :
    Option
      (Tensor α [seqLen, hiddenSize] ×
        Array (Tensor α [hiddenSize])) :=
    match layers with
    | [] => if index = initialHiddens.size then some (layerInput, #[]) else none
    | layer :: remainingLayers => do
      let hidden ← initialHiddens[index]?
      let layerOutput := gruSequenceSpec layer layerInput hidden
      have hLast : seqLen - 1 < seqLen := by
        simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
      let finalHidden := get layerOutput ⟨seqLen - 1, hLast⟩
      let (finalOutput, finalHiddens) ←
        processLayers remainingLayers (index + 1) layerOutput
      pure (finalOutput, #[finalHidden] ++ finalHiddens)
  let (gruOutput, finalHiddens) ← processLayers model.layers.toList 0 embedded
  let droppedOutput := dropoutInferenceSpec (p := model.dropoutRate) gruOutput
  let logits := Tensor.mapLeading ([seqLen])
    (linearSpec model.outputProjection) droppedOutput
  pure (logits, finalHiddens)

-- GRU Encoder-Decoder forward pass
/-- Encoder-decoder forward pass (GRU encoder + GRU decoder).

This is a small reference architecture:

- encode `sourceTokens` into a final hidden state,
- decode `targetTokens` starting from that hidden state (teacher forcing),
- project decoder states into output-vocabulary logits.

PyTorch analogy: `nn.GRU` encoder + `nn.GRU` decoder with a linear output projection.
-/
def EncoderDecoder.forward {srcSeqLen tgtSeqLen inputVocabSize hiddenSize outputVocabSize :
  Nat}
  (model : EncoderDecoder α inputVocabSize hiddenSize outputVocabSize)
  (sourceTokens : Tensor α [srcSeqLen, inputVocabSize])
  (targetTokens : Tensor α [tgtSeqLen, outputVocabSize])
  (encoderHidden : Tensor α [hiddenSize])
  (hSource : 0 < srcSeqLen) (hTarget : 0 < tgtSeqLen) :
  (Tensor α [tgtSeqLen, outputVocabSize] ×
   Tensor α [hiddenSize] × Tensor α [hiddenSize]) :=
  let sourceEmbedded := Tensor.mapLeading ([srcSeqLen])
    (linearSpec model.encoderEmbedding) sourceTokens
  let encoderStates := gruSequenceSpec model.encoderGru sourceEmbedded encoderHidden
  have hSourceLast : srcSeqLen - 1 < srcSeqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt hSource)
  let encoderFinal := get encoderStates ⟨srcSeqLen - 1, hSourceLast⟩
  let targetEmbedded := Tensor.mapLeading ([tgtSeqLen])
    (linearSpec model.decoderEmbedding) targetTokens
  let decoderStates := gruSequenceSpec model.decoderGru targetEmbedded encoderFinal
  have hTargetLast : tgtSeqLen - 1 < tgtSeqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt hTarget)
  let decoderFinal := get decoderStates ⟨tgtSeqLen - 1, hTargetLast⟩
  let outputs := Tensor.mapLeading ([tgtSeqLen])
    (linearSpec model.outputProjection) decoderStates
  (outputs, encoderFinal, decoderFinal)

-- Backward pass for simple GRU model with full BPTT
/-- Backward pass for `Gru.Model` using full backpropagation through time.

This assumes you already ran a forward pass that saved:
- `hidden_states`,
- the GRU intermediates (`resetGates`, `updateGates`, `newCandidates`, `resetHiddens`).

Those intermediates can be produced using `Spec.gruExtractIntermediateValues` from
`NN.Spec.Layers.Gru`.
-/
def Model.backward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : Model α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (hiddenStates : Tensor α [seqLen, hiddenSize])
  (outputGrad : Tensor α [seqLen, outputSize])
  (resetGates : Tensor α [seqLen, hiddenSize])
  (updateGates : Tensor α [seqLen, hiddenSize])
  (candidates : Tensor α [seqLen, hiddenSize])
  (h : seqLen ≠ 0) :
  Grads α inputSize hiddenSize outputSize ×
    Tensor α [seqLen, inputSize] :=
  let hiddenGrad := Tensor.mapLeading ([seqLen])
    (fun grad => linearInputDerivSpec model.outputLayer.weights grad) outputGrad
  let outputWeightGrad := Tensor.reduceSum 0
    (Tensor.zipEach ([seqLen])
      [outputSize, hiddenSize]
      linearWeightsDerivSpec hiddenStates outputGrad)
    (Shape.hasNonemptyAxisZeroOfNe h).proof
  let outputBiasGrad := Tensor.reduceSum 0 outputGrad
    (Shape.hasNonemptyAxisZeroOfNe h).proof
  let initialHidden := Tensor.full ([hiddenSize]) 0
  let (resetWeight, resetBias, updateWeight, updateBias,
       candidateWeight, candidateBias, inputGrad, _) :=
    gruSequenceBackwardFullSpec model.gru inputs hiddenStates hiddenGrad
      resetGates updateGates candidates initialHidden
  ({ cell := { resetWeight, resetBias, updateWeight, updateBias, candidateWeight, candidateBias }
     outputWeight := outputWeightGrad
     outputBias := outputBiasGrad }, inputGrad)

/--
Bundle of parameters for a residual GRU model.

This includes a projection from input space to hidden space so the input can be added as a residual
to the GRU hidden stream.
-/
structure ResidualModel (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize outputSize : Nat) where
  /-- Recurrent cell parameters. -/
  gru : GRUSpec α inputSize hiddenSize
  /-- Projection that gives the input stream the recurrent hidden width. -/
  residualProjection : LinearSpec α inputSize hiddenSize
  /-- Linear output projection. -/
  outputLayer : LinearSpec α hiddenSize outputSize

/--
Forward pass for `Gru.ResidualModel`.

This runs the GRU, adds a projected version of the input as a residual connection, and applies the
output head per timestep.
-/
def ResidualModel.forward {seqLen inputSize hiddenSize outputSize : Nat}
  (model : ResidualModel α inputSize hiddenSize outputSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize]) (h : 0 < seqLen) :
  (Tensor α [seqLen, outputSize] × Tensor α [hiddenSize]) :=
  let projectedInputs := Tensor.mapLeading ([seqLen])
    (linearSpec model.residualProjection) inputs
  let hiddenStates := gruSequenceSpec model.gru inputs initialHidden
  let residualStates := Tensor.zipEach ([seqLen]) ([hiddenSize])
    addSpec hiddenStates projectedInputs
  let outputs := Tensor.mapLeading ([seqLen])
    (linearSpec model.outputLayer) residualStates
  have hLast : seqLen - 1 < seqLen := by
    simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt h)
  let finalHidden := get residualStates ⟨seqLen - 1, hLast⟩
  (outputs, finalHidden)

/--
Package `Gru.Model` as a shape-indexed module.

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
  kind := "SimpleGRU",
  pythonExpr :=
    s!"SimpleGRU(input_size={inputSize}, hidden_size={hiddenSize}, " ++
      s!"output_size={outputSize})"
}

/--
Package `Gru.Classifier` as an `Spec.Module`.

PyTorch analogue: `nn.GRU` feeding a `nn.linear` classifier head.
-/
def Classifier.toModule {seqLen inputSize hiddenSize numClasses : Nat}
  (model : Classifier α inputSize hiddenSize numClasses) (h : 0 < seqLen) :
  Spec.Module α ([seqLen, inputSize]) ([numClasses]) :=
{
  forward := fun inputs =>
    let initialHidden := Tensor.full ([hiddenSize]) 0
    model.forward inputs initialHidden h,
  kind := "GRUClassifier",
  pythonExpr :=
    s!"GRUClassifier(input_size={inputSize}, hidden_size={hiddenSize}, " ++
      s!"num_classes={numClasses})"
}

/--
Package `Gru.BidirectionalModel` as an `Spec.Module`.

PyTorch analogue: `nn.GRU(..., bidirectional=true)` feeding a per-timestep linear head.
-/
def BidirectionalModel.toModule {seqLen inputSize hiddenSize outputSize : Nat}
  (model : BidirectionalModel α inputSize hiddenSize outputSize) :
  Spec.Module α ([seqLen, inputSize]) ([seqLen, outputSize]) :=
{
  forward := fun inputs =>
    let initialHidden := Tensor.full ([hiddenSize]) 0
    model.forward inputs initialHidden initialHidden,
  kind := "BiGRU",
  pythonExpr :=
    s!"SimpleGRU(input_size={inputSize}, hidden_size={hiddenSize}, " ++
      s!"output_size={outputSize}, bidirectional=True)"
}

/--
Package `Gru.Generator` as an `Spec.Module`.

PyTorch analogue: GRU language model (`nn.GRU` + vocabulary projection) producing a sequence of
logits.
-/
def Generator.toModule {seqLen vocabularySize hiddenSize : Nat}
  (model : Generator α vocabularySize hiddenSize) (h : 0 < seqLen) :
  Spec.Module α ([seqLen, vocabularySize]) ([seqLen, vocabularySize]) :=
{
  forward := fun inputs =>
    let initialHidden := Tensor.full ([hiddenSize]) 0
    (model.forward inputs initialHidden h).1,
  kind := "GRUGenerator",
  pythonExpr := s!"GRULanguageModel(vocab_size={vocabularySize}, hidden_size={hiddenSize})"
}

end Gru

end Spec
