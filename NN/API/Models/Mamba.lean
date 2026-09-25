/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Mamba Models

Configuration and a language-model constructor for selective Mamba-1 sequence models.

Each recurrent layer uses causal depthwise convolution, input-dependent time steps and B/C vectors,
learned negative state rates, and a gated readout. It is built from generic differentiable
operations shared by CPU and CUDA execution.
-/

@[expose] public section

namespace TorchLean


open Spec TorchLean TorchLean.Tensor

namespace nn
namespace models
namespace Mamba

/-- Configuration for the trainable one-hot-token Mamba language model. -/
structure Config where
  /-- Number of token categories accepted and predicted at each sequence position. -/
  vocabularySize : Nat
  /-- Feature widths of the Mamba layers. An empty list builds only the vocabulary projection. -/
  modelWidths : List Nat := []
  /-- Expanded channels per model feature in the convolution and recurrent path. -/
  expansion : Nat := 2
  /-- Diagonal recurrent states per expanded channel. -/
  stateWidth : Nat := 16
  /-- Newest-first taps in the causal depthwise convolution. -/
  kernelWidth : Nat := 4
deriving Repr

namespace Config

/-- Internal Mamba dimensions passed unchanged to the trainable layer. -/
def options (config : Config) : Runtime.Autograd.Model.Mamba.Options :=
  { expansion := config.expansion
    stateWidth := config.stateWidth
    kernelWidth := config.kernelWidth }

/-- Validate model dimensions before allocating recurrent or projection parameters. -/
def validate (config : Config) (sequenceLength : Nat) : Except String Unit := do
  if !config.modelWidths.isEmpty && sequenceLength = 0 then
    throw "Mamba: sequence length must be positive"
  if config.vocabularySize = 0 then
    throw "Mamba: vocabulary size must be positive"
  for modelWidth in config.modelWidths do
    if modelWidth = 0 then
      throw "Mamba: model width must be positive"
  unless config.modelWidths.isEmpty do
    config.options.validate

/-- One-hot input shape `batchShape × sequenceLength × vocabularySize`. -/
abbrev inputShape (config : Config) (sequenceLength : Nat)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat [sequenceLength, config.vocabularySize]

/-- Logit output shape `batchShape × sequenceLength × vocabularySize`. -/
abbrev outputShape (config : Config) (sequenceLength : Nat)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat [sequenceLength, config.vocabularySize]

end Config

/--
Trainable selective Mamba-1 language model over one-hot token inputs.

The layers map each `vocabularySize`-wide token through `modelWidths` in order, and a final affine
map produces vocabulary logits at every position. Each layer's internal width is
`expansion * modelWidth`.
Every sequence starts with zero hidden state and empty convolution history; each batch element has
its own recurrence while sharing the eleven tensors per Mamba layer and the vocabulary projection.
With no Mamba layers, the projection accepts empty sequences and ignores the unused core options.

The time-step projection is a dense matrix, matching `Models.SelectiveMambaBlockSpec`. The usual
low-rank Mamba checkpoint stores two factors instead; their product matches a forward map here, but
training a dense matrix gives a different parameterization. `Model.Mamba.runArray` exposes explicit
state and convolution history for streaming computations.
-/
def languageModel (config : Config) (sequenceLength : Nat) (batchShape : Shape := []) :
    nn.Builder
      (nn.Sequential
        (config.inputShape sequenceLength batchShape)
        (config.outputShape sequenceLength batchShape)) := by
  match config.validate sequenceLength with
  | .error message =>
      exact pure <| nn.Internal.invalidConfiguration
        (config.inputShape sequenceLength batchShape)
        (config.outputShape sequenceLength batchShape)
        "Mamba.languageModel" message
  | .ok () =>
      let rec buildLayers (inputWidth : Nat) (modelWidths : List Nat) :
          Builder (Sequential (batchShape.concat [sequenceLength, inputWidth])
            (config.outputShape sequenceLength batchShape)) :=
        match modelWidths with
        | [] => by
            simpa only [Config.outputShape, Shape.appendDim_appendDim_eq_concat] using
              (linear inputWidth config.vocabularySize
                (batchShape := batchShape.appendDim sequenceLength))
        | modelWidth :: rest => do
            let layer : Sequential (batchShape.concat [sequenceLength, inputWidth])
                (batchShape.concat [sequenceLength, modelWidth]) ← by
              simpa only [Shape.appendDim_appendDim_eq_concat] using
                (nn.mamba sequenceLength inputWidth modelWidth
                  (batchShape := batchShape) (options := config.options))
            let remaining ← buildLayers modelWidth rest
            pure (layer >>> remaining)
      exact buildLayers config.vocabularySize config.modelWidths

end Mamba
end models
end nn

end TorchLean
