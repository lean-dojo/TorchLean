/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.Runtime.Autograd.Model.Mamba

/-!
# Mamba Models

Configuration and a language-model constructor for selective Mamba-1 sequence models.

The recurrent core uses causal depthwise convolution, input-dependent time steps and B/C vectors,
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
  /-- Output feature width of the Mamba block, before the vocabulary projection. -/
  modelWidth : Nat
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
  if sequenceLength = 0 then
    throw "Mamba: sequence length must be positive"
  if config.vocabularySize = 0 then
    throw "Mamba: vocabulary size must be positive"
  if config.modelWidth = 0 then
    throw "Mamba: model width must be positive"
  config.options.validate

/-- One-hot input shape `batchShape × sequenceLength × vocabularySize`. -/
abbrev input (config : Config) (sequenceLength : Nat)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat [sequenceLength, config.vocabularySize]

/-- Logit output shape `batchShape × sequenceLength × vocabularySize`. -/
abbrev output (config : Config) (sequenceLength : Nat)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat [sequenceLength, config.vocabularySize]

end Config

/--
Trainable selective Mamba-1 language model over one-hot token inputs.

The block maps each `vocabularySize`-wide token to `modelWidth` features, and a final affine map
produces vocabulary logits at every position. Its internal width is `expansion * modelWidth`.
Every sequence starts with zero hidden state and empty convolution history; each batch element has
its own recurrence while sharing the eleven Mamba tensors and vocabulary projection.

The time-step projection is a dense matrix, matching `Models.SelectiveMambaBlockSpec`. The usual
low-rank Mamba checkpoint stores two factors instead; their product matches a forward map here, but
training a dense matrix gives a different parameterization. `Model.Mamba.runArray` exposes explicit
state and convolution history for streaming computations.
-/
def languageModel (config : Config) (sequenceLength : Nat) (batchShape : Shape := []) :
    nn.Builder
      (nn.Sequential
        (config.input sequenceLength batchShape)
        (config.output sequenceLength batchShape)) := by
  match config.validate sequenceLength with
  | .error message =>
      exact pure <| nn.Internal.invalidConfiguration
        (config.input sequenceLength batchShape)
        (config.output sequenceLength batchShape)
        "Mamba.languageModel" message
  | .ok () =>
      have model := do
        let recurrent ←
          nn.mamba sequenceLength config.vocabularySize config.modelWidth
            (batchShape := batchShape)
            (options := config.options)
        let outputProjection ← linear config.modelWidth config.vocabularySize
          (batchShape := batchShape.appendDim sequenceLength)
        pure (recurrent >>> outputProjection)
      simpa only [Config.input, Config.output, Shape.appendDim_appendDim_eq_concat] using model

end Mamba
end models
end nn

end TorchLean
