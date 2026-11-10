/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Recurrent
public import TorchLean.NN.Builder

/-!
# Mamba Models

`Mamba.Config` and `Mamba.languageModel` define a selective Mamba-1 language model over one-hot
tokens.

Each recurrent layer uses causal depthwise convolution, input-dependent time steps and B/C vectors,
learned negative state rates and a gated readout, all built from the generic differentiable
operations that CPU and CUDA execution share.

We stack the layers by hand. Gu and Dao repeat the Mamba block "interleaved with standard
normalization and residual connections" (Section 3.4), and `languageModel` composes the blocks with
neither, so each layer reads the previous layer's output as it is. A one-hot token times the first
layer's `xProj` or `zProj` is one row of that matrix, so the two double as the token embedding.
`expansion := 2` is their fixed `E = 2` (Section 3.4), `stateWidth := 16` is the `N = 16` of
Section 4.5, and `kernelWidth := 4` is `d_conv = 4` in the authors' `mamba_simple.py`. The layer
starts its rates at `1, ..., stateWidth`, which puts the paper's `A` at the S4D-Real `−(n + 1)` of
Section 3.6. Its time-step bias starts at `softplus⁻¹(0.01) ≈ −4.6`, where Section 3.6 takes
`softplus⁻¹` of a draw from `Uniform([0.001, 0.1])`. No theorem or TorchLeanTest test covers
`languageModel`.

References:
- Albert Gu and Tri Dao (2024), "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces", COLM 2024, arXiv:2312.00752, Sections 3.4, 3.6 and 4.5.
- `mamba_ssm/modules/mamba_simple.py` in github.com/state-spaces/mamba, the `Mamba` defaults.
-/

@[expose] public section

open scoped TorchLean.nn.Sequential

namespace TorchLean


open TorchLean.Tensor

namespace nn
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
def options (config : Config) : Mamba.Options :=
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
  batchShape ++ [sequenceLength, config.vocabularySize]

/-- Logit output shape `batchShape × sequenceLength × vocabularySize`. -/
abbrev outputShape (config : Config) (sequenceLength : Nat)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [sequenceLength, config.vocabularySize]

end Config

/--
Trainable selective Mamba-1 language model over one-hot token inputs.

The layers map each `vocabularySize`-wide token through `modelWidths` in order, and a final affine
map produces vocabulary logits at every position. Each layer's internal width is
`expansion * modelWidth`.
Every sequence starts with zero hidden state and empty convolution history. Each batch element has
its own recurrence while sharing the eleven tensors per Mamba layer and the vocabulary projection.
With no Mamba layers, the projection accepts empty sequences and ignores the unused core options.

The time-step projection is a dense matrix, matching `TorchLean.Mamba.SelectiveBlock`. The usual
Mamba checkpoint stores two low-rank factors, and over the reals their product gives the same
forward map here, but training a dense matrix is a different parameterization. `TorchLean.Mamba.run`
takes and returns the state and convolution history for streaming computations.
-/
def languageModel (config : Config) (sequenceLength : Nat) (batchShape : Shape := []) :
    nn.Builder
      (nn.Sequential
        (config.inputShape sequenceLength batchShape)
        (config.outputShape sequenceLength batchShape)) := by
  match config.validate sequenceLength with
  | .error message =>
      exact pure <| nn.Sequential.invalid
        (config.inputShape sequenceLength batchShape)
        (config.outputShape sequenceLength batchShape)
        "Mamba.languageModel" message
  | .ok () =>
      exact Recurrent.stack batchShape sequenceLength config.vocabularySize
        (fun inputWidth modelWidth => by
          simpa only [← List.append_cons] using
            (nn.mamba sequenceLength inputWidth modelWidth
              (batchShape := batchShape) (options := config.options)))
        config.vocabularySize config.modelWidths

end Mamba
end nn

end TorchLean
