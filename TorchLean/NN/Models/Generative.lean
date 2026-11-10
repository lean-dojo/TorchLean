/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder

/-!
# Generative Models

`Generative.autoencoder`, `Generative.generator` and `Generative.discriminator` build dense
backbones for the runnable generative examples from the widths in a `Generative.Config`.

Each model acts on a trailing feature axis and preserves the caller's `batchShape`. Examples may
flatten structured observations first, while convolutional or operator-based models have their
own shape-specific constructors.

We picked these stacks by hand for the examples, and they copy no paper's architecture. The
generator and the discriminator have three dense layers each, the autoencoder has four, and ReLU
sits between every pair.
-/

@[expose] public section

namespace TorchLean


open TorchLean.Tensor

namespace nn

namespace Generative

/-- Widths shared by dense generative models. -/
structure Config where
  /-- Width of the data feature axis. -/
  dataWidth : Nat
  /-- Width of the hidden layers. -/
  hiddenWidth : Nat
  /-- Width of the latent representation. -/
  latentWidth : Nat
deriving Repr

/-- Validate the data and hidden widths used by every dense generative model. -/
def validateSharedWidths (kind : String) (config : Config) : Except String Unit := do
  if config.dataWidth = 0 then
    throw s!"{kind}: data width must be positive"
  if config.hiddenWidth = 0 then
    throw s!"{kind}: hidden width must be positive"

/-- Validate all widths while naming the public constructor being built. -/
def validateConfig (kind : String) (config : Config) : Except String Unit := do
  validateSharedWidths kind config
  if config.latentWidth = 0 then
    throw s!"{kind}: latent width must be positive"

/-- Data tensor shape with an arbitrary batch shape. -/
abbrev Config.dataShape (config : Config)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.dataWidth]

/-- Latent tensor shape with an arbitrary batch shape. -/
abbrev Config.latentShape (config : Config)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.latentWidth]

/-- Scalar-score tensor shape with an arbitrary batch shape. -/
abbrev Config.scoreShape (_config : Config)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [1]

/--
Autoencoder backbone: `x -> hidden -> latent -> hidden -> reconstruction`.

The reconstruction is unconstrained. Append an output activation such as `nn.op .sigmoid` when the
data domain requires one.
-/
def autoencoder (config : Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.dataShape batchShape) (config.dataShape batchShape)) :=
  match validateConfig "Autoencoder" config with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.dataShape batchShape) (config.dataShape batchShape) "Autoencoder" message
  | .ok () =>
      nn.Sequential![
        linear config.dataWidth config.hiddenWidth (batchShape := batchShape),
        nn.op .relu,
        linear config.hiddenWidth config.latentWidth (batchShape := batchShape),
        nn.op .relu,
        linear config.latentWidth config.hiddenWidth (batchShape := batchShape),
        nn.op .relu,
        linear config.hiddenWidth config.dataWidth (batchShape := batchShape)
      ]

/--
Generator backbone `z -> x`.

The generated values are unconstrained. Choose an output activation at the call site to match the
training data and objective.
-/
def generator (config : Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.latentShape batchShape) (config.dataShape batchShape)) :=
  match validateConfig "Generator" config with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.latentShape batchShape) (config.dataShape batchShape) "Generator" message
  | .ok () =>
      nn.Sequential![
        linear config.latentWidth config.hiddenWidth (batchShape := batchShape),
        nn.op .relu,
        linear config.hiddenWidth config.hiddenWidth (batchShape := batchShape),
        nn.op .relu,
        linear config.hiddenWidth config.dataWidth (batchShape := batchShape)
      ]

/--
Discriminator `x -> logits`.

Logits suit numerically stable objectives such as `TorchLean.Loss.bceWithLogits`, and
`nn.op .sigmoid` can follow when probabilities are required. The model reads only the data and
hidden widths, so the latent width is neither used nor validated here.
-/
def discriminator (config : Config)
    (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.dataShape batchShape) (config.scoreShape batchShape)) :=
  match validateSharedWidths "Discriminator" config with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.dataShape batchShape) (config.scoreShape batchShape) "Discriminator" message
  | .ok () =>
      nn.Sequential![
        linear config.dataWidth config.hiddenWidth (batchShape := batchShape),
        nn.op .relu,
        linear config.hiddenWidth config.hiddenWidth (batchShape := batchShape),
        nn.op .relu,
        linear config.hiddenWidth 1 (batchShape := batchShape)
      ]

end Generative
end nn

end TorchLean
