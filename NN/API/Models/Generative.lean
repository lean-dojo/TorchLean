/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.API.Macros -- shake: keep
public import NN.Tensor -- shake: keep

/-!
# Generative Models

Config-style constructors for runnable generative examples.

These models act on a trailing feature axis and preserve the caller's `batchShape`. Examples may
flatten structured observations before applying them, while convolutional or operator-based
models can use their own shape-specific constructors.
-/

@[expose] public section

namespace TorchLean


open Spec TorchLean TorchLean.Tensor

namespace nn
namespace models

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

namespace Internal

/-- Validate shared dense-model widths while naming the public constructor being built. -/
def validateConfig (kind : String) (config : Config) : Except String Unit := do
  if config.dataWidth = 0 then
    throw s!"{kind}: data width must be positive"
  if config.hiddenWidth = 0 then
    throw s!"{kind}: hidden width must be positive"
  if config.latentWidth = 0 then
    throw s!"{kind}: latent width must be positive"

end Internal

namespace Config

/-- Validate every feature width before constructing or seeding a generative model. -/
def validate (config : Config) : Except String Unit :=
  Internal.validateConfig "Generative" config

end Config

/-- Data tensor shape with an arbitrary batch shape. -/
abbrev Config.data (config : Config)
    (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.dataWidth

/-- Latent tensor shape with an arbitrary batch shape. -/
abbrev Config.latent (config : Config)
    (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.latentWidth

/-- Scalar-score tensor shape with an arbitrary batch shape. -/
abbrev Config.score (_config : Config)
    (batchShape : Shape := []) : Shape :=
  batchShape.appendDim 1

/--
Autoencoder backbone: `x -> hidden -> latent -> hidden -> reconstruction`.

The reconstruction is unconstrained. Append an output activation such as `nn.sigmoid` when the
data domain requires one.
-/
def autoencoder (config : Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.data batchShape) (config.data batchShape)) :=
  match Internal.validateConfig "Autoencoder" config with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.data batchShape) (config.data batchShape) "Autoencoder" message
  | .ok () =>
      nn.Sequential![
        linear config.dataWidth config.hiddenWidth (batchShape := batchShape),
        relu,
        linear config.hiddenWidth config.latentWidth (batchShape := batchShape),
        relu,
        linear config.latentWidth config.hiddenWidth (batchShape := batchShape),
        relu,
        linear config.hiddenWidth config.dataWidth (batchShape := batchShape)
      ]

/--
Generator backbone `z -> x`.

The generated values are unconstrained. Choose an output activation at the call site to match the
training data and objective.
-/
def generator (config : Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.latent batchShape) (config.data batchShape)) :=
  match Internal.validateConfig "Generator" config with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.latent batchShape) (config.data batchShape) "Generator" message
  | .ok () =>
      nn.Sequential![
        linear config.latentWidth config.hiddenWidth (batchShape := batchShape),
        relu,
        linear config.hiddenWidth config.hiddenWidth (batchShape := batchShape),
        relu,
        linear config.hiddenWidth config.dataWidth (batchShape := batchShape)
      ]

/--
Discriminator `x -> logits`.

Returning logits keeps the model compatible with numerically stable objectives such as
`TorchLean.Loss.bceWithLogits`. Append `nn.sigmoid` only when probabilities are required.
-/
def discriminator (config : Config)
    (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.data batchShape) (config.score batchShape)) :=
  match Internal.validateConfig "Discriminator" config with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.data batchShape) (config.score batchShape) "Discriminator" message
  | .ok () =>
      nn.Sequential![
        linear config.dataWidth config.hiddenWidth (batchShape := batchShape),
        relu,
        linear config.hiddenWidth config.hiddenWidth (batchShape := batchShape),
        relu,
        linear config.hiddenWidth 1 (batchShape := batchShape)
      ]

end Generative
end models
end nn

end TorchLean
