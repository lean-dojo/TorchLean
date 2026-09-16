/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# PPO Actor-Critic Models

Reusable actor/critic MLP constructors for PPO examples.

These helpers cover the neural-network shape. Environment collection, trust-boundary checks,
advantage computation, and optimizer loops stay in the examples/runtime modules.
-/

@[expose] public section

namespace TorchLean


open Spec TorchLean TorchLean.Tensor

namespace nn
namespace models
namespace PPO

/-- Configuration for a simple PPO actor/critic pair over vector observations. -/
structure Config where
  /-- Number of features in each environment observation. -/
  observationWidth : Nat
  /-- Width of the actor and critic hidden layers. -/
  hiddenWidth : Nat
  /-- Number of discrete actions represented by the actor logits. -/
  actionCount : Nat
deriving Repr

namespace Internal

/-- Validate dimensions used by both PPO networks. -/
def validateNetwork (kind : String) (config : Config) : Except String Unit := do
  if config.observationWidth = 0 then
    throw s!"{kind}: observation width must be positive"
  if config.hiddenWidth = 0 then
    throw s!"{kind}: hidden width must be positive"

/-- Validate the actor, including its action-logit width. -/
def validateActor (config : Config) : Except String Unit := do
  validateNetwork "PPO.actor" config
  if config.actionCount = 0 then
    throw "PPO.actor: action count must be positive"

end Internal

namespace Config

/-- Validate the complete actor-critic configuration. -/
def validate (config : Config) : Except String Unit := do
  Internal.validateNetwork "PPO" config
  if config.actionCount = 0 then
    throw "PPO: action count must be positive"

/-- Observation tensor shape with an arbitrary batch shape. -/
abbrev input (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.observationWidth

/-- Actor-logit tensor shape with the same batch shape as the observations. -/
abbrev actorOutput (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.actionCount

/-- Critic-value tensor shape with the same batch shape as the observations. -/
abbrev criticOutput (_config : Config) (batchShape : Shape := []) : Shape :=
  batchShape.appendDim 1

end Config

/-- Actor MLP mapping observations to action logits. -/
def actor (config : Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.input batchShape) (config.actorOutput batchShape)) :=
  match Internal.validateActor config with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.actorOutput batchShape) "PPO.actor" message
  | .ok () =>
      nn.Sequential![
        linear config.observationWidth config.hiddenWidth (batchShape := batchShape),
        nn.tanh,
        linear config.hiddenWidth config.actionCount (batchShape := batchShape)
      ]

/-- Critic MLP mapping observations to a scalar value estimate. -/
def critic (config : Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.input batchShape) (config.criticOutput batchShape)) :=
  match Internal.validateNetwork "PPO.critic" config with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.criticOutput batchShape) "PPO.critic" message
  | .ok () =>
      nn.Sequential![
        linear config.observationWidth config.hiddenWidth (batchShape := batchShape),
        nn.tanh,
        linear config.hiddenWidth 1 (batchShape := batchShape)
      ]

end PPO
end models
end nn

end TorchLean
