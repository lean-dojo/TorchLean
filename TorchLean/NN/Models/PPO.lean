/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder

/-!
# PPO Actor-Critic Models

`PPO.actor` and `PPO.critic` build the one-hidden-layer `tanh` MLPs of a PPO actor-critic pair
from a shared `PPO.Config`.

Environment collection, trust-boundary checks, advantage computation and optimizer loops live in
the example and runtime modules.

Two choices come from Schulman et al. (2017), Section 6.1: `tanh` MLPs, and no parameters shared
between the policy and the value function. We keep one hidden layer where they use two of 64
units, and the actor returns logits of a categorical policy over `actionCount` actions. Their
continuous-control policy outputs the mean of a Gaussian.

Reference: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017),
"Proximal Policy Optimization Algorithms", arXiv:1707.06347, Section 6.1.
-/

@[expose] public section

namespace TorchLean


open TorchLean.Tensor

namespace nn
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

/-- Validate dimensions used by both PPO networks. -/
def validateNetwork (kind : String) (config : Config) : Except String Unit := do
  if config.observationWidth = 0 then
    throw s!"{kind}: observation width must be positive"
  if config.hiddenWidth = 0 then
    throw s!"{kind}: hidden width must be positive"

namespace Config

/-- Validate the complete actor-critic configuration. -/
def validate (config : Config) : Except String Unit := do
  validateNetwork "PPO" config
  if config.actionCount = 0 then
    throw "PPO: action count must be positive"

/-- Observation tensor shape with an arbitrary batch shape. -/
abbrev inputShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.observationWidth]

/-- Actor-logit tensor shape with the same batch shape as the observations. -/
abbrev actorOutputShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.actionCount]

/-- Critic-value tensor shape with the same batch shape as the observations. -/
abbrev criticOutputShape (_config : Config) (batchShape : Shape := []) : Shape :=
  batchShape ++ [1]

end Config

/-- Actor MLP mapping observations to action logits. -/
def actor (config : Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential
      (config.inputShape batchShape) (config.actorOutputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.actorOutputShape batchShape) "PPO.actor" message
  | .ok () =>
      nn.Sequential![
        linear config.observationWidth config.hiddenWidth (batchShape := batchShape),
        nn.op .tanh,
        linear config.hiddenWidth config.actionCount (batchShape := batchShape)
      ]

/-- Critic MLP mapping observations to a scalar value estimate. -/
def critic (config : Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential
      (config.inputShape batchShape) (config.criticOutputShape batchShape)) :=
  match validateNetwork "PPO.critic" config with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.criticOutputShape batchShape) "PPO.critic" message
  | .ok () =>
      nn.Sequential![
        linear config.observationWidth config.hiddenWidth (batchShape := batchShape),
        nn.op .tanh,
        linear config.hiddenWidth 1 (batchShape := batchShape)
      ]

end PPO
end nn

end TorchLean
