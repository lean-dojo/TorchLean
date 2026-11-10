/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.PPO
public import TorchLean.NN.RL.PPO.ActorCritic

/-!
# PPO Actor-Critic Setup

`PPOAgent.create` builds the `nn.PPO` actor and critic from one seed each, once for a batch
of `horizon` observations and once for a single observation. `RL.PPO.ActorCritic` trains the
batched networks, and `PPOAgent.policy` and `PPOAgent.value` run the single-observation graphs on
their state.
-/

@[expose] public section

namespace TorchLean.Examples

/-- A PPO actor-critic runtime with policy and value functions of one observation. -/
structure PPOAgent (obsShape : Shape) (nActions horizon : Nat) where
  /-- State shapes of the actor followed by those of the critic. -/
  stateShapes : List Shape
  /-- Read the current actor and critic state. -/
  state : IO (TensorPack Float stateShapes)
  /-- Action logits of one observation under the actor part of a state. -/
  policy : TensorPack Float stateShapes → Tensor Float obsShape → Tensor Float [nActions]
  /-- Value estimate of one observation under the critic part of a state. -/
  value : TensorPack Float stateShapes → Tensor Float obsShape → Float
  /-- One PPO optimizer step on a rollout batch. The optimizer history persists across calls. -/
  updateBatch : RL.PPO.TrainingBatch Float obsShape nActions horizon → IO Unit

namespace PPOAgent

/-- Build the actor and critic of `config` for batches of `horizon` observations and bind their PPO
update to `optimizer`. `sameActorState` and `sameCriticState` say that the batched and the
single-observation copy of each network have the same state shapes. -/
def create (config : nn.PPO.Config) (horizon : Nat) (runtime : Session.Config)
    (optimizer : optim.Optimizer)
    (sameActorState : ∀ seed,
      nn.Sequential.stateShapes (nn.build seed (nn.PPO.actor config [horizon])) =
        nn.Sequential.stateShapes (nn.build seed (nn.PPO.actor config)) :=
          by intro _; rfl)
    (sameCriticState : ∀ seed,
      nn.Sequential.stateShapes (nn.build seed (nn.PPO.critic config [horizon])) =
        nn.Sequential.stateShapes (nn.build seed (nn.PPO.critic config)) :=
          by intro _; rfl) :
    IO (PPOAgent (config.inputShape []) config.actionCount horizon) := do
  let seedActor ← Random.nextSeedGlobal
  let seedCritic ← Random.nextSeedGlobal
  let actorRollout := nn.build seedActor (nn.PPO.actor config [horizon])
  let criticRollout := nn.build seedCritic (nn.PPO.critic config [horizon])
  -- The single-observation copies only supply the computation: `policy` and `value` evaluate
  -- their graphs with the parameters that `actorCritic` trains.
  let actorGraph ← nn.Sequential.lowerToTypedGraph (α := Float)
    (nn.build seedActor (nn.PPO.actor config))
  let criticGraph ← nn.Sequential.lowerToTypedGraph (α := Float)
    (nn.build seedCritic (nn.PPO.critic config))
  let actorCritic ← RL.PPO.ActorCritic.create (α := Float) (options := runtime)
    (batch := horizon) (nActions := config.actionCount) actorRollout criticRollout
  let updateBatch ← actorCritic.bind optimizer
  pure
    { stateShapes :=
        nn.Sequential.stateShapes actorRollout ++ nn.Sequential.stateShapes criticRollout
      state := actorCritic.state
      policy := fun state =>
        RL.PPO.policy actorGraph actorRollout criticRollout state (sameActorState seedActor)
      value := fun state =>
        RL.PPO.value criticGraph actorRollout criticRollout state (sameCriticState seedCritic)
          (oneValue := rfl)
      updateBatch }

end PPOAgent

end TorchLean.Examples
