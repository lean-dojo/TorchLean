/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream modules reach the marked imports through this module. Import shaking cannot see
-- those downstream uses, so keep the marked imports.
module -- shake: keep-downstream

public import TorchLean.NN.RL.DQN -- shake: keep
public import TorchLean.NN.RL.DQN.Autograd -- shake: keep
public import TorchLean.NN.RL.Eval -- shake: keep
public import TorchLean.NN.RL.Replay -- shake: keep
public import TorchLean.NN.Module -- shake: keep
public import TorchLean.NN.Sequential.Execution -- shake: keep
public import TorchLean.NN.RL.Numerics.Advantage -- shake: keep
public import TorchLean.NN.RL.Numerics.Intervals -- shake: keep
public import TorchLean.NN.RL.Numerics.PPO -- shake: keep
public import TorchLean.NN.RL.PPO.Collect -- shake: keep
public import TorchLean.NN.RL.PPO.Training -- shake: keep

/-!
# PPO Actor-Critic Runtime

This module defines the PPO actor-critic runtime `RL.PPO.ActorCritic` and publicly imports the DQN,
replay, evaluation, PPO collection and training, and RL numerics modules.

We built that runtime by hand around Algorithm 1 of Schulman et al. (2017). Actor and critic keep
separate parameters, as in their MuJoCo runs, for which Section 6.1 calls $c_1$ irrelevant.
`RL.PPO.Autograd.create` adds the two losses so one optimizer step moves both networks. The value
loss reaches the critic alone, so the `valueCoef` of `RL.PPO.Autograd.loss` scales the critic's
gradient and nothing else. With SGD its default 1/2 halves the critic's step. Adam's
$\hat m / \sqrt{\hat v}$ cancels a positive constant factor, so with Adam or AdamW the coefficient
acts through `eps` alone.

References (background and terminology):
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.):
  http://incompleteideas.net/book/the-book-2nd.html
- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (Wiley, 1994),
  for finite discounted MDPs: https://doi.org/10.1002/9780470316887
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017), arXiv:1707.06347, Algorithm 1
  and Section 6.1: https://arxiv.org/abs/1707.06347
- Gymnasium API reference (reset/step, `terminated` vs `truncated`):
  https://gymnasium.farama.org/
-/

@[expose] public section

namespace TorchLean
namespace RL

namespace PPO

/--
PPO runtime state. Stateful layers update their buffers from the actual training forward pass.
-/
structure ActorCritic (α : Type) [TorchLean.Storage α] [Context α]
    (stateShapes : List Shape) (stateShape : Shape) (batch nActions : Nat) where
  private mk ::
  private objective : TorchLean.Module.Objective α (Fin nActions) stateShapes
    [stateShape, [batch], [batch], [batch, 1]] [[batch]]

/--
Instantiate the standard PPO actor-critic runtime, failing when `batch` or `nActions` is zero.

The actor and critic share one objective, and `ActorCritic.bind` attaches an optimizer to it and
returns an update callback with persistent optimizer history.
-/
@[no_expose] def ActorCritic.create
    {stateShape : Shape} {batch nActions : Nat} {α : Type}
    [TorchLean.Storage α] [Context α]
    [FromFloat α]
    [TensorTransfer α]
    (options : Session.Config)
    (actor : nn.Sequential stateShape [batch, nActions])
    (critic : nn.Sequential stateShape [batch, 1]) :
    IO (ActorCritic α
      (nn.Sequential.stateShapes actor
        ++ nn.Sequential.stateShapes critic)
      stateShape batch nActions) := do
  if hBatch : batch = 0 then
    throw <| IO.userError "PPO batch size must be positive"
  else if hActions : nActions = 0 then
    throw <| IO.userError "PPO action count must be positive"
  else
    letI : NeZero batch := ⟨hBatch⟩
    letI : NeZero nActions := ⟨hActions⟩
    do
      let objective ← TorchLean.Module.instantiate (α := α)
        (RL.PPO.Autograd.create
          (batch := batch) (nActions := nActions) actor critic)
        options
      pure ⟨objective⟩

/--
Bind a PPO actor-critic update function and preserve its optimizer history across calls.

`config` is checked with `optim.Optimizer.validateFor` for `α`, and on CUDA for binary32 as well,
before the optimizer state is created. Each call of the returned function, every repeated PPO
epoch over the same rollout batch included, updates the model buffers from the activations that
computed the gradients and then takes the optimizer step.
-/
@[no_expose] def ActorCritic.bind {α : Type}
    [TorchLean.Storage α] [Context α] [FromFloat α]
    {stateShapes : List Shape} {obsShape : Shape} {batch nActions : Nat}
    (m : ActorCritic α stateShapes
      (RL.PPO.stateBatchShape batch obsShape) batch nActions)
    (config : TorchLean.optim.Optimizer) :
    IO (RL.PPO.TrainingBatch α obsShape nActions batch → IO Unit) := do
  let step ← Module.packStep m.objective config
  pure fun trainingBatch => do
    step
      (RL.PPO.TrainingBatch.arguments trainingBatch)
      (.cons trainingBatch.actions .nil)

/-- Read concatenated actor-critic state without refreshing buffers. -/
@[no_expose] def ActorCritic.state {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Shape} {stateShape : Shape} {batch nActions : Nat}
    (m : ActorCritic α stateShapes stateShape batch nActions) :
    IO (TensorPack α stateShapes) :=
  m.objective.runtime.state

/--
Restore actor and critic parameters and persistent buffers.

Only the model state changes, so a callback that `ActorCritic.bind` returned earlier keeps its
optimizer history.
-/
@[no_expose] def ActorCritic.setState {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Shape} {stateShape : Shape} {batch nActions : Nat}
    (m : ActorCritic α stateShapes stateShape batch nActions)
    (newState : TensorPack α stateShapes) : IO Unit :=
  m.objective.runtime.loadState newState

/--
Build a single-observation actor policy from the state of a rollout-shaped actor-critic module.

The typed actor graph records its state layout, while `sameActorState` states that the rollout actor
uses that layout as well.
-/
def policy
    {obsShape logitsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : Shape}
    {actorStateShapes : List Shape}
    {α : Type} [TorchLean.Storage α]
    (actorGraph : nn.TypedGraphModel actorStateShapes obsShape logitsShape α)
    (actorRollout : nn.Sequential rolloutStateShape rolloutLogitsShape)
    (criticRollout : nn.Sequential rolloutStateShape rolloutValueShape)
    (state : TensorPack α
      (nn.Sequential.stateShapes actorRollout ++ nn.Sequential.stateShapes criticRollout))
    (sameActorState : nn.Sequential.stateShapes actorRollout = actorStateShapes := by rfl) :
    Tensor α obsShape → Tensor α logitsShape :=
  let actorState := state.split.1
  let actorState : TensorPack α actorStateShapes :=
    actorState.cast sameActorState
  fun obs => actorGraph.forward actorState obs

/--
Build a single-observation critic function from the state of a rollout-shaped actor-critic module.

The result is a scalar because `oneValue` proves that the output shape of the typed critic graph
has exactly one element, which holds for a scalar shape and for any number of singleton axes.
-/
def value
    {obsShape valueShape rolloutStateShape rolloutLogitsShape rolloutValueShape : Shape}
    {criticStateShapes : List Shape}
    {α : Type} [TorchLean.Storage α]
    (criticGraph : nn.TypedGraphModel criticStateShapes obsShape valueShape α)
    (actorRollout : nn.Sequential rolloutStateShape rolloutLogitsShape)
    (criticRollout : nn.Sequential rolloutStateShape rolloutValueShape)
    (state : TensorPack α
      (nn.Sequential.stateShapes actorRollout ++ nn.Sequential.stateShapes criticRollout))
    (sameCriticState : nn.Sequential.stateShapes criticRollout = criticStateShapes := by rfl)
    (oneValue : valueShape.size = 1 := by decide) :
    Tensor α obsShape → α :=
  let criticState := state.split.2
  let criticState : TensorPack α criticStateShapes :=
    criticState.cast sameCriticState
  fun obs =>
    Tensor.item (Tensor.reshape (criticGraph.forward criticState obs) [] oneValue)

end PPO

end RL
end TorchLean
