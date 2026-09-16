/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.Module -- shake: keep
public import NN.API.Neural.Execution -- shake: keep
public import NN.API.RL.Core -- shake: keep
public import NN.Runtime.RL.Boundary -- shake: keep
public import NN.Runtime.RL.Gymnasium -- shake: keep
public import NN.Runtime.RL.Numerics -- shake: keep
public import NN.Runtime.RL.PPO -- shake: keep

/-!
# RL Runtime

Rollout boundary checks, Gymnasium sessions, Float32 and interval numerics, and PPO actor-critic
wiring exposed under `TorchLean.rl`.
-/

@[expose] public section

namespace TorchLean
namespace rl

namespace boundary
export Runtime.RL.Boundary
  (isFiniteFloat tensorAll tensorFinite tensorInClosedInterval
   Contract Transition
   checkAction
   checkObservation checkReward checkDoneFlags
   checkTransitionFin checkTransition
   parseTransitionJson)
export Runtime.RL.Boundary.Transition (done)

/-!
## Casting to Other Scalar Backends

The trust-boundary checker validates rollout JSON in host `Float`, because that is the interchange
format. The functions below cast accepted rollouts into the element representation chosen for the
proof or training path.
-/

/-- Cast a `Float` observation tensor into a runtime element representation `α`. -/
def castObservation {α : Type} [TorchLean.Storage α] [Runtime.FromFloat α] {obsShape : Shape}
    (t : Tensor Float obsShape) : Tensor α obsShape :=
  TorchLean.Tensor.map (Runtime.ofFloat (α := α)) t

/-- Cast a validated `Float` transition into a runtime element representation `α`. -/
def castTransition {α : Type} [TorchLean.Storage α] [Runtime.FromFloat α]
    {obsShape : Shape} {nActions : Nat}
    (tr : Transition obsShape nActions) :
    Spec.RL.ObservedTransition (Tensor α obsShape) (Fin nActions) α :=
  { observation := castObservation (α := α) tr.observation
    action := tr.action
    reward := Runtime.ofFloat (α := α) tr.reward
    nextObservation := castObservation (α := α) tr.nextObservation
    terminated := tr.terminated
    truncated := tr.truncated }

/-- Cast a whole rollout into a runtime element representation `α`. -/
def castRollout {α : Type} [TorchLean.Storage α] [Runtime.FromFloat α]
    {obsShape : Shape} {nActions : Nat}
    (xs : Array (Transition obsShape nActions)) :
    Array (Spec.RL.ObservedTransition (Tensor α obsShape) (Fin nActions) α) :=
  xs.map (castTransition (α := α) (obsShape := obsShape) (nActions := nActions))

/--
Load and validate a rollout JSON file in the requested element type.

Host `Float` is the default interchange representation. Select another executable or proof-facing
representation with `(α := ...)`.
-/
def loadRollout {obsShape : Shape} {nActions : Nat}
    (path : String)
    (c : Contract obsShape nActions)
    (α : Type := Float)
    [TorchLean.Storage α] [Runtime.FromFloat α] :
    IO (Array (Spec.RL.ObservedTransition (Tensor α obsShape) (Fin nActions) α)) := do
  let xs ← Runtime.RL.Boundary.loadRollout path c
  pure (castRollout (α := α) xs)

end boundary

namespace numerics
namespace float32
export Runtime.RL.Numerics.Float32
  (Float32Exec Interval32
   ofFloatChecked castTensorChecked castTransitionChecked
   discountedBackupChecked discountedReturnsChecked
   tdResidualChecked
   generalizedAdvantageEstimationChecked
   normalizeZScoreChecked
   importanceRatioChecked
   ppoClippedObjectiveFromRatioChecked
   discountedBackupInterval tdResidualInterval
   ppoClippedObjectiveFromRatioInterval
   discountedReturnsIntervals generalizedAdvantageEstimationIntervals
   returnsWithinIntervals)
end float32
end numerics

namespace session
export Runtime.RL.Session (CheckedSession)
export Runtime.RL.Session.CheckedSession (gymnasium ofEnv)
end session

namespace gym
export Runtime.RL.Gymnasium (Client Session)

namespace client
-- Only export the stable high-level entry points. The JSON request/response protocol and raw-step
-- protocol remain behind `NN.Runtime.RL.Gymnasium`.
export Runtime.RL.Gymnasium.Client (spawn reset close withClient)
end client

namespace session
export Runtime.RL.Gymnasium.Session (start reset stepChecked close withSession)
end session

end gym

namespace ppo
export Runtime.RL.PPO
  (StateBatchShape LogitsBatchShape ScalarBatchShape ValueBatchShape
   Step Rollout TrainingBatch TrainConfig train
   collectRolloutFromCallbacks collectRolloutFromSession collectRolloutFromGymnasium)
export Runtime.RL.PPO.Rollout (trainingBatch)

/--
PPO runtime state. Stateful layers update their buffers from the actual training forward pass.
-/
structure ActorCritic (α : Type) [TorchLean.Storage α] [Context α]
    (stateShapes : List Spec.Shape) (stateShape : Spec.Shape) (batch nActions : Nat) where
  private mk ::
  private objective : TorchLean.Module.Objective α Unit stateShapes
    [stateShape, [batch, nActions], [batch], [batch], [batch, 1]]

/--
Instantiate the standard PPO actor-critic runtime.

The actor and critic share the objective's forward execution and optimizer history.
-/
@[no_expose] def instantiateActorCritic
    {stateShape : Spec.Shape} {batch nActions : Nat} {α : Type}
    [TorchLean.Storage α] [Context α]
    [TorchLean.Runtime.FromFloat α]
    [Runtime.TensorTransfer α]
    (options : Runtime.Autograd.Torch.Config)
    (actor : Runtime.Autograd.Model.Layers.Seq stateShape [batch, nActions])
    (critic : Runtime.Autograd.Model.Layers.Seq stateShape [batch, 1]) :
    IO (ActorCritic α
      (Runtime.Autograd.Model.Layers.Seq.stateShapes actor
        ++ Runtime.Autograd.Model.Layers.Seq.stateShapes critic)
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
        (Runtime.RL.PolicyGradient.Autograd.ppoActorCriticObjectiveDef
          (batch := batch) (nActions := nActions) actor critic)
        options
      pure ⟨objective⟩

/--
Bind a PPO actor-critic update function and preserve its optimizer history across calls.

Each call updates model buffers from the activations used to compute the gradients, then performs
the optimizer step. This includes every repeated PPO epoch over the same rollout batch.
-/
@[no_expose] def trainingStep {α : Type}
    [TorchLean.Storage α] [Context α] [TorchLean.Runtime.FromFloat α]
    {stateShapes : List Spec.Shape} {obsShape : Spec.Shape} {batch nActions : Nat}
    (m : ActorCritic α stateShapes
      (Runtime.RL.PPO.StateBatchShape batch obsShape) batch nActions)
    (config : TorchLean.optim.Optimizer) :
    IO (Runtime.RL.PPO.TrainingBatch α obsShape nActions batch → IO Unit) := do
  let step ← TorchLean.Module.Internal.packStep m.objective config
  pure fun trainingBatch => do
    step <| TorchLean.Arguments.Internal.fromTensorPack
      (Runtime.RL.PPO.TrainingBatch.Internal.arguments trainingBatch)

/-- Read concatenated actor-critic state without refreshing buffers. -/
@[no_expose] def state {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Spec.Shape} {stateShape : Spec.Shape} {batch nActions : Nat}
    (m : ActorCritic α stateShapes stateShape batch nActions) :
    IO (nn.State α stateShapes) :=
  TorchLean.Module.Objective.state m.objective

/--
Restore actor and critic parameters and persistent buffers.

An already-bound `trainingStep` keeps its optimizer history; this restores model state only.
-/
@[no_expose] def setState {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Spec.Shape} {stateShape : Spec.Shape} {batch nActions : Nat}
    (m : ActorCritic α stateShapes stateShape batch nActions)
    (newState : nn.State α stateShapes) : IO Unit :=
  TorchLean.Module.Objective.setState m.objective newState

/-- Actor and critic states, including parameters and persistent buffers. -/
structure ActorCriticState (α : Type) [TorchLean.Storage α]
    (actorShapes criticShapes : List Spec.Shape) where
  /-- Parameters and persistent buffers consumed by the actor graph. -/
  actor : nn.State α actorShapes
  /-- Parameters and persistent buffers consumed by the critic graph. -/
  critic : nn.State α criticShapes

/-- Split concatenated actor-critic state into its actor and critic components. -/
def splitState
    {σ₁ τ₁ σ₂ τ₂ : Spec.Shape}
    (actor : Runtime.Autograd.Model.Layers.Seq σ₁ τ₁)
    (critic : Runtime.Autograd.Model.Layers.Seq σ₂ τ₂)
    {α : Type} [TorchLean.Storage α]
    (state : nn.State α
        (Runtime.Autograd.Model.Layers.Seq.stateShapes actor ++
          Runtime.Autograd.Model.Layers.Seq.stateShapes critic)) :
    ActorCriticState α
      (Runtime.Autograd.Model.Layers.Seq.stateShapes actor)
      (Runtime.Autograd.Model.Layers.Seq.stateShapes critic) :=
  let partition := state.split
  { actor := partition.left
    critic := partition.right }

/--
Build a single-observation actor policy from the state of a rollout-shaped actor-critic module.

The typed actor graph records its state layout, while `sameActorState` states that the rollout actor
uses that layout as well.
-/
def actorPolicy
    {obsShape logitsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : Spec.Shape}
    {actorStateShapes : List Spec.Shape}
    {α : Type} [TorchLean.Storage α] [Context α]
    (actorGraph : nn.TypedGraphModel actorStateShapes obsShape logitsShape α)
    (actorRollout : nn.Sequential rolloutStateShape rolloutLogitsShape)
    (criticRollout : nn.Sequential rolloutStateShape rolloutValueShape)
    (state : nn.State α
      (nn.stateShapes actorRollout ++ nn.stateShapes criticRollout))
    (sameActorState : nn.stateShapes actorRollout = actorStateShapes := by rfl) :
    Tensor α obsShape → Tensor α logitsShape :=
  let actorState := (splitState actorRollout criticRollout state).actor
  let actorState : nn.State α actorStateShapes :=
    actorState.cast sameActorState
  fun obs => actorGraph.forward actorState obs

/--
Build a single-observation critic function from the state of a rollout-shaped actor-critic module.

The result is scalar because the typed critic graph has a checked one-element output shape.
-/
def criticValue
    {obsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : Spec.Shape}
    {criticStateShapes : List Spec.Shape}
    {α : Type} [TorchLean.Storage α] [Context α]
    (criticGraph : nn.TypedGraphModel criticStateShapes obsShape [1] α)
    (actorRollout : nn.Sequential rolloutStateShape rolloutLogitsShape)
    (criticRollout : nn.Sequential rolloutStateShape rolloutValueShape)
    (state : nn.State α
      (nn.stateShapes actorRollout ++ nn.stateShapes criticRollout))
    (sameCriticState : nn.stateShapes criticRollout = criticStateShapes := by rfl) :
    Tensor α obsShape → α :=
  let criticState := (splitState actorRollout criticRollout state).critic
  let criticState : nn.State α criticStateShapes :=
    criticState.cast sameCriticState
  fun obs =>
    (criticGraph.forward criticState obs)[0]

end ppo

end rl
end TorchLean
