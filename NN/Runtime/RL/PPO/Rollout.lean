/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Runtime.Autograd.TorchLean.NN
public import NN.Spec.Models.CommonHelpers

/-!
# PPO Rollouts (Discrete Actions)

This file defines:

- fixed-horizon PPO rollout records stored as typed tensors / arrays, and
- a conversion to the minibatch format expected by the PPO autograd loss module
  (`Runtime.RL.PolicyGradient.Autograd.ppoActorCriticScalarModuleDef`).

The goal is not to hide PPO’s math: the GAE/return definitions live in `NN.Spec.RL.Core` and the
tensor-shaped analogues live in `NN.Runtime.RL.Core`. This file is the typed rollout layer for
PPO training loops.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015):
  https://arxiv.org/abs/1506.02438
-/

@[expose] public section

namespace Runtime
namespace RL
namespace PPO

open _root_.Spec
open _root_.Spec.Tensor

variable {α : Type} [Context α] [DecidableEq Shape]

/-!
## Shapes

For a fixed horizon `T`, PPO minibatches are typically stored in "PyTorch-shaped" tensors:

- `states : (T × obsShape)`
- `actionsOneHot : (T × nActions)`
- `oldLogProb : (T)`
- `advantages : (T)`
- `valueTarget : (T × 1)`
-/

/-- Batch shape for a fixed-horizon sequence of observations: `horizon × obsShape`. -/
abbrev StateBatchShape (horizon : Nat) (obsShape : Shape) : Shape :=
  .dim horizon obsShape

/-- Batch shape for a fixed-horizon sequence of action logits: `horizon × nActions`. -/
abbrev LogitsBatchShape (horizon nActions : Nat) : Shape :=
  .dim horizon (.dim nActions .scalar)

/-- Batch shape for a fixed-horizon sequence of scalars: `horizon`. -/
abbrev ScalarBatchShape (horizon : Nat) : Shape :=
  .dim horizon .scalar

/-- Batch shape for a fixed-horizon sequence of scalar values stored as a column: `horizon × 1`. -/
abbrev ValueBatchShape (horizon : Nat) : Shape :=
  .dim horizon (.dim 1 .scalar)

/-!
## Rollouts
-/

/--
One fixed-horizon PPO step record.

This is the “typed parallel arrays” data layout commonly used in PPO implementations, but kept as
a single record so downstream code cannot accidentally desynchronize fields.
-/
structure Step (α : Type) (obsShape : Shape) (nActions : Nat) where
  /-- Observation `s_t` (already cast into the training scalar backend). -/
  state : Tensor α obsShape
  /-- Sampled action `a_t`. -/
  action : Fin nActions
  /-- Log-probability `log π_old(a_t | s_t)` under the behavior policy. -/
  oldLogProb : α
  /-- Reward `r_t`. -/
  reward : α
  /-- Episode boundary marker (Gym-style `terminated || truncated`). -/
  done : Bool
  /-- Baseline value prediction `V(s_t)`. -/
  value : α
  /-- Bootstrap value prediction `V(s_{t+1})` (before any auto-reset). -/
  nextValue : α

/--
Fixed-horizon rollout buffer for PPO.

The `steps_size_eq_horizon` field records the invariant that the buffer has exactly `horizon`
steps; this lets downstream tensor conversion be total without runtime bounds checks.
-/
structure Rollout (α : Type) (obsShape : Shape) (nActions horizon : Nat) where
  steps : Array (Step α obsShape nActions)
  /-- Invariant: fixed-horizon rollouts always have exactly `horizon` steps. -/
  steps_size_eq_horizon : steps.size = horizon

namespace Rollout

/--
Convert a fixed-horizon rollout into the PPO minibatch expected by
`Autograd.ppoActorCriticScalarModuleDef`.

Notes:

- Advantages are normalized (z-score) for the policy-gradient term, a common PPO
  variance-reduction practice.
  Value targets (lambda-returns) are computed from the *unnormalized* advantages.
-/
def toActorCriticSample {obsShape : Shape} {nActions horizon : Nat}
    [Fact (0 < horizon)] [Fact (0 < nActions)]
    (gamma lam : α)
    (r : Rollout α obsShape nActions horizon) :
    IO (_root_.Runtime.Autograd.Torch.TList α
      [StateBatchShape horizon obsShape,
       LogitsBatchShape horizon nActions,
       ScalarBatchShape horizon,
       ScalarBatchShape horizon,
       ValueBatchShape horizon]) := do
  let steps := r.steps

  let statesArr : Array (Tensor α obsShape) :=
    steps.map (fun st => st.state)
  let hStates : horizon = statesArr.size := by
    have : statesArr.size = horizon := by
      simpa [statesArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm
  let states : Tensor α (StateBatchShape horizon obsShape) :=
    Tensor.ofArrayDim (n := horizon) (s := obsShape) statesArr hStates

  let actionsOneHotArr : Array (Tensor α (.dim nActions .scalar)) :=
    steps.map (fun st => Core.oneHotAction (α := α) (nActions := nActions) st.action)
  let hActHot : horizon = actionsOneHotArr.size := by
    have : actionsOneHotArr.size = horizon := by
      simpa [actionsOneHotArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm
  let actionsOneHot : Tensor α (LogitsBatchShape horizon nActions) :=
    Tensor.ofArrayDim (n := horizon) (s := .dim nActions .scalar) actionsOneHotArr hActHot

  let oldLogProbArr : Array α := steps.map (fun st => st.oldLogProb)
  let hOldLP : horizon = oldLogProbArr.size := by
    have : oldLogProbArr.size = horizon := by
      simpa [oldLogProbArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm
  let oldLogProb : Tensor α (ScalarBatchShape horizon) :=
    Tensor.ofArray1D (α := α) (n := horizon) oldLogProbArr hOldLP

  let rewardsArr : Array α := steps.map (fun st => st.reward)
  let donesArr : Array Bool := steps.map (fun st => st.done)
  let valuesArr : Array α := steps.map (fun st => st.value)
  let nextValuesArr : Array α := steps.map (fun st => st.nextValue)

  let hRewards : horizon = rewardsArr.size := by
    have : rewardsArr.size = horizon := by
      simpa [rewardsArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm
  let hDones : horizon = donesArr.size := by
    have : donesArr.size = horizon := by
      simpa [donesArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm
  let hValues : horizon = valuesArr.size := by
    have : valuesArr.size = horizon := by
      simpa [valuesArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm
  let hNextValues : horizon = nextValuesArr.size := by
    have : nextValuesArr.size = horizon := by
      simpa [nextValuesArr, Array.size_map] using r.steps_size_eq_horizon
    simpa using this.symm

  let rewards : Tensor α (.dim horizon .scalar) :=
    Tensor.ofArray1D (α := α) (n := horizon) rewardsArr hRewards
  let dones : Tensor Bool (.dim horizon .scalar) :=
    Tensor.ofArray1D (α := Bool) (n := horizon) donesArr hDones
  let values : Tensor α (.dim horizon .scalar) :=
    Tensor.ofArray1D (α := α) (n := horizon) valuesArr hValues
  let nextValues : Tensor α (.dim horizon .scalar) :=
    Tensor.ofArray1D (α := α) (n := horizon) nextValuesArr hNextValues

  let advRaw :=
    Core.generalizedAdvantageEstimationVec (α := α) (n := horizon) gamma lam rewards values nextValues dones
  let returns := Core.returnsFromAdvantagesVec (α := α) (n := horizon) advRaw values
  let advantages := Spec.normalizeZscoreSpec (α := α) (n := horizon) advRaw

  let valueTarget : Tensor α (ValueBatchShape horizon) := Tensor.vecToCol returns
  let advantagesT : Tensor α (ScalarBatchShape horizon) := advantages

  pure <|
    .cons states <|
      .cons actionsOneHot <|
        .cons oldLogProb <|
          .cons advantagesT <|
            .cons valueTarget .nil

end Rollout

end PPO
end RL
end Runtime
