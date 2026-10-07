/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.PPO.Rollout
public import NN.Runtime.RL.Session
public import NN.Runtime.RL.Algorithms.PolicyGradient

/-!
# PPO Rollout Collection (Checked Sessions)

This file provides the rollout-collection loop used by executable PPO workflows. The key goals are:

- keep data collection typed and total (no “parallel arrays” that can desync),
- enforce the trust-boundary contract on every step (external Gymnasium or Lean-native env), and
- keep the API usable: callers should not need to thread a dozen actor/critic lowering details
  through every function call.

The unified session interface lives in `NN.Runtime.RL.Session` (`Session.CheckedSession`).
The lower-level Gymnasium subprocess protocol is implemented in `NN.Runtime.RL.Gymnasium`.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- Gymnasium API reference (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
-/

@[expose] public section

namespace Runtime
namespace RL
namespace PPO

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-!
## Rollout collection (ergonomic core API)
-/

/--
Collect a fixed-horizon rollout from any *stateful* environment session that can produce
fully-observed, contract-checked transitions.

The caller provides:

- `session`: initialization, observation, and checked stepping for the environment,
- `castObservation` to inject host `Float` observations into the chosen scalar backend `α`,
- `castReward` to inject host `Float` rewards into the chosen scalar backend `α`,
- `predictLogits` for the current actor,
- `predictValue` for the current critic (returns a scalar `α`).

The API supports the “typed graph + parameters” calling convention used throughout TorchLean.
Stored log-probabilities use `logSoftmax` without clamping. When using this rollout with
`Autograd.PPO.objective`, selected log-probabilities must lie inside its `[-10^30, 10^30]`
clamp and the policy reductions must agree for identical parameters to yield a unit ratio.
-/
def collect {obsShape : Shape} {nActions horizon : Nat}
    [NeZero horizon] [NeZero nActions]
    (session : Session.CheckedSession obsShape nActions)
    (castObservation : Float → α)
    (castReward : Float → α)
    (predictLogits : Tensor α obsShape → Tensor α [nActions])
    (predictValue : Tensor α obsShape → α)
    (rngSeed rngCounter : Nat) :
    IO (Rollout α obsShape nActions horizon × Nat) := do

  let mut state ← session.start

  let mut steps : Array (Step α obsShape nActions) := #[]
  let mut counter := rngCounter

  for _t in [0:horizon] do
    let obsF := session.observe state
    let obs : Tensor α obsShape := TorchLean.Tensor.map castObservation obsF

    let logits : Tensor α [nActions] := predictLogits obs
    let (counter', a) :=
      PolicyGradient.sample (α := α) (nActions := nActions)
        (seed := rngSeed) (counter := counter) logits (logits := true)
    counter := counter'

    -- Cache the raw behavior-policy value; the autograd objective clamps its new value.
    let lp : α := PolicyGradient.logSoftmax (α := α) (nActions := nActions) logits a
    let v : α := predictValue obs

    let (tr, nextState) ← session.stepChecked state a
    state := nextState

    let done : Bool := tr.terminated || tr.truncated
    let nextObs : Tensor α obsShape := TorchLean.Tensor.map castObservation tr.nextObservation
    let nv : α := predictValue nextObs

    steps := steps.push
      { state := obs
        action := tr.action
        oldLogProb := lp
        reward := castReward tr.reward
        done := done
        value := v
        nextValue := nv
        terminated := tr.terminated }

  if h : steps.size = horizon then
    pure ({ steps := steps, steps_size_eq_horizon := h }, counter)
  else
    throw <|
      IO.userError
        (s!"PPO.collect: internal error (steps.size={steps.size}, "
          ++ s!"horizon={horizon})")

/-!
## Rollout collection from Gymnasium (subprocess bridge)
-/

/--
Collect a fixed-horizon rollout from a Gymnasium subprocess environment.

This specializes `collect` to `Gymnasium.Session`.
-/
def collectRolloutFromGymnasium {obsShape : Shape} {nActions horizon : Nat}
    [NeZero horizon] [NeZero nActions]
    (castObservation : Float → α)
    (castReward : Float → α)
    (gym : Gymnasium.Client obsShape nActions)
    (predictLogits : Tensor α obsShape → Tensor α [nActions])
    (predictValue : Tensor α obsShape → α)
    (rngSeed rngCounter : Nat)
    (resetSeed : Nat) :
    IO (Rollout α obsShape nActions horizon × Nat) := do
  let sess : Session.CheckedSession obsShape nActions :=
    Session.CheckedSession.gymnasium (obsShape := obsShape) (nActions := nActions) gym
      (seed? := some resetSeed) (resetOnDone := true)
  collect (α := α) (obsShape := obsShape) (nActions := nActions)
    (horizon := horizon)
    sess castObservation castReward predictLogits predictValue rngSeed rngCounter

end PPO
end RL
end Runtime
