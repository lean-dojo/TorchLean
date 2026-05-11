/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.PPO.Rollout
public import NN.Runtime.RL.Gymnasium
public import NN.Runtime.RL.Session
public import NN.Runtime.RL.Algorithms.PolicyGradient

/-!
# PPO Rollout Collection (Checked Sessions)

This file provides the rollout-collection loop used by executable PPO workflows. The key goals are:

- keep data collection typed and total (no “parallel arrays” that can desync),
- enforce the trust-boundary contract on every step (external Gymnasium or Lean-native env), and
- keep the API usable: callers should not need to thread a dozen actor/critic compilation details
  through every function call.

The unified session interface lives in `NN.Runtime.RL.Session` (`Session.CheckedSession`).
The lower-level Gymnasium subprocess protocol is implemented in `NN.Runtime.RL.Gymnasium`.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- Gymnasium API docs (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
-/

@[expose] public section

namespace Runtime
namespace RL
namespace PPO

open Spec
open Tensor

variable {α : Type} [Context α] [DecidableEq Shape]

/-!
## Rollout collection (ergonomic core API)
-/

/--
Collect a fixed-horizon rollout from any *stateful* environment session that can produce
fully-observed, contract-checked transitions.

The caller provides:

- `start`: how to initialize the session (often `reset`),
- `observe`: how to read the current observation from the session,
- `stepChecked`: one checked step returning an observed transition and the updated session,
- `castObs` to inject host `Float` observations into the chosen scalar backend `α`,
- `castReward` to inject host `Float` rewards into the chosen scalar backend `α`,
- `predictLogits` for the current actor,
- `predictValue` for the current critic (returns a scalar `α`).

This keeps the PPO runtime API small while still supporting the “compiled model + parameters”
calling convention used throughout TorchLean.
-/
def collectRolloutSessionWith {obsShape : Shape} {nActions horizon : Nat} {Sess : Type}
    [Fact (0 < horizon)] [Fact (0 < nActions)]
    (start : IO Sess)
    (observe : Sess → Tensor Float obsShape)
    (stepChecked : Sess → Fin nActions → IO (Boundary.Transition obsShape nActions × Sess))
    (castObs : Float → α)
    (castReward : Float → α)
    (predictLogits : Tensor α obsShape → Tensor α (.dim nActions .scalar))
    (predictValue : Tensor α obsShape → α)
    (rngSeed rngCounter : Nat) :
    IO (Rollout α obsShape nActions horizon × Nat) := do

  let mut sess ← start

  let mut steps : Array (Step α obsShape nActions) := #[]
  let mut counter := rngCounter

  for _t in [0:horizon] do
    let obsF := observe sess
    let obs : Tensor α obsShape := Spec.mapTensor castObs obsF

    let logits : Tensor α (.dim nActions .scalar) := predictLogits obs
    let (counter', a) :=
      PolicyGradient.sampleActionFromLogits (α := α) (nActions := nActions)
        (seed := rngSeed) (counter := counter) logits
    counter := counter'

    let lp : α := PolicyGradient.actionLogProbability (α := α) (nActions := nActions) logits a
    let v : α := predictValue obs

    let (tr, sess') ← stepChecked sess a
    sess := sess'

    let done : Bool := tr.terminated || tr.truncated
    let nextObs : Tensor α obsShape := Spec.mapTensor castObs tr.nextObservation
    let nv : α := predictValue nextObs

    steps := steps.push
      { state := obs
        action := tr.action
        oldLogProb := lp
        reward := castReward tr.reward
        done := done
        value := v
        nextValue := nv }

  if h : steps.size = horizon then
    pure ({ steps := steps, steps_size_eq_horizon := h }, counter)
  else
    throw <|
      IO.userError
        s!"PPO.collectRolloutSessionWith: internal error (steps.size={steps.size}, horizon={horizon})"

/-!
## Rollout collection from a checked session
-/

/--
Collect a fixed-horizon rollout from a unified `Runtime.RL.Session.CheckedSession`.
-/
def collectRolloutCheckedSessionWith {obsShape : Shape} {nActions horizon : Nat}
    [Fact (0 < horizon)] [Fact (0 < nActions)]
    (sess : Session.CheckedSession obsShape nActions)
    (castObs : Float → α)
    (castReward : Float → α)
    (predictLogits : Tensor α obsShape → Tensor α (.dim nActions .scalar))
    (predictValue : Tensor α obsShape → α)
    (rngSeed rngCounter : Nat) :
    IO (Rollout α obsShape nActions horizon × Nat) :=
  collectRolloutSessionWith (α := α) (obsShape := obsShape) (nActions := nActions) (horizon := horizon)
    (Sess := sess.Sess)
    (start := sess.start) (observe := sess.observe) (stepChecked := sess.stepChecked)
    castObs castReward predictLogits predictValue rngSeed rngCounter

/-!
## Rollout collection from Gymnasium (subprocess bridge)
-/

/--
Collect a fixed-horizon rollout from a Gymnasium subprocess environment.

This is a thin wrapper around `collectRolloutSessionWith` specialized to `Gymnasium.Session`.
-/
def collectRolloutWith {obsShape : Shape} {nActions horizon : Nat}
    [Fact (0 < horizon)] [Fact (0 < nActions)]
    (castObs : Float → α)
    (castReward : Float → α)
    (gym : Gymnasium.Client obsShape nActions)
    (predictLogits : Tensor α obsShape → Tensor α (.dim nActions .scalar))
    (predictValue : Tensor α obsShape → α)
    (rngSeed rngCounter : Nat)
    (resetSeed : Nat) :
    IO (Rollout α obsShape nActions horizon × Nat) := do
  let sess : Session.CheckedSession obsShape nActions :=
    Session.CheckedSession.gymnasium (obsShape := obsShape) (nActions := nActions) gym
      (seed? := some resetSeed) (resetOnDone := true)
  collectRolloutCheckedSessionWith (α := α) (obsShape := obsShape) (nActions := nActions) (horizon := horizon)
    sess castObs castReward predictLogits predictValue rngSeed rngCounter

end PPO
end RL
end Runtime
