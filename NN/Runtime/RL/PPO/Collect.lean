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

- `start`: how to initialize the session (often `reset`),
- `observe`: how to read the current observation from the session,
- `stepChecked`: one checked step returning an observed transition and the updated session,
- `castObservation` to inject host `Float` observations into the chosen scalar backend `α`,
- `castReward` to inject host `Float` rewards into the chosen scalar backend `α`,
- `predictLogits` for the current actor,
- `predictValue` for the current critic (returns a scalar `α`).

The API supports the “typed graph + parameters” calling convention used throughout TorchLean.
-/
def collectRolloutFromCallbacks {obsShape : Shape} {nActions horizon : Nat} {Sess : Type}
    [NeZero horizon] [NeZero nActions]
    (start : IO Sess)
    (observe : Sess → Tensor Float obsShape)
    (stepChecked : Sess → Fin nActions → IO (Boundary.Transition obsShape nActions × Sess))
    (castObservation : Float → α)
    (castReward : Float → α)
    (predictLogits : Tensor α obsShape → Tensor α [nActions])
    (predictValue : Tensor α obsShape → α)
    (rngSeed rngCounter : Nat) :
    IO (Rollout α obsShape nActions horizon × Nat) := do

  let mut sess ← start

  let mut steps : Array (Step α obsShape nActions) := #[]
  let mut counter := rngCounter

  for _t in [0:horizon] do
    let obsF := observe sess
    let obs : Tensor α obsShape := TorchLean.Tensor.map castObservation obsF

    let logits : Tensor α [nActions] := predictLogits obs
    let (counter', a) :=
      PolicyGradient.sampleActionFromLogits (α := α) (nActions := nActions)
        (seed := rngSeed) (counter := counter) logits
    counter := counter'

    -- Same formula as the autograd objective's new log-probability, so the ratio starts at 1.
    let lp : α := PolicyGradient.actionLogSoftmax (α := α) (nActions := nActions) logits a
    let v : α := predictValue obs

    let (tr, sess') ← stepChecked sess a
    sess := sess'

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
        (s!"PPO.collectRolloutFromCallbacks: internal error (steps.size={steps.size}, "
          ++ s!"horizon={horizon})")

/-!
## Rollout collection from a checked session
-/

/--
Collect a fixed-horizon rollout from a unified `Runtime.RL.Session.CheckedSession`.
-/
def collectRolloutFromSession {obsShape : Shape} {nActions horizon : Nat}
    [NeZero horizon] [NeZero nActions]
    (sess : Session.CheckedSession obsShape nActions)
    (castObservation : Float → α)
    (castReward : Float → α)
    (predictLogits : Tensor α obsShape → Tensor α [nActions])
    (predictValue : Tensor α obsShape → α)
    (rngSeed rngCounter : Nat) :
    IO (Rollout α obsShape nActions horizon × Nat) :=
  collectRolloutFromCallbacks (α := α) (obsShape := obsShape) (nActions := nActions)
    (horizon := horizon)
    (Sess := sess.Sess)
    (start := sess.start) (observe := sess.observe) (stepChecked := sess.stepChecked)
    castObservation castReward predictLogits predictValue rngSeed rngCounter

/-!
## Rollout collection from Gymnasium (subprocess bridge)
-/

/--
Collect a fixed-horizon rollout from a Gymnasium subprocess environment.

This specializes `collectRolloutFromCallbacks` to `Gymnasium.Session`.
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
  collectRolloutFromSession (α := α) (obsShape := obsShape) (nActions := nActions)
    (horizon := horizon)
    sess castObservation castReward predictLogits predictValue rngSeed rngCounter

end PPO
end RL
end Runtime
