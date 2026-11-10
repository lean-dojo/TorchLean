/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PPO.Rollout
public import TorchLean.NN.RL.Session
public import TorchLean.NN.RL.PolicyGradient

/-!
# PPO Rollout Collection (Checked Sessions)

The rollout-collection loop of the executable PPO workflows. Collection is typed and total: each
step is one `Step` record, so there are no parallel arrays to fall out of sync. Every step passes
the trust-boundary contract check, for an external Gymnasium environment and a Lean-native one
alike, and callers hand over the actor and critic as two prediction functions, with none of their
lowering details threaded through the loop.

`collect` is the inner loop of Algorithm 1 in the PPO paper for one actor, written by hand: it runs
the current policy for `horizon` steps. Every call begins at `session.start`, a reset for both
session constructors, so no episode carries over from the previous rollout. An episode that ends
inside the horizon starts over at the next step when the session resets on `done`, as both
constructors do by default. We cache `PolicyGradient.logProb` of the sampled action, the log of
the sampling policy's probability in the denominator of their eq. (6), and read the critic at
`nextObservation` before any reset, so a truncated step can bootstrap from it.

The unified session interface is `Session.Checked` in `TorchLean.NN.RL.Session`, and
the Gymnasium subprocess protocol under it is implemented in
`TorchLean.NN.RL.Gymnasium.Client`.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017), arXiv:1707.06347, Algorithm 1
  and eq. (6): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- Gymnasium API reference (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace PPO

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-!
## Rollout collection (ergonomic core API)
-/

/--
Collect a fixed-horizon rollout from a stateful environment session whose steps return fully
observed, contract-checked transitions, and return it with the advanced sampler counter.

The caller provides:

- `session`: initialization, observation, and checked stepping for the environment,
- `castObservation` to inject host `Float` observations into the chosen scalar backend `α`,
- `castReward` to inject host `Float` rewards into the chosen scalar backend `α`,
- `predictLogits` for the current actor,
- `predictValue` for the current critic (returns a scalar `α`).

The actor and the critic are plain functions, so a caller can close over a typed graph and its
parameters. Each stored log-probability is `PolicyGradient.logProb` of the `predictLogits`
output, the program that `PPO.Autograd.objective` records, and that docstring states when the
ratio at the collection's parameters is exactly one.
-/
def collect {obsShape : Shape} {nActions horizon : Nat}
    [NeZero horizon] [NeZero nActions]
    (session : Session.Checked obsShape nActions)
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
    let obs : Tensor α obsShape := Tensor.map castObservation obsF

    let logits : Tensor α [nActions] := predictLogits obs
    let (counter', a) :=
      PolicyGradient.sample (α := α) (nActions := nActions)
        (seed := rngSeed) (counter := counter) logits (logits := true)
    counter := counter'

    -- Cache the behavior policy's selected log-probability.
    let lp : α := PolicyGradient.logProb (α := α) (nActions := nActions) logits a
    let v : α := predictValue obs

    let (tr, nextState) ← session.step state a
    state := nextState

    let done : Bool := tr.terminated || tr.truncated
    let nextObs : Tensor α obsShape := Tensor.map castObservation tr.nextObservation
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

end PPO
end RL
end TorchLean
