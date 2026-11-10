/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Environment
public import TorchLean.Tensor.Reduce

/-!
# RL Trust Boundary (External Rollouts)

TorchLean's RL update rules (Bellman backups, returns and GAE, PPO objectives) are Lean definitions,
so they can be reasoned about and proved correct inside Lean. Experience collected from an external
environment such as Python Gymnasium crosses a trust boundary. TorchLean does not, and in general
cannot, prove that the environment satisfies Markov, measurability or boundedness assumptions, or
that its runtime returns numerically well-behaved floating-point values.

The middle ground is a strict **contract** for externally provided rollouts (`Contract`) and a
checker that turns those assumptions into checked preconditions. A transition that passes the check
is a `TorchLean.RL.ObservedTransition`, the type `TorchLean.RL.Env.rollout` produces for Lean-native
environments, so downstream training code takes one input type from both sources.

We wrote the contract by hand from the `(observation, reward, terminated, truncated, info)` tuple
of Gymnasium's `Env.step`, leaving out `info`. `requireExclusiveDoneFlags` is off by default
because Gymnasium's `TimeLimit` wrapper sets `truncated` at the step limit whatever `terminated`
says, so an episode that terminates on its last allowed step reports both flags. A range check
rejects a NaN even with the matching finiteness check off, since every comparison with NaN is
false.

## References

- Gymnasium API (`reset`/`step`, `terminated` vs `truncated`): https://gymnasium.farama.org/
- Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments" (2024),
  NeurIPS Datasets and Benchmarks 2025, arXiv:2407.17032, Section 4.2 and eqs. (1) and (2) in
  arXiv v4: https://arxiv.org/abs/2407.17032
- Gymnasium v1.0.0, `TimeLimit.step` in `gymnasium/wrappers/common.py`:
  https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v1.0.0/gymnasium/wrappers/common.py
- Brockman et al., "OpenAI Gym" (2016), arXiv:1606.01540, background on the env interface:
  https://arxiv.org/abs/1606.01540
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Boundary

open TorchLean.Tensor

/-!
## Contract for external discrete-action rollouts

The contract leaves the environment dynamics unverified. It checks the syntactic and numeric
properties that can be validated locally and fails fast with a clear error.
-/

/-- Contract for validating externally provided Gym-style transitions. -/
structure Contract (obsShape : Shape) (nActions : Nat) where
  /-- Check that observations contain no NaNs/Infs. -/
  checkObsFinite : Bool := true
  /-- Check that rewards contain no NaNs/Infs. -/
  checkRewardFinite : Bool := true
  /-- Optional observation range check `[lo, hi]`. -/
  obsRange? : Option (Float × Float) := none
  /-- Optional reward range check `[lo, hi]`. -/
  rewardRange? : Option (Float × Float) := none
  /-- Optional strictness: reject steps with both `terminated=true` and `truncated=true`. -/
  requireExclusiveDoneFlags : Bool := false
  deriving Repr

/-- Validated transition type for discrete-action rollouts. -/
abbrev Transition (obsShape : Shape) (nActions : Nat) : Type :=
  RL.ObservedTransition (Tensor Float obsShape) (Fin nActions) Float

namespace Transition

/-- Whether the episode ends at this step, Gymnasium's `terminated || truncated`. A TD target
reads `terminated` alone, because a truncated step bootstraps from its next state. That's the split
of eqs. (1) and (2) in Towers et al. (2024). -/
def done {obsShape : Shape} {nActions : Nat} (t : Transition obsShape nActions) : Bool :=
  t.terminated || t.truncated

end Transition

/-!
## Checkers
-/

/-- Convert a raw action index into `Fin nActions` with a range check. -/
def checkAction (nActions : Nat) (action : Nat) : Except String (Fin nActions) :=
  if h : action < nActions then
    .ok ⟨action, h⟩
  else
    .error s!"RL boundary: action out of range: {action} (nActions={nActions})."

/--
Validate one observation tensor against the observation-related part of a `Contract`.

`checkTransitionFin` runs it on both observations of a transition, and the Gymnasium client runs it
on every `reset` observation.
-/
def checkObservation {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (field : String := "observation")
    (obs : Tensor Float obsShape) :
    Except String Unit := do
  if c.checkObsFinite then
    unless obs.all Context.isFinite do
      throw s!"RL boundary: expected finite {field} tensor (no NaNs/Infs)."
  match c.obsRange? with
  | none => pure ()
  | some (lo, hi) =>
      unless obs.all (fun x => lo ≤ x && x ≤ hi) do
        throw s!"RL boundary: {field} tensor out of range: expected all entries in [{lo}, {hi}]."

/-- Validate one reward against the reward-related part of a `Contract`. -/
def checkReward {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (reward : Float) :
    Except String Unit := do
  if c.checkRewardFinite then
    Context.requireWithin "RL boundary: reward" none none reward
  match c.rewardRange? with
  | none => pure ()
  | some (lo, hi) => Context.requireWithin "RL boundary: reward" (some lo) (some hi) reward

/-- Validate the Gymnasium-style done flags against a `Contract`. -/
def checkDoneFlags {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (terminated truncated : Bool) :
    Except String Unit := do
  if c.requireExclusiveDoneFlags && terminated && truncated then
    throw ("RL boundary: both `terminated` and `truncated` are true "
      ++ "(contract requires exclusivity).")

/--
Validate a transition whose action is already a `Fin nActions`, skipping the action range check.

The done flags, the reward, `observation` and `nextObservation` are checked in that order, and the
first failure is the error returned.
-/
def checkTransitionFin {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (observation nextObservation : Tensor Float obsShape)
    (action : Fin nActions)
    (reward : Float)
    (terminated truncated : Bool) :
    Except String (Transition obsShape nActions) := do
  checkDoneFlags (obsShape := obsShape) (nActions := nActions) c terminated truncated
  checkReward (obsShape := obsShape) (nActions := nActions) c reward
  checkObservation (obsShape := obsShape) (nActions := nActions) c (field := "observation")
    observation
  checkObservation (obsShape := obsShape) (nActions := nActions) c (field := "nextObservation")
    nextObservation
  pure
    { observation := observation
      action := action
      reward := reward
      nextObservation := nextObservation
      terminated := terminated
      truncated := truncated }

/--
One checked environment step: run `step`, validate its transition with `check` against the
observation before the step, and replace the next state by `reset` of it when `resetOnDone` is
set and the checked transition ends its episode (`done`).

Both runtime sessions pass `checkTransitionFin` as `check`. Passing `checkTransitionFinWithProof`
gives the same step and returns each transition with a proof of its contract.
-/
def stepChecked {State Checked : Type} {obsShape : Shape} {nActions : Nat}
    (observe : State → Tensor Float obsShape)
    (step : State → Fin nActions → IO (Tensor Float obsShape × Float × Bool × Bool × State))
    (reset : State → IO State)
    (check : Tensor Float obsShape → Tensor Float obsShape → Fin nActions → Float → Bool → Bool →
      Except String Checked)
    (done : Checked → Bool) (resetOnDone : Bool)
    (state : State) (action : Fin nActions) : IO (Checked × State) := do
  let (nextObservation, reward, terminated, truncated, next) ← step state action
  let checked ← IO.ofExcept
    (check (observe state) nextObservation action reward terminated truncated)
  let next ← if resetOnDone && done checked then reset next else pure next
  pure (checked, next)

/-- Validate one observed transition with a raw `Nat` action against a `Contract`: `checkAction`
range-checks the action, then `checkTransitionFin` checks the rest. -/
def checkTransition {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (observation nextObservation : Tensor Float obsShape)
    (action : Nat)
    (reward : Float)
    (terminated truncated : Bool) :
    Except String (Transition obsShape nActions) := do
  let action ← checkAction nActions action
  checkTransitionFin (obsShape := obsShape) (nActions := nActions) c observation nextObservation
    action reward terminated truncated

/-!
## Proposition-level contract

The checkers above run as programs and return `Except String ...` with runtime error messages. For
formal reasoning, `ObservationHolds`, `RewardHolds`, `DoneFlagsHolds` and `ContractHolds` state the
same conditions as propositions.

The bridge lemma `TorchLean.RL.Boundary.checkTransitionFin_sound` in
`TorchLean/NN/RL/Boundary/Soundness.lean` turns a successful `checkTransitionFin` into
`ContractHolds`.
-/

section PropContract

variable {obsShape : Shape} {nActions : Nat}

/--
Proposition-level version of the observation checks performed by `checkObservation`.

It covers the finiteness and range conditions on the tensor itself. The environment dynamics are
outside the contract.
-/
def ObservationHolds (c : Contract obsShape nActions) (obs : Tensor Float obsShape) : Prop :=
  (c.checkObsFinite = true → obs.all Context.isFinite = true) ∧
    match c.obsRange? with
    | none => True
    | some (lo, hi) => obs.all (fun x => lo ≤ x && x ≤ hi) = true

/--
Proposition-level version of the reward checks performed by `checkReward`.
-/
def RewardHolds (c : Contract obsShape nActions) (reward : Float) : Prop :=
  (c.checkRewardFinite = true → Context.isFinite reward = true) ∧
    match c.rewardRange? with
    | none => True
    | some (lo, hi) => lo ≤ reward ∧ reward ≤ hi

/--
Proposition-level version of the done-flag check performed by `checkDoneFlags`.

`terminated` and `truncated` follow Gymnasium's API: `terminated` means the environment reached a
terminal state, while `truncated` marks an external truncation such as a time limit. See the
Gymnasium `Env.step` documentation:
- https://gymnasium.farama.org/api/env/#gymnasium.Env.step
-/
def DoneFlagsHolds (c : Contract obsShape nActions) (terminated truncated : Bool) : Prop :=
  c.requireExclusiveDoneFlags = true → ¬ (terminated = true ∧ truncated = true)

/--
`ContractHolds c t` states that the transition `t` meets the contract `c`: its done flags, its
reward and both of its observations satisfy the corresponding `Holds` predicates.
-/
structure ContractHolds (c : Contract obsShape nActions) (t : Transition obsShape nActions) :
    Prop where
  doneFlags :
    DoneFlagsHolds (obsShape := obsShape) (nActions := nActions) c t.terminated t.truncated
  reward : RewardHolds (obsShape := obsShape) (nActions := nActions) c t.reward
  observation : ObservationHolds (obsShape := obsShape) (nActions := nActions) c t.observation
  nextObservation :
    ObservationHolds (obsShape := obsShape) (nActions := nActions) c t.nextObservation

end PropContract


end Boundary
end RL
end TorchLean
