/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.RL.Environment
public import NN.Tensor.API

/-!
# RL Trust Boundary (External Rollouts)

TorchLean’s RL update rules (Bellman backups, returns/GAE, PPO objectives, etc.) are defined in Lean
and can therefore be reasoned about and proved correct **inside Lean**. When you collect experience
with an **external** environment (for example Python Gymnasium), you cross a trust boundary:

- we do *not* (and generally cannot) prove the external environment satisfies Markov / measurability
  / boundedness assumptions, and
- we do *not* prove the external runtime returns numerically well-behaved floating-point values.

This module implements a practical middle ground: a strict, explicit **contract** for externally
provided rollouts and a checker that turns “assumptions” into “checked preconditions”.

The boundary emits `Spec.RL.ObservedTransition` as the validated output type. This
lets downstream training code share one common input type for both:

- Lean-native environments via `Spec.RL.rolloutFrom`, and
- external rollouts after passing this contract check.

## References

- Gymnasium API (`reset`/`step`, `terminated` vs `truncated`): https://gymnasium.farama.org/
- The original Gym API paper (background on the env interface): https://arxiv.org/abs/1606.01540
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Trust-boundary pattern used elsewhere in TorchLean (e.g. the Arb oracle): `NN.Floats.Arb`.
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Boundary

open Spec
open Tensor

/-!
## Basic numeric checks

We keep checks in `Bool` form and then use `Except String` wrappers for nice error messages.
This makes it easy to integrate with IO loaders (JSON, sockets, shared-memory, ...).
-/

/-- `true` iff a `Float` is neither `NaN` nor `±Inf`. -/
def isFiniteFloat (x : Float) : Bool :=
  !(x.isNaN || x.isInf)

/-- Apply a scalar boolean predicate to every entry of a tensor. -/
def tensorAll {α : Type} {s : Shape} (p : α → Bool) : Tensor α s → Bool
  | Tensor.scalar x => p x
  | Tensor.dim (n := n) (s := s') f =>
      let idxs : Array (Fin n) := Array.ofFn (fun i => i)
      idxs.foldl (fun ok i => ok && tensorAll (α := α) (s := s') p (f i)) true

/-- `true` iff every entry of the tensor is finite. -/
def tensorFinite {s : Shape} (t : Tensor Float s) : Bool :=
  tensorAll (α := Float) (s := s) isFiniteFloat t

/-- `true` iff every entry of the tensor lies in `[lo, hi]`. -/
def tensorInClosedInterval {s : Shape} (lo hi : Float) (t : Tensor Float s) : Bool :=
  tensorAll (α := Float) (s := s) (fun x => lo ≤ x && x ≤ hi) t

/-!
## Contract for external discrete-action rollouts

This contract is intentionally narrow:

- it does not try to verify the environment dynamics,
- it checks only *syntactic* and *numeric sanity* properties that can be validated locally,
- it is designed to fail fast with a human-readable error.
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
  Spec.RL.ObservedTransition (Tensor Float obsShape) (Fin nActions) Float

namespace Transition

/-- Gymnasium-style `done` flag: `terminated || truncated`. -/
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

namespace Internal

/-!
### Internal check helpers

These small routines are factored out so the exported checkers can stay readable and can assemble
good error messages.
-/

/-- Check that a `Float` is finite (not NaN/Inf), producing an error tagged with `field`. -/
def checkFloatFinite (field : String) (x : Float) : Except String Unit :=
  if isFiniteFloat x then
    .ok ()
  else
    .error s!"RL boundary: expected finite {field}, got {x}."

/-- Check that every tensor entry is finite, producing an error tagged with `field`. -/
def checkTensorFinite {s : Shape} (field : String) (t : Tensor Float s) :
    Except String Unit :=
  if tensorFinite (s := s) t then
    .ok ()
  else
    .error s!"RL boundary: expected finite {field} tensor (no NaNs/Infs)."

/-- Check that every tensor entry lies in `[lo, hi]`, producing an error tagged with `field`. -/
def checkTensorRange {s : Shape} (field : String) (lo hi : Float) (t : Tensor Float s) :
    Except String Unit :=
  if tensorInClosedInterval (s := s) lo hi t then
    .ok ()
  else
    .error s!"RL boundary: {field} tensor out of range: expected all entries in [{lo}, {hi}]."

/-- Check that a scalar lies in `[lo, hi]`, producing an error tagged with `field`. -/
def checkFloatRange (field : String) (lo hi : Float) (x : Float) : Except String Unit :=
  if lo ≤ x && x ≤ hi then
    .ok ()
  else
    .error s!"RL boundary: {field} out of range: got {x}, expected in [{lo}, {hi}]."

end Internal

/--
Validate one observation tensor against the observation-related part of a `Contract`.

This is mainly used to validate `reset` observations coming from external Gymnasium-like bridges.
-/
def checkObservation {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (field : String := "observation")
    (obs : Tensor Float obsShape) :
    Except String Unit := do
  if c.checkObsFinite then
    Internal.checkTensorFinite (field := field) obs
  match c.obsRange? with
  | none => pure ()
  | some (lo, hi) =>
      Internal.checkTensorRange (field := field) lo hi obs

/-- Validate one reward against the reward-related part of a `Contract`. -/
def checkReward {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (reward : Float) :
    Except String Unit := do
  if c.checkRewardFinite then
    Internal.checkFloatFinite (field := "reward") reward
  match c.rewardRange? with
  | none => pure ()
  | some (lo, hi) =>
      Internal.checkFloatRange (field := "reward") lo hi reward

/-- Validate the Gymnasium-style done flags against a `Contract`. -/
def checkDoneFlags {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (terminated truncated : Bool) :
    Except String Unit := do
  if c.requireExclusiveDoneFlags && terminated && truncated then
    throw s!"RL boundary: both `terminated` and `truncated` are true (contract requires exclusivity)."

/--
Validate a transition when the action is already range-checked (`Fin nActions`).

This avoids re-checking the action range at call sites that already work with `Fin` actions.
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
  checkObservation (obsShape := obsShape) (nActions := nActions) c (field := "observation") observation
  checkObservation (obsShape := obsShape) (nActions := nActions) c (field := "nextObservation") nextObservation
  pure
    { observation := observation
      action := action
      reward := reward
      nextObservation := nextObservation
      terminated := terminated
      truncated := truncated }

/-- Validate one observed transition against a `Contract`. -/
def checkTransition {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions)
    (observation nextObservation : Tensor Float obsShape)
    (action : Nat)
    (reward : Float)
    (terminated truncated : Bool) :
    Except String (Transition obsShape nActions) := do
  let action ← checkAction nActions action
  checkTransitionFin (obsShape := obsShape) (nActions := nActions) c observation nextObservation action reward
    terminated truncated

/-!
## Proposition-level contract

The checkers above are executable and return `Except String ...` for good runtime error messages.
For formal reasoning we also want a Prop-level view that states exactly what was checked.

The main bridge lemma is in `NN/Proofs/RL/Boundary.lean`:

`Proofs.RL.Boundary.contractHolds_of_checkTransitionFin_eq_ok` turns a successful runtime check
into `ContractHolds`.
-/

section PropContract

variable {obsShape : Shape} {nActions : Nat}

/--
Proposition-level version of the observation checks performed by `checkObservation`.

This is intentionally “syntactic”: it only states the locally checkable numeric/shape sanity
conditions, not any semantic assumptions about the environment dynamics.
-/
def ObservationHolds (c : Contract obsShape nActions) (obs : Tensor Float obsShape) : Prop :=
  (c.checkObsFinite = true → tensorFinite (s := obsShape) obs = true) ∧
    match c.obsRange? with
    | none => True
    | some (lo, hi) => tensorInClosedInterval (s := obsShape) lo hi obs = true

/--
Proposition-level version of the reward checks performed by `checkReward`.
-/
def RewardHolds (c : Contract obsShape nActions) (reward : Float) : Prop :=
  (c.checkRewardFinite = true → isFiniteFloat reward = true) ∧
    match c.rewardRange? with
    | none => True
    | some (lo, hi) => lo ≤ reward ∧ reward ≤ hi

/--
Proposition-level version of the done-flag check performed by `checkDoneFlags`.

`terminated` and `truncated` follow Gymnasium’s API: `terminated` means the environment reached a
terminal state, while `truncated` indicates an external truncation such as a time-limit.  See the
Gymnasium `Env.step` documentation:
- https://gymnasium.farama.org/api/env/#gymnasium.Env.step
-/
def DoneFlagsHolds (c : Contract obsShape nActions) (terminated truncated : Bool) : Prop :=
  c.requireExclusiveDoneFlags = true → ¬ (terminated = true ∧ truncated = true)

/--
`ContractHolds c t` means the external transition `t` satisfies the Prop-level trust-boundary
contract induced by `c`.
-/
structure ContractHolds (c : Contract obsShape nActions) (t : Transition obsShape nActions) : Prop where
  doneFlags : DoneFlagsHolds (obsShape := obsShape) (nActions := nActions) c t.terminated t.truncated
  reward : RewardHolds (obsShape := obsShape) (nActions := nActions) c t.reward
  observation : ObservationHolds (obsShape := obsShape) (nActions := nActions) c t.observation
  nextObservation : ObservationHolds (obsShape := obsShape) (nActions := nActions) c t.nextObservation

/-- Alias: Prop-level validity of a transition under `c`. -/
abbrev ValidTransition (c : Contract obsShape nActions) (t : Transition obsShape nActions) : Prop :=
  ContractHolds (obsShape := obsShape) (nActions := nActions) c t

end PropContract


end Boundary
end RL
end Runtime
