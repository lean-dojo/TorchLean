/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Session
public import NN.Runtime.Autograd.TorchLean.Metrics

/-!
# RL Evaluation Helpers (Executable Runtime)

This module factors out small *non-differentiable* evaluation helpers used in executable RL
workflows:

- choosing a greedy discrete action (`argmax`) from policy logits, and
- running episodes to compute the (undiscounted) total reward.

These helpers are intentionally written against the unified checked-session interface
`Runtime.RL.Session.CheckedSession`, so the same evaluation code can be reused with:

- external Python Gymnasium environments (via `Runtime.RL.Gymnasium`), and
- Lean-native environments (`Spec.RL.Env`) wrapped by `Runtime.RL.Session.CheckedSession.ofEnv`.

## Semantics and references

- We treat an episode as finished when `terminated || truncated`, mirroring Gymnasium's API.
  See Gymnasium docs: https://gymnasium.farama.org/
- The evaluation metric returned by this module is the **undiscounted** sum of rewards along the
  episode, which is the common “episode return” metric used in Gym-style benchmarks.
  See Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.), terminology on
  episodic return and discounting: http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Eval

open Spec
open Tensor

/-!
## Greedy action selection
 -/

/--
Pick a greedy discrete action from a logits vector by `argmax`.

If `nActions = 0` then `Fin nActions` is uninhabited, so we require `[Fact (0 < nActions)]`.
When `nActions > 0`, `Metrics.argmax?` never returns `none`; the `none` branch returns `0` only as
an unreachable totality fallback.
 -/
def greedyActionFromLogits {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {nActions : Nat} [Fact (0 < nActions)]
    (logits : Tensor α (.dim nActions .scalar)) : Fin nActions :=
  match Runtime.Autograd.TorchLean.Metrics.argmax? (α := α) (n := nActions) logits with
  | some a => a
  | none =>
      -- This branch is unreachable when `nActions > 0`, but it keeps the API total.
      ⟨0, (inferInstance : Fact (0 < nActions)).out⟩


/-!
## Episode evaluation
 -/

/--
Run one episode (up to `maxSteps`) using a greedy policy derived from `policyLogits`, and return
the total (undiscounted) reward.

The episode terminates early when the checked transition reports `terminated || truncated`.
 -/
def episodeTotalReward {obsShape : Shape} {nActions : Nat} [Fact (0 < nActions)]
    (sess : Session.CheckedSession obsShape nActions)
    (policyLogits : Tensor Float obsShape → Tensor Float (.dim nActions .scalar))
    (maxSteps : Nat := 1000) :
    IO Float := do
  let mut s ← sess.start
  let mut total : Float := 0.0
  for _t in [0:maxSteps] do
    let obs : Tensor Float obsShape := sess.observe s
    let logits := policyLogits obs
    let a := greedyActionFromLogits (α := Float) (nActions := nActions) logits
    let (tr, s') ← sess.stepChecked s a
    s := s'
    total := total + tr.reward
    if Boundary.Transition.done (obsShape := obsShape) (nActions := nActions) tr then
      break
  pure total

/--
Run one greedy-policy episode (up to `maxSteps`) and record the visited **session states**.

The returned path includes the initial session state from `sess.start`, and then each state after
every checked step.

Note: for `Session.CheckedSession.ofEnv`, `Sess` is the Lean-native environment's latent state.
For Gymnasium-backed sessions, `Sess` is the session record (it stores the last observation, etc.),
not the underlying Python environment's internal state.
-/
def episodeSessPath {obsShape : Shape} {nActions : Nat} [Fact (0 < nActions)]
    (sess : Session.CheckedSession obsShape nActions)
    (policyLogits : Tensor Float obsShape → Tensor Float (.dim nActions .scalar))
    (maxSteps : Nat := 1000) :
    IO (Array sess.Sess) := do
  let mut s ← sess.start
  let mut path : Array sess.Sess := #[s]
  for _t in [0:maxSteps] do
    let obs : Tensor Float obsShape := sess.observe s
    let logits := policyLogits obs
    let a := greedyActionFromLogits (α := Float) (nActions := nActions) logits
    let (tr, s') ← sess.stepChecked s a
    s := s'
    path := path.push s'
    if Boundary.Transition.done (obsShape := obsShape) (nActions := nActions) tr then
      break
  pure path

/--
Average `episodeTotalReward` across multiple episodes, seeding each episode by `baseSeed + k`.

The `mkSession` callback is responsible for interpreting the seed:
- for Gymnasium-backed sessions it typically passes the seed to `reset`, and
- for Lean-native environments it can ignore the seed.

When `episodes = 0`, this function returns `0` by convention.
 -/
def averageEpisodeTotalReward {obsShape : Shape} {nActions : Nat} [Fact (0 < nActions)]
    (mkSession : Nat → Session.CheckedSession obsShape nActions)
    (policyLogits : Tensor Float obsShape → Tensor Float (.dim nActions .scalar))
    (baseSeed : Nat) (episodes : Nat)
    (maxSteps : Nat := 1000) :
    IO Float := do
  if episodes = 0 then
    pure 0.0
  else
    let mut acc : Float := 0.0
    for k in [0:episodes] do
      let sess := mkSession (baseSeed + k)
      acc := acc + (← episodeTotalReward (obsShape := obsShape) (nActions := nActions)
        sess policyLogits (maxSteps := maxSteps))
    pure (acc / (episodes : Float))

end Eval
end RL
end Runtime
