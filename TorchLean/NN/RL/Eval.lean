/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Session
public import TorchLean.Tensor.ArgMax

/-!
# RL Evaluation Helpers (Executable Runtime)

Non-differentiable helpers for evaluating a policy in executable RL code: the greedy action of a
logits vector (`argmax`) and the undiscounted total reward of greedy episodes. They are written
against `TorchLean.RL.Session.Checked`, so the same evaluation runs on an external Python
Gymnasium environment (`TorchLean.RL.Gymnasium`) and on a Lean-native `TorchLean.RL.Env` wrapped by
`TorchLean.RL.Session.Checked.ofEnv`.

We wrote these helpers by hand. `totalReward` adds the rewards as `Float`s in step order, starting
from `0.0`, so 1000 steps of reward 0.1 come to 99.9999999999986.

## Semantics and references

- An episode ends at the first transition with `terminated || truncated`, or after `maxSteps`
  steps. The flags follow Gymnasium's API: https://gymnasium.farama.org/
- The metric is the **undiscounted** sum of rewards along the episode, the usual "episode return"
  of Gym-style benchmarks and the return $G_t$ of eq. (3.7) in Sutton and Barto, *Reinforcement
  Learning: An Introduction* (2nd ed., MIT Press, 2018), Section 3.3, whose (3.8) is the discounted
  form: http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Eval

open TorchLean.Tensor

/-!
## Greedy action selection
 -/

/--
Pick a greedy discrete action from a logits vector by `argmax`.

`[NeZero nActions]` is required because `Fin 0` is empty. Ties go to the lower action and the
first NaN logit wins, as for `Tensor.argmax`. The tie rule is that of `torch.argmax` (PyTorch 2.8
docs), which returns the first maximal index.
-/
def greedy {α : Type} [Storage α]
    [BEq α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {nActions : Nat} [NeZero nActions]
    (logits : Tensor α [nActions]) : Fin nActions :=
  (Tensor.argmax logits ⟨0, Nat.one_pos⟩ false (Nat.pos_of_ne_zero (NeZero.ne nActions))).item


/-!
## Episode evaluation
 -/

/--
Total undiscounted reward of one greedy episode under `policyLogits`, run for at most `maxSteps`
steps. The episode stops early at the first checked transition with `terminated || truncated`.
-/
def totalReward {obsShape : Shape} {nActions : Nat} [NeZero nActions]
    (sess : Session.Checked obsShape nActions)
    (policyLogits : Tensor Float obsShape → Tensor Float [nActions])
    (maxSteps : Nat := 1000) :
    IO Float := do
  let mut s ← sess.start
  let mut total : Float := 0.0
  for _t in [0:maxSteps] do
    let obs : Tensor Float obsShape := sess.observe s
    let logits := policyLogits obs
    let a := greedy (α := Float) (nActions := nActions) logits
    let (tr, s') ← sess.step s a
    s := s'
    total := total + tr.reward
    if Boundary.Transition.done (obsShape := obsShape) (nActions := nActions) tr then
      break
  pure total

/--
The session states of one greedy episode of at most `maxSteps` steps: the state from
`sess.start`, then the state each checked step returns. If the session resets on `done`, the
last entry is the state after that reset, so build evaluation sessions with
`resetOnDone := false` to end the path on the episode's last state.

For `Session.Checked.ofEnv` a state is the Lean-native environment's latent state. A
Gymnasium-backed session's state is the session record (the client and the last observation),
while the Python environment's own state stays in the subprocess.
-/
def path {obsShape : Shape} {nActions : Nat} [NeZero nActions]
    (sess : Session.Checked obsShape nActions)
    (policyLogits : Tensor Float obsShape → Tensor Float [nActions])
    (maxSteps : Nat := 1000) :
    IO (Array sess.State) := do
  let mut s ← sess.start
  let mut path : Array sess.State := #[s]
  for _t in [0:maxSteps] do
    let obs : Tensor Float obsShape := sess.observe s
    let logits := policyLogits obs
    let a := greedy (α := Float) (nActions := nActions) logits
    let (tr, s') ← sess.step s a
    s := s'
    path := path.push s'
    if Boundary.Transition.done (obsShape := obsShape) (nActions := nActions) tr then
      break
  pure path

/--
Mean of `totalReward` over `episodes` episodes, where episode `k` runs on the session
`mkSession (baseSeed + k)`.

A Gymnasium-backed `mkSession` usually passes the seed to `reset`, and a Lean-native one can
ignore it. The result is the mean return per episode. With `episodes = 0` it is `0`, and no
session is created or started.
-/
def meanReward {obsShape : Shape} {nActions : Nat} [NeZero nActions]
    (mkSession : Nat → Session.Checked obsShape nActions)
    (policyLogits : Tensor Float obsShape → Tensor Float [nActions])
    (baseSeed : Nat) (episodes : Nat)
    (maxSteps : Nat := 1000) :
    IO Float := do
  if episodes = 0 then
    pure 0.0
  else
    let mut acc : Float := 0.0
    for k in [0:episodes] do
      let sess := mkSession (baseSeed + k)
      acc := acc + (← totalReward (obsShape := obsShape) (nActions := nActions)
        sess policyLogits (maxSteps := maxSteps))
    pure (acc / (episodes : Float))

end Eval
end RL
end TorchLean
