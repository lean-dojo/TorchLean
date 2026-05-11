/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Basic

/-!
# Core Reinforcement-Learning Definitions

This module collects the small mathematical definitions that sit underneath TorchLean's RL
development.

These definitions are intentionally spec-level rather than runtime-level:

- Bellman-style backups,
- discounted returns,
- generalized advantage estimation (GAE),
- and simple typed rollout records.

That keeps the actual RL mathematics in a proof-friendly namespace and avoids duplicating it inside
runtime/trainer code.

## Why Lists (Not Tensors)?

Several helpers here operate on `List α` rather than `Tensor α (.dim n .scalar)` on purpose.

- A trajectory length is usually *data-dependent* (episode termination, truncation, variable rollout
  horizon), so a dependent tensor length is often the wrong abstraction.
- TorchLean uses typed tensors heavily for *fixed-shape* objects (value tables, Q-tables, logits,
  etc.). For variable-length traces, `List` is the lightweight, proof-friendly choice.

When you do have a fixed horizon `n`, it is reasonable to use `Fin n → α` or a vector tensor and
define specialized “returns/GAE” helpers on top. We keep the core definitions here compact and
general, and add fixed-horizon variants where they meaningfully improve downstream code.

Primary references:

- Sutton, "Learning to Predict by the Methods of Temporal Differences" (1988):
  https://doi.org/10.1023/A:1022633531479
- Watkins and Dayan, "Q-learning" (1992): https://doi.org/10.1007/BF00992698
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.):
  http://incompleteideas.net/book/the-book-2nd.html
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- TorchRL documentation (rollouts, tensordicts, and GAE-style objectives):
  https://pytorch.org/rl/
-/

@[expose] public section

namespace Spec
namespace RL

variable {α : Type}

/-- Small record used by generalized-advantage-estimation helpers. -/
structure AdvantageStep (α : Type) where
  /-- Immediate reward `r_t`. -/
  reward : α
  /-- Baseline / critic value estimate `V(s_t)`. -/
  value : α
  /-- Bootstrap value `V(s_{t+1})`. -/
  nextValue : α
  /-- Episode termination flag. -/
  done : Bool

/-- Convert a terminal flag into a multiplicative continuation mask (`1` for continue, `0` for
stop). -/
def continueMask [Zero α] [One α] (done : Bool) : α :=
  if done then 0 else 1

/-- Bellman-style one-step backup:
`r + γ * (1 - done) * bootstrap`. -/
def discountedBackup [Zero α] [One α] [Add α] [Mul α]
    (reward gamma bootstrap : α) (done : Bool) : α :=
  reward + gamma * continueMask (α := α) done * bootstrap

/-- One-step TD target for state-value or action-value updates. -/
def tdTarget [Zero α] [One α] [Add α] [Mul α]
    (reward gamma nextValue : α) (done : Bool) : α :=
  discountedBackup (α := α) reward gamma nextValue done

/-- TD residual / Bellman error:
`r + γ * (1-d) * nextValue - value`. -/
def tdResidual [Zero α] [One α] [Add α] [Mul α] [Sub α]
    (value reward gamma nextValue : α) (done : Bool) : α :=
  tdTarget (α := α) reward gamma nextValue done - value

/-- Discounted returns with a bootstrap value on the far right:
`G_t = r_t + γ G_{t+1}`. -/
def discountedReturnsFrom [Zero α] [Add α] [Mul α]
    (gamma : α) (rewards : List α) (bootstrap : α := 0) : List α :=
  let (_, returns) :=
    rewards.reverse.foldl
      (fun (acc : α × List α) reward =>
        let g := reward + gamma * acc.1
        (g, g :: acc.2))
      (bootstrap, [])
  returns

/-- Discounted returns for a terminal trajectory (bootstrap defaults to `0`). -/
def discountedReturns [Zero α] [Add α] [Mul α] (gamma : α) (rewards : List α) : List α :=
  discountedReturnsFrom (α := α) gamma rewards 0

/-- Discounted returns with explicit termination markers.

When `done = true`, the future return is reset before bootstrapping the current reward.
-/
def discountedReturnsDone [Zero α] [One α] [Add α] [Mul α]
    (gamma : α) (rewards : List α) (dones : List Bool) (bootstrap : α := 0) :
    List α :=
  let (_, returns) :=
    (List.zip rewards dones).reverse.foldl
      (fun (acc : α × List α) step =>
        let g := discountedBackup (α := α) step.1 gamma acc.1 step.2
        (g, g :: acc.2))
      (bootstrap, [])
  returns

/-- Generalized Advantage Estimation (GAE).

Each input step provides `r_t`, `V(s_t)`, `V(s_{t+1})`, and `done_t`. The resulting list contains
advantages in forward time order.
-/
def generalizedAdvantageEstimation [Zero α] [One α] [Add α] [Mul α] [Sub α]
    (gamma lam : α) (steps : List (AdvantageStep α)) : List α :=
  let (_, advantages) :=
    steps.reverse.foldl
      (fun (acc : α × List α) step =>
        let mask := continueMask (α := α) step.done
        let delta := step.reward + gamma * mask * step.nextValue - step.value
        let adv := delta + gamma * lam * mask * acc.1
        (adv, adv :: acc.2))
      (0, [])
  advantages

/-- Recover lambda-returns from advantages and baseline values via `R_t = A_t + V(s_t)`. -/
def returnsFromAdvantages [Add α] : List α → List α → List α
  | [], _ => []
  | _, [] => []
  | a :: as, v :: vs => (a + v) :: returnsFromAdvantages as vs

end RL
end Spec
