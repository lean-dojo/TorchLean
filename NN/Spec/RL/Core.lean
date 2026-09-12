/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Constructors

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

## Numerical Containers

Trajectories use `Tensor α [horizon]`, including when the horizon is chosen at runtime. Rewards,
values, and termination markers share that index, so mismatched trajectories cannot be silently
truncated. Scalar recurrences and their evaluation order are explicit below.

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

/-- Convert a terminal flag into a multiplicative continuation mask (`1` for continue, `0` for
stop). -/
def continueMask [Zero α] [One α] (done : Bool) : α :=
  if done then 0 else 1

/-- Bellman-style one-step backup:
$r+\gamma(1-\mathtt{done})\mathtt{bootstrap}$. -/
def discountedBackup [Zero α] [One α] [Add α] [Mul α]
    (reward gamma bootstrap : α) (done : Bool) : α :=
  reward + gamma * continueMask (α := α) done * bootstrap

/-- One-step TD target for state-value or action-value updates. -/
def tdTarget [Zero α] [One α] [Add α] [Mul α]
    (reward gamma nextValue : α) (done : Bool) : α :=
  discountedBackup (α := α) reward gamma nextValue done

/-- TD residual / Bellman error:
$r+\gamma(1-d)\mathtt{nextValue}-\mathtt{value}$. -/
def tdResidual [Zero α] [One α] [Add α] [Mul α] [Sub α]
    (value reward gamma nextValue : α) (done : Bool) : α :=
  tdTarget (α := α) reward gamma nextValue done - value

open TorchLean

variable [TorchLean.Storage α]

/-! ## Shape-indexed trajectory calculations -/

/-- Discounted returns with a far-right bootstrap, evaluated from right to left. -/
def discountedReturnsFrom [Zero α] [Add α] [Mul α] {n : Nat} (gamma : α)
    (rewards : Tensor α [n]) (bootstrap : α := 0) : Tensor α [n] :=
  Tensor.scanr (fun reward future => reward + gamma * future) bootstrap rewards

/-- Discounted returns for a terminal trajectory. -/
def discountedReturns [Zero α] [Add α] [Mul α] {n : Nat}
    (gamma : α) (rewards : Tensor α [n]) : Tensor α [n] :=
  discountedReturnsFrom gamma rewards 0

/-- Discounted returns with one termination marker per reward; unequal lengths are unrepresentable.

The multiplication order matches `discountedBackup`, including its floating-point behavior.
-/
def discountedReturnsDone [Zero α] [One α] [Add α] [Mul α] {n : Nat} (gamma : α)
    (rewards : Tensor α [n]) (dones : Tensor Bool [n]) (bootstrap : α := 0) : Tensor α [n] :=
  let indices : Tensor (Fin n) [n] := Tensor.ofFn id
  Tensor.scanr (fun i future => discountedBackup rewards[i] gamma future dones[i]) bootstrap indices

/-- Generalized Advantage Estimation, retaining the common horizon in all five tensor types. -/
def generalizedAdvantageEstimation [Zero α] [One α] [Add α] [Mul α] [Sub α]
    {n : Nat} (gamma lam : α)
    (rewards values nextValues : Tensor α [n]) (dones : Tensor Bool [n]) : Tensor α [n] :=
  let indices : Tensor (Fin n) [n] := Tensor.ofFn id
  Tensor.scanr (fun i nextAdvantage =>
    let mask := continueMask (α := α) dones[i]
    let delta := rewards[i] + gamma * mask * nextValues[i] - values[i]
    delta + gamma * lam * mask * nextAdvantage) 0 indices

/-- Lambda-returns `R_t = A_t + V_t`, with equal lengths enforced by the tensor shape. -/
def returnsFromAdvantages [Add α] {n : Nat} (advantages values : Tensor α [n]) : Tensor α [n] :=
  Tensor.ofFn fun i => advantages[i] + values[i]


end RL
end Spec
