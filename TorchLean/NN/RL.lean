/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Constructors

/-!
# Core Reinforcement-Learning Definitions

Bellman-style backups, discounted returns and generalized advantage estimation (GAE) are the small
mathematical definitions beneath TorchLean's RL development. They sit at the spec level, in a
proof-friendly namespace, and runtime code such as the PPO rollout calls them.

## Numerical Containers

Trajectories use `Tensor α [horizon]`, including when the horizon is chosen at runtime. Rewards,
values and termination markers share that index, so mismatched trajectories cannot be silently
truncated. The trajectory recurrences run from right to left with `Tensor.scanr`.

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

namespace TorchLean
namespace RL

variable {α : Type}

/--
Bellman-style one-step backup: the reward at a terminal step, otherwise
$r+\gamma\,\mathtt{bootstrap}$.

A terminal step does not read the bootstrap, so an infinite or NaN bootstrap cannot reach it.
-/
def discountedBackup [Add α] [Mul α]
    (reward gamma bootstrap : α) (done : Bool) : α :=
  if done then reward else reward + gamma * bootstrap

/-- TD residual / Bellman error: `discountedBackup reward gamma nextValue done - value`. -/
def tdResidual [Add α] [Mul α] [Sub α]
    (value reward gamma nextValue : α) (done : Bool) : α :=
  discountedBackup (α := α) reward gamma nextValue done - value

variable [Storage α]

/-! ## Shape-indexed trajectory calculations -/

/-- Discounted returns, evaluated from right to left with an optional far-right bootstrap.

The default bootstrap is `0`, representing a terminal trajectory.
-/
def discountedReturns [Zero α] [Add α] [Mul α] {n : Nat} (gamma : α)
    (rewards : Tensor α [n]) (bootstrap : α := 0) : Tensor α [n] :=
  Tensor.scanr (fun reward future => reward + gamma * future) bootstrap rewards

/-- Discounted returns with one termination marker per reward, both of length `n`.

Each step is `discountedBackup`, so a terminal step's return is its reward and a later episode's
return never reaches an earlier one.
-/
def discountedReturnsDone [Zero α] [Add α] [Mul α] {n : Nat} (gamma : α)
    (rewards : Tensor α [n]) (dones : Tensor Bool [n]) (bootstrap : α := 0) : Tensor α [n] :=
  let indices : Tensor (Fin n) [n] := Tensor.ofFn id
  Tensor.scanr (fun i future => discountedBackup rewards[i] gamma future dones[i]) bootstrap indices

/--
Generalized Advantage Estimation with separate termination and boundary masks.

The TD residual of step `t` is `tdResidual`, so `terminated[t]` drops the bootstrap value
`nextValues[t]`, and `boundaries[t]` stops the recursion from reading the advantage of step
`t + 1`. A time-limit truncation sets only `boundaries[t]`, so the step bootstraps from its
pre-reset next value without reading the next episode. Pass the same mask twice when every episode
end is a true termination. Neither branch multiplies an unused value by zero.
-/
def gae [Zero α] [Add α] [Mul α] [Sub α]
    {n : Nat} (gamma lam : α) (rewards values nextValues : Tensor α [n])
    (terminated boundaries : Tensor Bool [n]) : Tensor α [n] :=
  let indices : Tensor (Fin n) [n] := Tensor.ofFn id
  Tensor.scanr (fun i nextAdvantage =>
    let delta := tdResidual values[i] rewards[i] gamma nextValues[i] terminated[i]
    if boundaries[i] then delta else delta + gamma * lam * nextAdvantage) 0 indices

/-- Lambda-returns `R_t = A_t + V_t`, with equal lengths enforced by the tensor shape. -/
def returnsFromAdvantages [Add α] {n : Nat} (advantages values : Tensor α [n]) : Tensor α [n] :=
  Tensor.ofFn fun i => advantages[i] + values[i]


end RL
end TorchLean
