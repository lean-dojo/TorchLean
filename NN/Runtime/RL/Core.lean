/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.TensorOps
public import NN.Spec.RL.Core

public import NN.Tensor.API

/-!
# Core Reinforcement-Learning Runtime Helpers

This module adds the tensor-shaped and runtime-facing pieces that sit on top of the mathematical
RL core in `NN.Spec.RL.Core`.

Keeping Bellman / return / GAE definitions in the spec layer avoids an awkward split where the
same mathematics would otherwise exist in both runtime and proof namespaces. This file therefore
only keeps:

- typed transition records for tensor-valued or indexed rollouts,
- small action-encoding helpers,
- and scalar losses commonly used by deep RL objectives.
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Core

open Spec
open Tensor

export _root_.Spec.RL
  (AdvantageStep
   continueMask discountedBackup tdTarget tdResidual
   discountedReturns discountedReturnsFrom discountedReturnsDone
   generalizedAdvantageEstimation returnsFromAdvantages)

variable {α : Type} [Context α]

/-- A typed one-step transition for discrete-action RL over a tensor-valued state. -/
structure Transition (α : Type) (σ : Shape) (nActions : Nat) where
  /-- Current state `s_t`. -/
  state : Tensor α σ
  /-- Discrete action `a_t`. -/
  action : Fin nActions
  /-- Reward `r_t`. -/
  reward : α
  /-- Next state `s_{t+1}`. -/
  nextState : Tensor α σ
  /-- Episode termination flag. -/
  done : Bool

/-- A typed one-step transition for tabular RL over finite state/action spaces. -/
structure IndexedTransition (α : Type) (nStates nActions : Nat) where
  /-- Current state index `s_t`. -/
  state : Fin nStates
  /-- Discrete action `a_t`. -/
  action : Fin nActions
  /-- Reward `r_t`. -/
  reward : α
  /-- Next state index `s_{t+1}`. -/
  nextState : Fin nStates
  /-- Episode termination flag. -/
  done : Bool

/-- Build a typed one-hot action vector.

This is an alias for `NN.Tensor.oneHot`, kept under the RL namespace for ergonomics. -/
abbrev oneHotAction {nActions : Nat} (action : Fin nActions) : Tensor α (.dim nActions .scalar) :=
  NN.Tensor.oneHot (α := α) nActions action

/-!
## Fixed-Horizon Tensor Trajectory Helpers

Many RL implementations store a rollout buffer in fixed-size tensors (e.g. `T × ...` for a time
window, or `N × ...` for a batch). The spec-layer definitions in `NN.Spec.RL.Core` are list-based
because episode length is data-dependent, but it is still useful to have “PyTorch-shaped”
fixed-horizon variants when:

- you already committed to a horizon `n`, and
- you want to keep the data in typed tensors all the way through your update step.

The helpers below are the tensor analogues of:

- `discountedReturnsFrom` / `discountedReturns` / `discountedReturnsDone`,
- `generalizedAdvantageEstimation`,
- `returnsFromAdvantages`.

  They are total and use an internal array-backed right-to-left scan (no list conversion), while the
  public API stays tensor-shaped.

References:
- Sutton and Barto, *Reinforcement Learning: An Introduction*, Section 3 (returns) and 12
  (actor-critic).
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
-/

  /-- Fixed-horizon discounted returns with a bootstrap on the far right:
  `G_t = r_t + γ G_{t+1}`.

  This is the tensor-shaped sibling of `Spec.RL.discountedReturnsFrom`. -/
  def discountedReturnsVecFrom {n : Nat} (gamma : α) (rewards : Tensor α (.dim n .scalar))
      (bootstrap : α := 0) : Tensor α (.dim n .scalar) :=
    let rArr : Array α :=
      Array.ofFn (fun i : Fin n => Tensor.toScalar (get rewards i))
    let out : Array α :=
      Id.run do
        -- Array-backed right-to-left scan; `idx` is always in-bounds for arrays of size `n`.
        let mut returns : Array α := Array.replicate n (0 : α)
        let mut g : α := bootstrap
        for t in [0:n] do
          let idx := n - 1 - t
          g := rArr[idx]! + gamma * g
          returns := returns.set! idx g
        return returns
    Tensor.dim (fun i : Fin n => Tensor.scalar (out[i.val]!))

  /-- Fixed-horizon discounted returns for a terminal trajectory (bootstrap defaults to `0`). -/
  def discountedReturnsVec {n : Nat} (gamma : α) (rewards : Tensor α (.dim n .scalar)) :
      Tensor α (.dim n .scalar) :=
  discountedReturnsVecFrom (α := α) (n := n) gamma rewards 0

  /-- Fixed-horizon discounted returns with explicit termination markers.

  When `done_t = true`, the future return is reset before bootstrapping the current reward. -/
  def discountedReturnsVecDone {n : Nat} (gamma : α)
      (rewards : Tensor α (.dim n .scalar)) (dones : Tensor Bool (.dim n .scalar))
      (bootstrap : α := 0) : Tensor α (.dim n .scalar) :=
    let rArr : Array α :=
      Array.ofFn (fun i : Fin n => Tensor.toScalar (get rewards i))
    let dArr : Array Bool :=
      Array.ofFn (fun i : Fin n => Tensor.toScalar (get dones i))
    let out : Array α :=
      Id.run do
        let mut returns : Array α := Array.replicate n (0 : α)
        let mut g : α := bootstrap
        for t in [0:n] do
          let idx := n - 1 - t
          g := discountedBackup (α := α) (reward := rArr[idx]!) (gamma := gamma) (bootstrap := g)
            (done := dArr[idx]!)
          returns := returns.set! idx g
        return returns
    Tensor.dim (fun i : Fin n => Tensor.scalar (out[i.val]!))

  /-- Fixed-horizon Generalized Advantage Estimation (GAE) as a vector tensor.

  This is the tensor-shaped sibling of `Spec.RL.generalizedAdvantageEstimation`. -/
  def generalizedAdvantageEstimationVec {n : Nat} (gamma lam : α)
      (rewards values nextValues : Tensor α (.dim n .scalar))
      (dones : Tensor Bool (.dim n .scalar)) :
      Tensor α (.dim n .scalar) :=
    let rArr : Array α :=
      Array.ofFn (fun i : Fin n => Tensor.toScalar (get rewards i))
    let vArr : Array α :=
      Array.ofFn (fun i : Fin n => Tensor.toScalar (get values i))
    let nvArr : Array α :=
      Array.ofFn (fun i : Fin n => Tensor.toScalar (get nextValues i))
    let dArr : Array Bool :=
      Array.ofFn (fun i : Fin n => Tensor.toScalar (get dones i))
    let out : Array α :=
      Id.run do
        let mut advs : Array α := Array.replicate n (0 : α)
        let mut advNext : α := 0
        for t in [0:n] do
          let idx := n - 1 - t
          let done := dArr[idx]!
          let mask := continueMask (α := α) done
          let delta := rArr[idx]! + gamma * mask * nvArr[idx]! - vArr[idx]!
          let adv := delta + gamma * lam * mask * advNext
          advNext := adv
          advs := advs.set! idx adv
        return advs
    Tensor.dim (fun i : Fin n => Tensor.scalar (out[i.val]!))

/-- Fixed-horizon lambda-returns from advantages and baseline values via `R_t = A_t + V_t`. -/
def returnsFromAdvantagesVec {n : Nat}
    (advantages values : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  addSpec advantages values

/-- Squared-error helper used by critic / TD objectives. -/
def squaredError (prediction target : α) : α :=
  let d := prediction - target
  d * d

/-- Scalar Huber-style loss used by robust TD objectives.

We use the standard piecewise form:
- quadratic region: `(1 / (2 * delta)) * (pred - target)^2`
- linear region: `|pred - target| - delta / 2`
-/
def huberLoss (prediction target : α) (delta : α := 1) : α :=
  let d := prediction - target
  let ad := MathFunctions.abs d
  if delta > ad then
    (d * d) / (Numbers.two * delta)
  else
    ad - delta / Numbers.two

end Core
end RL
end Runtime
