/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.Spec.Core.TensorOps -- shake: keep
public import NN.Spec.RL.Core -- shake: keep
public import Batteries.Data.Array.Scan -- shake: keep
public import Mathlib.Algebra.Order.Field.Basic -- shake: keep
import Mathlib.Tactic.NormNum.Inv -- shake: keep
import Mathlib.Tactic.NormNum.Pow -- shake: keep
import Mathlib.Tactic.Positivity.Finset -- shake: keep
public import NN.Tensor.Internal.Elab.TensorLiteral -- shake: keep
public import NN.Spec.Core.Context -- shake: keep
public import NN.Tensor -- shake: keep

/-!
# Core Reinforcement-Learning Runtime Helpers

This module adds the tensor-shaped and runtime layer pieces that sit on top of the mathematical
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

open Spec TorchLean
open TorchLean TorchLean.Tensor

export Spec.RL
  (continueMask discountedBackup tdTarget tdResidual
   discountedReturnsFrom discountedReturns discountedReturnsDone
   generalizedAdvantageEstimation returnsFromAdvantages)

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- A typed one-step transition for discrete-action RL over a tensor-valued state. -/
structure Transition (α : Type) [TorchLean.Storage α]
    (σ : Shape) (nActions : Nat) where
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

/-- Squared-error helper used by critic / TD objectives. -/
def squaredError (prediction target : α) : α :=
  let d := prediction - target
  d * d

/-- Scalar Huber loss used by robust TD objectives.

We use the standard piecewise form:
- quadratic region: `(pred - target)^2 / 2`
- linear region: `delta * (|pred - target| - delta / 2)`

This is the `HuberLoss` convention, not the rescaled `SmoothL1Loss` convention. The intended domain
is `delta > 0`.
-/
def huberLoss (prediction target : α) (delta : α := 1) : α :=
  let d := prediction - target
  let ad := MathFunctions.abs d
  if delta > ad then
    (d * d) / 2
  else
    delta * (ad - delta / 2)

end Core
end RL
end Runtime
