/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Runtime.Autograd.TorchLean.Metrics
public import NN.Spec.Layers.Activation

/-!
# Tabular Reinforcement Learning

This module implements typed, total update rules for classic finite-state / finite-action RL:

- TD(0) state-value learning,
- SARSA,
- Expected SARSA,
- Q-learning,
- Double Q-learning.

The updates operate on shape-indexed vectors / Q-tables, so they fit naturally into the rest of
TorchLean's typed tensor surface.

Primary references:

- Sutton, "Learning to Predict by the Methods of Temporal Differences" (1988):
  https://doi.org/10.1023/A:1022633531479
- Rummery and Niranjan, "On-line Q-learning using connectionist systems" (1994) (SARSA precursor):
  https://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf
- Sutton, "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"
  (1996) (SARSA / function approximation example):
  http://www.cs.ualberta.ca/~sutton/papers/sutton-96.pdf
- Watkins and Dayan, "Q-learning" (1992): https://doi.org/10.1007/BF00992698
- van Hasselt, "Double Q-learning" (2010):
  https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.):
  http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Tabular

open Spec
open Tensor

variable {α : Type} [Context α]

/-- Extract the action-value row `Q[s, :]`. -/
def actionRow {nStates nActions : Nat} (q : Tensor α (.dim nStates (.dim nActions .scalar)))
    (state : Fin nStates) : Tensor α (.dim nActions .scalar) :=
  get q state

/-- Max action value at a state, defaulting to `0` for empty action spaces. -/
def maxActionValue {nStates nActions : Nat} (q : Tensor α (.dim nStates (.dim nActions .scalar)))
    (state : Fin nStates) : α :=
  let row := actionRow (α := α) q state
  match Runtime.Autograd.TorchLean.Metrics.argmax? (α := α) (n := nActions) row with
  | some action => Tensor.vecGet row action
  | none => 0

/-- Greedy action at a state, if the action space is nonempty. -/
def greedyAction? {nStates nActions : Nat} (q : Tensor α (.dim nStates (.dim nActions .scalar)))
    (state : Fin nStates) : Option (Fin nActions) :=
  Runtime.Autograd.TorchLean.Metrics.argmax? (α := α) (n := nActions) (actionRow (α := α) q state)

/-- Expected action value under an explicit policy over the next state. -/
def expectedActionValue {nStates nActions : Nat}
    (q : Tensor α (.dim nStates (.dim nActions .scalar)))
    (state : Fin nStates)
    (policy : Tensor α (.dim nActions .scalar)) : α :=
  sumSpec (mulSpec (actionRow (α := α) q state) policy)

/-- One TD(0) update for a state-value table. -/
def td0Update {nStates : Nat} (values : Tensor α (.dim nStates .scalar))
    (state nextState : Fin nStates) (reward gamma stepSize : α) (done : Bool := false) :
    Tensor α (.dim nStates .scalar) :=
  let current := Tensor.vecGet values state
  let target := Core.tdTarget (α := α) reward gamma (Tensor.vecGet values nextState) done
  let newValue := current + stepSize * (target - current)
  Tensor.updateSpec values [state.val] newValue

/-- SARSA target `r + γ Q(s', a')`. -/
def sarsaTarget {nStates nActions : Nat} (q : Tensor α (.dim nStates (.dim nActions .scalar)))
    (nextState : Fin nStates) (nextAction : Fin nActions) (reward gamma : α) (done : Bool := false) :
    α :=
  Core.tdTarget (α := α) reward gamma (get2 q nextState nextAction) done

/-- Expected SARSA target
`r + γ * E_{a' ~ π(.|s')}[Q(s', a')]`. -/
def expectedSarsaTarget {nStates nActions : Nat}
    (q : Tensor α (.dim nStates (.dim nActions .scalar)))
    (nextState : Fin nStates) (nextPolicy : Tensor α (.dim nActions .scalar))
    (reward gamma : α) (done : Bool := false) : α :=
  Core.tdTarget (α := α) reward gamma
    (expectedActionValue (α := α) q nextState nextPolicy) done

/-- Q-learning target `r + γ max_a Q(s', a)`. -/
def qLearningTarget {nStates nActions : Nat} (q : Tensor α (.dim nStates (.dim nActions .scalar)))
    (nextState : Fin nStates) (reward gamma : α) (done : Bool := false) : α :=
  Core.tdTarget (α := α) reward gamma (maxActionValue (α := α) q nextState) done

/-- Double Q-learning / Double DQN-style target:
choose the greedy action under `selector`, evaluate it under `evaluator`. -/
def doubleQTarget {nStates nActions : Nat}
    (selector evaluator : Tensor α (.dim nStates (.dim nActions .scalar)))
    (nextState : Fin nStates) (reward gamma : α) (done : Bool := false) : α :=
  match greedyAction? (α := α) selector nextState with
  | some action => Core.tdTarget (α := α) reward gamma (get2 evaluator nextState action) done
  | none => reward

/-- In-place style SARSA update on a Q-table, returned functionally. -/
def sarsaUpdate {nStates nActions : Nat} (q : Tensor α (.dim nStates (.dim nActions .scalar)))
    (state : Fin nStates) (action : Fin nActions)
    (reward : α) (nextState : Fin nStates) (nextAction : Fin nActions)
    (gamma stepSize : α) (done : Bool := false) :
    Tensor α (.dim nStates (.dim nActions .scalar)) :=
  let current := get2 q state action
  let target := sarsaTarget (α := α) q nextState nextAction reward gamma done
  let newValue := current + stepSize * (target - current)
  Tensor.updateTensorSpec q [state.val, action.val] newValue

/-- Expected SARSA update on a Q-table. -/
def expectedSarsaUpdate {nStates nActions : Nat}
    (q : Tensor α (.dim nStates (.dim nActions .scalar)))
    (state : Fin nStates) (action : Fin nActions)
    (reward : α) (nextState : Fin nStates)
    (nextPolicy : Tensor α (.dim nActions .scalar))
    (gamma stepSize : α) (done : Bool := false) :
    Tensor α (.dim nStates (.dim nActions .scalar)) :=
  let current := get2 q state action
  let target := expectedSarsaTarget (α := α) q nextState nextPolicy reward gamma done
  let newValue := current + stepSize * (target - current)
  Tensor.updateTensorSpec q [state.val, action.val] newValue

/-- Q-learning update on a Q-table. -/
def qLearningUpdate {nStates nActions : Nat} (q : Tensor α (.dim nStates (.dim nActions .scalar)))
    (state : Fin nStates) (action : Fin nActions)
    (reward : α) (nextState : Fin nStates)
    (gamma stepSize : α) (done : Bool := false) :
    Tensor α (.dim nStates (.dim nActions .scalar)) :=
  let current := get2 q state action
  let target := qLearningTarget (α := α) q nextState reward gamma done
  let newValue := current + stepSize * (target - current)
  Tensor.updateTensorSpec q [state.val, action.val] newValue

/-- Update the left table in Double Q-learning. -/
def doubleQUpdateLeft {nStates nActions : Nat}
    (qLeft qRight : Tensor α (.dim nStates (.dim nActions .scalar)))
    (state : Fin nStates) (action : Fin nActions)
    (reward : α) (nextState : Fin nStates)
    (gamma stepSize : α) (done : Bool := false) :
    Tensor α (.dim nStates (.dim nActions .scalar)) :=
  let current := get2 qLeft state action
  let target := doubleQTarget (α := α) qLeft qRight nextState reward gamma done
  let newValue := current + stepSize * (target - current)
  Tensor.updateTensorSpec qLeft [state.val, action.val] newValue

/-- Update the right table in Double Q-learning. -/
def doubleQUpdateRight {nStates nActions : Nat}
    (qLeft qRight : Tensor α (.dim nStates (.dim nActions .scalar)))
    (state : Fin nStates) (action : Fin nActions)
    (reward : α) (nextState : Fin nStates)
    (gamma stepSize : α) (done : Bool := false) :
    Tensor α (.dim nStates (.dim nActions .scalar)) :=
  let current := get2 qRight state action
  let target := doubleQTarget (α := α) qRight qLeft nextState reward gamma done
  let newValue := current + stepSize * (target - current)
  Tensor.updateTensorSpec qRight [state.val, action.val] newValue

end Tabular
end RL
end Runtime
