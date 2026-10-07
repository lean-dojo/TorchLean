/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Algorithms.ValueLearning

/-!
# Tabular Reinforcement Learning

This module implements typed, total update rules for classic finite-state / finite-action RL:

- TD(0) state-value learning,
- SARSA,
- Expected SARSA,
- Q-learning,
- Double Q-learning.

Choose the bootstrap target with the appropriate rule, then apply `update` to a tensor coordinate.
The same update works for state-value vectors, Q-tables, and higher-dimensional value tensors.

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

open Spec TorchLean
open TorchLean TorchLean.Tensor

/-- Move one tensor entry toward a supplied target by `stepSize`.

The target rule determines the algorithm; the update itself is independent of the table's shape.
For Double Q-learning, update either table with a target computed using that table as selector and
the other as evaluator. Arithmetic keeps the order `current + stepSize * (target - current)`.
-/
def update {α : Type} [Storage α] [Storage.Update α] [Add α] [Sub α] [Mul α]
    {shape : Shape} (values : Tensor α shape) (coordinate : shape.Coord)
    (target stepSize : α) : Tensor α shape :=
  let current := values coordinate
  Tensor.set values coordinate (current + stepSize * (target - current))

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Extract the action-value row `Q[s, :]`. -/
def row {nStates nActions : Nat} (q : Tensor α [nStates, nActions])
    (state : Fin nStates) : Tensor α [nActions] :=
  get q state

/-- Max action value at a state, defaulting to `0` for empty action spaces. -/
def maximum {nStates nActions : Nat} (q : Tensor α [nStates, nActions])
    (state : Fin nStates) : α :=
  let row := row (α := α) q state
  ValueLearning.maximum (α := α) row

/-- Greedy action at a state, if the action space is nonempty. -/
def greedy? {nStates nActions : Nat} (q : Tensor α [nStates, nActions])
    (state : Fin nStates) : Option (Fin nActions) :=
  (TorchLean.Metrics.argmax? (α := α) (row (α := α) q state)).map
    (Fin.cast (by simp [Shape.size]))

/-- Policy-weighted action value. Weights are used as supplied, without normalization. -/
def expectation {nStates nActions : Nat}
    (q : Tensor α [nStates, nActions])
    (state : Fin nStates)
    (policy : Tensor α [nActions]) : α :=
  sumSpec (mulSpec (row (α := α) q state) policy)

/-- SARSA target `r + γ Q(s', a')`. -/
def sarsaTarget {nStates nActions : Nat} (q : Tensor α [nStates, nActions])
    (nextState : Fin nStates) (nextAction : Fin nActions) (reward gamma : α)
    (done : Bool := false) : α :=
  Core.discountedBackup (α := α) reward gamma (get2 q nextState nextAction) done

/-- Expected SARSA target
`r + γ * E_{a' ~ π(.|s')}[Q(s', a')]`. -/
def expectedSarsaTarget {nStates nActions : Nat}
    (q : Tensor α [nStates, nActions])
    (nextState : Fin nStates) (nextPolicy : Tensor α [nActions])
    (reward gamma : α) (done : Bool := false) : α :=
  Core.discountedBackup (α := α) reward gamma
    (expectation (α := α) q nextState nextPolicy) done

/-- Q-learning target `r + γ max_a Q(s', a)`. -/
def qLearningTarget {nStates nActions : Nat} (q : Tensor α [nStates, nActions])
    (nextState : Fin nStates) (reward gamma : α) (done : Bool := false) : α :=
  Core.discountedBackup (α := α) reward gamma (maximum (α := α) q nextState) done

/-- Double Q-learning / Double DQN-style target:
choose the greedy action under `selector`, evaluate it under `evaluator`. -/
def doubleQTarget {nStates nActions : Nat}
    (selector evaluator : Tensor α [nStates, nActions])
    (nextState : Fin nStates) (reward gamma : α) (done : Bool := false) : α :=
  match greedy? (α := α) selector nextState with
  | some action =>
      Core.discountedBackup (α := α) reward gamma (get2 evaluator nextState action) done
  | none => reward

end Tabular
end RL
end Runtime
