/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.ValueLearning
public import TorchLean.Tensor.Operations

/-!
# Tabular Reinforcement Learning

Typed, total update rules for classic finite-state, finite-action RL: TD(0) state-value
learning, SARSA, Expected SARSA, Q-learning and Double Q-learning. Pick the bootstrap target with
the matching rule (TD(0) applies `RL.discountedBackup` to `V(s')`), then apply `update` at a
tensor coordinate. One update serves state-value vectors, Q-tables and higher-dimensional value
tensors. The targets go through `RL.discountedBackup`, so a terminal step (`done`) gives the
reward alone.

We wrote the rules by hand from Sutton and Barto. `update` is their incremental step (2.4), and
with the matching target it gives TD(0) (6.2), Sarsa (6.7), Q-learning (6.8), Expected Sarsa
(6.9) and Double Q-learning (6.10). `doubleQTarget` is the target in the `UPDATE(A)` branch of
van Hasselt's Algorithm 1 when the selector is $Q^A$. Ties in `greedy?` and in the Double Q pick
go to the lowest action index, through `Tensor.argmax`. The step size is an argument of each call,
so a per-pair rate $\alpha(s,a)$ is the caller's to keep. No theorem or test covers these rules.

Primary references:

- Sutton, "Learning to Predict by the Methods of Temporal Differences" (1988), Machine Learning
  3(1):9-44: https://doi.org/10.1023/A:1022633531479
- Rummery and Niranjan, "On-line Q-learning using connectionist systems" (1994), Cambridge
  University Engineering Department, CUED/F-INFENG/TR 166 (SARSA precursor):
  https://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf
- Sutton, "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"
  (1996), Advances in Neural Information Processing Systems 8, pp. 1038-1044, Figure 1 (SARSA /
  function approximation example): http://incompleteideas.net/papers/sutton-96.pdf
- Watkins and Dayan, "Q-learning" (1992), Machine Learning 8(3-4):279-292:
  https://doi.org/10.1007/BF00992698
- van Hasselt, "Double Q-learning" (2010), Advances in Neural Information Processing Systems 23,
  pp. 2613-2621, Algorithm 1:
  https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html
  (author manuscript: https://ir.cwi.nl/pub/16889)
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  Sections 2.4 and 6.1 to 6.7: http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Tabular

open Tensor

/-- Move one tensor entry toward a supplied target by `stepSize`, evaluated in the order
`current + stepSize * (target - current)`.

The target rule picks the algorithm, and the update is the same for every table shape. For Double
Q-learning, update either table with a target that uses it as the selector and the other table as
the evaluator.
-/
def update {α : Type} [Storage α] [Storage.Update α] [Add α] [Sub α] [Mul α]
    {shape : Shape} (values : Tensor α shape) (coordinate : shape.Coord)
    (target stepSize : α) : Tensor α shape :=
  let current := values coordinate
  Tensor.set values coordinate (current + stepSize * (target - current))

variable {α : Type} [Storage α] [Context α]

/-- Greedy action at a state, if the action space is nonempty. -/
def greedy? {nStates nActions : Nat} (q : Tensor α [nStates, nActions])
    (state : Fin nStates) : Option (Fin nActions) :=
  if h : 0 < nActions then
    some (Tensor.argmax (Tensor.unstack q state) ⟨0, Nat.one_pos⟩ false h).item
  else none

/-- Policy-weighted action value `Σ_a π(a) Q(s, a)`. The weights are not normalized. -/
def expectation {nStates nActions : Nat}
    (q : Tensor α [nStates, nActions])
    (state : Fin nStates)
    (policy : Tensor α [nActions]) : α :=
  Tensor.sum (Tensor.unstack q state * policy)

/-- SARSA target `r + γ Q(s', a')`. -/
def sarsaTarget {nStates nActions : Nat} (q : Tensor α [nStates, nActions])
    (nextState : Fin nStates) (nextAction : Fin nActions) (reward gamma : α)
    (done : Bool := false) : α :=
  RL.discountedBackup (α := α) reward gamma (get2 q nextState nextAction) done

/-- Expected SARSA target
`r + γ * E_{a' ~ π(.|s')}[Q(s', a')]`. -/
def expectedSarsaTarget {nStates nActions : Nat}
    (q : Tensor α [nStates, nActions])
    (nextState : Fin nStates) (nextPolicy : Tensor α [nActions])
    (reward gamma : α) (done : Bool := false) : α :=
  RL.discountedBackup (α := α) reward gamma
    (expectation (α := α) q nextState nextPolicy) done

/-- Q-learning target `r + γ max_a Q(s', a)`: the DQN target on the table row of `nextState`. -/
def qLearningTarget {nStates nActions : Nat} (q : Tensor α [nStates, nActions])
    (nextState : Fin nStates) (reward gamma : α) (done : Bool := false) : α :=
  ValueLearning.dqnTarget (α := α) reward gamma done (Tensor.unstack q nextState)

/-- Double Q-learning target: the Double DQN target with the `selector` row choosing the greedy
action and the `evaluator` row valuing it. -/
def doubleQTarget {nStates nActions : Nat}
    (selector evaluator : Tensor α [nStates, nActions])
    (nextState : Fin nStates) (reward gamma : α) (done : Bool := false) : α :=
  ValueLearning.doubleDQNTarget (α := α) reward gamma done (Tensor.unstack selector nextState)
    (Tensor.unstack evaluator nextState)

end Tabular
end RL
end TorchLean
