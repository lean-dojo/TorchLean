/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Algorithms.Bandits
public import NN.Runtime.RL.Algorithms.Tabular
public import NN.Runtime.RL.Algorithms.ValueLearning
public import NN.Runtime.RL.Algorithms.DQN
public import NN.Runtime.RL.Algorithms.PolicyGradient

/-!
# RL Algorithm Equations

This umbrella collects the runtime-side equations for finite-action reinforcement-learning
algorithms:

- multi-armed bandits;
- tabular TD/SARSA/Q-learning updates;
- deep value-learning targets and scalar losses; and
- replay-minibatch DQN helpers; and
- categorical policy-gradient / PPO objectives.

These modules are "runtime" because they use TorchLean's typed tensor surface and executable scalar
classes, but they are still mostly pure mathematical equations. Environment IO, Gymnasium sessions,
rollout collection, and trust-boundary checks live outside this folder.

References:
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
- Watkins and Dayan, "Q-learning", 1992.
- Williams, "Simple Statistical Gradient-Following Algorithms", 1992.
- Mnih et al., "Human-level control through deep reinforcement learning", 2015.
- van Hasselt, Guez, and Silver, "Deep Reinforcement Learning with Double Q-learning", 2016.
- Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
-/

@[expose] public section
