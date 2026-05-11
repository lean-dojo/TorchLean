/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.PPO.Rollout
public import NN.Runtime.RL.PPO.Collect

/-!
# PPO Helpers (Discrete Actions)

Umbrella import for TorchLean’s PPO rollout/training helpers.

The implementation is split into two focused submodules:

- `NN.Runtime.RL.PPO.Rollout`: rollout record + minibatch conversion (GAE/returns live in `Runtime.RL.Core`).
- `NN.Runtime.RL.PPO.Collect`: data collection from `Runtime.RL.Gymnasium.Session`.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015):
  https://arxiv.org/abs/1506.02438
-/

@[expose] public section
