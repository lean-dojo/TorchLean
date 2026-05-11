/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.RL.PPOCartPoleView
public import NN.Examples.RL.PPOGridWorldView
public import NN.Examples.RL.PPOPongRamView
public import NN.Examples.RL.GymnasiumRolloutView

/-!
# RL Example Artifacts

This umbrella collects the editor-side RL artifact viewers.

The executable trainers live under `NN.Examples.Models` because they are runnable model/training
examples:

- `lake exe torchlean ppo_gridworld`
- `lake exe torchlean ppo_cartpole`
- `lake exe torchlean ppo_pong_ram`
- `lake exe torchlean dqn_replay`

The files under `NN/Examples/RL` are the companion layer: widget viewers, Python Gymnasium boundary
helpers, and rollout exporters. Keeping that split prevents the example tree from having two
competing PPO implementations while still making RL artifacts easy to inspect.
-/

@[expose] public section
