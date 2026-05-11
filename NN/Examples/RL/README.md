# RL Examples

This folder is the companion layer for TorchLean's runnable RL examples.

The executable trainers are under `NN/Examples/Models` and are selected through the shared runner:

- `lake exe torchlean ppo_gridworld`
- `lake exe torchlean ppo_cartpole`
- `lake exe torchlean ppo_pong_ram`
- `lake exe torchlean dqn_replay`

The files here do three different jobs:

- `PPOGridWorldView.lean`, `PPOCartPoleView.lean`, `PPOPongRamView.lean`: editor widgets for logs,
  GridWorld policies, and episode paths written by the trainers.

The Python boundary helpers live under `scripts/rl/` so runtime code does not depend on
`Examples/` paths:

- `scripts/rl/gymnasium_server.py`: JSON lines bridge used by Lean to step external Gymnasium
  environments behind a checked boundary contract.
- `scripts/rl/export_gymnasium_rollout.py`: exporter for offline rollout JSON accepted by the Lean
  RL boundary loader.
- `scripts/rl/train_ppo_cartpole_sb3.py`: Stable Baselines3 baseline for checking target
  performance.

## Recommended Workflow

1. Train or smoke test a Lean example:

```bash
lake exe torchlean ppo_gridworld --updates 200
lake exe torchlean ppo_cartpole
```

2. Open the corresponding `*View.lean` file in the editor and put the cursor on the widget command.

3. For external Gymnasium environments, treat Python as an untrusted producer: TorchLean checks
   observations, rewards, actions, and done flags before consuming rollout data.

## Dependencies

For CartPole:

```bash
python3 -m pip install --user 'gymnasium>=1.0'
```

For ALE/Pong RAM:

```bash
python3 -m pip install --user 'gymnasium>=1.0' ale-py
```

For the Python SB3 baseline:

```bash
python3 -m pip install --user 'gymnasium>=1.0' stable-baselines3
```

References:

- Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
  2015.
- Machado et al., "Revisiting the Arcade Learning Environment", 2018.
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
