# Reinforcement-Learning Model Examples

Here are the runnable RL commands and the editor views for their saved artifacts. Pure RL specs,
runtime sessions, and proof hooks live in `TorchLean/NN/RL.lean` and `TorchLean/NN/RL/`.

## Files

- `PPOGridWorld.lean`: PPO on a GridWorld written in Lean.
- `PPOCartPole.lean`: PPO on external Gymnasium `CartPole-v1`, with every step checked before it
  becomes training data.
- `PPOAgent.lean`: the actor-critic setup of `ppo_gridworld` and `ppo_cartpole`. It builds the
  `nn.PPO` actor and critic for rollout batches, binds their PPO update, and evaluates their
  state on single observations.
- `DQNReplay.lean`: a small replay buffer and DQN minibatch loss over hand-written Q functions,
  with no trainable neural DQN.
- `PPOGridWorld/View.lean` and `GymnasiumRollout.lean`: editor widgets for training logs, policies,
  episode paths, and checked external rollouts.
- [`GridWorld/`](GridWorld/README.md): the Lean-native environment (`Env.lean`), its reward
  bounds and MDP validity (`step_reward_bounds` and `toStochasticMDP_valid` in
  `Properties.lean`), the JSON policy and path snapshots of `ppo_gridworld` (`Artifacts.lean`)
  and their infoview widgets (`Widgets.lean`).

## Commands

Lean-native GridWorld:

```bash
scripts/lake.sh -Kcuda=true exe torchlean ppo_gridworld --device cuda \
  --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8
```

Gymnasium CartPole:

```bash
python3 -m pip install --user 'gymnasium>=1.0'
scripts/lake.sh -Kcuda=true exe torchlean ppo_cartpole --device cuda \
  --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8
```

DQN replay mini-example:

```bash
scripts/lake.sh exe torchlean dqn_replay
```

## Artifacts

PPO commands write JSON artifacts under `data/rl/` by default. Inspect the CartPole training
log with the generic widget in a Lean file:

```lean
import TorchLean.NN.Widget.Training

#train_log_file_view "data/rl/ppo_cartpole_trainlog.json"
```

Place the cursor on a widget command to inspect its saved artifact in the Lean infoview. For
GridWorld's policy/path artifacts and Gymnasium transition checks, open the corresponding file:

| Viewer | Artifact producer | What it shows |
| --- | --- | --- |
| `PPOGridWorld/View.lean` | `torchlean ppo_gridworld` | Return curve, before/after policies, and episode paths |
| `GymnasiumRollout.lean` | `scripts/rl/gymnasium_server.py --out PATH` | Recorded transitions and boundary-check results |

The views read saved data, so opening one starts no training. If you choose a custom output path,
update the widget's path too. A missing file shows as an error panel, so the modules build before
any training run. `GymnasiumRollout.lean` includes the export command for its expected
`data/rl/gym_cartpole_rollout.json` file. Its observation, action, reward, and termination checks
do not show that the external simulator obeys an MDP model or that a policy is optimal.

## What Is Checked

The RL examples are executable algorithms with formal hooks. The checks cover the
environment/rollout boundary and the Lean-native MDP structure that downstream code consumes:

- Gymnasium transitions are decoded into typed observations and actions, then checked against the
  configured contract. CartPole requires finite observations and rewards and permits
  simultaneous termination and truncation.
- The Lean-native GridWorld goes through the same boundary checker, so downstream code sees one
  data shape.
- Generic MDP and boundary facts live in `TorchLean/NN/RL/`, and the GridWorld theorems in
  `GridWorld/Properties.lean`. The example's `realGridWorld` uses sparse rewards, while its
  executable trains with shaped rewards. Its validity theorem, `realGridWorld_valid`, concerns
  `realGridWorld` alone.

The same rollout data feeds runtime training, widget inspection, and theorem statements. External
simulators are named producers of that data, and TorchLean owns the typed rollout records,
boundary checks, Lean-native environment specs, and the proof hooks built on those records.
