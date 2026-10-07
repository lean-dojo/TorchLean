# Reinforcement-Learning Model Examples

This folder contains runnable RL commands and the views for their saved artifacts. Pure RL specs
live in `NN/Spec/RL/`, runtime sessions in `NN/Runtime/RL/`, and proof hooks in `NN/Proofs/RL/`.

## Files

- `PPOGridWorld.lean`: PPO on a Lean-native GridWorld. Even though the environment is Lean code,
  transitions are still checked through the same boundary shape used by external environments.
- `PPOCartPole.lean`: PPO on external Gymnasium `CartPole-v1`, with every step checked before it is
  consumed as training data.
- `PPOPongRam.lean`: optional ALE/Gymnasium Pong RAM path. This depends on external `ale-py` and is
  not part of the default quick-check tier.
- `DQNReplay.lean`: small replay-buffer and DQN minibatch-loss example using hand-written Q
  functions rather than a full trainable neural DQN.
- `Views/`: editor widgets for training logs, policies, episode paths, and checked external
  rollouts.

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

Optional Pong RAM path:

```bash
python3 -m pip install --user 'gymnasium>=1.0' ale-py
scripts/lake.sh -Kcuda=true exe torchlean ppo_pong_ram --device cuda --updates 1
```

## Artifacts

PPO commands write JSON artifacts under `data/rl/` by default. Open the corresponding file in
`Views/` and place the cursor on its widget command to inspect the artifact in the Lean infoview.

| Viewer under `Views/` | Artifact producer | What it shows |
| --- | --- | --- |
| `PPOCartPole.lean` | `torchlean ppo_cartpole` | Greedy evaluation return by training checkpoint |
| `PPOGridWorld.lean` | `torchlean ppo_gridworld` | Return curve, before/after policies, and episode paths |
| `PPOPongRam.lean` | `torchlean ppo_pong_ram` | Evaluation return from the optional ALE/Gymnasium run |
| `GymnasiumRollout.lean` | `scripts/rl/gymnasium_server.py --out PATH` | Recorded transitions and boundary-check results |

These views display saved data; opening them does not launch training. If you choose a custom
output path, update the viewer path too. Missing files appear as error panels, so the modules can
build before a training run. `GymnasiumRollout.lean` includes the export command for its expected
`data/rl/gym_cartpole_rollout.json` file. Its observation, action, reward, and termination checks
do not prove that the external simulator obeys an MDP model or that a policy is optimal.

## What Is Checked

The RL examples are executable algorithm examples with formal hooks. The checked surface is the
environment/rollout boundary and the Lean-native MDP structure that downstream code consumes:

- Gymnasium transitions are decoded into typed observations and actions, then checked against the
  configured contract. CartPole requires finite observations and rewards; Pong additionally checks
  that RAM values lie in `[0, 255]`. Both permit simultaneous termination and truncation.
- Lean-native GridWorld also goes through the boundary checker so downstream code sees one data
  shape.
- MDP and boundary facts live in `NN/Spec/RL` and `NN/Proofs/RL`. The example's `proofGridWorld`
  uses sparse rewards, while its executable trains with shaped rewards; the validity theorem
  concerns the former.

That separation lets the same rollout data be used for runtime training, widget inspection, and
future theorem statements. External simulators stay named producers; TorchLean owns the typed
rollout records, boundary checks, Lean-native environment specs, and proof hooks built on top of
those records.
