# Lean-native GridWorld

This directory holds the GridWorld environment that `torchlean ppo_gridworld` trains on, its
reward and MDP theorems, the JSON snapshots a PPO run saves, and the infoview widgets that draw
them.

## Files

- `Env.lean` (namespace `TorchLean.Examples`): `GridWorld width height` with a `start` cell, a
  `goal` cell and a real `discount`. A state is a `(row, col)` pair and the actions up, down, left
  and right are 0 to 3. A move past the border leaves that coordinate unchanged. `step` gives
  reward 0 and terminates at the goal or on reaching it, and gives reward -1 otherwise. `truncated`
  is always false. `toEnv`, `toFiniteMDP` and `toStochasticMDP` view the grid as an environment and
  as finite MDPs over `height * width` states, numbered row-major by `encode` and `decode`.
- `Properties.lean` (namespace `TorchLean.Examples.GridWorld`): `step_reward_bounds` (every reward
  lies in `[-1, 0]`) and `toStochasticMDP_valid` (the one-hot stochastic view is a valid MDP
  when `0 ≤ discount` and `discount < 1`). Two more theorems show that its expected next values and
  action values agree with those of `toFiniteMDP`.
- `Artifacts.lean` (namespace `TorchLean.Examples.GridWorld`): `Diff`, a before/after snapshot
  with the grid size and notes, and its JSON codecs `GridWorld.Codec.policy` (one action below 4 per
  cell, row-major) and `GridWorld.Codec.path` (`[row, col]` positions inside the grid).
  `Diff.writeJson` validates before writing and creates parent directories, and `Diff.readJson`
  validates what it reads.
- `Widgets.lean` (the same namespace): `#gridworld_view gw, pos`,
  `#gridworld_policy_view gw, policy` and `#gridworld_path_view gw, path` draw a grid, a policy and
  a path. `#gridworld_policy_file_view gw, path` and `#gridworld_path_file_view gw, path` draw a
  saved snapshot, or an error panel when the file is missing or invalid. No widget runs training.

## Producing snapshots

A short CPU run from the repository root:

```bash
scripts/lake.sh exe torchlean ppo_gridworld --updates 1 --eval-every 1 \
  --eval-episodes 1 --eval-max-steps 8
```

It writes the training log `data/rl/ppo_gridworld_trainlog.json` and the snapshots
`data/rl/ppo_gridworld_policy.json` and `data/rl/ppo_gridworld_path.json`. `--log`, `--policy` and
`--path` change the three paths.

## Users

- [`../PPOGridWorld.lean`](../PPOGridWorld.lean) trains PPO on a 4×4 grid from (0, 0) to (3, 3)
  with horizon 64 and saves its snapshots with `Diff.writeJson`. Its `realGridWorld` (discount
  99/100) has the rewards of `step`, and `realGridWorld_valid` applies
  `toStochasticMDP_valid` to it. Training runs on the shaped rewards of `stepState`: 1 for
  reaching the goal, and otherwise the decrease in Manhattan distance minus 0.05. No theorem
  relates the two.
- [`../PPOGridWorld/View.lean`](../PPOGridWorld/View.lean) draws the start cell and the saved
  snapshots of a run.
- [`../../../DeepDives/WidgetGallery.lean`](../../../DeepDives/WidgetGallery.lean) shows the
  three direct widgets on a 4×4 grid with an always-right policy.
- The guide chapters `ReinforcementLearning.lean` and `Widgets.lean` under
  `home_page/blueprint/TorchLeanBlueprint/Guide/Ch5_Applications/` import `Properties.lean` and
  `Widgets.lean`.
