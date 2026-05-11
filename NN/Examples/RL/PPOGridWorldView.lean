/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.RL.Envs.GridWorld
public import NN.Entrypoint.Widgets
public import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# PPO GridWorld Artifacts

This file visualizes the artifacts produced by
`NN/Examples/Models/RL/PPOGridWorld.lean` (`lake exe torchlean ppo_gridworld`).

GridWorld is the smallest RL artifact path because the environment itself is Lean-native: the
executable can both train and emit artifacts, while the proof layer can reason about the finite MDP
model.

Tip:
- For CUDA: `lake build -R -K cuda=true && lake exe torchlean ppo_gridworld --cuda`
- For a quick run that still writes artifacts: add `--updates 200`

The executable writes three JSON files by default:
- `data/rl/ppo_gridworld_trainlog.json` (greedy-policy evaluation return curve)
- `data/rl/ppo_gridworld_policy.json` (before/after greedy policy snapshot)
- `data/rl/ppo_gridworld_path.json` (before/after greedy episode path)

Put the cursor on the `#*_file_view` commands below to render them in the infoview.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015):
  https://arxiv.org/abs/1506.02438
-/

open Spec.RL.Envs

/-- A 4x4 GridWorld layout matching the executable example defaults. -/
def gw44 : GridWorld 4 4 :=
  { start := (⟨0, by decide⟩, ⟨0, by decide⟩)
    goal := (⟨3, by decide⟩, ⟨3, by decide⟩)
    -- Discount is not used by the widgets.
    discount := 0 }

/-- Default training-log path written by `torchlean ppo_gridworld` (override with `--log`). -/
def trainLogPath : System.FilePath :=
  Runtime.RL.Artifacts.DefaultPaths.ppoGridWorldTrainLog

/-- Default greedy-policy snapshot path written by `torchlean ppo_gridworld` (override with `--policy`). -/
def policyPath : System.FilePath :=
  Runtime.RL.Artifacts.DefaultPaths.ppoGridWorldPolicy

/-- Default greedy-episode path snapshot written by `torchlean ppo_gridworld` (override with `--path`). -/
def pathPath : System.FilePath :=
  Runtime.RL.Artifacts.DefaultPaths.ppoGridWorldPath

#gridworld_view gw44, gw44.start

#train_log_file_view trainLogPath

#gridworld_policy_file_view gw44, policyPath

#gridworld_path_file_view gw44, pathPath
