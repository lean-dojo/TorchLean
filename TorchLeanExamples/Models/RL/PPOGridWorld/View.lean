/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import TorchLeanExamples.Models.RL.GridWorld.Env
import TorchLeanExamples.Models.Training
meta import TorchLeanExamples.Models.Training
public meta import TorchLeanExamples.Models.RL.GridWorld.Artifacts
import TorchLean.NN.RL.Transition
import TorchLeanExamples.Models.RL.GridWorld.Widgets
import TorchLean.NN.Widget.Training

/-!
# PPO GridWorld View

The commands below display the artifacts produced by `TorchLeanExamples/Models/RL/PPOGridWorld.lean`
(`scripts/lake.sh exe torchlean ppo_gridworld`).

GridWorld is the smallest RL artifact path because the environment itself is Lean-native: the
executable can both train and emit artifacts, while the proof layer can reason about the finite MDP
model.

Tip:
- For a short run that writes artifacts:
  ```bash
  scripts/lake.sh -Kcuda=true exe torchlean ppo_gridworld --device cuda --updates 1 --eval-every 1 \
    --eval-episodes 1 --eval-max-steps 8
  ```

The executable writes three JSON files by default:
- `data/rl/ppo_gridworld_trainlog.json` (greedy-policy evaluation return curve)
- `data/rl/ppo_gridworld_policy.json` (before/after greedy policy snapshot)
- `data/rl/ppo_gridworld_path.json` (before/after greedy episode path)

`#gridworld_view` draws the grid and highlights the start, the goal and the given position, here
the start cell. Put the cursor on the `#*_file_view` commands below to render the three artifacts
in the infoview.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
-/

namespace TorchLean.Examples.Views.PPOGridWorld

/-- A 4x4 GridWorld layout matching the executable example defaults. -/
def defaultGridWorld : GridWorld 4 4 :=
  { start := (⟨0, by decide⟩, ⟨0, by decide⟩)
    goal := (⟨3, by decide⟩, ⟨3, by decide⟩)
    -- Discount is not used by the widgets.
    discount := 0 }

/-- Default training-log path written by `torchlean ppo_gridworld` (override with `--log`). -/
def trainLogPath : System.FilePath :=
  Training.artifactPath "ppo_gridworld" (dir := "rl")

/--
Default greedy-policy snapshot path written by `torchlean ppo_gridworld` (override with `--policy`).
-/
def policyPath : System.FilePath :=
  Training.artifactPath "ppo_gridworld" "policy" "rl"

/--
Default greedy-episode path snapshot written by `torchlean ppo_gridworld` (override with `--path`).
-/
def episodePath : System.FilePath :=
  Training.artifactPath "ppo_gridworld" "path" "rl"

#gridworld_view defaultGridWorld, defaultGridWorld.start

#train_log_file_view trainLogPath

#gridworld_policy_file_view defaultGridWorld, policyPath

#gridworld_path_file_view defaultGridWorld, episodePath

end TorchLean.Examples.Views.PPOGridWorld
