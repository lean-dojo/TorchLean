/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Default Artifact Paths for RL Examples

TorchLean's runnable RL examples write small JSON artifacts to `data/rl/` by default:

- training curves (`TrainLog` JSON),
- optional policy snapshots, and
- optional episode path snapshots.

Both the executable trainers (under `NN/Examples/Models/RL`) and the editor-side viewers (under
`NN/Examples/RL`) need to agree on these defaults. Keeping the strings here gives both sides one
shared path convention.

Users can always override these paths with the corresponding CLI flags (e.g. `--log`, `--policy`,
`--path`).
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Artifacts
namespace DefaultPaths

/-- Default training-log path written by `torchlean ppo_cartpole` (override with `--log`). -/
def ppoCartPoleTrainLog : System.FilePath :=
  ("data/rl/ppo_cartpole_trainlog.json" : System.FilePath)

/-- Default training-log path written by `torchlean ppo_pong_ram` (override with `--log`). -/
def ppoPongRamTrainLog : System.FilePath :=
  ("data/rl/ppo_pong_ram_trainlog.json" : System.FilePath)

/-- Default training-log path written by `torchlean ppo_gridworld` (override with `--log`). -/
def ppoGridWorldTrainLog : System.FilePath :=
  ("data/rl/ppo_gridworld_trainlog.json" : System.FilePath)

/-- Default policy snapshot path written by `torchlean ppo_gridworld` (override with `--policy`). -/
def ppoGridWorldPolicy : System.FilePath :=
  ("data/rl/ppo_gridworld_policy.json" : System.FilePath)

/-- Default episode-path snapshot path written by `torchlean ppo_gridworld` (override with `--path`). -/
def ppoGridWorldPath : System.FilePath :=
  ("data/rl/ppo_gridworld_path.json" : System.FilePath)

end DefaultPaths
end Artifacts
end RL
end Runtime
