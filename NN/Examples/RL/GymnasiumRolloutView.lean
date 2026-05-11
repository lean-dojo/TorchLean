/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Entrypoint.Widgets
public import NN.Runtime.RL.Boundary.Core

/-!
# Gymnasium Rollout Boundary Viewer

This is the “first thing to open” when an external Gymnasium rollout looks wrong.

Workflow:

1. Export a rollout in Python:

```bash
python3 -m pip install --user 'gymnasium>=1.0'
python3 scripts/rl/export_gymnasium_rollout.py --env-id CartPole-v1 --steps 256 --seed 0 \
  --out data/rl/gym_cartpole_rollout.json
```

2. Open this file in an editor and put the cursor on the `#rl_boundary_rollout_file_view` command.

The widget validates every transition against the contract and summarizes any violations. This is
the trust boundary that sits between untrusted Python environments and Lean-side PPO code.
-/

open Spec Tensor

namespace NN.Examples.RL.GymnasiumRolloutView

def rolloutPath : System.FilePath :=
  ("data/rl/gym_cartpole_rollout.json" : System.FilePath)

def obsShape : Shape := shape![4]
def nActions : Nat := 2

def contract : Runtime.RL.Boundary.Contract obsShape nActions :=
  { checkObsFinite := true
    checkRewardFinite := true
    obsRange? := none
    rewardRange? := none
    requireExclusiveDoneFlags := false }

#rl_boundary_rollout_file_view rolloutPath, contract, 12

end NN.Examples.RL.GymnasiumRolloutView
