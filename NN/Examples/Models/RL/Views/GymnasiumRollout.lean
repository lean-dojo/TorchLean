/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Runtime.RL.Boundary.Core
import NN.Runtime.RL.Core
import NN.Widgets.RL.Boundary

/-!
# Gymnasium Rollout Boundary Viewer

This is the “first thing to open” when an external Gymnasium rollout looks wrong.

Workflow:

1. Export a rollout in Python:

```bash
python3 -m pip install --user 'gymnasium>=1.0'
python3 scripts/rl/gymnasium_server.py --env-id CartPole-v1 --steps 256 --seed 0 \
  --out data/rl/gym_cartpole_rollout.json
```

2. Open this file in an editor and put the cursor on the `#rl_boundary_rollout_file_view` command.

The widget validates every transition against the contract and summarizes any violations. This is
the trust boundary that sits between untrusted Python environments and Lean side PPO code.
-/

open Spec TorchLean TorchLean.Tensor

namespace NN.Examples.Models.RL.Views.GymnasiumRollout

/--
A recorded CartPole rollout, checked in so this view renders without needing Gymnasium installed.
-/
def rolloutPath : System.FilePath :=
  ("data/rl/gym_cartpole_rollout.json" : System.FilePath)

/-- CartPole's four observation features: position, velocity, pole angle, pole angular velocity. -/
def observation : Shape := [4]
/-- Two actions: push left or push right. -/
def nActions : Nat := 2

/--
The trust boundary for this rollout: finite observations and rewards, no range restriction.

CartPole's observations are unbounded in principle, so a range check would be wrong here even though
the recorded values happen to be small.
-/
def contract : Runtime.RL.Boundary.Contract observation nActions :=
  { checkObsFinite := true
    checkRewardFinite := true
    obsRange? := none
    rewardRange? := none
    requireExclusiveDoneFlags := false }

#rl_boundary_rollout_file_view rolloutPath, contract, 12

end NN.Examples.Models.RL.Views.GymnasiumRollout
