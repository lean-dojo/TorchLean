/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import TorchLean.NN.RL.Boundary
import TorchLean.NN.RL.Transition
import TorchLean.NN.Widget.RL.Boundary

/-!
# Gymnasium Rollout Boundary Viewer

Open this view first when an external Gymnasium rollout looks wrong.

Workflow:

1. Export a rollout in Python:

```bash
python3 -m pip install --user 'gymnasium>=1.0'
python3 scripts/rl/gymnasium_server.py --env-id CartPole-v1 --steps 256 --seed 0 \
  --out data/rl/gym_cartpole_rollout.json
```

2. Open this file in an editor and put the cursor on the `#rl_boundary_rollout_file_view` command.

The widget validates every transition against the contract, counts the passing and failing ones,
and lists up to 12 violations. The same contract check separates untrusted Python environments
from the Lean-side PPO code.
-/

open TorchLean.Tensor

namespace TorchLean.Examples.GymnasiumRollout

/--
Path of the local CartPole rollout. Run the export command above before opening the view.
-/
def rolloutPath : System.FilePath :=
  ("data/rl/gym_cartpole_rollout.json" : System.FilePath)

/-- CartPole's four observation features: position, velocity, pole angle, pole angular velocity. -/
def observation : Shape := [4]
/-- Two actions: push left or push right. -/
def nActions : Nat := 2

/--
The trust boundary for this rollout: finite observations and rewards, no range restriction.

The two CartPole velocity features are unbounded in principle, so a fixed range check could reject
valid transitions even though the recorded values happen to be small.
-/
def contract : RL.Boundary.Contract observation nActions :=
  { checkObsFinite := true
    checkRewardFinite := true
    obsRange? := none
    rewardRange? := none
    requireExclusiveDoneFlags := false }

#rl_boundary_rollout_file_view rolloutPath, contract, 12

end TorchLean.Examples.GymnasiumRollout
