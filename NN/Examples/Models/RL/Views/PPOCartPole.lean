/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Widgets
import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# PPO CartPole View

This file visualizes the training curve produced by
`NN/Examples/Models/RL/PPOCartPole.lean`
(`lake -R -K cuda=true exe torchlean ppo_cartpole --device cuda ...`).

The executable trainer and this editor-side view live together under `Examples/Models/RL`.

Workflow:
1. Run:

```bash
python3 -m pip install --user 'gymnasium>=1.0'
lake -R -K cuda=true exe torchlean ppo_cartpole --device cuda --updates 1 --eval-every 1 \
  --eval-episodes 1 --eval-max-steps 8
```

2. Put the cursor on the command below in an editor. The infoview will render the saved log.

Notes:
- The executable writes `data/rl/ppo_cartpole_trainlog.json` by default (override with `--log`).
- This viewer is pure: if the file is missing, it shows an error panel instead of failing to build.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
-/

/-- Default training-log path written by `torchlean ppo_cartpole` (override with `--log`). -/
def trainLogPath : System.FilePath :=
  Runtime.RL.Artifacts.DefaultPaths.ppoCartPoleTrainLog

#train_log_file_view trainLogPath
