/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Widgets
import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# PPO Atari Pong RAM View

This file visualizes the training curve produced by the optional Pong RAM PPO module in
`NN/Examples/Models/RL/PPOPongRam.lean`.

Pong RAM uses the same Gymnasium boundary as CartPole, with ALE registration and a
higher-dimensional observation. The viewer focuses on saved-artifact inspection for the Atari RAM
path rather than benchmark-specific PPO tuning.

Dependency setup for the optional ALE path:

```bash
python3 -m pip install --user 'gymnasium>=1.0' ale-py
```

After producing a log, put the cursor on the command below in an editor. The infoview will render
the saved log.

Notes:
- The executable writes `data/rl/ppo_pong_ram_trainlog.json` by default (override with `--log`).
- This viewer is pure: if the file is missing, it shows an error panel instead of failing to build.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
- Machado et al., "Revisiting the Arcade Learning Environment" (2018):
  https://arxiv.org/abs/1709.06009
- ALE docs: https://ale.farama.org/
-/

/-- Default training-log path for the optional Pong RAM artifact viewer. -/
def trainLogPath : System.FilePath :=
  Runtime.RL.Artifacts.DefaultPaths.ppoPongRamTrainLog

#train_log_file_view trainLogPath
