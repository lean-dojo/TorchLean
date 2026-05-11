/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Entrypoint.Widgets
public import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# PPO Atari Pong RAM Artifacts

This file visualizes the training curve produced by
`NN/Examples/Models/RL/PPOPongRam.lean` (`lake exe torchlean ppo_pong_ram`).

Pong RAM intentionally uses the same Gymnasium boundary as CartPole, but with ALE registration and
a higher-dimensional observation. It is an integration/regression example, not a tuned Atari agent.

Workflow:
1. Run:

```bash
python3 -m pip install --user 'gymnasium>=1.0' ale-py
lake exe torchlean ppo_pong_ram
lake build -R -K cuda=true && lake exe torchlean ppo_pong_ram --cuda
```

2. Put the cursor on the command below in an editor. The infoview will render the saved log.

Notes:
- The executable writes `data/rl/ppo_pong_ram_trainlog.json` by default (override with `--log`).
- This viewer is pure: if the file is missing, it shows an error panel instead of failing to build.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Machado et al., "Revisiting the Arcade Learning Environment" (2018): https://arxiv.org/abs/1709.06009
- ALE docs: https://ale.farama.org/
-/

/-- Default training-log path written by `torchlean ppo_pong_ram` (override with `--log`). -/
def trainLogPath : System.FilePath :=
  Runtime.RL.Artifacts.DefaultPaths.ppoPongRamTrainLog

#train_log_file_view trainLogPath
