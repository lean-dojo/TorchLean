# RL artifact views

Open these Lean files in the editor after producing the corresponding JSON artifacts. Put the
cursor on a `#..._file_view` command to render it in the infoview. These files display saved data;
they do not launch training.

| File | Artifact producer | What the view shows |
| --- | --- | --- |
| `PPOCartPole.lean` | `torchlean ppo_cartpole` | Greedy evaluation return by training checkpoint |
| `PPOGridWorld.lean` | `torchlean ppo_gridworld` | Return curve, before/after policies, and episode paths |
| `PPOPongRam.lean` | `torchlean ppo_pong_ram` | Evaluation return from the optional ALE/Gymnasium run |
| `GymnasiumRollout.lean` | `scripts/rl/gymnasium_server.py --out PATH` | Recorded transitions and boundary-check results |

The first three producers write under `data/rl/` by default. Their parent
[RL guide](../README.md) gives the training commands. If a producer uses a custom output path,
update the corresponding viewer path too. Missing files appear as error panels, so the example
modules can still build before any training run.

`GymnasiumRollout.lean` includes the export command for its expected
`data/rl/gym_cartpole_rollout.json` file. Its checks cover the declared observation, action, reward,
and termination contract. Passing those checks does not prove that the external simulator obeys
an MDP model or that a trained policy is optimal.
