/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Gymnasium.Client
public import NN.Runtime.RL.Gymnasium.Session

/-!
# Gymnasium Bridge (Subprocess, JSON Lines)

Umbrella import for TorchLean’s small Gymnasium subprocess bridge.

The implementation is split into two focused submodules:

- `NN.Runtime.RL.Gymnasium.Client`: JSON-lines protocol, startup handshake, and low-level
  `reset`/`stepRaw` operations.
- `NN.Runtime.RL.Gymnasium.Session`: stores the previous observation so `stepChecked` can emit a
  full observed transition validated by the trust-boundary contract.

References:
- Gymnasium API docs (reset/step, terminated vs truncated): https://gymnasium.farama.org/
- The original Gym API paper (background on the env interface): https://arxiv.org/abs/1606.01540
- Gymnasium source repository (implementation reference): https://github.com/Farama-Foundation/Gymnasium
- Trust-boundary rationale: see `NN.Runtime.RL.Boundary`.
-/

@[expose] public section
