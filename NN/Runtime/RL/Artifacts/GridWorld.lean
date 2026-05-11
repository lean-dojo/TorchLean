/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Artifacts.GridWorld.Position
public import NN.Runtime.RL.Artifacts.GridWorld.Policy
public import NN.Runtime.RL.Artifacts.GridWorld.Path

/-!
# GridWorld Run Artifacts (Umbrella)

Stable import for small GridWorld JSON artifacts used by TorchLean RL workflows and widgets. The
implementation is split into:

- `Position`: shared `(row, col)` JSON helpers;
- `Policy`: before/after greedy policy snapshots;
- `Path`: before/after rollout trajectory snapshots.

Primitive Nat/String array codecs come from `Runtime.Training.JsonCodec`, so training logs and RL
artifacts share one JSON vocabulary.
-/

@[expose] public section
