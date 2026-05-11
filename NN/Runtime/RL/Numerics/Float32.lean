/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Numerics.Float32.Types
public import NN.Runtime.RL.Numerics.Float32.Returns
public import NN.Runtime.RL.Numerics.Float32.Advantage
public import NN.Runtime.RL.Numerics.Float32.PPO
public import NN.Runtime.RL.Numerics.Float32.Intervals

/-!
# RL Float32 Numeric Checks (Umbrella)

This is the stable import for TorchLean's explicit binary32 RL diagnostics. The implementation is
split by concern:

- `Types`: `IEEE32Exec`/`Interval32` aliases, boundary casts, and checked scalar primitives;
- `Returns`: checked discounted backups and fixed-horizon returns;
- `Advantage`: checked TD residuals, GAE(λ), and advantage normalization;
- `PPO`: checked importance ratios and clipped PPO objective pieces;
- `Intervals`: outward-rounded interval enclosures for return/GAE/PPO diagnostics.

The public namespace remains `Runtime.RL.Numerics.Float32`, so existing downstream imports and names
continue to work while the source layout is easier to review.

References:
- IEEE 754-2019 (binary32 arithmetic): https://doi.org/10.1109/IEEESTD.2019.8766229
- IEEE 1788-2015 (interval arithmetic API/semantics): https://doi.org/10.1109/IEEESTD.2015.7140721
- Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic” (1991):
  https://doi.org/10.1145/103162.103163
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.).
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015), and "Proximal Policy Optimization Algorithms" (2017).
-/

@[expose] public section
