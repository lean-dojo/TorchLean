/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Numerics.Float32

/-!
# RL Numeric Diagnostics

This umbrella collects optional numeric hardening tools for runtime RL code. These modules do not
replace the ordinary scalar-polymorphic RL equations; they add checked execution paths for cases
where we want explicit floating-point semantics or interval-style diagnostics.

Current contents:
- `NN.Runtime.RL.Numerics.Float32.Types`: checked `Float → IEEE32Exec` casting.
- `NN.Runtime.RL.Numerics.Float32.Returns`: explicit binary32 discounted backups and returns.
- `NN.Runtime.RL.Numerics.Float32.Advantage`: TD residuals, GAE, and normalization with
  finite-intermediate checks.
- `NN.Runtime.RL.Numerics.Float32.PPO`: checked PPO scalar objective helpers.
- `NN.Runtime.RL.Numerics.Float32.Intervals`: outward-rounded interval enclosures.

References:
- IEEE 754-2019, floating-point arithmetic.
- IEEE 1788-2015, interval arithmetic.
- Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", 1991.
-/

@[expose] public section
