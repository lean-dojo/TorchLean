/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Boundary.Core
public import NN.Runtime.RL.Boundary.Json

/-!
# RL Trust Boundary (Umbrella)

Stable import for TorchLean's RL trust-boundary layer. The implementation is split into:

- `NN.Runtime.RL.Boundary.Core`: contracts, executable checkers, and Prop-level validity predicates;
- `NN.Runtime.RL.Boundary.Json`: a small JSON rollout schema plus parser/validator for external
  producers such as Gymnasium scripts.

Use this umbrella when you want the full runtime surface. Proof modules that do not parse JSON can
import `Boundary.Core` directly.
-/

@[expose] public section
