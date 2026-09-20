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

Umbrella import for TorchLean's RL trust-boundary layer. The boundary has two parts:

- `NN.Runtime.RL.Boundary.Core`: contracts, executable checkers, and Prop-level validity predicates;
- `NN.Runtime.RL.Boundary.Json`: a small JSON rollout schema plus parser/validator for external
  producers such as Gymnasium scripts.

Use this umbrella when you want the full runtime boundary API. Proof modules that do not parse JSON
can import `Boundary.Core` directly.
-/

@[expose] public section
