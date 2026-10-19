/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Models.Attention
public import NN.Proofs.Models.MoE
public import NN.Proofs.Models.Mlp

/-!
# Model-Level Proofs

This module is the stable proof umbrella for properties of complete model components.

The separation is intentional:
- `NN.Spec.*` defines reusable mathematical and executable specifications;
- `NN.Proofs.Autograd.*` proves differentiation and tape/runtime correctness;
- `NN.Proofs.Models.*` proves model-facing invariants such as attention mask laws, attention-weight
  normalization, architecture equivariances, and the routing and gating laws of mixtures of
  experts.

Keeping model theorems behind this one import gives public users a clean path without asking them to
guess which attention leaf file contains the fact they need.
-/

@[expose] public section
