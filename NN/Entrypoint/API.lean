/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.API.Adapters
public import NN.API.Models.Generative
public import NN.API.SelfSupervised

/-!
# API entrypoint

This entrypoint exists mainly for symmetry with the other curated `NN.Entrypoint.*` umbrellas.
It re-exports the primary user-facing facade `NN.API.Public`.

Most users should still prefer `import NN`; use this module when you want only the public
PyTorch-shaped API without the broader library umbrella.
-/

@[expose] public section
