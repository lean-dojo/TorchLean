/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.PyTorch.ParamStore

/-!
# PINN PyTorch

Verification-facing PyTorch import surface for trained PINNs.

This module exists so verification call sites can import one file:
`NN.Verification.PINN.PyTorch`.

Implementation is split for readability:

- `NN.Verification.PINN.PyTorch.Load` parses JSON into a shape-checked `PinnState`.
- `NN.Verification.PINN.PyTorch.ParamStore` turns a `PinnState` into the CROWN graph/parameter
  representation used by the verifier.
-/

@[expose] public section
