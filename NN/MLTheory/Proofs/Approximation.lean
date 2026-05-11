/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.FloatInterval
public import NN.MLTheory.Proofs.Approximation.Universal

/-!
# Approximation proofs

This entrypoint collects TorchLean's approximation-theory proof chapter. We separate the two main
threads while keeping them importable together:

- constructive ReLU universal approximation over real semantics, with quantitative and
  finite-precision variants; and
- exact interval semantics for rounded binary32 targets.

Both threads expose the mathematical assumptions and numerical side conditions explicitly rather
than burying them inside runtime behavior.
-/

@[expose] public section
