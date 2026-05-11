/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.FloatInterval.ConstantTarget
public import NN.MLTheory.Proofs.Approximation.FloatInterval.ExactImageTheorem
public import NN.MLTheory.Proofs.Approximation.FloatInterval.Semantics

/-!
# Floating-point interval approximation proofs

This entrypoint collects the `IEEE32Exec` interval-semantics development used by the floating-point
universal-approximation development. The files underneath separate the work into:

- interval-domain semantics for executable binary32 networks;
- exact interval-image statements for rounded targets; and
- the constant-target base theorem.

The design keeps the trusted/executable float representation visible while proving the interval
claims in Lean over the concrete `IEEE32Exec` semantics.

Reference:
- Hwang, Lee, Park, Park, and Saad, "Floating-Point Neural Networks Are Provably Robust Universal
  Approximators", arXiv:2506.16065.
-/

@[expose] public section
