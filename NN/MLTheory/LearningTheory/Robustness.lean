/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Robustness.Spec
public import NN.MLTheory.LearningTheory.Robustness.Runtime

/-!
# Robustness

This entrypoint pairs the two layers of TorchLean's robustness vocabulary:

- `Robustness.Spec` gives scalar-polymorphic definitions over shape-indexed tensors: tensor norms,
  distances, adversarial robustness, certified robustness, Lipschitz continuity, and contractions.
- `Robustness.Runtime` specializes those definitions to `Float` and provides finite, empirical
  diagnostics for examples and command-line checks.

We keep these layers separate on purpose. The spec layer is the mathematical language used by proof
developments; the runtime layer computes observed quantities from finite samples and does not claim
certification by itself.

References:
- Szegedy et al., "Intriguing properties of neural networks", ICLR 2014.
- Goodfellow, Shlens, and Szegedy, "Explaining and Harnessing Adversarial Examples", ICLR 2015.
- Wong and Kolter, "Provable defenses against adversarial examples via the convex outer
  adversarial polytope", ICML 2018.
-/

@[expose] public section
