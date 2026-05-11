/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Verification.Robustness.LipschitzCertified
public import NN.MLTheory.Proofs.Verification.Robustness.MlpRobustness

/-!
# Robustness verification proofs

This entrypoint collects proof-level links between robustness specifications and analytic
certificates. The files underneath prove:

- Lipschitz continuity implies adversarial robustness;
- logit margins plus output perturbation bounds preserve the `argmax` classifier; and
- basic MLP/ReLU Lipschitz lemmas used by certified-robustness statements.

The executable verifiers live elsewhere; this chapter supplies the mathematical statements that
make those certificates meaningful.

References:
- Hein and Andriushchenko, "Formal guarantees on the robustness of a classifier against
  adversarial manipulation", NeurIPS 2017.
- Wong and Kolter, "Provable defenses against adversarial examples via the convex outer
  adversarial polytope", ICML 2018.
-/

@[expose] public section
