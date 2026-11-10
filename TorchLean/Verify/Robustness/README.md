# `TorchLean/Verify/Robustness`

This directory holds the theorems that say what robustness certificates mean over `ℝ`. A Lipschitz
bound turns an input perturbation into an output perturbation, and a positive logit margin keeps the
`argmax` class under small enough perturbations. The robustness specifications are in
`TorchLean/NN`, and the dataset workflows are in [the robustness
examples](../../../TorchLeanExamples/Verification/README.md).

- `LipschitzCertified.lean`: the Lipschitz and margin argument, with the bound `2 * (L * ε) < m`.
- `MLP.lean`: Frobenius-norm L2 Lipschitz bounds of linear layers and 2-layer ReLU MLPs.

Import the module of the theorem you use, for example the Lipschitz margin argument:

```lean
import TorchLean.Verify.Robustness.LipschitzCertified
```

## References

- Hein and Andriushchenko, "Formal guarantees on the robustness of a classifier against
  adversarial manipulation," NeurIPS 2017.
- Wong and Kolter, "Provable defenses against adversarial examples via the convex outer
  adversarial polytope," ICML 2018.
