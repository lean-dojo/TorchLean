/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.Universal.IEEE32ExecCore
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationFP32
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationIEEE32Exec
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationIEEE32ExecMlp2
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationND
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationRate

/-!
# Universal-approximation proofs

This entrypoint collects the constructive ReLU approximation layer. The chapter is organized around
one mathematical spine:

- one-dimensional constructive ReLU approximation by hinge sums;
- quantitative width/rate refinements;
- finite-precision lifts through `FP32` and executable `IEEE32Exec` semantics; and
- an `n`-dimensional Stone-Weierstrass bridge through coordinate polynomials.

The point is not only that a ReLU MLP exists. We also record how the exact real-valued construction
connects to TorchLean's spec-level MLP and to executable binary32 arithmetic.

References:
- Cybenko, "Approximation by superpositions of a sigmoidal function", 1989.
- Leshno, Lin, Pinkus, and Schocken, "Multilayer feedforward networks with a nonpolynomial
  activation function can approximate any function", 1993.
- Pinkus, *Approximation Theory of the MLP Model in Neural Networks*, 1999.
- Yarotsky, "Error bounds for approximations with deep ReLU networks", 2017.
-/

@[expose] public section
