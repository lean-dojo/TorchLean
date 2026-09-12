/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Analysis.Neighbors
public import NN.Floats.NeuralFloat.Analysis.SterbenzFLT

/-!
# Floating-Point Analysis

This layer studies spacing rather than defining formats or rounding modes.  It provides ULP
functions, adjacent representable values, and exact-subtraction results used in error-free
transformations and reduction proofs.

## References

- P. H. Sterbenz, *Floating-Point Computation*, Prentice-Hall, 1974.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, second edition, SIAM, 2002.
-/

@[expose] public section
