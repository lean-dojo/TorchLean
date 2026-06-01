/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Core
public import NN.Proofs.Tensor.Basic.Folds
public import NN.Proofs.Tensor.Basic.LinearAlgebra
public import NN.Proofs.Tensor.Basic.Factorizations
public import NN.Proofs.Tensor.Basic.FactorizationsReconstruction
public import NN.Proofs.Tensor.Basic.FactorizationsSolve
public import NN.Proofs.Tensor.Basic.FactorizationsVariational
public import NN.Proofs.Tensor.Basic.FactorizationsDecision
public import NN.Proofs.Tensor.Basic.FactorizationsZTest
public import NN.Proofs.Tensor.Basic.FactorizationsZAsymptotic
public import NN.Proofs.Tensor.Basic.FactorizationsKernels
public import NN.Proofs.Tensor.Basic.FactorizationsOrthonormal
public import NN.Proofs.Tensor.Basic.FactorizationsJacobi
public import NN.Proofs.Tensor.Basic.FactorizationsJacobiDecrease
public import NN.Proofs.Tensor.Basic.FactorizationsJacobiRate
public import NN.Proofs.Tensor.Basic.BoundsNorms
public import NN.Proofs.Tensor.Basic.Algebra

/-!
Basic tensor proof entry point.

The submodules group algebraic, folding, bound, and linear-algebra facts about dependent tensors so
other proof developments can import a coherent tensor toolkit.
-/
