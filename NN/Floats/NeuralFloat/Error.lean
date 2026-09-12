/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Error.Addition
public import NN.Floats.NeuralFloat.Error.Bounds
public import NN.Floats.NeuralFloat.Error.Directed
public import NN.Floats.NeuralFloat.Error.DivisionSqrt
public import NN.Floats.NeuralFloat.Error.Exactness
public import NN.Floats.NeuralFloat.Error.Multiplication
public import NN.Floats.NeuralFloat.Error.Relative

/-!
# Rounded-Arithmetic Error Theory

This umbrella collects the generic error results used by TorchLean's `FP32` specialization,
runtime-refinement proofs, verifier margins, and numerical analyses of neural-network operations.
Import a child module when developing the theory itself; downstream users can import this module.
-/

@[expose] public section
