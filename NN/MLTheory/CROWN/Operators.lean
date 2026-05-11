/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Operators.Activations
public import NN.MLTheory.CROWN.Operators.Arithmetic
public import NN.MLTheory.CROWN.Operators.Batchnorm
public import NN.MLTheory.CROWN.Operators.Gelu
public import NN.MLTheory.CROWN.Operators.Pooling
public import NN.MLTheory.CROWN.Operators.Reduce
public import NN.MLTheory.CROWN.Operators.Slice

/-!
# CROWN Operator Index

This module re-exports the operator-level transfer rules used by the graph-based LiRPA/CROWN engine
(`NN.MLTheory.CROWN.Graph`): activations, arithmetic, batch normalization, pooling, reductions, and
shape/indexing operations.

Trigonometric operators remain opt-in because `tan`/`atan` require an extra scalar-function
interface beyond the project-wide `Context`. Import them explicitly when needed:

`import NN.MLTheory.CROWN.Operators.Trigonometric`
-/

@[expose] public section
