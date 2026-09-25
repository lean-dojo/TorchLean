/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Primitives.Core
public import NN.GraphSpec.DAG.Primitives.LinearAlgebra
public import NN.GraphSpec.DAG.Primitives.Nonlinear
public import NN.GraphSpec.DAG.Primitives.Normalization
public import NN.GraphSpec.DAG.Primitives.Shape
public import NN.GraphSpec.DAG.Term

/-!
# Typed Computation DAGs

This is the public entry point for GraphSpec models with shared values or multi-input operations.
It imports the typed term language together with the standard primitive families:

- constants and elementwise arithmetic;
- matrix, vector, and batched linear algebra;
- reductions and shape transformations;
- normalization and nonlinear activation functions;
- multi-head attention;
- finite sums of same-shaped graph terms.

For a chain of unary layers, `NN.GraphSpec.Chain.Lowering` provides the lighter sequential
notation, and `NN.GraphSpec.Chain.ToDAG.Model` lowers it into this representation. Use the DAG
language directly for residual connections, shared subexpressions, caches, attention blocks, and
other architectures whose dataflow is not a simple chain.
-/
