/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Context
public import NN.Proofs.Autograd.Tape.Nodes.Elementwise
public import NN.Proofs.Autograd.Tape.Nodes.Arithmetic
public import NN.Proofs.Autograd.Tape.Nodes.Matrix
public import NN.Proofs.Autograd.Tape.Nodes.Softmax
public import NN.Proofs.Autograd.Tape.Nodes.Reductions
public import NN.Proofs.Autograd.Tape.Nodes.Losses
public import NN.Proofs.Autograd.Tape.Nodes.Piecewise
public import NN.Proofs.Autograd.Tape.Nodes.GraphComposition

/-!
# Tape Nodes

Public entrypoint for the analytic tape-node proof library.

The modules below provide `NodeFDerivCorrect` and `NodeFDerivCorrectAt` facts for the primitive
nodes used by graph-level reverse-mode autograd proofs:

- `Context`: vectorized contexts, projections, injections, and generic node construction;
- `Elementwise`: pointwise scalar functions and activations;
- `Arithmetic`: affine maps, pointwise arithmetic, and fixed-mask stochastic nodes;
- `Matrix`: matmul, transpose, and row/column matrix adapters;
- `Softmax`: last-axis softmax and log-softmax;
- `Reductions`: sums, broadcasts, reductions, concatenation, and shape adapters;
- `Losses`: training losses;
- `Piecewise`: branch-selected min/max nodes;
- `GraphComposition`: the `DGraph` layer for composing node-local proofs into model-level VJP
  theorems.
-/

@[expose] public section
