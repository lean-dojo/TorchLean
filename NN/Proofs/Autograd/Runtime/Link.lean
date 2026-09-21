/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Accumulation
public import NN.Proofs.Autograd.Runtime.Link.BackwardDense
public import NN.Proofs.Autograd.Runtime.Link.BackwardDenseGraph
public import NN.Proofs.Autograd.Runtime.Link.BackwardGraph
public import NN.Proofs.Autograd.Runtime.Link.BackwardGraphData
public import NN.Proofs.Autograd.Runtime.Link.BackwardLeaves
public import NN.Proofs.Autograd.Runtime.Link.BackwardSnoc
public import NN.Proofs.Autograd.Runtime.Link.Checked
public import NN.Proofs.Autograd.Runtime.Link.Core
public import NN.Proofs.Autograd.Runtime.Link.FDeriv
public import NN.Proofs.Autograd.Runtime.Link.GraphComposition
public import NN.Proofs.Autograd.Runtime.Link.HigherOrder
public import NN.Proofs.Autograd.Runtime.Link.HigherOrderFDeriv
public import NN.Proofs.Autograd.Runtime.Link.HigherOrderReverse
public import NN.Proofs.Autograd.Runtime.Link.Invariants

/-!
Runtime-to-tape autograd link proofs.

These modules connect executable runtime graph bookkeeping to the proof-oriented autograd tape
semantics used by correctness theorems.

`BackwardGraph` and `FDeriv` are about the proved sweep `Tape.backwardDenseFrom`. `BackwardDense`
and `BackwardDenseGraph` extend the same results to the sweep the eager trainer actually executes,
`Tape.backwardDenseAll`, which skips nodes that never receive a cotangent.
-/
