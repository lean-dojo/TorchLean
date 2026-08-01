/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Core
public import NN.Proofs.Autograd.Runtime.Link.Invariants
public import NN.Proofs.Autograd.Runtime.Link.Accumulation
public import NN.Proofs.Autograd.Runtime.Link.BackwardGraph
public import NN.Proofs.Autograd.Runtime.Link.BackwardGraphData
public import NN.Proofs.Autograd.Runtime.Link.FDeriv

/-!
Runtime-to-tape autograd link proofs.

These modules connect executable runtime graph bookkeeping to the proof-oriented autograd tape
semantics used by correctness theorems.
-/
