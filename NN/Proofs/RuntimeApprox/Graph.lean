/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.BackwardApprox
public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox
public import NN.Proofs.RuntimeApprox.Graph.LinkAutogradAlgebra

/-!
# Runtime Approximation Graphs

Backend-independent composition theorems for approximation bounds over typed SSA/tape graphs.

The leaf modules define forward graph composition, reverse-mode/backward graph composition, and the
bridge from proof-level graphs to the executable autograd-algebra `GraphData` representation.
Concrete numeric backends instantiate these graph theorems by supplying local per-op approximation
lemmas.
-/

@[expose] public section

