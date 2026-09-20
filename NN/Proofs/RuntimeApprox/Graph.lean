/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.BackwardApprox
public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox
public import NN.Proofs.RuntimeApprox.Graph.LinkAutogradAlgebra
public import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Certificate

/-!
# Runtime Approximation Graphs

Backend-independent composition theorems for approximation bounds over typed SSA/tape graphs.

The leaf modules define forward graph composition, reverse-mode/backward graph composition, and the
bridge from proof-level graphs to the executable autograd-algebra `GraphData` representation.
Concrete numeric backends instantiate these graph theorems by supplying local per-op approximation
lemmas. Architectures are compositions of those nodes; there is no MLP, convolution, or attention
case in the graph induction. `NumericalCertificate` complements the analytic error bounds with an
operation-keyed IEEE-754 range registry and the backend audit for the selected kernel capsules.
Its coverage pass names every unsupported primitive before certificate propagation begins.
-/

@[expose] public section
