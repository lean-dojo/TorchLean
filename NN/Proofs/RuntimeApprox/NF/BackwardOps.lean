/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.BackwardOps.Backend
public import NN.Proofs.RuntimeApprox.NF.BackwardOps.Linalg
public import NN.Proofs.RuntimeApprox.NF.BackwardOps.Main
public import NN.Proofs.RuntimeApprox.NF.BackwardOps.Sparse

/-!
# BackwardOps

NF reverse-mode runtime-to-spec approximation lemmas.

The public theorem is `NFBackend.backprop_approx`: a reverse graph built from sound local NF nodes
has a backpropagated rounded context within explicit per-entry error bounds of the supplied spec
reverse pass. This is approximation, not a derivative-correctness proof. The local node
families cover sparse VJP contexts, context accumulation, primitive operations, and linear algebra.
-/
