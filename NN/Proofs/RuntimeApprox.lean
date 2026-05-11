/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Core
public import NN.Proofs.RuntimeApprox.FP32
public import NN.Proofs.RuntimeApprox.Graph
public import NN.Proofs.RuntimeApprox.NF
public import NN.Proofs.RuntimeApprox.Rounding
public import NN.Proofs.RuntimeApprox.Scale

/-!
# Runtime Approximation Proofs

Umbrella import for TorchLean's executable-runtime-to-real-spec approximation theorems.

The stack is intentionally layered:
- `Core`: tolerance objects and tensor/context approximation predicates;
- `Rounding`: scalar `neural_round` error lemmas;
- `Graph`: forward and reverse graph composition theorems;
- `NF`: proof-relevant rounded tensor/operator backend;
- `FP32`: convenient FP32-specialized layer/MLP/CROWN statements;
- `Scale`: optional magnitude propagation for abs/rel tolerance reporting.

Leaf modules stay available for developers working on one operator family, but public entrypoints
and CI should import this umbrella rather than listing every runtime-approximation file by hand.
-/

@[expose] public section
