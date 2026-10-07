/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Core
public import NN.Proofs.RuntimeApprox.FP32
public import NN.Proofs.RuntimeApprox.FP32.CROWN
public import NN.Proofs.RuntimeApprox.Graph
public import NN.Proofs.RuntimeApprox.IEEE32.Expressions
public import NN.Proofs.RuntimeApprox.IEEE32.MinMaxTotal
public import NN.Proofs.RuntimeApprox.NF
public import NN.Proofs.RuntimeApprox.Optimizer
public import NN.Proofs.RuntimeApprox.Reductions.IEEE32
public import NN.Proofs.RuntimeApprox.Rounding
public import NN.Proofs.RuntimeApprox.Scale

/-!
# Runtime Approximation Proofs

Umbrella import for TorchLean's runtime approximation contracts and composition theorems.

The runtime-approximation library is intentionally layered:
- `Core`: tolerance objects and scalar/tensor approximation predicates;
- `Rounding`: scalar FloatLib rounding-error lemmas;
- `Graph`: forward and reverse graph composition theorems;
- `NF`: proof-relevant rounded tensor/operator backend;
- `FP32`: convenient FP32-specialized layer/MLP/CROWN statements;
- `Optimizer`: one finite-run contract for numerical optimizer updates;
- `IEEE32`: finite binary32 refinements of arithmetic, expression trees, and min/max contracts;
- `Reductions`: error bounds for rounded-addition trees and finite binary32 schedules;
- `Scale`: optional magnitude propagation for abs/rel tolerance reporting.

Import this umbrella for the full proof bundle, or a leaf module for one operator family.
Native-runtime agreement still requires the provider assumptions stated by the relevant contract.
-/

@[expose] public section
