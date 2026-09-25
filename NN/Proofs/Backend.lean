/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Backend.Attention
public import NN.Proofs.Backend.Grouping
public import NN.Proofs.Backend.NumericalPolicy

/-!
# Backend planning and contract preservation

Successful plans respect the requested operations, registry, availability, and policy.
Coalescing preserves node order, multiplicity, and complete capsules, so group acceptance
applies to every original node. These are proofs about Lean planning and declared contracts;
the numerical certificate consumer retains its selected capsule's reduction-policy guard.
They do not establish correctness of native kernels, numerical algorithms, or the FFI.
-/

@[expose] public section
