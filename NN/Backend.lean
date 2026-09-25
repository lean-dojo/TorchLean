/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Types
public import NN.Backend.Capsule
public import NN.Backend.Availability
public import NN.Backend.Planner
public import NN.Backend.Audit
public import NN.Backend.Attention
public import NN.Backend.Reference
public import NN.Backend.LibTorch
public import NN.Backend.Registry
public import NN.Backend.IR
public import NN.Backend.Grouping
public import NN.Backend.ContractCheck
public import NN.Backend.Profile
public import NN.Backend.Report

/-!
# Backend Contracts

Contract-carrying backend vocabulary for TorchLean runtimes.

The semantic graph and specs stay in Lean. LibTorch and reference operations enter through named
capsules that record shape, layout, value, and VJP contracts, a reduction-order policy, and an
explicit trust level. The planner consumes these capsules under a `KernelPolicy`, the contract check
confirms that every selected contract rests on evidence the policy accepts, and the eager runtime
binds the selected capsule to a handler with the same operation, provider, and device.
-/

@[expose] public section
