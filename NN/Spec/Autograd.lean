/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.AutogradSpec
public import NN.Spec.Autograd.Ops

/-!
# Spec autograd

Umbrella import for spec-level reverse-mode contracts.

`OpSpec` records pure forward and backward meanings for operations. Runtime autograd can cache or
schedule these operations differently, but its behavior is expected to refine these contracts.
-/

@[expose] public section

