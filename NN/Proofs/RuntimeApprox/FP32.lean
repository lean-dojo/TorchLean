/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.FP32.CROWN
public import NN.Proofs.RuntimeApprox.FP32.Layers
public import NN.Proofs.RuntimeApprox.FP32.MLP

/-!
# FP32 Runtime Approximation

Specialization of the backend-generic approximation stack to TorchLean's FP32 rounding model.

These modules package layer, MLP, and CROWN/IBP margin lemmas so downstream examples can state
“real spec result plus explicit FP32 error budget” without reassembling the numeric backend each
time.
-/

@[expose] public section

