/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Verification.TorchLean.TorchLeanCrownOps
public import NN.Examples.Verification.TorchLean.TorchLeanIBP
public import NN.Examples.Verification.TorchLean.TorchLeanMlpWorkflow
public import NN.Examples.Verification.TorchLean.TorchLeanTransformerIBP

/-!
# TorchLean Verification Workflows

End-to-end examples that build TorchLean models, lower them into verification artifacts, and run
IBP/CROWN-style checks.
-/

@[expose] public section
