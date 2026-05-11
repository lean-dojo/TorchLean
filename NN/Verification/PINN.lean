/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.Architecture
public import NN.Verification.PINN.CLI
public import NN.Verification.PINN.Certificate
public import NN.Verification.PINN.Core
public import NN.Verification.PINN.DatasetCheck
public import NN.Verification.PINN.PdeAst
public import NN.Verification.PINN.PdeParse
public import NN.Verification.PINN.ResidualAffine

/-!
# PINN Verification

Reusable TorchLean support for physics-informed neural network verification.

The files under this namespace are library code: graph construction, PDE expression parsing,
residual-bound helpers, certificate checking, and dataset-backed interval containment checks.
Example directories should provide assets and Python producers only.
-/

@[expose] public section
