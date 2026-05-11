/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.ODE.Ast
public import NN.Verification.ODE.Parse
public import NN.Verification.ODE.Verify

/-!
# ODE Verification

Public umbrella import for TorchLean's ODE corridor verification stack:

- a small ODE expression AST,
- parsers and JSON loaders for certificate artifacts, and
- the executable checker that replays corridor bounds inside Lean.

Keeping this index module makes `NN.Verification` mirror the other subfolders (`PINN`,
`Robustness`, etc.): users can import `NN.Verification.ODE` without remembering the per-file split.
-/

@[expose] public section

