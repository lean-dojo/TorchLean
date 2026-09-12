/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert
public import NN.Verification.Geometry3D
public import NN.Verification.Monotonicity.Json
public import NN.Verification.ODE
public import NN.Verification.PINN
public import NN.Verification.Robustness
public import NN.Verification.Splines
public import NN.Verification.Builtin.Lowering
public import NN.Verification.Builtin.ExecutableLowering
public import NN.Verification.Builtin.Proved
public import NN.Verification.Util.Json
public import NN.Verification.Util.TextCursor
public import NN.Verification.VNNComp

/-!
# Verification

Import this file for TorchLean’s verification infrastructure: JSON utilities, certificate formats,
ODE/PINN-style checkers, proof-backed certificate soundness, and proof-backed TorchLean-to-IR
forward lowering.

Their expression parsers share the fuel-bounded byte cursor and decimal scanners from
`NN.Verification.Util.TextCursor`; each verifier keeps only its grammar and AST-specific logic.

The lowering API is imported through `NN.Verification.Builtin.Proved`, which contains both the
lowering functions and their correctness theorems.

Runnable CLIs stay out of this umbrella. If you want a command-line tool, import
the registry explicitly (for example `NN.Verification.CLI`).

The underlying CROWN/LiRPA soundness development enters here too. Executable examples can parse
JSON artifacts and recompute bounds inside Lean; theorem-level credit comes from those imported
soundness modules, where locally valid certificates are connected to Lean graph semantics.
-/

@[expose] public section
