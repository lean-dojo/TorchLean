/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Splines.PiecewisePolyCert

/-!
# Spline Verification

Public umbrella import for spline / piecewise-polynomial certificate checking.

This namespace is intentionally narrow: spline certificates are treated as untrusted artifacts that
are checked by recomputation inside Lean against the same spec-layer evaluation used elsewhere in
TorchLean verification.
-/

@[expose] public section

