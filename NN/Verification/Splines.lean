/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Splines.PiecewisePolyCert

/-!
# Spline Verification

Public import for exact-rational piecewise-polynomial artifact checks and optional binary32 replay.
The checker evaluates local-coordinate polynomials by Horner's rule; it does not establish a
general approximation-error bound for the fitted function.
-/

@[expose] public section
