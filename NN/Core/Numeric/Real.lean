/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import NN.Core.Numeric
public import NN.Core.Numeric.Angle.Real

/-!
# Exact-real instances for the foundational numeric interfaces

`NN.Core.Numeric` is kept free of Mathlib so that the bulk of the library can use `MathFunctions`
without loading the real-analysis hierarchy. This module supplies the `ℝ` instances
that specification code needs, and is the one place that pays for those imports.
-/

@[expose] public section

/-- Exact-real interpretations of the scalar transcendental interface. -/
noncomputable instance : MathFunctions ℝ where
  exp := Real.exp
  tanh := Real.tanh
  cosh := Real.cosh
  sinh := Real.sinh
  sqrt := Real.sqrt
  abs := fun x => |x|
  log := Real.log
  pi := Real.pi
  cos := Real.cos
  sin := Real.sin
