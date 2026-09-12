/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.Numeric.Angle
public import Mathlib.Analysis.SpecialFunctions.Complex.Arg

/-!
# Exact real polar angle

The real instance uses Mathlib's principal complex argument in `(-π, π]`, with angle zero at the
origin. This module keeps complex analysis out of executable scalar interfaces.
-/

@[expose] public section

noncomputable instance : Atan2 ℝ := ⟨fun y x => _root_.Complex.arg ⟨x, y⟩⟩
