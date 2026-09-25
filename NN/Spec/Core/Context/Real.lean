/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.Numeric
public import NN.Core.Numeric.Angle.Real
public import NN.Spec.Core.Context

/-!
# The real scalar dictionary

`Spec.SpecScalar` is `ℝ`, so every "paper theorem" ultimately runs through the instances here. They
are kept out of `NN.Spec.Core.Context` because `Context ℝ` needs `MathFunctions ℝ`, and that drags
in the whole real-analysis hierarchy; modules working at `Float` or at a general `[Context α]`
should not pay for it.
-/

@[expose] public section

/-- Full `Context` instance for `ℝ` (proof backend, noncomputable). -/
noncomputable instance : Context ℝ :=
  { defaultEpsilon := 1e-6, decidableGT := Classical.decRel _, ratCast := Rat.cast }

/-- The real `Context` is built from Mathlib's field structure, so every law holds by `rfl`. -/
instance : LawfulContext ℝ where
  add_eq _ _ := rfl
  mul_eq _ _ := rfl
  sub_eq _ _ := rfl
  div_eq _ _ := rfl
  neg_eq _ := rfl
  zero_eq := rfl
  one_eq := rfl
  lt_iff _ _ := Iff.rfl
  le_iff _ _ := Iff.rfl
  max_eq _ _ := rfl
  min_eq _ _ := rfl
  beq_iff _ _ := beq_iff_eq
  natCast_eq _ := rfl
  ratCast_eq _ := rfl
  abs_eq _ := rfl
  defaultEpsilon_pos := by
    show (0 : ℝ) < 1e-6
    norm_num
