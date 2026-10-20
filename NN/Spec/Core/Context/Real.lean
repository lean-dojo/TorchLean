/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.Numeric.Angle.Real
public import NN.Spec.Core.Context
public import NN.Spec.Core.Erf

/-!
# The real scalar dictionary

This module supplies the noncomputable `Context ℝ` instance and its compatibility laws with
Mathlib's real field. Keeping these instances separate lets callers import the general `Context`
interface without selecting the real scalar instance.

The instance also gives the exact meaning of the pointwise operators in `Spec.PointwiseOp`. At a
pole or outside the real domain, such as a zero divisor or the reciprocal square root of a negative
number, PyTorch returns an infinity or a NaN in every format and no real value is faithful to it, so
the operator returns an error instead of Lean's junk value. Elsewhere the operator returns the real
value, also where a binary format would round it to an infinity.
-/

@[expose] public section

namespace Context

/-- The integer nearest to `q` in the direction of zero. -/
noncomputable def truncReal (q : ℝ) : ℤ :=
  if 0 ≤ q then ⌊q⌋ else ⌈q⌉

/--
Exact real meanings of the pointwise operators that depend on the scalar type.

Division requires a nonzero divisor and the reciprocal square root a positive argument. A power
requires either a positive base, a zero base with a nonnegative exponent, or a negative base
with an integral exponent; there it agrees with `Real.rpow`, and `0 ^ 0 = 1` as in PyTorch.
-/
noncomputable def realPointwise : Spec.PointwiseTable ℝ
  | .rsqrt => some <| Spec.PointwiseFn.unary fun x =>
      if 0 < x then .ok (1 / √x)
      else .error "rsqrt: a nonpositive input has no real reciprocal square root"
  | .erf => some <| Spec.PointwiseFn.unary fun x => .ok (Spec.erf x)
  | .div .none => some <| Spec.PointwiseFn.binary fun x y =>
      if y = 0 then .error "div: a zero divisor has no real quotient" else .ok (x / y)
  | .div .trunc => some <| Spec.PointwiseFn.binary fun x y =>
      if y = 0 then .error "div: a zero divisor has no real quotient"
      else .ok (truncReal (x / y) : ℝ)
  | .div .floor => some <| Spec.PointwiseFn.binary fun x y =>
      if y = 0 then .error "div: a zero divisor has no real quotient"
      else .ok (⌊x / y⌋ : ℝ)
  | .pow => some <| Spec.PointwiseFn.binary fun x y =>
      if x = 0 ∧ y < 0 then
        .error "pow: a zero base with a negative exponent has no real power"
      else if x < 0 ∧ Int.fract y ≠ 0 then
        .error "pow: a negative base with a nonintegral exponent has no real power"
      else .ok (x ^ y)
  | .neg | .gelu _ | .clampMin => none

end Context

/-- Full `Context` instance for `ℝ` (proof backend, noncomputable). -/
noncomputable instance : Context ℝ :=
  { defaultEpsilon := 1e-6, decidableGT := Classical.decRel _, ratCast := Rat.cast,
    pointwise := Context.realPointwise }

/-- The real dictionary's pointwise operators are `Context.realPointwise`. -/
@[simp] theorem Context.pointwise_real : Context.pointwise (α := ℝ) = Context.realPointwise := rfl

/-- The real dictionary uses Mathlib's field operations and has strictly positive tolerance. -/
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
