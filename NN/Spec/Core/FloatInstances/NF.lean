/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.Flocq.Theory.Scalar.NF
public import NN.Spec.Core.Context
public import NN.Core.Numeric.Angle.Real

/-!
# Rounded-Real Scalars As Specification Contexts

FloatLib's `NF` models a real value rounded onto a selected radix/exponent grid. This adapter
supplies TorchLean's specification dictionary for that same carrier; it is noncomputable.
Executable configured binary tensors use `NN.Spec.Core.FloatInstances` instead.
-/

@[expose] public section

namespace FloatLib.Floats.Formats.Flocq

namespace NF

variable {β : FloatLib.Numerics.Radix} {fexp : ℤ → ℤ} {rnd : ℝ → ℤ}
variable [ValidExp fexp] [ValidRnd rnd]

/-- Evaluate the principal polar angle in the reals, then round onto the selected grid. -/
noncomputable instance : Atan2 (NF β fexp rnd) where
  atan2 y x := ofReal (Atan2.atan2 y.val x.val)

omit [ValidRnd rnd] in
/-- Unfold the natural-number cast into the rounded real it denotes. -/
@[simp] theorem natCast_eq_ofReal (n : Nat) :
    ((n : Nat) : NF β fexp rnd) = ofReal (β := β) (fexp := fexp) (rnd := rnd) (n : ℝ) :=
  rfl

/--
Use rounded-real `NF` arithmetic as a TorchLean specification scalar.

The general scalar interface requires a total `α ^ α`. Its adapter uses `NF.checkedRealPow`, which
handles arbitrary exponents on positive bases, integer exponents on negative bases, and positive
exponents at zero. The adapter selects its rounded-zero fallback only when `checkedRealPow` rejects
the domain, such as a negative base with a noninteger exponent or zero with a negative exponent;
an accepted computation can independently round to zero. Direct numerical code should inspect the
checked result, or use the unambiguous `NF.powNat`, rather than relying on that compatibility
fallback.

Rational casts round the exact real fraction once. The default safeguard rounds `1e-6` onto the
chosen grid and can be zero: a general exponent function and rounding rule do not supply the
smallest-positive-value contract of a configured binary format. This dictionary supplies no
`LawfulContext` or positive-tolerance theorem.
-/
noncomputable instance : Context (NF β fexp rnd) where
  ratCast value := ofReal (β := β) (fexp := fexp) (rnd := rnd) (value : ℝ)
  defaultEpsilon := ofReal (β := β) (fexp := fexp) (rnd := rnd) 1e-6
  pow a b :=
    (checkedRealPow (β := β) (fexp := fexp) (rnd := rnd) a b).getD
      (ofReal (β := β) (fexp := fexp) (rnd := rnd) 0)
  decidableGT := Classical.decRel _

end NF

end FloatLib.Floats.Formats.Flocq
