/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.Flocq.Theory.Scalar.NF
public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Numeric.InverseTrig.Real

/-!
# Rounded-Real Scalars As Specification Contexts

FloatLib's `NF` models a real value rounded onto a selected radix/exponent grid. Its noncomputable
`Context` and `InverseTrig` instances make that carrier a TorchLean specification scalar, and
executable configured binary tensors use `TorchLean.Tensor.ConfiguredBinary`.

The grid is Flocq's (Boldo and Melquiond, 2011, "Flocq: A Unified Library for Proving
Floating-Point Algorithms in Coq", ARITH 2011, pp. 243-252, DOI 10.1109/ARITH.2011.40): a format
is the set of reals that an exponent function `fexp` allows. We wrote the instances below by hand
in that model, so each operation takes the real result and rounds it once with `NF.ofReal`. A
real zero has no sign, which makes `atan2 0 (-1)` the rounded `π`, where C's `atan2 (-0, -1)` is
`-π`.
-/

@[expose] public section

namespace TorchLean.NF

section

open FloatLib.Floats.Formats.Flocq (NF ValidRnd ValidExp)
open FloatLib.Floats.Formats.Flocq.NF (ofReal)

variable {β : FloatLib.Numerics.Radix} {fexp : ℤ → ℤ} {rnd : ℝ → ℤ}
variable [ValidExp fexp] [ValidRnd rnd]

/-- Evaluate the principal polar angle and the inverse trigonometric functions in the reals, then
round onto the selected grid. -/
noncomputable instance : InverseTrig (NF β fexp rnd) where
  atan2 y x := ofReal (InverseTrig.atan2 y.val x.val)
  atan x := ofReal (InverseTrig.atan x.val)
  asin x := ofReal (InverseTrig.asin x.val)
  acos x := ofReal (InverseTrig.acos x.val)

omit [ValidRnd rnd] in
/-- Unfold the natural-number cast into the rounded real it denotes. -/
@[simp] theorem natCast_eq_ofReal (n : Nat) :
    ((n : Nat) : NF β fexp rnd) = ofReal (β := β) (fexp := fexp) (rnd := rnd) (n : ℝ) :=
  rfl

/--
Use rounded-real `NF` arithmetic as a TorchLean specification scalar.

The scalar interface requires a total `α ^ α`. This instance's `pow` uses `NF.checkedRealPow`,
which handles arbitrary exponents on positive bases, integer exponents on negative bases and
positive exponents at zero. A domain that `checkedRealPow` rejects, such as a negative base with a
noninteger exponent or zero with a negative exponent, gives the rounded zero. An accepted power can
round to zero as well, so direct numerical code should inspect the checked result or use the
unambiguous `NF.powNat`.

Rational casts and division by a count (`divNat`) round the exact real fraction once, without
rounding the count first. The default safeguard rounds `1e-6` onto the chosen grid and can be
zero, since a general exponent function and rounding rule do not supply the smallest-positive-value
contract of a configured binary format. This dictionary supplies no `LawfulContext` or
positive-tolerance theorem.
-/
noncomputable instance : Context (NF β fexp rnd) where
  ratCast value := ofReal (β := β) (fexp := fexp) (rnd := rnd) (value : ℝ)
  divNat x n := ofReal (β := β) (fexp := fexp) (rnd := rnd) (x.val / n)
  defaultEpsilon := ofReal (β := β) (fexp := fexp) (rnd := rnd) 1e-6
  pow a b :=
    (NF.checkedRealPow (β := β) (fexp := fexp) (rnd := rnd) a b).getD
      (ofReal (β := β) (fexp := fexp) (rnd := rnd) 0)
  decidableGT := Classical.decRel _

omit [ValidRnd rnd] in
/-- A rounded real is never a NaN, so the IEEE maximum `Context.maximum` is `max`. -/
theorem maximum_eq_max (x y : NF β fexp rnd) : Context.maximum x y = max x y :=
  Context.maximum_eq_max (fun _ => by simp [Context.isNaN]) (fun _ _ _ => rfl)
    (fun _ _ => Iff.rfl) x y

omit [ValidRnd rnd] in
/-- A rounded real is never a NaN, so the IEEE minimum `Context.minimum` is `min`. -/
theorem minimum_eq_min (x y : NF β fexp rnd) : Context.minimum x y = min x y :=
  Context.minimum_eq_min (fun _ => by simp [Context.isNaN]) (fun _ _ _ => rfl)
    (fun _ _ => Iff.rfl) x y

end

end TorchLean.NF
