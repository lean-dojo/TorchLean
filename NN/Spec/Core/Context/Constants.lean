/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context

/-!
# Scalar Constants

Model settings such as normalization epsilon are stored as rational numbers. This lets us use the
same configuration when running a model with `Float` or `Float32` and when reasoning about it
over the reals.

`Context.ofRat` uses the backend's rational cast. Over the reals this gives the exact fraction.
`Float` and `Float32` round the fraction once, without first rounding its numerator and denominator.
Values outside the backend's exponent range can still round to zero or infinity.
-/

@[expose] public section

namespace Context

/--
Convert a rational constant using the cast supplied by `Context`.

Keeping this conversion in `Context` lets a model choose its constants before the forward program
chooses a scalar type.
-/
def ofRat {α : Type} [Context α] (value : Rat) : α :=
  value

/--
In a lawful context, rational constants agree with the usual field cast.
We can then use Mathlib's rational-cast lemmas for configuration values, such as the positive
epsilon used by normalization layers.

The context is an ordinary dictionary argument here: we are comparing its arithmetic with the
field structure on the same scalar type.
-/
@[simp] theorem ofRat_eq_cast {α : Type}
    [scalarField : Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {context : Context α} [lawful : @LawfulContext α context _ _ _]
    (value : Rat) :
    @ofRat α context value = @Rat.cast α scalarField.toRatCast value := by
  exact @LawfulContext.ratCast_eq α context _ _ _ lawful value

end Context
