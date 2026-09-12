/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Floats.IEEEExec.Exec32.Instances

/-!
# Floating-Point Adapters For Tensor Specifications

The numerical types in `NN.Floats` are independent of TorchLean's tensor and model interfaces.
This module supplies the one-way adapters that let those types instantiate the broader `Context`
expected by scalar-polymorphic specifications.

Only the executable binary32 adapters live here. The rounded-real `NF` adapters are in
`NN.Spec.Core.FloatInstances.NF`, which is where the real-analysis dependency stays.
-/

@[expose] public section

namespace TorchLean.Floats

namespace IEEE754.IEEE32Exec

/-- Natural-number casts round the exact integer to binary32, as the `Coe Nat` path does. -/
instance instNatCast : NatCast IEEE32Exec where
  natCast n := roundDyadicToIEEE32 { sign := false, mant := n, exp := 0 }

/-- Unfold the natural-number cast into the binary32 rounding it denotes. -/
@[simp] theorem natCast_eq_roundDyadic (n : Nat) :
    ((n : Nat) : IEEE32Exec) = roundDyadicToIEEE32 { sign := false, mant := n, exp := 0 } :=
  rfl

/-- Use executable binary32 arithmetic as a TorchLean specification scalar. -/
instance : Context IEEE32Exec where
  defaultEpsilon := ofFloat 1e-6
  decidableGT := fun x y => inferInstanceAs (Decidable (x > y))
  ratCast value := roundRatToIEEE32 (value.num < 0) value.num.natAbs value.den

end IEEE754.IEEE32Exec
end TorchLean.Floats
