/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Scalar.NF
public import NN.Spec.Core.Context
public import NN.Core.Numeric.Angle.Real

/-!
# Rounded-Real Scalars As Specification Contexts

`NF` is the rounded-real float model, so its adapters need the Mathlib real-analysis stack that
`NN.Floats.NeuralFloat` is built on. They live apart from the binary32 adapters in
`NN.Spec.Core.FloatInstances` for that reason: the runtime and the API only ever instantiate
specifications at `IEEE32Exec`, and pulling real analysis in behind them costs every module
downstream of the tensor API about a thousand extra Mathlib modules to load.
-/

@[expose] public section

namespace TorchLean.Floats

namespace NF

variable {β : NeuralRadix} {fexp : ℤ → ℤ} {rnd : ℝ → ℤ}
variable [NeuralValidExp fexp] [NeuralValidRnd rnd]

/-- Evaluate the principal polar angle in the reals, then round onto the selected grid. -/
noncomputable instance : Atan2 (NF β fexp rnd) where
  atan2 y x := ofReal (Atan2.atan2 y.val x.val)

/-- Natural-number casts round the exact integer onto the `NF` grid, as the `Coe Nat` path does. -/
noncomputable instance instNatCast : NatCast (NF β fexp rnd) where
  natCast n := ofReal (β := β) (fexp := fexp) (rnd := rnd) (n : ℝ)

omit [NeuralValidRnd rnd] in
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
-/
noncomputable instance : Context (NF β fexp rnd) where
  defaultEpsilon := ofReal (β := β) (fexp := fexp) (rnd := rnd) 1e-6
  pow a b :=
    (checkedRealPow (β := β) (fexp := fexp) (rnd := rnd) a b).getD
      (ofReal (β := β) (fexp := fexp) (rnd := rnd) 0)
  decidableGT := Classical.decRel _

end NF

end TorchLean.Floats
