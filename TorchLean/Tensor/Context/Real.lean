/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Numeric.InverseTrig.Real
public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Erf
public import Mathlib.Analysis.SpecialFunctions.Arcosh
public import Mathlib.Analysis.SpecialFunctions.Artanh
public import Mathlib.Analysis.SpecialFunctions.Arsinh
public import Mathlib.Analysis.SpecialFunctions.Log.Base
public import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Core

/-!
# The real scalar dictionary

The noncomputable `Context ℝ` instance and its compatibility laws with Mathlib's real field. These
instances live apart from the general `Context` interface, so a caller can import that interface
without selecting the real scalar instance.

Alongside them are the real integral roundings, `Rounding ℝ` (`⌊x⌋`, `⌈x⌉`,
`FloatLib.Floats.Formats.Flocq.truncRound`, the nearest integer with ties to even, and C's `fmod`
as `TorchLean.Context.fmodReal`), the real error function, `Erf ℝ` (`TorchLean.erf`), and the real
cube root, `Cbrt ℝ` (`TorchLean.Context.cbrtReal`).
-/

@[expose] public section

namespace TorchLean.Context

/-- C's `fmod` over the reals: `x - y * trunc (x / y)`. At `y = 0` it's `x`, because Lean's `x / 0`
is `0`, where C99 F.9.7.1 (WG14 N1256) returns a NaN. -/
noncomputable def fmodReal (x y : ℝ) : ℝ := x - y * FloatLib.Floats.Formats.Flocq.truncRound (x / y)

/-- The real cube root: `x ^ (1 / 3)` from zero and `-(-x) ^ (1 / 3)` below zero. -/
noncomputable def cbrtReal (x : ℝ) : ℝ := if x < 0 then -(-x) ^ (1 / 3 : ℝ) else x ^ (1 / 3 : ℝ)

end TorchLean.Context

namespace TorchLean

/-- The real integral roundings: `⌊x⌋`, `⌈x⌉`, truncation, the nearest integer with ties to even,
and `x - y * trunc (x / y)`. -/
noncomputable instance : Rounding ℝ where
  floor x := ⌊x⌋
  ceil x := ⌈x⌉
  trunc x := FloatLib.Floats.Formats.Flocq.truncRound x
  round x := FloatLib.Floats.Formats.Flocq.nearestEven x
  fmod := Context.fmodReal

/-- The real error function. -/
noncomputable instance : Erf ℝ := ⟨erf⟩

/-- The real cube root. -/
noncomputable instance : Cbrt ℝ := ⟨Context.cbrtReal⟩

/-- Full `Context` instance for `ℝ` (proof backend, noncomputable). -/
noncomputable instance : Context ℝ :=
  { defaultEpsilon := 1e-6, decidableGT := Classical.decRel _, ratCast := Rat.cast,
    tan := Real.tan, exp2 := fun x => (2 : ℝ) ^ x, log2 := Real.logb 2, log10 := Real.logb 10,
    asinh := Real.arsinh, acosh := Real.arcosh, atanh := Real.artanh }

/-- The real dictionary divides by a count as the field divides by its cast. -/
@[simp] theorem Context.divNat_real (x : ℝ) (n : Nat) : Context.divNat x n = x / n := rfl

/-- The real dictionary is an unsupported quotient carrier, so it uses the ordinary operations. -/
theorem Numeric.QuotientArithmetic.supported_real :
    TorchLean.Numeric.QuotientArithmetic.supported (α := ℝ) = false := rfl

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
  divNat_eq _ _ := rfl
  ratCast_eq _ := rfl
  abs_eq _ := rfl
  defaultEpsilon_pos := by
    show (0 : ℝ) < 1e-6
    norm_num

end TorchLean
