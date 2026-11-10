/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Elementwise
public import Mathlib.Analysis.SpecialFunctions.Arcosh
public import Mathlib.Analysis.SpecialFunctions.Arsinh
public import Mathlib.Analysis.SpecialFunctions.Complex.Arg
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.ArctanDeriv
public import Mathlib.Basic.Real.Sign

/-!
# Real meanings of the elementwise operators

Over `ℝ` the scalar functions of `TorchLean.Tensor.Elementwise` are the mathematical
functions they round in a binary format: `floor` is `⌊x⌋`, `round` the nearest integer with ties to
even, `tan` is `Real.tan`, `log2` is `Real.logb 2`, and so on. No real number is a NaN or an
infinity, so the NaN branches of `maximum`, `clamp` and `nanToNum` disappear and `maximum` is `max`.
Both branches of ATen's `lerp` are `a + w (b - a)`. The arctangent is `Real.arctan`, which is also
the polar angle of `(1, x)`, and the error function is `erf`.

The pullbacks `asinhBackward`, `acoshBackward`, `tanBackward` and `erfBackward` multiply the
upstream gradient by the derivative at the input: `(√(1 + x²))⁻¹` for `arsinh`, `(√(x² - 1))⁻¹`
for `arcosh` above one, `1 / cos² x` for `tan` where `cos x ≠ 0`, and `2 / √π · exp (-x²)` for
`erf`.
-/

@[expose] public section

namespace TorchLean.Context

variable (x y z : ℝ)

/-- Over `ℝ`, `Rounding.floor x` is `⌊x⌋`. -/
@[simp] theorem floor_real : Rounding.floor x = ⌊x⌋ := rfl
/-- Over `ℝ`, `Rounding.ceil x` is `⌈x⌉`. -/
@[simp] theorem ceil_real : Rounding.ceil x = ⌈x⌉ := rfl
/-- Over `ℝ`, `Rounding.trunc x` is Flocq's `truncRound x`. -/
@[simp] theorem trunc_real : Rounding.trunc x = FloatLib.Floats.Formats.Flocq.truncRound x := rfl
/-- Over `ℝ`, `Rounding.round x` is Flocq's `nearestEven x`. -/
@[simp] theorem round_real : Rounding.round x = FloatLib.Floats.Formats.Flocq.nearestEven x := rfl
/-- Over `ℝ`, `Context.tan x` is `Real.tan x`. -/
@[simp] theorem tan_real : Context.tan x = Real.tan x := rfl
/-- Over `ℝ`, `Context.exp2 x` is `2 ^ x`. -/
@[simp] theorem exp2_real : Context.exp2 x = (2 : ℝ) ^ x := rfl
/-- Over `ℝ`, `Context.log2 x` is `Real.logb 2 x`. -/
@[simp] theorem log2_real : Context.log2 x = Real.logb 2 x := rfl
/-- Over `ℝ`, `Context.log10 x` is `Real.logb 10 x`. -/
@[simp] theorem log10_real : Context.log10 x = Real.logb 10 x := rfl
/-- Over `ℝ`, `Context.asinh x` is `Real.arsinh x`. -/
@[simp] theorem asinh_real : Context.asinh x = Real.arsinh x := rfl
/-- Over `ℝ`, `Context.acosh x` is `Real.arcosh x`. -/
@[simp] theorem acosh_real : Context.acosh x = Real.arcosh x := rfl
/-- Over `ℝ`, `Context.atanh x` is `Real.artanh x`. -/
@[simp] theorem atanh_real : Context.atanh x = Real.artanh x := rfl
/-- Over `ℝ`, `Context.hypot x y` is `√(x * x + y * y)`. -/
@[simp] theorem hypot_real : Context.hypot x y = √(x * x + y * y) := rfl
/-- Over `ℝ`, `Rounding.fmod x y` is `x - y * truncRound (x / y)`, with Flocq's `truncRound`. -/
@[simp] theorem fmod_real :
    Rounding.fmod x y = x - y * FloatLib.Floats.Formats.Flocq.truncRound (x / y) := rfl

/-- No real number is a NaN. -/
@[simp] theorem isNaN_real : Context.isNaN x = false := by
  simp [Context.isNaN]

/-- Over `ℝ` the tie rule of an extremum returns the chosen value. -/
@[simp] theorem extremaTie_real : Context.extremaTie x y z = z := rfl

/-- No real number is an infinity. -/
@[simp] theorem isInf_real : Context.isInf x = false := by
  simp [Context.isInf]

/-- Every real number is finite. -/
@[simp] theorem isFinite_real : Context.isFinite x = true := by
  simp [Context.isFinite]

/-- The real `sign` is `Real.sign`. -/
theorem sign_real : Context.sign x = Real.sign x := by
  unfold Context.sign Real.sign
  rcases lt_trichotomy x 0 with h | h | h
  · simp [h, not_lt.mpr h.le]
  · simp [h, Context.stopGradient]
    rfl
  · simp [h, not_lt.mpr h.le]

/-- The real `maximum` is `max`. -/
theorem maximum_real : Context.maximum x y = max x y :=
  maximum_eq_max isNaN_real extremaTie_real (fun _ _ => Iff.rfl) x y

/-- The real `minimum` is `min`. -/
theorem minimum_real : Context.minimum x y = min x y :=
  minimum_eq_min isNaN_real extremaTie_real (fun _ _ => Iff.rfl) x y

/-- The real `fmax` is `max`. -/
theorem fmax_real : Context.fmax x y = max x y := by
  rcases lt_trichotomy x y with h | rfl | h
  · simp [Context.fmax, Context.fmaxTakesFirst, Context.gtBool, h, max_eq_right h.le]
  · simp [Context.fmax]
  · simp [Context.fmax, Context.fmaxTakesFirst, Context.gtBool, h, lt_asymm h,
      max_eq_left h.le]

/-- The real `fmin` is `min`. -/
theorem fmin_real : Context.fmin x y = min x y := by
  rcases lt_trichotomy x y with h | rfl | h
  · simp [Context.fmin, Context.fminTakesFirst, Context.gtBool, h, lt_asymm h,
      min_eq_left h.le]
  · simp [Context.fmin]
  · simp [Context.fmin, Context.fminTakesFirst, Context.gtBool, h, min_eq_right h.le]

/-- The real `clamp` is `min (max x lo) hi`. -/
theorem clamp_real : Context.clamp x y z = min (max x y) z := by
  simp [Context.clamp, maximum_real, minimum_real]

/-- Both branches of the real `lerp` are `a + w (b - a)`. -/
theorem lerp_real : Context.lerp x y z = x + z * (y - x) := by
  unfold Context.lerp
  split_ifs <;> ring

/-- The real `frac` is `x - trunc x`. -/
theorem frac_real : Context.frac x = x - FloatLib.Floats.Formats.Flocq.truncRound x := rfl

/-- The real arctangent is `Real.arctan`. -/
@[simp] theorem atan_real : InverseTrig.atan x = Real.arctan x := rfl

/-- The polar angle of `(1, x)` is the arctangent of `x`. -/
theorem atan2_one_real : InverseTrig.atan2 x 1 = Real.arctan x := by
  change Complex.arg ⟨1, x⟩ = _
  rw [Complex.arg_of_re_nonneg (by norm_num), Real.arctan_eq_arcsin, Complex.norm_def,
    Complex.normSq_mk]
  congr 3
  ring

/-- `nanToNum` leaves every real number unchanged. -/
@[simp] theorem nanToNum_real (nan posinf neginf : ℝ) :
    Context.nanToNum x nan posinf neginf = x := by
  simp [Context.nanToNum]

end TorchLean.Context

namespace TorchLean.Tensor

/-- The pullback of `asinh` at `x` is `arsinh' x · g`. -/
theorem asinhBackward_real {s : Shape} (x g : Tensor ℝ s) (c : s.Coord) :
    HasDerivAt Real.arsinh (√(1 + x c ^ 2))⁻¹ (x c) ∧
      asinhBackward x g c = (√(1 + x c ^ 2))⁻¹ * g c := by
  refine ⟨Real.hasDerivAt_arsinh (x c), ?_⟩
  simp only [asinhBackward, Tensor.zipWith_apply]
  change g c / √(x c * x c + 1 * 1) = _
  rw [show x c * x c + 1 * 1 = 1 + x c ^ 2 by ring, div_eq_inv_mul]

/-- Above one, the pullback of `acosh` at `x` is `arcosh' x · g`. -/
theorem acoshBackward_real {s : Shape} (x g : Tensor ℝ s) (c : s.Coord) (hx : 1 < x c) :
    HasDerivAt Real.arcosh (√(x c ^ 2 - 1))⁻¹ (x c) ∧
      acoshBackward x g c = (√(x c ^ 2 - 1))⁻¹ * g c := by
  refine ⟨Real.hasDerivAt_arcosh hx, ?_⟩
  simp only [acoshBackward, Tensor.zipWith_apply]
  change g c / (√(x c - 1) * √(x c - (0 - 1))) = _
  rw [← Real.sqrt_mul (by linarith), show (x c - 1) * (x c - (0 - 1)) = x c ^ 2 - 1 by ring,
    div_eq_inv_mul]

/-- Where `cos x ≠ 0`, the pullback of `tan` at `x` is `tan' x · g`. -/
theorem tanBackward_real {s : Shape} (x g : Tensor ℝ s) (c : s.Coord)
    (hx : Real.cos (x c) ≠ 0) :
    HasDerivAt Real.tan (1 / Real.cos (x c) ^ 2) (x c) ∧
      tanBackward x g c = 1 / Real.cos (x c) ^ 2 * g c := by
  refine ⟨Real.hasDerivAt_tan hx, ?_⟩
  simp only [tanBackward, Tensor.zipWith_apply]
  change g c * (1 + Real.tan (x c) * Real.tan (x c)) = _
  rw [show 1 + Real.tan (x c) * Real.tan (x c) = 1 / Real.cos (x c) ^ 2 by
    rw [one_div, ← Real.inv_one_add_tan_sq hx, inv_inv, sq], mul_comm]

/-- The pullback of `erf` at `x` is `erf' x · g`. -/
theorem erfBackward_real {s : Shape} (x g : Tensor ℝ s) (c : s.Coord) :
    HasDerivAt TorchLean.erf (2 / √Real.pi * Real.exp (-x c ^ 2)) (x c) ∧
      erfBackward x g c = 2 / √Real.pi * Real.exp (-x c ^ 2) * g c := by
  refine ⟨hasDerivAt_erf (x c), ?_⟩
  simp only [erfBackward, Tensor.zipWith_apply]
  change (1 + 1) / √Real.pi * Real.exp (-(x c * x c)) * g c = _
  rw [show (1 : ℝ) + 1 = 2 by norm_num, show -(x c * x c) = -x c ^ 2 by ring]

end TorchLean.Tensor
