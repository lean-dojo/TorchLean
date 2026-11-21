/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Int.Log
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.Calculus.Deriv.Mul

/-!
# The binary exponent and mantissa of a real number

`torch.frexp` splits `x` into a mantissa `m` and an integer exponent `e` with `x = m 2ᵉ`, and its
PyTorch 2.13 docstring puts `m` in "the open interval (-1, 1)". C's `frexp` (C99 7.12.6.4, WG14
N1256) fixes `1/2 ≤ |m| < 1`, and both are zero at a zero `x`.
We wrote `frexpExponent` and `frexpMantissa` by hand from that clause, with the exponent from
Mathlib's `Int.log 2`, the floor of the base-two logarithm. Over the reals there are no
infinities or NaNs, for which C leaves the exponent unspecified.

Proved here: `x = frexpMantissa x * 2 ^ frexpExponent x` for every `x` (`frexp_reconstruct`), and
`1/2 ≤ |frexpMantissa x| < 1` for `x ≠ 0` (`frexpMantissa_mem`). Between `2 ^ a` and `2 ^ b` in
absolute value the exponent lies in `[a + 1, b]` (`frexpExponent_mem_Icc`). Away from zero and
from the powers of two in absolute value, the exponent is constant near `x`
(`frexpExponent_eventuallyEq`), so the mantissa has derivative `1 / 2 ^ frexpExponent x`
(`hasDerivAt_frexpMantissa`).

## References

- WG14 N1256 (C99 committee draft), 7.12.6.4, the `frexp` functions:
  `https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf`
- PyTorch 2.13 documentation, `torch.frexp`:
  `https://docs.pytorch.org/docs/2.13/generated/torch.frexp.html`
-/

@[expose] public section

noncomputable section

namespace TorchLean

/-- The exponent `e` of `torch.frexp`: `⌊log₂ |x|⌋ + 1`, and `0` at `0`. -/
def frexpExponent (x : ℝ) : ℤ := if x = 0 then 0 else Int.log 2 |x| + 1

/-- The mantissa `x / 2 ^ frexpExponent x` of `torch.frexp`, `0` at `0`. -/
def frexpMantissa (x : ℝ) : ℝ := x / 2 ^ frexpExponent x

/-- The mantissa and exponent rebuild `x`. -/
theorem frexp_reconstruct (x : ℝ) : x = frexpMantissa x * 2 ^ frexpExponent x := by
  rw [frexpMantissa, div_mul_cancel₀ _ (zpow_ne_zero _ two_ne_zero)]

/-- A nonzero mantissa lies in `[1/2, 1)` in absolute value. -/
theorem frexpMantissa_mem {x : ℝ} (hx : x ≠ 0) :
    1 / 2 ≤ |frexpMantissa x| ∧ |frexpMantissa x| < 1 := by
  have hAbs : 0 < |x| := abs_pos.mpr hx
  have hLow := Int.zpow_log_le_self (b := 2) (by norm_num) hAbs
  have hHigh := Int.lt_zpow_succ_log_self (b := 2) (by norm_num) |x|
  push_cast at hLow hHigh
  have hPow : (0 : ℝ) < 2 ^ (Int.log 2 |x| + 1) := zpow_pos two_pos _
  simp only [frexpMantissa, frexpExponent, hx, ↓reduceIte, abs_div, abs_of_pos hPow]
  constructor
  · rw [le_div_iff₀ hPow, zpow_add_one₀ two_ne_zero]
    linarith
  · rw [div_lt_one hPow]
    exact hHigh

/-- If `2 ^ a ≤ |x| < 2 ^ b`, the exponent lies in `[a + 1, b]`. So the finite nonzero values of
a binary format have exponents in a fixed range: at binary64, `2 ^ -1074 ≤ |x| < 2 ^ 1024`
(binary64's `emax = 1023` and `p = 53`, IEEE 754-2019 clause 3) gives `[-1073, 1024]`. -/
theorem frexpExponent_mem_Icc {x : ℝ} {a b : ℤ} (hLow : (2 : ℝ) ^ a ≤ |x|) (hHigh : |x| < 2 ^ b) :
    frexpExponent x ∈ Set.Icc (a + 1) b := by
  have hPos : 0 < |x| := (zpow_pos two_pos _).trans_le hLow
  have hLe : a ≤ Int.log 2 |x| :=
    (Int.zpow_le_iff_le_log (b := 2) (by norm_num) hPos).1 (by exact_mod_cast hLow)
  have hLt : Int.log 2 |x| < b :=
    (Int.lt_zpow_iff_log_lt (b := 2) (by norm_num) hPos).1 (by exact_mod_cast hHigh)
  simp only [Set.mem_Icc, frexpExponent, abs_pos.mp hPos, ↓reduceIte]
  omega

/-- Away from zero and from the powers of two in absolute value, the exponent is constant near
`x`: `|x|` lies strictly between `2 ^ (e - 1)` and `2 ^ e`, and so do the points near it. -/
theorem frexpExponent_eventuallyEq {x : ℝ} (hx : x ≠ 0) (hpow : ∀ k : ℤ, |x| ≠ 2 ^ k) :
    frexpExponent =ᶠ[nhds x] fun _ => frexpExponent x := by
  have hAbs : 0 < |x| := abs_pos.mpr hx
  have hLow := Int.zpow_log_le_self (b := 2) (by norm_num) hAbs
  have hHigh := Int.lt_zpow_succ_log_self (b := 2) (by norm_num) |x|
  push_cast at hLow hHigh
  have hOpen : IsOpen {y : ℝ | (2 : ℝ) ^ Int.log 2 |x| < |y| ∧ |y| < 2 ^ (Int.log 2 |x| + 1)} :=
    (isOpen_lt continuous_const continuous_abs).inter (isOpen_lt continuous_abs continuous_const)
  filter_upwards [hOpen.mem_nhds ⟨lt_of_le_of_ne hLow (hpow _).symm, hHigh⟩] with y hy
  have hy' := Set.mem_Icc.1 (frexpExponent_mem_Icc hy.1.le hy.2)
  have hx' : frexpExponent x = Int.log 2 |x| + 1 := by simp only [frexpExponent, hx, ↓reduceIte]
  simp only [hx']
  omega

/-- Away from zero and from the powers of two in absolute value, the mantissa is `x / 2 ^ e` for
the exponent `e` of `x`, constant nearby, so its derivative is `1 / 2 ^ e`. -/
theorem hasDerivAt_frexpMantissa {x : ℝ} (hx : x ≠ 0) (hpow : ∀ k : ℤ, |x| ≠ 2 ^ k) :
    HasDerivAt frexpMantissa (1 / 2 ^ frexpExponent x) x := by
  refine ((hasDerivAt_id x).div_const _).congr_of_eventuallyEq ?_
  filter_upwards [frexpExponent_eventuallyEq hx hpow] with y hy
  simp only [frexpMantissa, hy, id]
