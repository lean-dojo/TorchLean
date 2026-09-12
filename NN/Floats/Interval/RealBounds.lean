/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic

/-!
# Real helper lemmas for interval corner bounds

Several IEEEExec32 interval-soundness modules need the same basic real-analysis fact:

If $x\in[a,b]$ and $y\in[c,d]$, then $xy$ is enclosed by the minimum and maximum of the four
corner products $\{ac,ad,bc,bd\}$.

This module keeps the interval arithmetic fact separate from the IEEEExec32 soundness proofs.
-/

@[expose] public section

namespace TorchLean.Floats.Interval

/-- Minimum of four real numbers, grouped as `min (min a b) (min c d)`. -/
def minOfFourReal (a b c d : ℝ) : ℝ :=
  min (min a b) (min c d)

/-- Maximum of four real numbers, grouped as `max (max a b) (max c d)`. -/
def maxOfFourReal (a b c d : ℝ) : ℝ :=
  max (max a b) (max c d)

/--
Corner enclosure for real multiplication on intervals.

If $x\in[a,b]$ and $y\in[c,d]$, then
$$
\min(ac,ad,bc,bd)\le xy\le\max(ac,ad,bc,bd),
$$

where the min/max are represented by `minOfFourReal`/`maxOfFourReal` with the same grouping used by the IEEE32Exec
4-corner rule implementations.
-/
theorem mul_bounds_Icc (a b c d x y : ℝ)
    (hx : x ∈ Set.Icc a b) (hy : y ∈ Set.Icc c d) :
    minOfFourReal (a * c) (a * d) (b * c) (b * d) ≤ x * y ∧
      x * y ≤ maxOfFourReal (a * c) (a * d) (b * c) (b * d) := by
  rcases hx with ⟨hax, hxb⟩
  rcases hy with ⟨hcy, hyd⟩

  have h_upper_xy : x * y ≤ max (x * c) (x * d) := by
    by_cases hx0 : 0 ≤ x
    · have : x * y ≤ x * d := mul_le_mul_of_nonneg_left hyd hx0
      exact le_trans this (le_max_right _ _)
    · have hx0' : x ≤ 0 := le_of_not_ge hx0
      have : x * y ≤ x * c := mul_le_mul_of_nonpos_left hcy hx0'
      exact le_trans this (le_max_left _ _)

  have h_lower_xy : min (x * c) (x * d) ≤ x * y := by
    by_cases hx0 : 0 ≤ x
    · have hxc : x * c ≤ x * y := mul_le_mul_of_nonneg_left hcy hx0
      have hxd : x * y ≤ x * d := mul_le_mul_of_nonneg_left hyd hx0
      have hxcd : x * c ≤ x * d := le_trans hxc hxd
      -- `min (x*c) (x*d) = x*c`
      simpa [min_eq_left hxcd] using hxc
    · have hx0' : x ≤ 0 := le_of_not_ge hx0
      have hxd : x * d ≤ x * y := mul_le_mul_of_nonpos_left hyd hx0'
      have hxc : x * y ≤ x * c := mul_le_mul_of_nonpos_left hcy hx0'
      have hdc : x * d ≤ x * c := le_trans hxd hxc
      simpa [min_eq_right hdc] using hxd

  have hxc_upper : x * c ≤ max (a * c) (b * c) := by
    by_cases hc0 : 0 ≤ c
    · have : x * c ≤ b * c := mul_le_mul_of_nonneg_right hxb hc0
      exact le_trans this (le_max_right _ _)
    · have hc0' : c ≤ 0 := le_of_not_ge hc0
      have : x * c ≤ a * c := mul_le_mul_of_nonpos_right hax hc0'
      exact le_trans this (le_max_left _ _)

  have hxd_upper : x * d ≤ max (a * d) (b * d) := by
    by_cases hd0 : 0 ≤ d
    · have : x * d ≤ b * d := mul_le_mul_of_nonneg_right hxb hd0
      exact le_trans this (le_max_right _ _)
    · have hd0' : d ≤ 0 := le_of_not_ge hd0
      have : x * d ≤ a * d := mul_le_mul_of_nonpos_right hax hd0'
      exact le_trans this (le_max_left _ _)

  have hxc_lower : min (a * c) (b * c) ≤ x * c := by
    by_cases hc0 : 0 ≤ c
    · have : a * c ≤ x * c := mul_le_mul_of_nonneg_right hax hc0
      have hab : a * c ≤ b * c := mul_le_mul_of_nonneg_right (le_trans hax hxb) hc0
      -- `min (a*c) (b*c) = a*c`
      simpa [min_eq_left hab] using this
    · have hc0' : c ≤ 0 := le_of_not_ge hc0
      have : b * c ≤ x * c := mul_le_mul_of_nonpos_right hxb hc0'
      have hba : b * c ≤ a * c := mul_le_mul_of_nonpos_right (le_trans hax hxb) hc0'
      -- `min (a*c) (b*c) = b*c`
      simpa [min_eq_right hba] using this

  have hxd_lower : min (a * d) (b * d) ≤ x * d := by
    by_cases hd0 : 0 ≤ d
    · have : a * d ≤ x * d := mul_le_mul_of_nonneg_right hax hd0
      have hab : a * d ≤ b * d := mul_le_mul_of_nonneg_right (le_trans hax hxb) hd0
      simpa [min_eq_left hab] using this
    · have hd0' : d ≤ 0 := le_of_not_ge hd0
      have : b * d ≤ x * d := mul_le_mul_of_nonpos_right hxb hd0'
      have hba : b * d ≤ a * d := mul_le_mul_of_nonpos_right (le_trans hax hxb) hd0'
      simpa [min_eq_right hba] using this

  -- Lower bound: min of corner mins ≤ min(x*c,x*d) ≤ x*y.
  have h_lower_corners :
      min (min (a * c) (b * c)) (min (a * d) (b * d)) ≤ min (x * c) (x * d) := by
    have h1 : min (min (a * c) (b * c)) (min (a * d) (b * d)) ≤ x * c :=
      le_trans (min_le_left _ _) hxc_lower
    have h2 : min (min (a * c) (b * c)) (min (a * d) (b * d)) ≤ x * d :=
      le_trans (min_le_right _ _) hxd_lower
    exact le_min h1 h2

  have h_lower :
      minOfFourReal (a * c) (a * d) (b * c) (b * d) ≤ x * y := by
    -- Switch groupings to match the helper bound.
    have hgrp :
        minOfFourReal (a * c) (a * d) (b * c) (b * d) =
          min (min (a * c) (b * c)) (min (a * d) (b * d)) := by
      -- This is the same `min` of 4 numbers, regrouped.
      simp [minOfFourReal, min_left_comm, min_comm]
    rw [hgrp]
    exact le_trans h_lower_corners h_lower_xy

  -- Upper bound: x*y ≤ max(x*c,x*d) ≤ max corner maxes (regroup to our maxOfFourReal).
  have h_upper_corners :
      max (x * c) (x * d) ≤ max (max (a * c) (b * c)) (max (a * d) (b * d)) :=
    max_le_max hxc_upper hxd_upper

  have h_upper :
      x * y ≤ maxOfFourReal (a * c) (a * d) (b * c) (b * d) := by
    have hgrp :
        max (max (a * c) (b * c)) (max (a * d) (b * d)) =
          maxOfFourReal (a * c) (a * d) (b * c) (b * d) := by
      -- Regroup `max` of 4 numbers.
      simp [maxOfFourReal, max_left_comm, max_comm]
    exact le_trans (le_trans h_upper_xy h_upper_corners) (by simp [hgrp])

  exact ⟨h_lower, h_upper⟩

/--
Reciprocal enclosure for an interval that lies strictly on one side of zero.

If `y ∈ [c, d]` and either `d < 0` or `0 < c`, then inversion reverses the endpoints:
`y⁻¹ ∈ [d⁻¹, c⁻¹]`.
-/
theorem inv_mem_Icc (c d y : ℝ) (hy : y ∈ Set.Icc c d) (h0 : d < 0 ∨ 0 < c) :
    y⁻¹ ∈ Set.Icc d⁻¹ c⁻¹ := by
  rcases hy with ⟨hcy, hyd⟩
  rcases h0 with hd | hc
  · have hyneg : y < 0 := lt_of_le_of_lt hyd hd
    constructor
    · simpa [one_div] using one_div_le_one_div_of_neg_of_le hd hyd
    · simpa [one_div] using one_div_le_one_div_of_neg_of_le hyneg hcy
  · have hypos : 0 < y := lt_of_lt_of_le hc hcy
    constructor
    · simpa [one_div] using one_div_le_one_div_of_le hypos hyd
    · simpa [one_div] using one_div_le_one_div_of_le hc hcy

/--
Four-corner enclosure for division by a sign-stable interval.

The denominator hypothesis excludes zero. The result follows by enclosing `y⁻¹` with
`inv_mem_Icc` and applying `mul_bounds_Icc` to `x * y⁻¹`.
-/
theorem div_bounds_Icc (a b c d x y : ℝ)
    (hx : x ∈ Set.Icc a b) (hy : y ∈ Set.Icc c d) (h0 : d < 0 ∨ 0 < c) :
    minOfFourReal (a / c) (a / d) (b / c) (b / d) ≤ x / y ∧
      x / y ≤ maxOfFourReal (a / c) (a / d) (b / c) (b / d) := by
  have hinv := inv_mem_Icc c d y hy h0
  have hmul := mul_bounds_Icc a b d⁻¹ c⁻¹ x y⁻¹ hx hinv
  simpa [div_eq_mul_inv, minOfFourReal, maxOfFourReal, min_comm, max_comm] using hmul

end TorchLean.Floats.Interval
