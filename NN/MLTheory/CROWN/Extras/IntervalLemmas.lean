/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Interval arithmetic lemmas (ℝ)

This file provides foundational lemmas for interval arithmetic operations
used in CROWN/LiRPA bound propagation soundness proofs.

It is primarily intended as a small toolbox for proof scripts and examples; it lives under
`NN/MLTheory/CROWN/Extras/` to keep the main CROWN modules focused on the bound propagation API.

-/

@[expose] public section


namespace NN.MLTheory.CROWN.IntervalLemmas

/-! ### Basic Interval Membership -/

/-- A value is in an interval [lo, hi] -/
def inInterval (x lo hi : ℝ) : Prop := lo ≤ x ∧ x ≤ hi

theorem inInterval_refl (x : ℝ) : inInterval x x x :=
  ⟨le_refl x, le_refl x⟩

theorem inInterval_of_bounds {x lo hi : ℝ}
    (hlo : lo ≤ x) (hhi : x ≤ hi) : inInterval x lo hi := ⟨hlo, hhi⟩

/-! ### Addition Interval Soundness -/

/-- If x ∈ [a,b] and y ∈ [c,d], then x + y ∈ [a+c, b+d] -/
theorem interval_add_sound {x y a b c d : ℝ}
    (hx : inInterval x a b) (hy : inInterval y c d) :
    inInterval (x + y) (a + c) (b + d) := by
  constructor
  · exact add_le_add hx.1 hy.1
  · exact add_le_add hx.2 hy.2

/-! ### Subtraction Interval Soundness -/

/-- If x ∈ [a,b] and y ∈ [c,d], then x - y ∈ [a-d, b-c] -/
theorem interval_sub_sound {x y a b c d : ℝ}
    (hx : inInterval x a b) (hy : inInterval y c d) :
    inInterval (x - y) (a - d) (b - c) := by
  constructor
  · exact sub_le_sub hx.1 hy.2
  · exact sub_le_sub hx.2 hy.1

/-! ### Multiplication Interval Soundness -/

/-- Helper: minimum of four values -/
def min4 (p q r s : ℝ) : ℝ := min (min p q) (min r s)

/-- Helper: maximum of four values -/
def max4 (p q r s : ℝ) : ℝ := max (max p q) (max r s)

/-- For multiplication, the interval is [min(ac,ad,bc,bd), max(ac,ad,bc,bd)].

    The proof requires 9-case analysis based on signs of interval endpoints:
    1. a >= 0, c >= 0: min=ac, max=bd
    2. a >= 0, c < 0 <= d: min=bc, max=bd
    3. a >= 0, d < 0: min=bc, max=ad
    4. a < 0 <= b, c >= 0: min=ad, max=bd
    5. a < 0 <= b, c < 0 <= d: min=min(ad,bc), max=max(ac,bd)
    6. a < 0 <= b, d < 0: min=bc, max=ac
    7. b < 0, c >= 0: min=ad, max=bc
    8. b < 0, c < 0 <= d: min=ad, max=ac
    9. b < 0, d < 0: min=bd, max=ac

    The key insight is that extrema of the bilinear function x*y over a rectangle
    [a,b] x [c,d] occur at the corners.
-/
theorem interval_mul_sound {x y a b c d : ℝ}
    (hx : inInterval x a b) (hy : inInterval y c d) :
    inInterval (x * y) (min4 (a*c) (a*d) (b*c) (b*d)) (max4 (a*c) (a*d) (b*c) (b*d)) := by
  -- The key insight: for bilinear f(x,y) = xy over [a,b] × [c,d], extrema occur at corners.
  -- For fixed y, f(x,y) = xy is linear in x, achieving min/max at x=a or x=b.
  -- Similarly for fixed x. So global extrema are at corners.
  unfold inInterval min4 max4
  have hab : a ≤ b := le_trans hx.1 hx.2
  have hcd : c ≤ d := le_trans hy.1 hy.2
  constructor
  · -- Lower bound: x*y ≥ min of corner products
    -- Step 1: For fixed y, min over x∈[a,b] of xy is min(ay, by)
    have h_xy_ge_min_ab : min (a*y) (b*y) ≤ x * y := by
      by_cases hy_nonneg : 0 ≤ y
      · -- y ≥ 0: min(ay, by) = ay since a ≤ b
        have hmin_eq : min (a*y) (b*y) = a*y := min_eq_left (mul_le_mul_of_nonneg_right hab
          hy_nonneg)
        rw [hmin_eq]
        exact mul_le_mul_of_nonneg_right hx.1 hy_nonneg
      · -- y < 0: min(ay, by) = by since a ≤ b implies ay ≥ by
        push Not at hy_nonneg
        have hmin_eq : min (a*y) (b*y) = b*y := min_eq_right (mul_le_mul_of_nonpos_right hab
          (le_of_lt hy_nonneg))
        rw [hmin_eq]
        exact mul_le_mul_of_nonpos_right hx.2 (le_of_lt hy_nonneg)
    -- Step 2: Show min4(...) ≤ min(ay, by)
    have h_min4_le : min (min (a*c) (a*d)) (min (b*c) (b*d)) ≤ min (a*y) (b*y) := by
      by_cases hy_nonneg : 0 ≤ y
      · -- y ≥ 0: min(ay, by) = ay, need min4 ≤ ay
        have hmin_ab : min (a*y) (b*y) = a*y := min_eq_left (mul_le_mul_of_nonneg_right hab
          hy_nonneg)
        rw [hmin_ab]
        by_cases ha_nonneg : 0 ≤ a
        · -- a ≥ 0, y ≥ 0: ac ≤ ay since c ≤ y
          have : a*c ≤ a*y := mul_le_mul_of_nonneg_left hy.1 ha_nonneg
          exact le_trans (le_trans (min_le_left _ _) (min_le_left _ _)) this
        · -- a < 0, y ≥ 0: ad ≤ ay since y ≤ d and a < 0
          push Not at ha_nonneg
          have : a*d ≤ a*y := mul_le_mul_of_nonpos_left hy.2 (le_of_lt ha_nonneg)
          exact le_trans (le_trans (min_le_left _ _) (min_le_right _ _)) this
      · -- y < 0: min(ay, by) = by
        push Not at hy_nonneg
        have hmin_ab : min (a*y) (b*y) = b*y := min_eq_right (mul_le_mul_of_nonpos_right hab
          (le_of_lt hy_nonneg))
        rw [hmin_ab]
        by_cases hb_nonneg : 0 ≤ b
        · -- b ≥ 0, y < 0: bc ≤ by since c ≤ y and b ≥ 0
          have : b*c ≤ b*y := mul_le_mul_of_nonneg_left hy.1 hb_nonneg
          exact le_trans (le_trans (min_le_right _ _) (min_le_left _ _)) this
        · -- b < 0, y < 0: bd ≤ by since y ≤ d and b < 0
          push Not at hb_nonneg
          have : b*d ≤ b*y := mul_le_mul_of_nonpos_left hy.2 (le_of_lt hb_nonneg)
          exact le_trans (le_trans (min_le_right _ _) (min_le_right _ _)) this
    exact le_trans h_min4_le h_xy_ge_min_ab
  · -- Upper bound: x*y ≤ max of corner products
    -- Step 1: For fixed y, max over x∈[a,b] of xy is max(ay, by)
    have h_xy_le_max_ab : x * y ≤ max (a*y) (b*y) := by
      by_cases hy_nonneg : 0 ≤ y
      · -- y ≥ 0: max(ay, by) = by since a ≤ b
        have hmax_eq : max (a*y) (b*y) = b*y := max_eq_right (mul_le_mul_of_nonneg_right hab
          hy_nonneg)
        rw [hmax_eq]
        exact mul_le_mul_of_nonneg_right hx.2 hy_nonneg
      · -- y < 0: max(ay, by) = ay since a ≤ b implies ay ≥ by
        push Not at hy_nonneg
        have hmax_eq : max (a*y) (b*y) = a*y := max_eq_left (mul_le_mul_of_nonpos_right hab
          (le_of_lt hy_nonneg))
        rw [hmax_eq]
        exact mul_le_mul_of_nonpos_right hx.1 (le_of_lt hy_nonneg)
    -- Step 2: Show max(ay, by) ≤ max4(...)
    have h_max_le_max4 : max (a*y) (b*y) ≤ max (max (a*c) (a*d)) (max (b*c) (b*d)) := by
      by_cases hy_nonneg : 0 ≤ y
      · -- y ≥ 0: max(ay, by) = by, need by ≤ max4
        have hmax_ab : max (a*y) (b*y) = b*y := max_eq_right (mul_le_mul_of_nonneg_right hab
          hy_nonneg)
        rw [hmax_ab]
        by_cases hb_nonneg : 0 ≤ b
        · -- b ≥ 0, y ≥ 0: by ≤ bd since y ≤ d
          have : b*y ≤ b*d := mul_le_mul_of_nonneg_left hy.2 hb_nonneg
          exact le_trans this (le_trans (le_max_right _ _) (le_max_right _ _))
        · -- b < 0, y ≥ 0: by ≤ bc since c ≤ y and b < 0
          push Not at hb_nonneg
          have : b*y ≤ b*c := mul_le_mul_of_nonpos_left hy.1 (le_of_lt hb_nonneg)
          exact le_trans this (le_trans (le_max_left _ _) (le_max_right _ _))
      · -- y < 0: max(ay, by) = ay
        push Not at hy_nonneg
        have hmax_ab : max (a*y) (b*y) = a*y := max_eq_left (mul_le_mul_of_nonpos_right hab
          (le_of_lt hy_nonneg))
        rw [hmax_ab]
        by_cases ha_nonneg : 0 ≤ a
        · -- a ≥ 0, y < 0: ay ≤ ad since y ≤ d and a ≥ 0
          have : a*y ≤ a*d := mul_le_mul_of_nonneg_left hy.2 ha_nonneg
          exact le_trans this (le_trans (le_max_right _ _) (le_max_left _ _))
        · -- a < 0, y < 0: ay ≤ ac since c ≤ y and a < 0
          push Not at ha_nonneg
          have : a*y ≤ a*c := mul_le_mul_of_nonpos_left hy.1 (le_of_lt ha_nonneg)
          exact le_trans this (le_trans (le_max_left _ _) (le_max_left _ _))
    exact le_trans h_xy_le_max_ab h_max_le_max4

/-! ### ReLU Interval Soundness -/

/-- ReLU function: max(0, x) -/
def relu (x : ℝ) : ℝ := max 0 x

/-- ReLU is monotone -/
theorem relu_monotone {x y : ℝ} (h : x ≤ y) : relu x ≤ relu y := by
  unfold relu
  exact max_le_max_left 0 h

/-- ReLU is non-negative -/
theorem relu_nonneg (x : ℝ) : 0 ≤ relu x := by
  unfold relu
  exact le_max_left 0 x

/-- If x ≤ 0, then relu(x) = 0 -/
theorem relu_of_nonpos {x : ℝ} (h : x ≤ 0) : relu x = 0 := by
  unfold relu
  exact max_eq_left h

/-- If 0 ≤ x, then relu(x) = x -/
theorem relu_of_nonneg {x : ℝ} (h : 0 ≤ x) : relu x = x := by
  unfold relu
  exact max_eq_right h

/-- ReLU interval: if x ∈ [l, u], then relu(x) ∈ [max(0,l), max(0,u)] -/
theorem interval_relu_sound {x l u : ℝ} (h : inInterval x l u) :
    inInterval (relu x) (max 0 l) (max 0 u) := by
  unfold inInterval relu
  constructor
  · exact max_le_max_left 0 h.1
  · exact max_le_max_left 0 h.2

/-! ### Square Interval Soundness -/

/-- Square function -/
def square (x : ℝ) : ℝ := x * x

/-- Square is non-negative -/
theorem square_nonneg (x : ℝ) : 0 ≤ square x := mul_self_nonneg x

/-- If 0 ≤ a ≤ b, then a² ≤ b² -/
theorem square_le_square_of_nonneg {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) :
    square a ≤ square b := mul_self_le_mul_self ha hab

/-- Minimum square in an interval -/
noncomputable def intervalSquareMin (l u : ℝ) : ℝ :=
  if l < 0 then
    if 0 < u then 0
    else min (l * l) (u * u)
  else
    l * l

/-- Maximum square in an interval -/
noncomputable def intervalSquareMax (l u : ℝ) : ℝ := max (l * l) (u * u)

/-- Square interval soundness: if x ∈ [l, u], then x² ∈ [minSq, maxSq] -/
theorem interval_square_sound {x l u : ℝ} (h : inInterval x l u) :
    inInterval (square x) (intervalSquareMin l u) (intervalSquareMax l u) := by
  unfold square intervalSquareMin intervalSquareMax inInterval
  constructor
  · split_ifs with hl_neg hu_pos
    · exact mul_self_nonneg x
    · have hx_nonpos : x ≤ 0 := le_trans h.2 (le_of_not_gt hu_pos)
      have hu_neg : u ≤ 0 := le_of_not_gt hu_pos
      have hu_sq_le : u * u ≤ x * x := by
        have h1 : 0 ≤ -u := neg_nonneg.mpr hu_neg
        have h2 : -u ≤ -x := neg_le_neg h.2
        calc u * u = (-u) * (-u) := by ring
             _ ≤ (-x) * (-x) := mul_self_le_mul_self h1 h2
             _ = x * x := by ring
      exact min_le_of_right_le hu_sq_le
    · have hl_nonneg : 0 ≤ l := le_of_not_gt hl_neg
      exact mul_self_le_mul_self hl_nonneg h.1
  · by_cases hx_nonneg : 0 ≤ x
    · have hu_nonneg : 0 ≤ u := le_trans hx_nonneg h.2
      exact le_max_of_le_right (mul_self_le_mul_self hx_nonneg h.2)
    · push Not at hx_nonneg
      have hx_nonpos : x ≤ 0 := le_of_lt hx_nonneg
      have habs : -x ≤ -l := neg_le_neg h.1
      have hx_sq_le_l_sq : x * x ≤ l * l := by
        have h1 : 0 ≤ -x := neg_nonneg.mpr hx_nonpos
        calc x * x = (-x) * (-x) := by ring
             _ ≤ (-l) * (-l) := mul_self_le_mul_self h1 habs
             _ = l * l := by ring
      exact le_max_of_le_left hx_sq_le_l_sq

/-! ### Negation Interval -/

/-- Negation flips and swaps the interval: -[a,b] = [-b, -a] -/
theorem interval_neg_sound {x a b : ℝ} (h : inInterval x a b) :
    inInterval (-x) (-b) (-a) := by
  constructor
  · exact neg_le_neg h.2
  · exact neg_le_neg h.1

/-! ### Absolute Value Interval -/

/-- If x ∈ [l, u], then |x| is bounded -/
theorem interval_abs_sound {x l u : ℝ} (h : inInterval x l u) :
    0 ≤ |x| ∧ |x| ≤ max |l| |u| := by
  constructor
  · exact abs_nonneg x
  · by_cases hx : 0 ≤ x
    · have : |x| = x := abs_of_nonneg hx
      rw [this]
      calc x ≤ u := h.2
           _ ≤ |u| := le_abs_self u
           _ ≤ max |l| |u| := le_max_right _ _
    · push Not at hx
      have : |x| = -x := abs_of_neg hx
      rw [this]
      calc -x ≤ -l := neg_le_neg h.1
           _ ≤ |l| := neg_le_abs l
           _ ≤ max |l| |u| := le_max_left _ _

end NN.MLTheory.CROWN.IntervalLemmas
