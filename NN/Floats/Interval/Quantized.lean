/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Complex.Trigonometric
public import Mathlib.Data.EReal.Basic
public import NN.Floats.Interval.Rounders

/-!
# Quantized interval arithmetic (rounding-on-`ℝ`, overflow-aware)

This file provides a **sound enclosure** layer for real computations where interval endpoints are
snapped outward to a chosen representable grid (a Flocq-style `(β,fexp)` format).

The main intended use is verification/bounding:

- You start from a real interval `x ∈ [lo,hi]`.
- You propagate it through primitive operations and monotone nonlinearities.
- After each primitive, you quantize the endpoint outward via a `Rounder` (down/up).

Compared to an executable IEEE kernel:
- This is proof-friendly (everything lives in `ℝ`/`EReal` and uses Flocq-style rounding).
- It is *not* a bit-level IEEE-754 model (no NaN payloads, signed zero rules, etc.).

Overflow-awareness:
- We use `EReal` (extended reals) to represent unbounded intervals arising from division by an
  interval that contains `0`.

References:
- IEEE 1788-2015 (interval arithmetic).
- Moore, Kearfott, Cloud, *Introduction to Interval Analysis* (2009).
- Rump (INTLAB) for outward rounding.
-/

@[expose] public section


namespace TorchLean.Floats.Interval

open scoped BigOperators

/-- A closed real interval `[lo,hi]`. -/
structure RInterval where
  /-- lo. -/
  lo : ℝ
  /-- hi. -/
  hi : ℝ

namespace RInterval

/-- Membership predicate: `x ∈ I` means `I.lo ≤ x ≤ I.hi`. -/
def mem (I : RInterval) (x : ℝ) : Prop :=
  I.lo ≤ x ∧ x ≤ I.hi

/-- Enable `x ∈ I` notation for real intervals. -/
instance : Membership ℝ RInterval where
  mem I x := I.mem x

/-- Unfold membership: `x ∈ I ↔ I.lo ≤ x ∧ x ≤ I.hi`. -/
@[simp] theorem mem_iff (I : RInterval) (x : ℝ) : x ∈ I ↔ I.lo ≤ x ∧ x ≤ I.hi :=
  Iff.rfl

/-- Validity predicate: endpoints are ordered (`lo ≤ hi`). -/
def Valid (I : RInterval) : Prop :=
  I.lo ≤ I.hi

/-- Degenerate interval `[x,x]`. -/
@[inline] def point (x : ℝ) : RInterval := ⟨x, x⟩

/-- Membership in a point interval is equality. -/
@[simp] theorem mem_point (x y : ℝ) : x ∈ point y ↔ x = y := by
  constructor
  · intro h
    exact le_antisymm h.2 h.1
  · intro h
    subst h
    constructor <;> rfl

/-- Interval negation: `-[lo,hi] = [-hi,-lo]`. -/
@[inline] def neg (I : RInterval) : RInterval := ⟨-I.hi, -I.lo⟩

/-- Outward-rounded interval addition, using the provided endpoint `Rounder`. -/
@[inline] def add (R : Rounder) (A B : RInterval) : RInterval :=
  ⟨R.down (A.lo + B.lo), R.up (A.hi + B.hi)⟩

/-- Interval subtraction, implemented as `A + (-B)` so soundness reuses `mem_add` and `mem_neg`. -/
@[inline] def sub (R : Rounder) (A B : RInterval) : RInterval :=
  add R A (neg B)

/--
Maximum absolute endpoint magnitude: `max |lo| |hi|`.

This is a convenient (and very proof-stable) way to get a coarse enclosure for products.
-/
noncomputable def absMax (I : RInterval) : ℝ :=
  max |I.lo| |I.hi|

/-- If `x ∈ I`, then `|x| ≤ absMax I`. -/
theorem abs_le_absMax {I : RInterval} {x : ℝ} (hx : x ∈ I) :
    |x| ≤ absMax I := by
  -- `abs` is convex on ℝ; max on an interval occurs at endpoints.
  exact abs_le_max_abs_abs hx.1 hx.2

/--
Loose-but-sound multiplication enclosure.

For `x ∈ A`, `y ∈ B` we have `|x*y| ≤ absMax A * absMax B`, hence
`x*y ∈ [-M, M]`. We then outward-round the endpoints via `R`.

This bound is conservative and serves as a generally applicable, proof-stable fallback. The
executable IEEE interval layer provides the tighter four-corner multiplication theorem when that
extra precision matters.
-/
noncomputable def mul (R : Rounder) (A B : RInterval) : RInterval :=
  let m : ℝ := absMax A * absMax B
  ⟨R.down (-m), R.up m⟩

/-- Soundness of `add`: membership is preserved by real addition. -/
theorem mem_add {R : Rounder} {A B : RInterval} {x y : ℝ}
    (hx : x ∈ A) (hy : y ∈ B) :
    x + y ∈ add R A B := by
  constructor
  · have : A.lo + B.lo ≤ x + y := add_le_add hx.1 hy.1
    exact le_trans (R.down_le _) this
  · have : x + y ≤ A.hi + B.hi := add_le_add hx.2 hy.2
    exact le_trans this (R.le_up _)

/-- Soundness of `sub`: membership is preserved by real subtraction. -/
theorem mem_sub {R : Rounder} {A B : RInterval} {x y : ℝ}
    (hx : x ∈ A) (hy : y ∈ B) :
    x - y ∈ sub R A B := by
  have hy' : (-y) ∈ neg B := by
    constructor
    · have : -B.hi ≤ -y := neg_le_neg hy.2
      simpa [neg] using this
    · have : -y ≤ -B.lo := neg_le_neg hy.1
      simpa [neg] using this
  -- `sub` is defined via `add` + `neg`.
  simpa [sub, sub_eq_add_neg] using (mem_add (R := R) (A := A) (B := neg B) hx hy')

/-- Soundness of the conservative `mul` enclosure: membership is preserved by real multiplication.
  -/
theorem mem_mul {R : Rounder} {A B : RInterval} {x y : ℝ}
    (hx : x ∈ A) (hy : y ∈ B) :
    x * y ∈ mul R A B := by
  have hxabs : |x| ≤ absMax A := abs_le_absMax (I := A) hx
  have hyabs : |y| ≤ absMax B := abs_le_absMax (I := B) hy
  have hxyabs : |x * y| ≤ absMax A * absMax B := by
    -- `|x*y| = |x|*|y|` and each factor is bounded by `absMax`.
    have : |x * y| = |x| * |y| := by simp [abs_mul]
    rw [this]
    exact mul_le_mul hxabs hyabs (abs_nonneg y) (le_trans (abs_nonneg x) hxabs)
  have hbounds : - (absMax A * absMax B) ≤ x * y ∧ x * y ≤ absMax A * absMax B := by
    -- Turn an `abs` bound into a two-sided bound.
    have := (abs_le).1 hxyabs
    simpa [sub_eq_add_neg] using this
  constructor
  · exact le_trans (R.down_le _) hbounds.1
  · exact le_trans hbounds.2 (R.le_up _)

/-- Outward-rounded interval enclosure for `Real.exp`, using monotonicity. -/
noncomputable def exp (R : Rounder) (A : RInterval) : RInterval :=
  ⟨R.down (Real.exp A.lo), R.up (Real.exp A.hi)⟩

/-- Soundness of `exp`: membership is preserved by `Real.exp`. -/
theorem mem_exp {R : Rounder} {A : RInterval} {x : ℝ} (hx : x ∈ A) :
    Real.exp x ∈ exp R A := by
  constructor
  · have : Real.exp A.lo ≤ Real.exp x := by exact Real.exp_monotone hx.1
    exact le_trans (R.down_le _) this
  · have : Real.exp x ≤ Real.exp A.hi := by exact Real.exp_monotone hx.2
    exact le_trans this (R.le_up _)

/--
Coarse `tanh` enclosure `[-1, 1]` with outward-rounded endpoints.

We use this instead of a tighter monotone bound because `tanh` is bounded everywhere and the
uniform bound is often enough for verification pipelines.
-/
noncomputable def tanhBounds (R : Rounder) : RInterval :=
  ⟨R.down (-1), R.up 1⟩

/-- Soundness of `tanhBounds`: `Real.tanh x ∈ [-1,1]`. -/
theorem mem_tanhBounds {R : Rounder} (x : ℝ) :
    Real.tanh x ∈ tanhBounds R := by
  -- `tanh x = sinh x / cosh x`, `cosh x > 0`, and `sinh x < cosh x`.
  have hcosh : 0 < Real.cosh x := Real.cosh_pos x
  have hs : Real.sinh x < Real.cosh x := Real.sinh_lt_cosh (x := x)
  have ht_lt : Real.tanh x < 1 := by
    have hfrac : Real.sinh x / Real.cosh x < 1 := by
      have : Real.sinh x / Real.cosh x < Real.cosh x / Real.cosh x :=
        div_lt_div_of_pos_right hs hcosh
      simpa [div_self (ne_of_gt hcosh)] using this
    simpa [Real.tanh_eq_sinh_div_cosh] using hfrac
  have ht_gt : -1 < Real.tanh x := by
    -- Apply the same argument at `-x` and use `tanh (-x) = -tanh x`.
    have hcosh' : 0 < Real.cosh (-x) := Real.cosh_pos (-x)
    have hs' : Real.sinh (-x) < Real.cosh (-x) := Real.sinh_lt_cosh (x := -x)
    have hfrac : Real.sinh (-x) / Real.cosh (-x) < 1 := by
      have : Real.sinh (-x) / Real.cosh (-x) < Real.cosh (-x) / Real.cosh (-x) :=
        div_lt_div_of_pos_right hs' hcosh'
      simpa [Real.sinh_neg, Real.cosh_neg, div_self (ne_of_gt hcosh)] using this
    have ht_neg : Real.tanh (-x) < 1 := by
      simpa [Real.tanh_eq_sinh_div_cosh] using hfrac
    have : -Real.tanh x < 1 := by simpa [Real.tanh_neg] using ht_neg
    linarith
  constructor
  · exact le_trans (R.down_le _) (le_of_lt ht_gt)
  · exact le_trans (le_of_lt ht_lt) (R.le_up _)

/--
Interval `tanh` enclosure.

This deliberately returns the global codomain enclosure `tanhBounds`. It is a proof-stable fallback
for pipelines that only need a sound bound; monotone interval refinements can live beside it without
changing this simple contract.
-/
noncomputable def tanh (R : Rounder) (_A : RInterval) : RInterval :=
  tanhBounds R

/-- Soundness of `tanh`: membership is preserved by `Real.tanh` (via `tanhBounds`). -/
theorem mem_tanh {R : Rounder} {A : RInterval} {x : ℝ} (_hx : x ∈ A) :
    Real.tanh x ∈ tanh R A := by
  simpa [tanh] using (mem_tanhBounds (R := R) x)

/-- Outward-rounded interval enclosure for `Real.sqrt` (requires `0 ≤ lo`). -/
noncomputable def sqrt (R : Rounder) (A : RInterval) : RInterval :=
  ⟨R.down (Real.sqrt A.lo), R.up (Real.sqrt A.hi)⟩

/-- Soundness of `sqrt`: membership is preserved by `Real.sqrt` when `A` is nonnegative. -/
theorem mem_sqrt {R : Rounder} {A : RInterval} {x : ℝ} (hA : 0 ≤ A.lo) (hx : x ∈ A) :
    Real.sqrt x ∈ sqrt R A := by
  have _hx0 : 0 ≤ x := le_trans hA hx.1
  constructor
  · have : Real.sqrt A.lo ≤ Real.sqrt x := by exact Real.sqrt_le_sqrt hx.1
    exact le_trans (R.down_le _) this
  · have : Real.sqrt x ≤ Real.sqrt A.hi := by exact Real.sqrt_le_sqrt hx.2
    exact le_trans this (R.le_up _)

/-- Outward-rounded interval enclosure for `Real.log` (requires `0 < lo`). -/
noncomputable def log (R : Rounder) (A : RInterval) : RInterval :=
  ⟨R.down (Real.log A.lo), R.up (Real.log A.hi)⟩

/-- Soundness of `log`: membership is preserved by `Real.log` on positive intervals. -/
theorem mem_log {R : Rounder} {A : RInterval} {x : ℝ} (hA : 0 < A.lo) (hx : x ∈ A) :
    Real.log x ∈ log R A := by
  have hxpos : 0 < x := lt_of_lt_of_le hA hx.1
  constructor
  · have : Real.log A.lo ≤ Real.log x := by
      exact Real.log_le_log hA hx.1
    exact le_trans (R.down_le _) this
  · have : Real.log x ≤ Real.log A.hi := by
      exact Real.log_le_log hxpos hx.2
    exact le_trans this (R.le_up _)

end RInterval

/-!
## Extended-real intervals (for division by an interval containing 0)
-/

/--
An `EReal` interval `[lo,hi]`.

We use this for operations like division where a single interval may need to represent unbounded
results (`-∞`/`+∞`) in a sound-but-coarse way.
-/
structure EInterval where
  /-- lo. -/
  lo : EReal
  /-- hi. -/
  hi : EReal

noncomputable section

namespace EInterval

/-- Membership predicate: `x ∈ I` means `I.lo ≤ x ≤ I.hi` in `EReal`. -/
def mem (I : EInterval) (x : EReal) : Prop :=
  I.lo ≤ x ∧ x ≤ I.hi

/-- Enable `x ∈ I` notation for `EReal` intervals. -/
instance : Membership EReal EInterval where
  mem I x := I.mem x

/-- Unfold membership: `x ∈ I ↔ I.lo ≤ x ∧ x ≤ I.hi`. -/
@[simp] theorem mem_iff (I : EInterval) (x : EReal) : x ∈ I ↔ I.lo ≤ x ∧ x ≤ I.hi :=
  Iff.rfl

/-- Top interval `[-∞,+∞]` (the most conservative enclosure). -/
noncomputable def top : EInterval := ⟨⊥, ⊤⟩

/-- Every value lies in `top = [-∞,+∞]`. -/
@[simp] theorem mem_top (x : EReal) : x ∈ top := by
  simp [top]

/-- Embed a real interval into an extended-real interval. -/
@[inline] def ofRInterval (I : RInterval) : EInterval := ⟨(I.lo : EReal), (I.hi : EReal)⟩

/--
Membership in `ofRInterval` is the same as membership in the underlying real interval.

This is a small simp lemma that helps bridge between `ℝ`-level and `EReal`-level enclosures.
-/
@[simp] theorem mem_ofRInterval (I : RInterval) (x : ℝ) :
    ((x : ℝ) : EReal) ∈ ofRInterval I ↔ x ∈ I := by
  constructor
  · intro hx
    constructor
    · exact (EReal.coe_le_coe_iff).1 hx.1
    · exact (EReal.coe_le_coe_iff).1 hx.2
  · intro hx
    constructor
    · exact (EReal.coe_le_coe_iff).2 hx.1
    · exact (EReal.coe_le_coe_iff).2 hx.2
end EInterval

namespace RInterval

open EInterval

/-!
Division with overflow-awareness.

If the denominator interval contains `0`, the result is unbounded (`[-∞,+∞]`).
Otherwise we return a loose enclosure using absolute-value bounds.
-/
/--
Outward-rounded interval division as an `EReal` enclosure.

If the denominator interval contains `0`, we return `EInterval.top = [-∞,+∞]`. Otherwise we use a
coarse-but-sound absolute-value bound and outward-round the endpoints with `R`.
-/
noncomputable def div (R : Rounder) (A B : RInterval) : EInterval :=
  if _h0 : (B.lo ≤ 0 ∧ 0 ≤ B.hi) then
    EInterval.top
  else
    -- If `0 ∉ B`, we can bound `|1/y|` by `1 / min(|B.lo|,|B.hi|)` (loose but sound).
    let mA : ℝ := absMax A
    let mB : ℝ := min |B.lo| |B.hi|
    let bound : ℝ := mA / mB
    EInterval.ofRInterval ⟨R.down (-bound), R.up bound⟩

/--
Soundness of `div` in the nonzero-denominator case (`0 ∉ B`).

Under the hypothesis `¬ (B.lo ≤ 0 ∧ 0 ≤ B.hi)`, the constructed `EInterval` contains the true real
quotient `x/y` (as an `EReal`).
-/
theorem mem_div_of_nozero {R : Rounder} {A B : RInterval} {x y : ℝ}
    (hx : x ∈ A) (hy : y ∈ B) (h0 : ¬ (B.lo ≤ 0 ∧ 0 ≤ B.hi)) :
    ((x / y : ℝ) : EReal) ∈ (div R A B) := by
  -- We prove a conservative enclosure: `|x/y| ≤ absMax A / min(|B.lo|,|B.hi|)`.
  simp (config := { zeta := true }) [div, h0, EInterval.ofRInterval]
  have hxabs : |x| ≤ absMax A := abs_le_absMax (I := A) hx
  have hcase : 0 < B.lo ∨ B.hi < 0 := by
    by_contra h
    have hlo : B.lo ≤ 0 := le_of_not_gt (by intro hlo; exact h (Or.inl hlo))
    have hhi : 0 ≤ B.hi := le_of_not_gt (by intro hhi; exact h (Or.inr hhi))
    exact h0 ⟨hlo, hhi⟩
  have hmin_pos : 0 < min |B.lo| |B.hi| := by
    cases hcase with
    | inl hloPos =>
        have hypos : 0 < y := lt_of_lt_of_le hloPos hy.1
        have hhiPos : 0 < B.hi := lt_of_lt_of_le hypos hy.2
        have hloAbs : 0 < |B.lo| := abs_pos.2 (ne_of_gt hloPos)
        have hhiAbs : 0 < |B.hi| := abs_pos.2 (ne_of_gt hhiPos)
        exact (lt_min_iff).2 ⟨hloAbs, hhiAbs⟩
    | inr hhiNeg =>
        have hyneg : y < 0 := lt_of_le_of_lt hy.2 hhiNeg
        have hloNeg : B.lo < 0 := lt_of_le_of_lt hy.1 hyneg
        have hloAbs : 0 < |B.lo| := abs_pos.2 (ne_of_lt hloNeg)
        have hhiAbs : 0 < |B.hi| := abs_pos.2 (ne_of_lt hhiNeg)
        exact (lt_min_iff).2 ⟨hloAbs, hhiAbs⟩
  have hy_lower : min |B.lo| |B.hi| ≤ |y| := by
    cases hcase with
    | inl hloPos =>
        have hypos : 0 < y := lt_of_lt_of_le hloPos hy.1
        have hylo : B.lo ≤ |y| := by simpa [abs_of_pos hypos] using hy.1
        have hmin_le : min |B.lo| |B.hi| ≤ B.lo := by
          rw [abs_of_pos hloPos]
          exact min_le_left _ _
        exact le_trans hmin_le hylo
    | inr hhiNeg =>
        have hyneg : y < 0 := lt_of_le_of_lt hy.2 hhiNeg
        have hyhi' : -B.hi ≤ -y := by linarith [hy.2]
        have hyhiAbs : |B.hi| ≤ |y| := by
          simpa [abs_of_neg hhiNeg, abs_of_neg hyneg] using hyhi'
        exact le_trans (min_le_right _ _) hyhiAbs

  have habs_div : |x / y| ≤ absMax A / min |B.lo| |B.hi| := by
    have : |x / y| = |x| / |y| := by simp [abs_div]
    rw [this]
    have hinv : 0 ≤ (|y|)⁻¹ := inv_nonneg.2 (abs_nonneg y)
    have hnum : |x| / |y| ≤ absMax A / |y| := by
      simpa [div_eq_mul_inv, mul_assoc] using (mul_le_mul_of_nonneg_right hxabs hinv)
    have hInv : (1 / |y|) ≤ (1 / min |B.lo| |B.hi|) :=
      one_div_le_one_div_of_le hmin_pos hy_lower
    have hA : 0 ≤ absMax A := by
      have : 0 ≤ |A.lo| := abs_nonneg A.lo
      exact le_trans this (le_max_left _ _)
    have hdenMul := mul_le_mul_of_nonneg_left hInv hA
    have hden : absMax A / |y| ≤ absMax A / min |B.lo| |B.hi| := by
      simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using hdenMul
    exact le_trans (le_trans hnum hden) (le_rfl)

  have hbounds : - (absMax A / min |B.lo| |B.hi|) ≤ x / y ∧ x / y ≤ absMax A / min |B.lo| |B.hi| :=
    (abs_le).1 habs_div

  constructor
  · have hreal : (R.down (-(absMax A / min |B.lo| |B.hi|)) : ℝ) ≤ x / y :=
      le_trans (R.down_le _) hbounds.1
    have hE : ((R.down (-(absMax A / min |B.lo| |B.hi|)) : ℝ) : EReal) ≤ ((x / y : ℝ) : EReal) := by
      exact_mod_cast hreal
    simpa using hE
  · have hreal : x / y ≤ (R.up (absMax A / min |B.lo| |B.hi|) : ℝ) :=
      le_trans hbounds.2 (R.le_up _)
    have hE : ((x / y : ℝ) : EReal) ≤ ((R.up (absMax A / min |B.lo| |B.hi|) : ℝ) : EReal) := by
      exact_mod_cast hreal
    simpa using hE

end RInterval

end

end Interval
