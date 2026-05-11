/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Core
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Rounding modes and a half-ULP error bound

We use this file as the “rounding” half of the Flocq-style rounding-on-`ℝ` model:

- rounding modes `rnd : ℝ → ℤ` (floor/ceil/trunc/nearest-even),
- the validity typeclasses `NeuralValidRnd` and `NeuralValidRndToNearest`,
- the core rounding operator `neural_round`,
- the standard bound `abs(neural_round … x - x) ≤ ulp(x)/2` for round-to-nearest.

These definitions are used by `NF` (the rounded scalar type), by the `FP32` model, and by the
bridge layer that connects proofs to executable IEEE-754 behavior.

## References

- IEEE Std 754-2019, "IEEE Standard for Floating-Point Arithmetic".
- D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic",
  ACM Computing Surveys, 1991.
- N. J. Higham, "Accuracy and Stability of Numerical Algorithms", SIAM, 2nd ed., 2002.
- The Flocq Coq library is a classic example of axiomatizing "rounding on R"; TorchLean's
  `NeuralFloat` layer is inspired by that style (but is implemented natively in Lean).
-/

@[expose] public section


namespace TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Round toward negative infinity (floor) -/
noncomputable def neuralFloorRound : ℝ → ℤ := fun x => ⌊x⌋

/-- Round toward positive infinity (ceiling) -/
noncomputable def neuralCeilRound : ℝ → ℤ := fun x => ⌈x⌉

/-- Round toward zero (truncation) -/
noncomputable def neuralTruncRound : ℝ → ℤ := fun x =>
  if x < 0 then ⌈x⌉ else ⌊x⌋

/-- Round to nearest, ties to even -/
noncomputable def neuralNearestEven : ℝ → ℤ := fun x =>
  let f := ⌊x⌋
  if x - f < 1/2 then f
  else if x - f > 1/2 then f + 1
  else if Even f then f else f + 1

/--
Valid rounding mode predicate.
-/
class NeuralValidRnd (rnd : ℝ → ℤ) : Prop where
  monotone : ∀ x y, x ≤ y → rnd x ≤ rnd y
  id : ∀ n : ℤ, rnd n = n

/--
Rounding modes with a half-unit error bound on the rounded integer.

This matches "round-to-nearest" style roundings (ties can be resolved arbitrarily):
`|rnd x - x| ≤ 1/2` for all `x`.
-/
class NeuralValidRndToNearest (rnd : ℝ → ℤ) : Prop extends NeuralValidRnd rnd where
  abs_sub_le_half : ∀ x : ℝ, abs ((rnd x : ℝ) - x) ≤ (2⁻¹ : ℝ)

-- Instance for floor rounding
/-- `neural_floor_round` is a valid rounding mode (monotone and fixes integers). -/
instance : NeuralValidRnd neuralFloorRound where
  monotone := fun _ _ h => Int.floor_mono h
  id := fun n => Int.floor_intCast n

-- Instance for ceiling rounding
/-- `neural_ceil_round` is a valid rounding mode (monotone and fixes integers). -/
instance : NeuralValidRnd neuralCeilRound where
  monotone := fun _ _ h => Int.ceil_mono h
  id := fun n => Int.ceil_intCast n

-- Helper lemmas for nearest-even rounding monotonicity proof
/--
Basic bounds for nearest-even rounding.

In words: `neural_nearest_even x` always lands in `{⌊x⌋, ⌊x⌋ + 1}`.
This is the key fact used in the monotonicity proof and in IEEE-style interval bounds.
-/
lemma neural_nearest_even_bounds (x : ℝ) :
    ⌊x⌋ ≤ neuralNearestEven x ∧ neuralNearestEven x ≤ ⌊x⌋ + 1 := by
  simp only [neuralNearestEven]
  split_ifs with h1 h2 h3
  · -- Case: x - ⌊x⌋ < 1/2, so neural_nearest_even x = ⌊x⌋
    simp
  · -- Case: x - ⌊x⌋ > 1/2, so neural_nearest_even x = ⌊x⌋ + 1
    simp
  · -- Case: x - ⌊x⌋ = 1/2 and ⌊x⌋ is even, so neural_nearest_even x = ⌊x⌋
    simp
  · -- Case: x - ⌊x⌋ = 1/2 and ⌊x⌋ is odd, so neural_nearest_even x = ⌊x⌋ + 1
    simp

/--
Nearest-even rounds down when the fractional part is strictly less than `1/2`.

In words: if `x` is closer to `⌊x⌋` than to `⌊x⌋+1`, then it rounds to `⌊x⌋`.
-/
lemma neural_nearest_even_eq_floor_of_frac_lt_half (x : ℝ) (h : x - ⌊x⌋ < 1/2) :
    neuralNearestEven x = ⌊x⌋ := by
  simp only [neuralNearestEven, h, if_true]

/--
Nearest-even rounds up when the fractional part is strictly greater than `1/2`.

In words: if `x` is closer to `⌊x⌋+1` than to `⌊x⌋`, then it rounds to `⌊x⌋+1`.
-/
lemma neural_nearest_even_eq_ceil_of_frac_gt_half (x : ℝ) (h : x - ⌊x⌋ > 1/2) :
    neuralNearestEven x = ⌊x⌋ + 1 := by
  simp only [neuralNearestEven]
  have h1 : ¬(x - ⌊x⌋ < 1/2) := not_lt.mpr (le_of_lt h)
  simp only [h1, if_false, h, if_true]

/--
Nearest-even tie-breaking: when the fractional part is exactly `1/2` and the floor is even,
round down to the even integer.
-/
lemma neural_nearest_even_eq_floor_of_frac_half_even (x : ℝ) (h1 : x - ⌊x⌋ = 1/2) (h2 : Even ⌊x⌋) :
    neuralNearestEven x = ⌊x⌋ := by
  simp only [neuralNearestEven]
  have h3 : ¬(x - ⌊x⌋ < 1/2) := by rw [h1]; norm_num
  have h4 : ¬(x - ⌊x⌋ > 1/2) := by rw [h1]; norm_num
  simp only [h3, h4, if_false, h2, if_true]

/--
Nearest-even tie-breaking: when the fractional part is exactly `1/2` and the floor is odd,
round up to the even integer.
-/
lemma neural_nearest_even_eq_ceil_of_frac_half_odd (x : ℝ) (h1 : x - ⌊x⌋ = 1/2) (h2 : ¬Even ⌊x⌋) :
    neuralNearestEven x = ⌊x⌋ + 1 := by
  simp only [neuralNearestEven]
  have h3 : ¬(x - ⌊x⌋ < 1/2) := by rw [h1]; norm_num
  have h4 : ¬(x - ⌊x⌋ > 1/2) := by rw [h1]; norm_num
  simp only [h3, h4, if_false, h2, if_false]

-- Instance for nearest even rounding with complete mathematical foundation
/-- `neural_nearest_even` is a valid rounding mode (monotone and fixes integers). -/
instance : NeuralValidRnd neuralNearestEven where
  monotone := by
    intros x y h
    -- Monotonicity of nearest-even rounding.
    --
    -- Idea: nearest-even always lands in `{⌊t⌋, ⌊t⌋+1}`. If `x` and `y` are in the same unit
    -- interval,
    -- compare their fractional parts and check the tie-breaking. If they are in different
    -- intervals,
    -- the `{⌊t⌋, ⌊t⌋+1}` bounds are enough to finish.

    -- First establish bounds for both values
    have bounds_x := neural_nearest_even_bounds x
    have bounds_y := neural_nearest_even_bounds y

    -- Case analysis: either ⌊x⌋ = ⌊y⌋ or ⌊x⌋ < ⌊y⌋
    by_cases h_floors : ⌊x⌋ = ⌊y⌋

    -- CASE 1: Same unit interval (⌊x⌋ = ⌊y⌋)
    · -- Since x ≤ y and ⌊x⌋ = ⌊y⌋, we have x - ⌊x⌋ ≤ y - ⌊y⌋
      have frac_order : x - ⌊x⌋ ≤ y - ⌊y⌋ := by
        rw [h_floors]
        linarith [h]

      -- Define fractional parts for clarity
      let fx := x - ⌊x⌋
      let fy := y - ⌊y⌋
      have fx_def : fx = x - ⌊x⌋ := rfl
      have fy_def : fy = y - ⌊y⌋ := rfl
      have frac_le : fx ≤ fy := by rw [fx_def, fy_def]; exact frac_order

      -- Subcase analysis based on fractional parts
      by_cases hx_half : fx < 1/2
      · -- Subcase 1.1: fx < 1/2
        by_cases hy_half : fy < 1/2
        · -- Both fractional parts < 1/2 → both round down
          rw [neural_nearest_even_eq_floor_of_frac_lt_half x (by rwa [←fx_def])]
          rw [neural_nearest_even_eq_floor_of_frac_lt_half y (by rwa [←fy_def])]
          rw [h_floors]
        · -- fx < 1/2, fy ≥ 1/2 → x rounds down, y rounds up or stays
          push Not at hy_half
          rw [neural_nearest_even_eq_floor_of_frac_lt_half x (by rwa [←fx_def])]
          by_cases hy_gt_half : fy > 1/2
          · -- fy > 1/2 → y rounds up
            rw [neural_nearest_even_eq_ceil_of_frac_gt_half y (by rwa [←fy_def])]
            rw [h_floors]
            simp
          · -- fy = 1/2 → y uses tie-breaking
            push Not at hy_gt_half
            have fy_eq_half : fy = 1/2 := le_antisymm hy_gt_half hy_half
            by_cases hy_even : Even ⌊y⌋
            · -- y rounds down (to even)
              rw [neural_nearest_even_eq_floor_of_frac_half_even y (by rwa [←fy_def]) hy_even]
              rw [h_floors]
            · -- y rounds up (to even)
              rw [neural_nearest_even_eq_ceil_of_frac_half_odd y (by rwa [←fy_def]) hy_even]
              rw [h_floors]
              simp

      · -- Subcase 1.2: fx ≥ 1/2
        push Not at hx_half
        by_cases hx_gt_half : fx > 1/2
        · -- fx > 1/2 → x rounds up
          by_cases hy_half : fy < 1/2
          · -- This is impossible: fx > 1/2 but fy < 1/2 contradicts fx ≤ fy
            exfalso
            linarith [hx_gt_half, hy_half, frac_le]
          · -- fy ≥ 1/2
            push Not at hy_half
            by_cases hy_gt_half : fy > 1/2
            · -- Both > 1/2 → both round up
              rw [neural_nearest_even_eq_ceil_of_frac_gt_half x (by rwa [←fx_def])]
              rw [neural_nearest_even_eq_ceil_of_frac_gt_half y (by rwa [←fy_def])]
              rw [h_floors]
            · -- fx > 1/2, fy = 1/2 → impossible since fx ≤ fy
              push Not at hy_gt_half
              have fy_eq_half : fy = 1/2 := le_antisymm hy_gt_half hy_half
              exfalso
              linarith [hx_gt_half, fy_eq_half, frac_le]

        · -- fx = 1/2 (tie-breaking case)
          push Not at hx_gt_half
          have fx_eq_half : fx = 1/2 := le_antisymm hx_gt_half hx_half
          by_cases hy_half : fy < 1/2
          · -- Impossible: fx = 1/2 but fy < 1/2 contradicts fx ≤ fy
            exfalso
            linarith [fx_eq_half, hy_half, frac_le]
          · -- fy ≥ 1/2
            push Not at hy_half
            by_cases hy_gt_half : fy > 1/2
            · -- fx = 1/2, fy > 1/2 → x uses tie-breaking, y rounds up
              by_cases hx_even : Even ⌊x⌋
              · -- x rounds down (to even), y rounds up
                rw [neural_nearest_even_eq_floor_of_frac_half_even x (by rwa [←fx_def]) hx_even]
                rw [neural_nearest_even_eq_ceil_of_frac_gt_half y (by rwa [←fy_def])]
                rw [h_floors]
                simp
              · -- x rounds up (to even), y rounds up
                rw [neural_nearest_even_eq_ceil_of_frac_half_odd x (by rwa [←fx_def]) hx_even]
                rw [neural_nearest_even_eq_ceil_of_frac_gt_half y (by rwa [←fy_def])]
                rw [h_floors]
            · -- Both = 1/2 → both use same tie-breaking rule
              push Not at hy_gt_half
              have fy_eq_half : fy = 1/2 := le_antisymm hy_gt_half hy_half
              by_cases hx_even : Even ⌊x⌋
              · -- Both round down (both floors are even since ⌊x⌋ = ⌊y⌋)
                have hy_even : Even ⌊y⌋ := by rwa [←h_floors]
                rw [neural_nearest_even_eq_floor_of_frac_half_even x (by rwa [←fx_def]) hx_even]
                rw [neural_nearest_even_eq_floor_of_frac_half_even y (by rwa [←fy_def]) hy_even]
                rw [h_floors]
              · -- Both round up (both floors are odd since ⌊x⌋ = ⌊y⌋)
                have hy_odd : ¬Even ⌊y⌋ := by rwa [←h_floors]
                rw [neural_nearest_even_eq_ceil_of_frac_half_odd x (by rwa [←fx_def]) hx_even]
                rw [neural_nearest_even_eq_ceil_of_frac_half_odd y (by rwa [←fy_def]) hy_odd]
                rw [h_floors]

    -- CASE 2: Different intervals (⌊x⌋ < ⌊y⌋)
    · -- Since ⌊x⌋ ≠ ⌊y⌋ and x ≤ y, we must have ⌊x⌋ < ⌊y⌋
      have floor_lt : ⌊x⌋ < ⌊y⌋ := by
        exact Int.lt_iff_le_and_ne.mpr ⟨Int.floor_mono h, h_floors⟩

      -- Use the bounds: neural_nearest_even x ≤ ⌊x⌋ + 1 ≤ ⌊y⌋ ≤ neural_nearest_even y
      have key_ineq : ⌊x⌋ + 1 ≤ ⌊y⌋ := by
        exact Int.add_one_le_iff.mpr floor_lt

      -- Chain the inequalities
      calc neuralNearestEven x
        ≤ ⌊x⌋ + 1     := bounds_x.2
        _ ≤ ⌊y⌋       := key_ineq
        _ ≤ neuralNearestEven y := bounds_y.1
  id := by
    intro n
    simp only [neuralNearestEven]
    -- For integers, the fractional part is 0, so we round to the integer itself
    have h : (n : ℝ) - ⌊(n : ℝ)⌋ = 0 := by simp
    simp only [h]
    -- Since 0 < 1/2, we take the first branch
    norm_num

/--
Nearest-even is a “round-to-nearest” mode in the integer sense: `|rnd(x) - x| ≤ 1/2`.

This is the key property required by `NeuralValidRndToNearest`.
-/
lemma neural_nearest_even_abs_sub_le_half (x : ℝ) :
    abs ((neuralNearestEven x : ℝ) - x) ≤ (2⁻¹ : ℝ) := by
  simp [neuralNearestEven]
  split_ifs with hlt hgt heven
  · -- frac < 1/2, result = floor
    have hf_le : (⌊x⌋ : ℝ) ≤ x := Int.floor_le x
    have habs : abs ((⌊x⌋ : ℝ) - x) = x - (⌊x⌋ : ℝ) := by
      have : (⌊x⌋ : ℝ) - x ≤ 0 := sub_nonpos.mpr hf_le
      simpa [sub_eq_add_neg] using (abs_of_nonpos this)
    have hle : x - (⌊x⌋ : ℝ) ≤ (2⁻¹ : ℝ) := le_of_lt (by simpa using hlt)
    simpa [habs] using hle
  · -- frac > 1/2, result = floor+1
    have hx_lt : x < (⌊x⌋ : ℝ) + 1 := by
      have : x - (⌊x⌋ : ℝ) < 1 := Int.fract_lt_one x
      linarith
    have habs : abs ((⌊x⌋ : ℝ) + 1 - x) = (⌊x⌋ : ℝ) + 1 - x := by
      have : 0 ≤ (⌊x⌋ : ℝ) + 1 - x := sub_nonneg.mpr (le_of_lt hx_lt)
      simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using (abs_of_nonneg this)
    have hrewrite : (⌊x⌋ : ℝ) + 1 - x = 1 - (x - (⌊x⌋ : ℝ)) := by ring
    have hgt' : (2⁻¹ : ℝ) < x - (⌊x⌋ : ℝ) := by simpa using hgt
    have hlt' : 1 - (x - (⌊x⌋ : ℝ)) < (2⁻¹ : ℝ) := by linarith
    have hle' : 1 - (x - (⌊x⌋ : ℝ)) ≤ (2⁻¹ : ℝ) := le_of_lt hlt'
    have : abs ((⌊x⌋ : ℝ) + 1 - x) ≤ (2⁻¹ : ℝ) := by
      simpa [habs, hrewrite] using hle'
    simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using this
  · -- tie, even floor -> floor
    have hge : (2⁻¹ : ℝ) ≤ x - (⌊x⌋ : ℝ) := le_of_not_gt (by simpa using hlt)
    have hle : x - (⌊x⌋ : ℝ) ≤ (2⁻¹ : ℝ) := le_of_not_gt (by simpa using hgt)
    have h_eq : x - (⌊x⌋ : ℝ) = (2⁻¹ : ℝ) := le_antisymm hle hge
    have hf_le : (⌊x⌋ : ℝ) ≤ x := Int.floor_le x
    have habs : abs ((⌊x⌋ : ℝ) - x) = x - (⌊x⌋ : ℝ) := by
      have : (⌊x⌋ : ℝ) - x ≤ 0 := sub_nonpos.mpr hf_le
      simpa [sub_eq_add_neg] using (abs_of_nonpos this)
    simp [habs, h_eq]
  · -- tie, odd floor -> floor+1
    have hge : (2⁻¹ : ℝ) ≤ x - (⌊x⌋ : ℝ) := le_of_not_gt (by simpa using hlt)
    have hle : x - (⌊x⌋ : ℝ) ≤ (2⁻¹ : ℝ) := le_of_not_gt (by simpa using hgt)
    have h_eq : x - (⌊x⌋ : ℝ) = (2⁻¹ : ℝ) := le_antisymm hle hge
    have hhalf : 1 - (x - (⌊x⌋ : ℝ)) = (2⁻¹ : ℝ) := by
      calc
        1 - (x - (⌊x⌋ : ℝ)) = 1 - (2⁻¹ : ℝ) := by simp [h_eq]
        _ = (2⁻¹ : ℝ) := by norm_num
    have hval : (⌊x⌋ : ℝ) + 1 - x = (2⁻¹ : ℝ) := by
      have : (⌊x⌋ : ℝ) + 1 - x = 1 - (x - (⌊x⌋ : ℝ)) := by ring
      simpa [this] using hhalf
    calc
      abs ((⌊x⌋ : ℝ) + 1 - x) = abs (2⁻¹ : ℝ) := by simp [hval]
      _ = (2⁻¹ : ℝ) := by simp
      _ ≤ (2⁻¹ : ℝ) := le_rfl

/-- `neural_nearest_even` satisfies the half-unit error bound `|rnd x - x| ≤ 1/2`. -/
instance : NeuralValidRndToNearest neuralNearestEven where
  monotone := NeuralValidRnd.monotone (rnd := neuralNearestEven)
  id := NeuralValidRnd.id (rnd := neuralNearestEven)
  abs_sub_le_half := neural_nearest_even_abs_sub_le_half

/--
Core rounding operator (“compute in `ℝ`, then round back to the grid”).

We build a `NeuralFloat` record and then interpret it via `neural_to_real`.
Only `mantissa`/`exponent` affect the value; the metadata fields are set to conventional defaults.
-/
noncomputable def neuralRound (rnd : ℝ → ℤ) (x : ℝ) : ℝ :=
  neuralToReal (β := β) { mantissa := rnd (neuralScaledMantissa β fexp x),
                            exponent := neuralCexp β fexp x,
                            precision := NeuralPrecision.ieee_single,
                            error_bound := 0,
                            phase := TrainingPhase.forward,
                            layer_id := 0,
                            batch_id := 0 }

-- Proof-only helper: if `x` is in generic format, then its scaled mantissa is an integer.
/--
If `x` is already in the generic format grid, then `neural_scaled_mantissa β fexp x` is an integer.

In words: exact representability means “no fractional bits at the chosen exponent scale”.
-/
private lemma neural_scaled_mantissa_int_of_generic (x : ℝ) (hx : neuralGenericFormat β fexp x) :
    ∃ n : ℤ, neuralScaledMantissa β fexp x = n := by
  -- From neural_generic_format definition, we have:
  -- x = neural_to_real { mantissa := ⌊neural_scaled_mantissa β fexp x⌋, exponent := neural_cexp β
  -- fexp x, ... }
  -- This means: x = ⌊neural_scaled_mantissa β fexp x⌋ * neural_bpow β (neural_cexp β fexp x)

  -- Let's call the floor of the scaled mantissa 'm' and the canonical exponent 'e'
  let m := ⌊neuralScaledMantissa β fexp x⌋
  let e := neuralCexp β fexp x
  use m

  -- From hx, we know that x can be written as m * neural_bpow β e
  have h_repr : x = m * neuralBpow β e := by
    simp only [neuralGenericFormat, neuralToReal] at hx
    exact hx

  -- Now, by definition of neural_scaled_mantissa:
  -- neural_scaled_mantissa β fexp x = x * neural_bpow β (-neural_cexp β fexp x)
  -- Since neural_cexp β fexp x = e, we have:
  -- neural_scaled_mantissa β fexp x = x * neural_bpow β (-e)

  calc neuralScaledMantissa β fexp x
    = x * neuralBpow β (-neuralCexp β fexp x)             := by rfl
    _ = x * neuralBpow β (-e)                              := by rfl
    _ = (m * neuralBpow β e) * neuralBpow β (-e)          := by rw [h_repr]
    _ = m * (neuralBpow β e * neuralBpow β (-e))          := by rw [mul_assoc]
    _ = m * neuralBpow β (e + (-e))                        := by rw [← neuralBpow.add_exp]
    _ = m * neuralBpow β 0                                 := by simp [add_neg_cancel]
    _ = m * 1                                               := by simp [neuralBpow,
      NeuralRadix.toReal, zpow_zero]
    _ = m                                                   := by rw [mul_one]
    _ = ↑m                                                  := by simp

/--
Rounding preserves exactly-representable numbers.

In words: if `x` lies on the grid described by `(β,fexp)` (`neural_generic_format`), then
rounding it with any valid `rnd` is a no-op.

This is the Flocq-style “round_generic” lemma.
-/
@[simp] theorem neural_round_preserves_generic (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ)
    (hx : neuralGenericFormat β fexp x) :
    neuralRound (β := β) (fexp := fexp) rnd x = x := by
  -- This is Flocq's round_generic theorem
  -- The key insight: if x is in generic format, then rounding doesn't change it
  -- because the scaled mantissa is already an integer
  simp only [neuralRound]

  -- Get that the scaled mantissa is an integer
  obtain ⟨n, hn⟩ := neural_scaled_mantissa_int_of_generic x hx

  -- Since the scaled mantissa equals an integer n, rounding doesn't change it
  have h_rnd : rnd (neuralScaledMantissa β fexp x) = n := by
    rw [hn]
    exact NeuralValidRnd.id n

  -- Also, we have n = ⌊neural_scaled_mantissa β fexp x⌋
  have h_floor : n = ⌊neuralScaledMantissa β fexp x⌋ := by
    rw [hn]
    simp only [Int.floor_intCast]

  -- Therefore rnd (neural_scaled_mantissa β fexp x) = ⌊neural_scaled_mantissa β fexp x⌋
  rw [h_rnd, h_floor]

  -- Now the result is exactly what neural_generic_format says x equals
  simp only [neuralGenericFormat] at hx
  exact hx.symm

/--
Scaled mantissa times base power equals the original value.

In words: `scaled_mantissa(x) * β^{cexp(x)} = x`.
This is the algebraic identity that justifies the scaling used before rounding.
-/
lemma neural_scaled_mantissa_mul_bpow (x : ℝ) :
    neuralScaledMantissa β fexp x * neuralBpow β (neuralCexp β fexp x) = x := by
  simp only [neuralScaledMantissa]
  rw [mul_assoc, ← neuralBpow.add_exp]
  simp only [neuralBpow]
  simp [NeuralRadix.toReal, zpow_zero, mul_one]

/--
Half-ULP error bound for `neural_round` under round-to-nearest.

This is the basic “one-step” bound used by most error propagation arguments:
`neural_round` deviates from `x` by at most half an ulp at the chosen exponent scale.
-/
theorem neural_error_bound_ulp (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd x - x) ≤ neuralUlp β fexp x TrainingPhase.forward
      / 2 := by
  by_cases hx : x = 0
  · subst hx
    simp [neuralUlp, neuralRound, neuralScaledMantissa, neuralCexp, neuralMagnitude,
      neuralToReal]
    have hr0 : rnd 0 = 0 := by simpa using (NeuralValidRnd.id (rnd := rnd) (0 : ℤ))
    simp [hr0]
    exact div_nonneg (neuralBpow.nonneg β (fexp 0)) (by norm_num)
  · have hphase : TrainingPhase.forward.requiresHighPrecision = false := by rfl
    simp [neuralUlp, hx, hphase]
    simp [neuralRound, neuralToReal]
    set s : ℝ := neuralScaledMantissa β fexp x
    set e : ℤ := neuralCexp β fexp x
    have hxrepr : s * neuralBpow β e = x := by
      subst s e
      simpa using (neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x)
    rw [← hxrepr]
    rw [← sub_mul]
    rw [abs_mul]
    have hbpos : 0 < neuralBpow β e := neuralBpow.pos β e
    simp [abs_of_pos hbpos]
    have h := NeuralValidRndToNearest.abs_sub_le_half (rnd := rnd) s
    have hbnonneg : 0 ≤ neuralBpow β e := le_of_lt hbpos
    have hmul := mul_le_mul_of_nonneg_right h hbnonneg
    simpa [div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm, mul_right_comm] using hmul

end TorchLean.Floats
