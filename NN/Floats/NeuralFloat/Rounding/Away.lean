/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding.Core
public import NN.Floats.NeuralFloat.Rounding.Predicates
public import NN.Floats.NeuralFloat.Rounding.Order

/-!
# Rounding Away from Zero

Away-from-zero rounding uses ceiling on nonnegative inputs and floor on negative inputs.  Together
with toward-zero rounding, it supplies the second endpoint decomposition for arbitrary valid
integer rounding modes.
-/

@[expose] public section

namespace TorchLean.Floats

/-- Integer rounding away from zero. -/
noncomputable def neuralAwayRound (x : ℝ) : ℤ :=
  if x < 0 then ⌊x⌋ else ⌈x⌉

/-- Away-from-zero agrees with floor on negative inputs. -/
theorem neuralAwayRound_eq_floor {x : ℝ} (hx : x < 0) :
    neuralAwayRound x = neuralFloorRound x := by
  simp [neuralAwayRound, neuralFloorRound, hx]

/-- Away-from-zero agrees with ceiling on nonnegative inputs. -/
theorem neuralAwayRound_eq_ceil {x : ℝ} (hx : 0 ≤ x) :
    neuralAwayRound x = neuralCeilRound x := by
  simp [neuralAwayRound, neuralCeilRound, not_lt.mpr hx]

/-- On nonpositive inputs, away-from-zero agrees with floor, including at zero. -/
theorem neuralAwayRound_eq_floor_of_nonpos {x : ℝ} (hx : x ≤ 0) :
    neuralAwayRound x = neuralFloorRound x := by
  rcases hx.eq_or_lt with rfl | hx
  · simp [neuralAwayRound, neuralFloorRound]
  · exact neuralAwayRound_eq_floor hx

/-- Away-from-zero rounding is monotone and fixes integers. -/
instance neuralAwayRoundValid : NeuralValidRnd neuralAwayRound where
  monotone := by
    intro x y hxy
    by_cases hx : x < 0
    · by_cases hy : y < 0
      · rw [neuralAwayRound_eq_floor hx, neuralAwayRound_eq_floor hy]
        exact Int.floor_mono hxy
      · rw [neuralAwayRound_eq_floor hx, neuralAwayRound_eq_ceil (le_of_not_gt hy)]
        have hleft : neuralFloorRound x ≤ 0 := Int.floor_nonpos hx.le
        have hright : 0 ≤ neuralCeilRound y := Int.ceil_nonneg (le_of_not_gt hy)
        exact hleft.trans hright
    · have hx0 : 0 ≤ x := le_of_not_gt hx
      have hy0 : 0 ≤ y := hx0.trans hxy
      rw [neuralAwayRound_eq_ceil hx0, neuralAwayRound_eq_ceil hy0]
      exact Int.ceil_mono hxy
  id := by
    intro n
    by_cases hn : n < 0
    · simp [neuralAwayRound, hn]
    · simp [neuralAwayRound, hn]

/-- Arbitrary valid integer rounding chooses toward-zero or away-from-zero. -/
theorem neural_valid_round_eq_trunc_or_away (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) :
    rnd x = neuralTruncRound x ∨ rnd x = neuralAwayRound x := by
  rcases neural_valid_round_eq_floor_or_ceil rnd x with hfloor | hceil
  · by_cases hx : x < 0
    · exact Or.inr (hfloor.trans (neuralAwayRound_eq_floor hx).symm)
    · exact Or.inl (hfloor.trans (neuralTruncRound_eq_floor (le_of_not_gt hx)).symm)
  · by_cases hx : 0 ≤ x
    · exact Or.inr (hceil.trans (neuralAwayRound_eq_ceil hx).symm)
    · exact Or.inl (hceil.trans (neuralTruncRound_eq_ceil (le_of_not_ge hx)).symm)

/-- Away-from-zero uses upward rounding for nonnegative values and downward rounding otherwise. -/
def NeuralRoundAwayFromZeroPoint (F : ℝ → Prop) (x f : ℝ) : Prop :=
  (0 ≤ x → NeuralRoundUpPoint F x f) ∧ (x ≤ 0 → NeuralRoundDownPoint F x f)

/-- Generic away-from-zero rounding satisfies its semantic point specification. -/
theorem neuralRound_away_point {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
    (x : ℝ) :
    NeuralRoundAwayFromZeroPoint (neuralGenericFormat β fexp) x
      (neuralRound (β := β) (fexp := fexp) neuralAwayRound x) := by
  constructor
  · intro hx
    have hs : 0 ≤ neuralScaledMantissa β fexp x := by
      rw [neuralScaledMantissa_eq_div]
      exact div_nonneg hx (neuralBpow.nonneg β _)
    have hr : neuralRound (β := β) (fexp := fexp) neuralAwayRound x =
        neuralRound (β := β) (fexp := fexp) neuralCeilRound x := by
      unfold neuralRound
      rw [neuralAwayRound_eq_ceil hs]
    rw [hr]
    exact neuralRound_ceil_point x
  · intro hx
    have hs : neuralScaledMantissa β fexp x ≤ 0 := by
      rw [neuralScaledMantissa_eq_div]
      exact div_nonpos_of_nonpos_of_nonneg hx (neuralBpow.nonneg β _)
    have hr : neuralRound (β := β) (fexp := fexp) neuralAwayRound x =
        neuralRound (β := β) (fexp := fexp) neuralFloorRound x := by
      unfold neuralRound
      rw [neuralAwayRound_eq_floor_of_nonpos hs]
    rw [hr]
    exact neuralRound_floor_point x

end TorchLean.Floats
