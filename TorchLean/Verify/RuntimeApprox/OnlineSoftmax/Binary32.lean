/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.OnlineSoftmax
public meta import TorchLean.Tensor.Context
public meta import TorchLean.Tensor.Layers.Softmax

/-!
# Online softmax bounds at binary32 sizes

These theorems evaluate the closed-form bounds of `TorchLean.Verify.RuntimeApprox.OnlineSoftmax` for
`binary32` precision at 4096 and 16384 keys, with the stored-weight bounds of the recipe that sums
its stored weights (`stored_4096_128`) and of FlashAttention-2's recipe with `bfloat16` weights
(`flash2_4096_128`). Each numeric theorem states that the numerical budget and range premises of the
closed-form theorem hold at those sizes, for values of magnitude at most one, and bounds its
conclusion. `online_4096_128_output`, `stored_4096_128_output` and `flash2_4096_128_output` combine
them with the closed-form theorems into bounds on the computed output. `binary32` meets their format
hypotheses (`unitRoundoff_binary32`, `underflow_binary32`, `maxFinite_binary32`). The exponentials
and logarithms are bounded by `exp x ≤ 1 / (1 - x)` and `log 2 < 0.6931471808`, and `norm_num`
decides the remaining rational inequalities in the kernel.
-/

public section

namespace TorchLean.OnlineSoftmax

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open Softmax (onlineAverage)

/-- The largest finite `binary32` value is at least `2 ^ 100`. -/
theorem maxFinite_binary32 :
    (2 : ℝ) ^ 100 ≤ Model.toReal (Model.posMaxFinite FloatFormat.binary32) := by
  rw [Model.toReal_posMaxFinite]
  have hexp : FloatFormat.binary32.maxFiniteExpField = 254 := by decide
  have hmask : FloatFormat.binary32.fracMaskNat = 8388607 := by decide
  have hfrac : FloatFormat.binary32.fracWidth = 23 := rfl
  have hbias : FloatFormat.binary32.exponentBias = 127 := rfl
  have henc : FloatFormat.binary32.encoding = FloatFormat.Encoding.ieee := rfl
  norm_num [hfrac, hbias, henc, Model.bpow, FloatFormat.maxNormalExponent,
    FloatFormat.maxFiniteFracField, Model.pow2, hexp, hmask, Formats.Flocq.bpow,
    FloatLib.Numerics.binaryRadix, FloatLib.Numerics.Radix.toReal]

/-- `log (2 ^ k)` is at most `k * 0.6931471808`. -/
theorem log_two_pow_le (k : ℕ) : Real.log ((2 ^ k : ℕ) : ℝ) ≤ k * 0.6931471808 := by
  push_cast
  rw [Real.log_pow]
  exact mul_le_mul_of_nonneg_left Real.log_two_lt_d9.le (Nat.cast_nonneg k)

/-- The unit roundoff of `binary32` is `2 ^ (-24)`. -/
theorem unitRoundoff_binary32 : Model.unitRoundoffAt FloatFormat.binary32 = 2 ^ (-24 : ℤ) := by
  rw [Model.unitRoundoffAt_eq]; rfl

/-- The underflow allowance of `binary32` is `2 ^ (-150)`. -/
theorem underflow_binary32 : Model.underflowErrorAt FloatFormat.binary32 = 2 ^ (-150 : ℤ) := by
  rw [Model.underflowErrorAt_eq]; rfl

/-- The unit roundoff of `bfloat16` is `2 ^ (-8)`. -/
theorem unitRoundoff_bfloat16 : Model.unitRoundoffAt FloatFormat.bfloat16 = 2 ^ (-8 : ℤ) := by
  rw [Model.unitRoundoffAt_eq]; rfl

/-- The underflow allowance of `bfloat16` is `2 ^ (-134)`. -/
theorem underflow_bfloat16 : Model.underflowErrorAt FloatFormat.bfloat16 = 2 ^ (-134 : ℤ) := by
  rw [Model.underflowErrorAt_eq]; rfl

/-- Online softmax, 4096 keys in blocks of 128, weights kept in `binary32`, with values of magnitude
at most one, for a working format with the unit roundoff and underflow allowance of `binary32` and
largest finite value at least `2 ^ 100`: the numerical budget and range premises of the closed form
hold and its bound is at most 2.9e-5. -/
theorem online_4096_128 {u η mx : ℝ} (hu : u = 2 ^ (-24 : ℤ)) (hη : η = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ mx) :
    let uL : ℝ := 0
    let ηL : ℝ := 0
    let cl := u * ((128 : ℕ) + 3 * (32 : ℕ) - 1)
    let co := u * ((128 : ℕ) + 3 * (32 : ℕ)) + uL
    let al := (Real.exp cl - 1) + 2 * u * Real.exp cl * Real.log (4096 : ℕ)
    let ao := (Real.exp co - 1) + 2 * u * Real.exp co * Real.log (4096 : ℕ)
    let ρl := (32 : ℕ) * residualGrowth u η ^ 32 * blockResidual u η 4096 128 1 cl η
    let ρo := (32 : ℕ) * residualGrowth u η ^ 32 *
      blockResidual u η 4096 128 1 co (productResidual u η uL ηL 1)
    al + ρl < 1 ∧ ((4096 : ℕ) : ℝ) * 1 * Real.exp cl + ρl ≤ mx ∧
      ((4096 : ℕ) : ℝ) * 1 * Real.exp co + ρo ≤ mx ∧ (1 + 1 * ao + ρo) / (1 - al - ρl) ≤ mx ∧
      (1 + u) * ((1 * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * 1 + η ≤ 29 / 1000000 := by
  intro uL ηL cl co al ao ρl ρo
  have huL : uL = 0 := rfl
  have hηL : ηL = 0 := rfl
  have hcl : cl = 223 * 2 ^ (-24 : ℤ) := by simp only [cl, hu]; push_cast; ring
  have hco : co = 224 * 2 ^ (-24 : ℤ) := by simp only [co, hu, huL]; push_cast; ring
  have hcl0 : 0 ≤ cl := by rw [hcl]; positivity
  have hco0 : 0 ≤ co := by rw [hco]; positivity
  have hcl1 : cl < 1 := by rw [hcl]; norm_num
  have hco1 : co < 1 := by rw [hco]; norm_num
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  have hu1 : u < 1 := by rw [hu]; norm_num
  have key := closed_form_le_of_bounds (u := u) (η := η) (V := 1) (uL := uL) (ηL := ηL)
    (N := 4096) (B := 128) (T := 32) (Pw := (1 + u) ^ (128 + 1))
    hu0 (by rw [hη]; positivity) zero_le_one (le_rfl) (le_rfl)
    hcl0 hco0 (Real.exp_bound_div_one_sub_of_interval hcl0 hcl1)
    (Real.exp_bound_div_one_sub_of_interval hco0 hco1)
    (Real.exp_bound_div_one_sub_of_interval hu0 hu1) (log_two_pow_le 12) le_rfl
  have hmax := hmx
  obtain ⟨h1, h2, h3, h4, h5⟩ := key (by rw [hcl, hu, hη]; norm_num)
  refine ⟨h1, h2.trans (le_trans (by rw [hcl, hu, hη]; norm_num) hmax),
    h3.trans (le_trans (by unfold productResidual; rw [hco, hu, hη, huL, hηL]; norm_num) hmax),
    h4.trans (le_trans (by unfold productResidual; rw [hcl, hco, hu, hη, huL, hηL]; norm_num)
      hmax), h5.trans ?_⟩
  unfold productResidual; rw [hcl, hco, hu, hη, huL, hηL]; norm_num

/-- Online softmax, 4096 keys in blocks of 128, weights stored in `bfloat16`, with values of
magnitude at most one, for a working format with the unit roundoff and underflow allowance of
`binary32` and largest finite value at least `2 ^ 100`: the numerical budget and range premises of
the closed form hold and its bound is at most 0.004. -/
theorem online_4096_128_bfloat16 {u η mx : ℝ} (hu : u = 2 ^ (-24 : ℤ)) (hη : η = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ mx) :
    let uL : ℝ := Model.unitRoundoffAt FloatFormat.bfloat16
    let ηL : ℝ := Model.underflowErrorAt FloatFormat.bfloat16
    let cl := u * ((128 : ℕ) + 3 * (32 : ℕ) - 1)
    let co := u * ((128 : ℕ) + 3 * (32 : ℕ)) + uL
    let al := (Real.exp cl - 1) + 2 * u * Real.exp cl * Real.log (4096 : ℕ)
    let ao := (Real.exp co - 1) + 2 * u * Real.exp co * Real.log (4096 : ℕ)
    let ρl := (32 : ℕ) * residualGrowth u η ^ 32 * blockResidual u η 4096 128 1 cl η
    let ρo := (32 : ℕ) * residualGrowth u η ^ 32 *
      blockResidual u η 4096 128 1 co (productResidual u η uL ηL 1)
    al + ρl < 1 ∧ ((4096 : ℕ) : ℝ) * 1 * Real.exp cl + ρl ≤ mx ∧
      ((4096 : ℕ) : ℝ) * 1 * Real.exp co + ρo ≤ mx ∧ (1 + 1 * ao + ρo) / (1 - al - ρl) ≤ mx ∧
      (1 + u) * ((1 * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * 1 + η ≤ 4 / 1000 := by
  intro uL ηL cl co al ao ρl ρo
  have huL : uL = 2 ^ (-8 : ℤ) := unitRoundoff_bfloat16
  have hηL : ηL = 2 ^ (-134 : ℤ) := underflow_bfloat16
  have hcl : cl = 223 * 2 ^ (-24 : ℤ) := by simp only [cl, hu]; push_cast; ring
  have hco : co = 224 * 2 ^ (-24 : ℤ) + 2 ^ (-8 : ℤ) := by simp only [co, hu, huL]; push_cast; ring
  have hcl0 : 0 ≤ cl := by rw [hcl]; positivity
  have hco0 : 0 ≤ co := by rw [hco]; positivity
  have hcl1 : cl < 1 := by rw [hcl]; norm_num
  have hco1 : co < 1 := by rw [hco]; norm_num
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  have hu1 : u < 1 := by rw [hu]; norm_num
  have key := closed_form_le_of_bounds (u := u) (η := η) (V := 1) (uL := uL) (ηL := ηL)
    (N := 4096) (B := 128) (T := 32) (Pw := (1 + u) ^ (128 + 1))
    hu0 (by rw [hη]; positivity) zero_le_one (by rw [huL]; positivity) (by rw [hηL]; positivity)
    hcl0 hco0 (Real.exp_bound_div_one_sub_of_interval hcl0 hcl1)
    (Real.exp_bound_div_one_sub_of_interval hco0 hco1)
    (Real.exp_bound_div_one_sub_of_interval hu0 hu1) (log_two_pow_le 12) le_rfl
  have hmax := hmx
  obtain ⟨h1, h2, h3, h4, h5⟩ := key (by rw [hcl, hu, hη]; norm_num)
  refine ⟨h1, h2.trans (le_trans (by rw [hcl, hu, hη]; norm_num) hmax),
    h3.trans (le_trans (by unfold productResidual; rw [hco, hu, hη, huL, hηL]; norm_num) hmax),
    h4.trans (le_trans (by unfold productResidual; rw [hcl, hco, hu, hη, huL, hηL]; norm_num)
      hmax), h5.trans ?_⟩
  unfold productResidual; rw [hcl, hco, hu, hη, huL, hηL]; norm_num

/-- The blocked two-pass softmax, 4096 keys in blocks of 128, with values of magnitude at most one,
for a working format with the unit roundoff and underflow allowance of `binary32` and largest finite
value at least `2 ^ 100`: the numerical budget and range premises of the closed form hold and its
bound is at most 2.2e-5. -/
theorem twoPass_4096_128 {u η mx : ℝ} (hu : u = 2 ^ (-24 : ℤ)) (hη : η = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ mx) :
    let uL : ℝ := 0
    let ηL : ℝ := 0
    let cl := u * ((128 : ℕ) + (32 : ℕ) + 1)
    let co := u * ((128 : ℕ) + (32 : ℕ) + 2) + uL
    let al := (Real.exp cl - 1) + 2 * u * Real.exp cl * Real.log (4096 : ℕ)
    let ao := (Real.exp co - 1) + 2 * u * Real.exp co * Real.log (4096 : ℕ)
    let ρl := (32 : ℕ) * residualGrowth u η ^ 32 * blockResidual u η 4096 128 1 cl η
    let ρo := (32 : ℕ) * residualGrowth u η ^ 32 *
      blockResidual u η 4096 128 1 co (productResidual u η uL ηL 1)
    al + ρl < 1 ∧ ((4096 : ℕ) : ℝ) * 1 * Real.exp cl + ρl ≤ mx ∧
      ((4096 : ℕ) : ℝ) * 1 * Real.exp co + ρo ≤ mx ∧ (1 + 1 * ao + ρo) / (1 - al - ρl) ≤ mx ∧
      (1 + u) * ((1 * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * 1 + η ≤ 22 / 1000000 := by
  intro uL ηL cl co al ao ρl ρo
  have huL : uL = 0 := rfl
  have hηL : ηL = 0 := rfl
  have hcl : cl = 161 * 2 ^ (-24 : ℤ) := by simp only [cl, hu]; push_cast; ring
  have hco : co = 162 * 2 ^ (-24 : ℤ) := by simp only [co, hu, huL]; push_cast; ring
  have hcl0 : 0 ≤ cl := by rw [hcl]; positivity
  have hco0 : 0 ≤ co := by rw [hco]; positivity
  have hcl1 : cl < 1 := by rw [hcl]; norm_num
  have hco1 : co < 1 := by rw [hco]; norm_num
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  have hu1 : u < 1 := by rw [hu]; norm_num
  have key := closed_form_le_of_bounds (u := u) (η := η) (V := 1) (uL := uL) (ηL := ηL)
    (N := 4096) (B := 128) (T := 32) (Pw := (1 + u) ^ (128 + 1))
    hu0 (by rw [hη]; positivity) zero_le_one (le_rfl) (le_rfl)
    hcl0 hco0 (Real.exp_bound_div_one_sub_of_interval hcl0 hcl1)
    (Real.exp_bound_div_one_sub_of_interval hco0 hco1)
    (Real.exp_bound_div_one_sub_of_interval hu0 hu1) (log_two_pow_le 12) le_rfl
  have hmax := hmx
  obtain ⟨h1, h2, h3, h4, h5⟩ := key (by rw [hcl, hu, hη]; norm_num)
  refine ⟨h1, h2.trans (le_trans (by rw [hcl, hu, hη]; norm_num) hmax),
    h3.trans (le_trans (by unfold productResidual; rw [hco, hu, hη, huL, hηL]; norm_num) hmax),
    h4.trans (le_trans (by unfold productResidual; rw [hcl, hco, hu, hη, huL, hηL]; norm_num)
      hmax), h5.trans ?_⟩
  unfold productResidual; rw [hcl, hco, hu, hη, huL, hηL]; norm_num

/-- One block of 4096 keys, the two-pass softmax with sequential sums, with values of magnitude at
most one, for a working format with the unit roundoff and underflow allowance of `binary32` and
largest finite value at least `2 ^ 100`: the numerical budget and range premises of the closed form
hold and its bound is at most 0.000492. -/
theorem online_4096_4096 {u η mx : ℝ} (hu : u = 2 ^ (-24 : ℤ)) (hη : η = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ mx) :
    let uL : ℝ := 0
    let ηL : ℝ := 0
    let cl := u * ((4096 : ℕ) + 3 * (1 : ℕ) - 1)
    let co := u * ((4096 : ℕ) + 3 * (1 : ℕ)) + uL
    let al := (Real.exp cl - 1) + 2 * u * Real.exp cl * Real.log (4096 : ℕ)
    let ao := (Real.exp co - 1) + 2 * u * Real.exp co * Real.log (4096 : ℕ)
    let ρl := (1 : ℕ) * residualGrowth u η ^ 1 * blockResidual u η 4096 4096 1 cl η
    let ρo := (1 : ℕ) * residualGrowth u η ^ 1 *
      blockResidual u η 4096 4096 1 co (productResidual u η uL ηL 1)
    al + ρl < 1 ∧ ((4096 : ℕ) : ℝ) * 1 * Real.exp cl + ρl ≤ mx ∧
      ((4096 : ℕ) : ℝ) * 1 * Real.exp co + ρo ≤ mx ∧ (1 + 1 * ao + ρo) / (1 - al - ρl) ≤ mx ∧
      (1 + u) * ((1 * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * 1 + η ≤ 492 / 1000000 := by
  intro uL ηL cl co al ao ρl ρo
  have huL : uL = 0 := rfl
  have hηL : ηL = 0 := rfl
  have hcl : cl = 4098 * 2 ^ (-24 : ℤ) := by simp only [cl, hu]; push_cast; ring
  have hco : co = 4099 * 2 ^ (-24 : ℤ) := by simp only [co, hu, huL]; push_cast; ring
  have hcl0 : 0 ≤ cl := by rw [hcl]; positivity
  have hco0 : 0 ≤ co := by rw [hco]; positivity
  have hcl1 : cl < 1 := by rw [hcl]; norm_num
  have hco1 : co < 1 := by rw [hco]; norm_num
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  have hu1 : u < 1 := by rw [hu]; norm_num
  have key := closed_form_le_of_bounds (u := u) (η := η) (V := 1) (uL := uL) (ηL := ηL)
    (N := 4096) (B := 4096) (T := 1) (Pw := 1 / (1 - ((4096 + 1 : ℕ) : ℝ) * u))
    hu0 (by rw [hη]; positivity) zero_le_one (le_rfl) (le_rfl)
    hcl0 hco0 (Real.exp_bound_div_one_sub_of_interval hcl0 hcl1)
    (Real.exp_bound_div_one_sub_of_interval hco0 hco1)
    (Real.exp_bound_div_one_sub_of_interval hu0 hu1) (log_two_pow_le 12)
    (one_add_pow_le_inv (4096 + 1) hu0 (by rw [hu]; norm_num))
  have hmax := hmx
  obtain ⟨h1, h2, h3, h4, h5⟩ := key (by rw [hcl, hu, hη]; norm_num)
  refine ⟨h1, h2.trans (le_trans (by rw [hcl, hu, hη]; norm_num) hmax),
    h3.trans (le_trans (by unfold productResidual; rw [hco, hu, hη, huL, hηL]; norm_num) hmax),
    h4.trans (le_trans (by unfold productResidual; rw [hcl, hco, hu, hη, huL, hηL]; norm_num)
      hmax), h5.trans ?_⟩
  unfold productResidual; rw [hcl, hco, hu, hη, huL, hηL]; norm_num

/-- Online softmax, 16384 keys in blocks of 128, weights kept in `binary32`, with values of
magnitude at most one, for a working format with the unit roundoff and underflow allowance of
`binary32` and largest finite value at least `2 ^ 100`: the numerical budget and range premises of
the closed form hold and its bound is at most 6.4e-5. -/
theorem online_16384_128 {u η mx : ℝ} (hu : u = 2 ^ (-24 : ℤ)) (hη : η = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ mx) :
    let uL : ℝ := 0
    let ηL : ℝ := 0
    let cl := u * ((128 : ℕ) + 3 * (128 : ℕ) - 1)
    let co := u * ((128 : ℕ) + 3 * (128 : ℕ)) + uL
    let al := (Real.exp cl - 1) + 2 * u * Real.exp cl * Real.log (16384 : ℕ)
    let ao := (Real.exp co - 1) + 2 * u * Real.exp co * Real.log (16384 : ℕ)
    let ρl := (128 : ℕ) * residualGrowth u η ^ 128 * blockResidual u η 16384 128 1 cl η
    let ρo := (128 : ℕ) * residualGrowth u η ^ 128 *
      blockResidual u η 16384 128 1 co (productResidual u η uL ηL 1)
    al + ρl < 1 ∧ ((16384 : ℕ) : ℝ) * 1 * Real.exp cl + ρl ≤ mx ∧
      ((16384 : ℕ) : ℝ) * 1 * Real.exp co + ρo ≤ mx ∧ (1 + 1 * ao + ρo) / (1 - al - ρl) ≤ mx ∧
      (1 + u) * ((1 * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * 1 + η ≤ 64 / 1000000 := by
  intro uL ηL cl co al ao ρl ρo
  have huL : uL = 0 := rfl
  have hηL : ηL = 0 := rfl
  have hcl : cl = 511 * 2 ^ (-24 : ℤ) := by simp only [cl, hu]; push_cast; ring
  have hco : co = 512 * 2 ^ (-24 : ℤ) := by simp only [co, hu, huL]; push_cast; ring
  have hcl0 : 0 ≤ cl := by rw [hcl]; positivity
  have hco0 : 0 ≤ co := by rw [hco]; positivity
  have hcl1 : cl < 1 := by rw [hcl]; norm_num
  have hco1 : co < 1 := by rw [hco]; norm_num
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  have hu1 : u < 1 := by rw [hu]; norm_num
  have key := closed_form_le_of_bounds (u := u) (η := η) (V := 1) (uL := uL) (ηL := ηL)
    (N := 16384) (B := 128) (T := 128) (Pw := (1 + u) ^ (128 + 1))
    hu0 (by rw [hη]; positivity) zero_le_one (le_rfl) (le_rfl)
    hcl0 hco0 (Real.exp_bound_div_one_sub_of_interval hcl0 hcl1)
    (Real.exp_bound_div_one_sub_of_interval hco0 hco1)
    (Real.exp_bound_div_one_sub_of_interval hu0 hu1) (log_two_pow_le 14) le_rfl
  have hmax := hmx
  obtain ⟨h1, h2, h3, h4, h5⟩ := key (by rw [hcl, hu, hη]; norm_num)
  refine ⟨h1, h2.trans (le_trans (by rw [hcl, hu, hη]; norm_num) hmax),
    h3.trans (le_trans (by unfold productResidual; rw [hco, hu, hη, huL, hηL]; norm_num) hmax),
    h4.trans (le_trans (by unfold productResidual; rw [hcl, hco, hu, hη, huL, hηL]; norm_num)
      hmax), h5.trans ?_⟩
  unfold productResidual; rw [hcl, hco, hu, hη, huL, hηL]; norm_num

/-- Online softmax over 4096 keys in 32 blocks of at most 128 that sums the weights it stores, with
values of magnitude at most one, for a working format with the unit roundoff and underflow
allowance of `binary32` and largest finite value at least `2 ^ 100`: the range and budget premises
of `onlineAverage_stored_error_of_eq` hold and its bound is at most 2.31e-5. -/
theorem stored_4096_128 {u η mx : ℝ} (hu : u = 2 ^ (-24 : ℤ)) (hη : η = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ mx) :
    let al := Real.exp (u * ((128 : ℕ) + 2 * (32 : ℕ))) - 1
    let ao := Real.exp (u * ((128 : ℕ) + 1 + 2 * (32 : ℕ))) - 1
    let ρl := ((32 : ℕ) : ℝ) * (1 + u) ^ (2 * 32) * η
    let ρo := ((32 : ℕ) : ℝ) * (1 + u) ^ (2 * 32) * (η + (1 + u) ^ 128 * ((128 : ℕ) * η))
    ((4096 : ℕ) : ℝ) * Real.exp (u * ((128 : ℕ) + 2 * (32 : ℕ))) + ρl ≤ mx ∧
      ((4096 : ℕ) : ℝ) * 1 * Real.exp (u * ((128 : ℕ) + 1 + 2 * (32 : ℕ))) + ρo ≤ mx ∧
      al + ρl < 1 ∧ (1 + 1 * ao + ρo) / (1 - al - ρl) ≤ mx ∧
      (1 + u) * ((1 * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * 1 + η ≤ 231 / 10000000 := by
  intro al ao ρl ρo
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  have hη0 : 0 ≤ η := by rw [hη]; positivity
  have hEcl : Real.exp (u * ((128 : ℕ) + 2 * (32 : ℕ))) ≤ 1 / (1 - 192 * u) :=
    Real.exp_bound_div_one_sub_of_interval (by positivity) (by rw [hu]; norm_num) |>.trans_eq'
      (by push_cast; ring_nf)
  have hEco : Real.exp (u * ((128 : ℕ) + 1 + 2 * (32 : ℕ))) ≤ 1 / (1 - 193 * u) :=
    Real.exp_bound_div_one_sub_of_interval (by positivity) (by rw [hu]; norm_num) |>.trans_eq'
      (by push_cast; ring_nf)
  obtain ⟨h1, h2, h3, h4, h5⟩ := stored_closed_form_le_of_bounds (N := 4096) (V := 1) (V' := 1)
    hu0 hη0 zero_le_one zero_le_one hEcl hEco
    (one_add_pow_le_inv (2 * 32) hu0 (by rw [hu]; norm_num))
    (one_add_pow_le_inv 128 hu0 (by rw [hu]; norm_num)) (by rw [hu, hη]; norm_num)
  refine ⟨h2.trans (le_trans (by rw [hu, hη]; norm_num) hmx),
    h3.trans (le_trans (by rw [hu, hη]; norm_num) hmx), h1,
    h4.trans (le_trans (by rw [hu, hη]; norm_num) hmx), h5.trans (by rw [hu, hη]; norm_num)⟩

/-- FlashAttention-2's recipe over 4096 keys in 32 blocks of at most 128, weights stored in a format
with the unit roundoff and underflow allowance of `bfloat16`, with values of magnitude at most one,
for a working format with those of `binary32` and largest finite value at least `2 ^ 100`: the
premises of `onlineAverage_id_roundTo_error` hold, its scale error `uL + N ηL` is at most
`2 ^ (-8) + 2 ^ (-122)` and its bound is at most 2.32e-5. -/
theorem flash2_4096_128 {u η uL ηL mx : ℝ} (hu : u = 2 ^ (-24 : ℤ)) (hη : η = 2 ^ (-150 : ℤ))
    (huL : uL = 2 ^ (-8 : ℤ)) (hηL : ηL = 2 ^ (-134 : ℤ)) (hmx : (2 : ℝ) ^ 100 ≤ mx) :
    let al := Real.exp (u * ((128 : ℕ) + 2 * (32 : ℕ))) - 1
    let ao := Real.exp (u * ((128 : ℕ) + 1 + 2 * (32 : ℕ))) - 1
    let ρl := ((32 : ℕ) : ℝ) * (1 + u) ^ (2 * 32) * η
    let ρo := ((32 : ℕ) : ℝ) * (1 + u) ^ (2 * 32) * (η + (1 + u) ^ 128 * ((128 : ℕ) * η))
    let V' := 1 * (1 + uL + ((4096 : ℕ) : ℝ) * ηL)
    ((4096 : ℕ) : ℝ) * Real.exp (u * ((128 : ℕ) + 2 * (32 : ℕ))) + ρl ≤ mx ∧
      ((4096 : ℕ) : ℝ) * 1 * Real.exp (u * ((128 : ℕ) + 1 + 2 * (32 : ℕ))) + ρo ≤ mx ∧
      al + ρl < 1 ∧ (V' + V' * ao + ρo) / (1 - al - ρl) ≤ mx ∧
      uL + ((4096 : ℕ) : ℝ) * ηL = 2 ^ (-8 : ℤ) + 2 ^ (-122 : ℤ) ∧
      (1 + u) * ((V' * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * V' + η ≤ 232 / 10000000 := by
  intro al ao ρl ρo V'
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  have hη0 : 0 ≤ η := by rw [hη]; positivity
  have hV' : V' = 1 + 2 ^ (-8 : ℤ) + 4096 * 2 ^ (-134 : ℤ) := by
    simp only [V', huL, hηL]; push_cast; ring
  have hEcl : Real.exp (u * ((128 : ℕ) + 2 * (32 : ℕ))) ≤ 1 / (1 - 192 * u) :=
    Real.exp_bound_div_one_sub_of_interval (by positivity) (by rw [hu]; norm_num) |>.trans_eq'
      (by push_cast; ring_nf)
  have hEco : Real.exp (u * ((128 : ℕ) + 1 + 2 * (32 : ℕ))) ≤ 1 / (1 - 193 * u) :=
    Real.exp_bound_div_one_sub_of_interval (by positivity) (by rw [hu]; norm_num) |>.trans_eq'
      (by push_cast; ring_nf)
  obtain ⟨h1, h2, h3, h4, h5⟩ := stored_closed_form_le_of_bounds (N := 4096) (V := 1) (V' := V')
    hu0 hη0 zero_le_one (by rw [hV']; positivity) hEcl hEco
    (one_add_pow_le_inv (2 * 32) hu0 (by rw [hu]; norm_num))
    (one_add_pow_le_inv 128 hu0 (by rw [hu]; norm_num)) (by rw [hu, hη]; norm_num)
  refine ⟨h2.trans (le_trans (by rw [hu, hη]; norm_num) hmx),
    h3.trans (le_trans (by rw [hu, hη]; norm_num) hmx), h1,
    h4.trans (le_trans (by rw [hu, hη, hV']; norm_num) hmx), by rw [huL, hηL]; push_cast; norm_num,
    h5.trans (by rw [hu, hη, hV']; norm_num)⟩

section Output

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

/-- Online softmax over 4096 keys in blocks of 128, in any IEEE format with the precision and
underflow of `binary32` and largest finite value at least `2 ^ 100`, with weights kept in the
working format, scores of magnitude at most `2 ^ 23` and values of magnitude at most one, is
finite and within `2.9e-5` of the softmax average. -/
theorem online_4096_128_output (hformat : format.isIEEE = true)
    (hu : Model.unitRoundoffAt format = 2 ^ (-24 : ℤ))
    (hη : Model.underflowErrorAt format = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ Model.toReal (Model.posMaxFinite format))
    (first : ExecFloat (Configured.Family format code plan) ×
      ExecFloat (Configured.Family format code plan))
    (rest : List (ExecFloat (Configured.Family format code plan) ×
      ExecFloat (Configured.Family format code plan)))
    (hlen : (first :: rest).length = 4096)
    (hkeys : ∀ e ∈ first :: rest, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ 2 ^ 23 ∧
      |ExecFloat.Binary.toReal e.2| ≤ 1) :
    let M := (first :: rest).foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal first.1)
    let S := ((first :: rest).map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let O := ((first :: rest).map fun e =>
      Real.exp (ExecFloat.Binary.toReal e.1 - M) * ExecFloat.Binary.toReal e.2).sum
    ExecFloat.Binary.isFinite (onlineAverage
        (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id id first.1
        (List.toChunks 128 (first :: rest))) = true ∧
      |ExecFloat.Binary.toReal (onlineAverage
          (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id id first.1
          (List.toChunks 128 (first :: rest))) - O / S| ≤ 29 / 1000000 := by
  intro M S O
  have hK : 2 * (2 : ℝ) ^ 23 ≤ Model.toReal (Model.posMaxFinite format) :=
    le_trans (by norm_num) hmx
  have hKu : 2 * (2 : ℝ) ^ 23 * Model.unitRoundoffAt format ≤ 1 := by rw [hu]; norm_num
  have main := onlineAverage_blocks_error hformat id (uL := 0) (ηL := 0) le_rfl le_rfl
    (fun x hx h01 => ⟨hx, h01, by simp⟩) (B := 128) (by norm_num) first rest hK hKu zero_le_one
    hkeys
  simp only [hlen, Nat.reduceAdd, Nat.reduceSub, Nat.reduceDiv] at main
  obtain ⟨h1, h2, h3, h4, h5⟩ := online_4096_128 hu hη hmx
  obtain ⟨hfinite, hbound⟩ := main h2 h3 h1 h4
  exact ⟨hfinite, hbound.trans h5⟩

/-- Online softmax over 4096 keys in 32 blocks of at most 128 that stores its weights in `bfloat16`
and sums the stored weights, in any IEEE format with the precision and underflow of `binary32`,
largest finite value at least `2 ^ 100` and the `bfloat16` grid, with scores of magnitude at most
`2 ^ 23` and values in `[lo, hi]` of magnitude at most one, is finite and within `2.31e-5` of some
`z ∈ [lo, hi]`. -/
theorem stored_4096_128_output (hformat : format.isIEEE = true)
    (hu : Model.unitRoundoffAt format = 2 ^ (-24 : ℤ))
    (hη : Model.underflowErrorAt format = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ Model.toReal (Model.posMaxFinite format))
    (hfrac : FloatFormat.bfloat16.fracWidth ≤ format.fracWidth)
    (hmin : format.minSubnormalExponent ≤ FloatFormat.bfloat16.minSubnormalExponent)
    (start : ExecFloat (Configured.Family format code plan))
    (blocks : List (List (ExecFloat (Configured.Family format code plan) ×
      ExecFloat (Configured.Family format code plan)))) {lo hi : ℝ}
    (hstart : ExecFloat.Binary.isFinite start = true)
    (hstartK : |ExecFloat.Binary.toReal start| ≤ 2 ^ 23)
    (hstart_le : ∃ e ∈ blocks.flatten, ExecFloat.Binary.toReal start ≤ ExecFloat.Binary.toReal e.1)
    (hN : blocks.flatten.length ≤ 4096) (hB : ∀ block ∈ blocks, block.length ≤ 128)
    (hT : blocks.length = 32)
    (hkeys : ∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ 2 ^ 23 ∧
      lo ≤ ExecFloat.Binary.toReal e.2 ∧ ExecFloat.Binary.toReal e.2 ≤ hi ∧
      |ExecFloat.Binary.toReal e.2| ≤ 1) :
    ∃ z ∈ Set.Icc lo hi,
      ExecFloat.Binary.isFinite (onlineAverage
        (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) (roundTo FloatFormat.bfloat16)
        (roundTo FloatFormat.bfloat16) start blocks) = true ∧
      |ExecFloat.Binary.toReal (onlineAverage
          (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) (roundTo FloatFormat.bfloat16)
          (roundTo FloatFormat.bfloat16) start blocks) - z| ≤ 231 / 10000000 := by
  have hK : 2 * (2 : ℝ) ^ 23 ≤ Model.toReal (Model.posMaxFinite format) :=
    le_trans (by norm_num) hmx
  have main := onlineAverage_roundTo_error hformat (low := FloatFormat.bfloat16)
    (by decide) hfrac hmin start blocks hK zero_le_one hstart hstartK hstart_le hN hB hT hkeys
  obtain ⟨h1, h2, h3, h4, h5⟩ := stored_4096_128 hu hη hmx
  obtain ⟨z, hz, hfinite, hbound⟩ := main h1 h2 h3 h4
  exact ⟨z, hz, hfinite, hbound.trans h5⟩

/-- FlashAttention-2's recipe over 4096 keys in 32 blocks of at most 128, the normalizer summed
from the unrounded weights and the weights stored in `bfloat16`, under the hypotheses of
`stored_4096_128_output`, is finite and within `2.32e-5` of `β z` with `z ∈ [lo, hi]` and
`|β - 1| ≤ 2 ^ (-8) + 2 ^ (-122)`. -/
theorem flash2_4096_128_output (hformat : format.isIEEE = true)
    (hu : Model.unitRoundoffAt format = 2 ^ (-24 : ℤ))
    (hη : Model.underflowErrorAt format = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ Model.toReal (Model.posMaxFinite format))
    (hfrac : FloatFormat.bfloat16.fracWidth ≤ format.fracWidth)
    (hmin : format.minSubnormalExponent ≤ FloatFormat.bfloat16.minSubnormalExponent)
    (start : ExecFloat (Configured.Family format code plan))
    (blocks : List (List (ExecFloat (Configured.Family format code plan) ×
      ExecFloat (Configured.Family format code plan)))) {lo hi : ℝ}
    (hstart : ExecFloat.Binary.isFinite start = true)
    (hstartK : |ExecFloat.Binary.toReal start| ≤ 2 ^ 23)
    (hstart_le : ∃ e ∈ blocks.flatten, ExecFloat.Binary.toReal start ≤ ExecFloat.Binary.toReal e.1)
    (hN : blocks.flatten.length ≤ 4096) (hB : ∀ block ∈ blocks, block.length ≤ 128)
    (hT : blocks.length = 32)
    (hkeys : ∀ e ∈ blocks.flatten, ExecFloat.Binary.isFinite e.1 = true ∧
      ExecFloat.Binary.isFinite e.2 = true ∧ |ExecFloat.Binary.toReal e.1| ≤ 2 ^ 23 ∧
      lo ≤ ExecFloat.Binary.toReal e.2 ∧ ExecFloat.Binary.toReal e.2 ≤ hi ∧
      |ExecFloat.Binary.toReal e.2| ≤ 1) :
    ∃ β z : ℝ, z ∈ Set.Icc lo hi ∧ |β - 1| ≤ 2 ^ (-8 : ℤ) + 2 ^ (-122 : ℤ) ∧
      ExecFloat.Binary.isFinite (onlineAverage
        (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id (roundTo FloatFormat.bfloat16)
        start blocks) = true ∧
      |ExecFloat.Binary.toReal (onlineAverage
          (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id (roundTo FloatFormat.bfloat16)
          start blocks) - β * z| ≤ 232 / 10000000 := by
  have hK : 2 * (2 : ℝ) ^ 23 ≤ Model.toReal (Model.posMaxFinite format) :=
    le_trans (by norm_num) hmx
  have main := onlineAverage_id_roundTo_error hformat (low := FloatFormat.bfloat16)
    (by decide) hfrac hmin start blocks hK zero_le_one hstart hstartK hstart_le hN hB hT hkeys
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ :=
    flash2_4096_128 hu hη unitRoundoff_bfloat16 underflow_bfloat16 hmx
  obtain ⟨β, z, hz, hβ, hfinite, hbound⟩ := main h1 h2 h3 h4
  exact ⟨β, z, hz, hβ.trans_eq h5, hfinite, hbound.trans h6⟩

end Output

end TorchLean.OnlineSoftmax
