/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel
public import TorchLean.Verify.RuntimeApprox.OnlineSoftmax
public import TorchLean.Verify.RuntimeApprox.TransformerBlock

/-!
# Transformer block with nearest rounding

`TorchLean.Verify.RuntimeApprox.TransformerBlock` bounds the block for any arithmetic meeting
`Arithmetic.Contract`. Here nearest-even rounding onto the real grid of an IEEE format meets it,
with the unit roundoff and underflow allowance of the format and the left-fold summation constants
of `abs_foldl_round_add_sub_sum_le`. The grid has no upper exponent bound, so a result of the
executable operations agrees with it only when nothing overflows. Even then the rounded block is
`Arithmetic.block`, whose layer normalization and affine maps differ from `Tensor.layerNorm`, which
rescales by `Tensor.normalizationScale`, and from `Linear.forward`, which adds the bias last.

The evaluation of the block bound at the sizes of GPT-2 small in `binary32` lives in
`TorchLeanExamples/Models/Sequence/GPT2BlockBound.lean`.
-/

@[expose] public section

namespace TorchLean.TransformerBlock

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

/-- The arithmetic that rounds every scalar operation, product, exponential and activation to the
nearest value of `work`, writes stage outputs through `store`, and sums lists from zero left to
right with rounded additions. -/
noncomputable def nearest (work : FloatFormat) (store act : ℝ → ℝ) : Arithmetic where
  round t := Model.roundAt work t
  roundMul t := Model.roundAt work t
  store := store
  sum l := l.foldl (fun s t => Model.roundAt work (s + t)) 0
  exp t := Model.roundAt work (Real.exp t)
  act t := Model.roundAt work (act t)

/-- `nearest work store act` meets the contract with the unit roundoff `u` and underflow allowance
`η` of `work` for each operation, `g n = (1 + u) ^ n - 1` and `h n = n * (1 + u) ^ n * η` for sums,
the errors `us` and `ηs` of `store`, and an activation with `|act t| ≤ |t|` and Lipschitz constant
`La`. -/
noncomputable def nearestContract (work : FloatFormat) {store : ℝ → ℝ} {us ηs : ℝ} (hus : 0 ≤ us)
    (hηs : 0 ≤ ηs) (hstore : ∀ t, |store t - t| ≤ us * |t| + ηs) {act : ℝ → ℝ}
    (hact : ∀ t, |act t| ≤ |t|) {La : ℝ} (hLa : 0 ≤ La)
    (hlip : ∀ s t, |act s - act t| ≤ La * |s - t|) : (nearest work store act).Contract act where
  u := Model.unitRoundoffAt work
  η := Model.underflowErrorAt work
  um := Model.unitRoundoffAt work
  ηm := Model.underflowErrorAt work
  us := us
  ηs := ηs
  ue := Model.unitRoundoffAt work
  ηe := Model.underflowErrorAt work
  ua := Model.unitRoundoffAt work
  ηa := Model.underflowErrorAt work
  la := La
  g n := (1 + Model.unitRoundoffAt work) ^ n - 1
  h n := n * (1 + Model.unitRoundoffAt work) ^ n * Model.underflowErrorAt work
  u_nonneg := (Model.unitRoundoffAt_pos work).le
  η_nonneg := (Model.underflowErrorAt_pos work).le
  um_nonneg := (Model.unitRoundoffAt_pos work).le
  ηm_nonneg := (Model.underflowErrorAt_pos work).le
  us_nonneg := hus
  ηs_nonneg := hηs
  ue_nonneg := (Model.unitRoundoffAt_pos work).le
  ηe_nonneg := (Model.underflowErrorAt_pos work).le
  ua_nonneg := (Model.unitRoundoffAt_pos work).le
  ηa_nonneg := (Model.underflowErrorAt_pos work).le
  la_nonneg := hLa
  g_nonneg n := sub_nonneg.2 (one_le_pow₀ (by linarith [Model.unitRoundoffAt_pos work]))
  h_nonneg n := by
    have := Model.unitRoundoffAt_pos work
    have := Model.underflowErrorAt_pos work
    positivity
  round_le := Model.abs_roundAt_sub_le_unitRoundoffAt work
  roundMul_le := Model.abs_roundAt_sub_le_unitRoundoffAt work
  store_le := hstore
  exp_le t := by
    show |Model.roundAt work (Real.exp t) - Real.exp t| ≤ _
    have h := Model.abs_roundAt_sub_le_unitRoundoffAt work (Real.exp t)
    rwa [abs_of_pos (Real.exp_pos t)] at h
  sum_le := abs_foldl_round_add_sub_sum_le (Model.unitRoundoffAt_pos work).le
    (Model.underflowErrorAt_pos work).le (Model.abs_roundAt_sub_le_unitRoundoffAt work)
  act_le t := by
    show |Model.roundAt work (act t) - act t| ≤ _
    have h := Model.abs_roundAt_sub_le_unitRoundoffAt work (act t)
    have := mul_le_mul_of_nonneg_left (hact t) (Model.unitRoundoffAt_pos work).le
    linarith
  act_lipschitz := hlip

/-- Writing stage outputs without rounding meets the store contract with `us = 0` and `ηs = 0`. -/
theorem abs_id_sub_le (t : ℝ) : |id t - t| ≤ 0 * |t| + 0 := by simp

/-- `binary32` arithmetic: the unit roundoff is `2 ^ (-24)` and the underflow allowance
`2 ^ (-150)`. -/
theorem nearestContract_binary32 {store : ℝ → ℝ} {us ηs : ℝ} (hus : 0 ≤ us) (hηs : 0 ≤ ηs)
    (hstore : ∀ t, |store t - t| ≤ us * |t| + ηs) {act : ℝ → ℝ} (hact : ∀ t, |act t| ≤ |t|)
    {La : ℝ} (hLa : 0 ≤ La) (hlip : ∀ s t, |act s - act t| ≤ La * |s - t|) :
    let C := nearestContract FloatFormat.binary32 hus hηs hstore hact hLa hlip
    C.u = 2 ^ (-24 : ℤ) ∧ C.η = 2 ^ (-150 : ℤ) := by
  refine ⟨?_, ?_⟩
  · show Model.unitRoundoffAt FloatFormat.binary32 = _
    rw [Model.unitRoundoffAt_eq]; rfl
  · show Model.underflowErrorAt FloatFormat.binary32 = _
    rw [Model.underflowErrorAt_eq]; rfl

/-- Rounding stage outputs to `bfloat16` meets the store contract with `us = 2 ^ (-8)` and
`ηs = 2 ^ (-134)`, and to `binary16` with `us = 2 ^ (-11)` and `ηs = 2 ^ (-25)`. -/
theorem roundAt_bfloat16_binary16 (t : ℝ) :
    |Model.roundAt FloatFormat.bfloat16 t - t| ≤ 2 ^ (-8 : ℤ) * |t| + 2 ^ (-134 : ℤ) ∧
      |Model.roundAt FloatFormat.binary16 t - t| ≤ 2 ^ (-11 : ℤ) * |t| + 2 ^ (-25 : ℤ) := by
  have hb := Model.abs_roundAt_sub_le_unitRoundoffAt FloatFormat.bfloat16 t
  have hh := Model.abs_roundAt_sub_le_unitRoundoffAt FloatFormat.binary16 t
  rw [Model.unitRoundoffAt_eq, Model.underflowErrorAt_eq] at hb hh
  exact ⟨hb, hh⟩

/-- The left-fold constant `g n` of `binary32` is at most `1 / (1 - m * 2 ^ (-24)) - 1` for
`n ≤ m`. -/
theorem one_add_pow_sub_one_le {n m : ℕ} (hn : n ≤ m) (hm : (m : ℝ) * 16777216⁻¹ < 1) :
    (1 + (16777216 : ℝ)⁻¹) ^ n - 1 ≤ 1 / (1 - m * 16777216⁻¹) - 1 :=
  sub_le_sub_right ((pow_le_pow_right₀ (by norm_num) hn).trans
    (OnlineSoftmax.one_add_pow_le_inv m (by norm_num) hm)) 1

/-- The left-fold constant `h n` of `binary32` is at most `m / (1 - m * 2 ^ (-24)) * η` for
`n ≤ m`. -/
theorem mul_one_add_pow_mul_le {n m : ℕ} (hn : n ≤ m) (hm : (m : ℝ) * 16777216⁻¹ < 1) {η : ℝ}
    (hη : 0 ≤ η) : n * (1 + (16777216 : ℝ)⁻¹) ^ n * η ≤ m * (1 / (1 - m * 16777216⁻¹)) * η :=
  mul_le_mul_of_nonneg_right (mul_le_mul (by exact_mod_cast hn)
    ((pow_le_pow_right₀ (by norm_num) hn).trans
      (OnlineSoftmax.one_add_pow_le_inv m (by norm_num) hm))
    (by positivity) (by positivity)) hη

/-- `2 * (exp x - 1) / (exp x + 1) ≤ exp x - 1` for `0 ≤ x`. -/
theorem two_mul_exp_sub_one_div_le {x : ℝ} (hx : 0 ≤ x) :
    2 * (Real.exp x - 1) / (Real.exp x + 1) ≤ Real.exp x - 1 := by
  have h1 : 1 ≤ Real.exp x := Real.one_le_exp hx
  rw [div_le_iff₀ (by linarith)]
  nlinarith

end TorchLean.TransformerBlock
