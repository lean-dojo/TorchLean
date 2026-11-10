/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError
public import TorchLean.Verify.RuntimeApprox.Roundoff
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Layers.Activation
import TorchLean.Tensor.LinearAlgebra.Real

/-!
# Softmax replay

Softmax normalization and its pullback replay their arithmetic over exact rationals when the
ordinary floating-point candidate loses range. The theorems here concern the configured scalar
codec, whose coefficients have depth zero, in every IEEE format and for any tensor storage. They do
not cover the native `Float` and `Float32` instances of `QuotientArithmetic` or the nested
coefficient trees of `Dual`.

Normalization keeps the ordinary quotients when the left-to-right denominator is finite, nonzero
and normal and no nonzero weight was absorbed into it. Otherwise each weight is the exact
normalized weight rounded once, even when the discarded denominator overflowed: nonnegative
weights with a positive exact total keep every coefficient in `[0, 1]`, so each decoded weight is
finite. For every length, both paths give finite weights in `[0, 1]`. When the ordinary path's
`n - 1` roundings stay below a factor of two, both paths also have a mixed error against the exact
weights. Rounded weights need not sum to one, and `abs_sum_normalizeWeights_sub_one_le` bounds
their mass defect under the same condition.

For the pullback, the replay does exact rational arithmetic on the computed weights, centered at
the anchor's upstream value, and rounds each coordinate once. Before that rounding, the distance
from the exact softmax VJP comes from the weight errors alone, and `softmaxPullback_coefficient_eq`
shows where the mass defect enters. When no ordinary coordinate is flagged, the pullback keeps its
ordinary result, whose coordinates the check itself proves finite and whose error the ordinary
analysis bounds.

Inputs are decoded as stored: errors made before normalization, such as those of the exponentials
that produce the weights, enter through the stated weight errors.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open TorchLean.Tensor TorchLean.Numeric

/-- If each `w i` approximates `p i` with relative error `r` and absolute error `a`, and the `p i`
are nonnegative and sum to one, then `|∑ i, w i - 1|` is at most `r + n * a`. -/
theorem abs_sum_sub_one_le {n : ℕ} {w p : Fin n → ℝ} {r a : ℝ}
    (herror : ∀ i, MixedError (w i) (p i) r a) (hp : ∀ i, 0 ≤ p i) (hsum : ∑ i, p i = 1) :
    |∑ i, w i - 1| ≤ r + n * a := by
  calc |∑ i, w i - 1| = |∑ i, (w i - p i)| := by rw [Finset.sum_sub_distrib, hsum]
    _ ≤ ∑ i, |w i - p i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, (r * p i + a) := Finset.sum_le_sum fun i _ => by
      simpa only [abs_of_nonneg (hp i)] using (herror i).bound
    _ = r + n * a := by simp [Finset.sum_add_distrib, ← Finset.mul_sum, hsum]

/-- If every weight has magnitude at most one, the centered contraction
`w i * (h i - ∑ j, h j * w j)` is at most `|h i| + ∑ j, |h j|` in magnitude. -/
theorem abs_mul_sub_sum_le {n : ℕ} (w h : Fin n → ℝ) (hw : ∀ j, |w j| ≤ 1) (i : Fin n) :
    |w i * (h i - ∑ j, h j * w j)| ≤ |h i| + ∑ j, |h j| := by
  rw [abs_mul]
  refine (mul_le_of_le_one_left (abs_nonneg _) (hw i)).trans ((abs_sub _ _).trans
    (add_le_add le_rfl ((Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j _ => ?_))))
  rw [abs_mul]
  exact mul_le_of_le_one_right (abs_nonneg _) (hw j)

/-- Centering at the anchor `a` changes coordinate `i` of the VJP in the computed weights `w` by
`w i * a` times their mass defect `1 - ∑ j, w j`. -/
theorem softmaxPullback_coefficient_eq {n : ℕ} (w g : Fin n → ℝ) (a : ℝ) (i : Fin n) :
    w i * ((g i - a) - ∑ j, (g j - a) * w j) =
      w i * (g i - ∑ j, g j * w j) - w i * a * (1 - ∑ j, w j) := by
  simp only [sub_mul, Finset.sum_sub_distrib, ← Finset.mul_sum]
  ring

/-- If the exact weights `p` sum to one and `|w j - p j| ≤ e j`, the centered coefficient in the
computed weights `w` differs from the exact softmax VJP by terms in the weight errors `e` alone,
with no hypothesis that `w` sums to one. -/
theorem abs_softmaxPullback_coefficient_sub_le {n : ℕ} (w p e g : Fin n → ℝ) (a : ℝ)
    (hp : ∑ j, p j = 1) (he : ∀ j, |w j - p j| ≤ e j) (i : Fin n) :
    |w i * ((g i - a) - ∑ j, (g j - a) * w j) - p i * (g i - ∑ j, p j * g j)| ≤
      e i * (|g i - a| + ∑ j, |g j - a| * |w j|) + |p i| * ∑ j, e j * |g j - a| := by
  have center : ∑ j, p j * g j = ∑ j, p j * (g j - a) + a := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hp, one_mul, sub_add_cancel]
  have difference : ∑ j, (g j - a) * (p j - w j) =
      ∑ j, p j * (g j - a) - ∑ j, (g j - a) * w j := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have split : w i * ((g i - a) - ∑ j, (g j - a) * w j) - p i * (g i - ∑ j, p j * g j) =
      (w i - p i) * ((g i - a) - ∑ j, (g j - a) * w j) +
        p i * ∑ j, (g j - a) * (p j - w j) := by
    rw [center, difference]
    ring
  rw [split]
  refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
  · rw [abs_mul]
    refine mul_le_mul (he i) ((abs_sub _ _).trans ?_) (abs_nonneg _) ((abs_nonneg _).trans (he i))
    gcongr
    exact (Finset.abs_sum_le_sum_abs _ _).trans_eq (by simp only [abs_mul])
  · rw [abs_mul]
    refine mul_le_mul_of_nonneg_left ((Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum fun j _ => ?_)) (abs_nonneg _)
    rw [abs_mul, abs_sub_comm (p j), mul_comm]
    exact mul_le_mul_of_nonneg_right (he j) (abs_nonneg _)

section Configured

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-! ## Zero tests and range flags -/

/-- A range check `x + 0` that raises no flag certifies that `x` is finite. -/
theorem isFinite_of_addWithFlags_zero {x : Value}
    (hformat : format.isIEEE = true)
    (hflags : (@QuotientArithmetic.addWithFlags Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) x 0).2 = 0) :
    ExecFloat.Binary.isFinite x = true := by
  have sum : Model.isFinite (ExecFloat.Binary.toModel (x + 0)) = true :=
    (isFinite_add_of_flags_eq_zero hflags).1
  rw [ExecFloat.Binary.toModel_hAdd x 0] at sum
  exact (Model.isFinite_of_isFinite_add hformat sum).1

/-- A replayed coordinate keeps an ordinary zero for a zero coefficient and otherwise decodes the
coefficient. Within the largest finite magnitude, both give the coefficient rounded once. -/
private theorem configured_replay_rounding (value : ℚ) (original : Value)
    (hformat : format.isIEEE = true)
    (hrange : |(value : ℝ)| ≤ Model.toReal (Model.posMaxFinite format)) :
    let ops := QuotientArithmetic.configured (format := format) (plan := plan) (code := code)
    let candidate := (ops.decode (.scalar value)).getD original
    let replayed := if (value == 0 && original == 0) = true then ops.copyPrimal original candidate
      else candidate
    ExecFloat.Binary.isFinite replayed = true ∧
      ExecFloat.Binary.toReal replayed = Model.roundAt format value := by
  intro ops candidate replayed
  by_cases chosen : (value == 0 && original == 0) = true
  · simp only [Bool.and_eq_true, beq_iff_eq] at chosen
    obtain ⟨rfl, zero⟩ := chosen
    simp only [replayed, beq_self_eq_true, zero, Bool.and_self, ↓reduceIte, Rat.cast_zero,
      Model.roundAt_zero]
    exact ExecFloat.Binary.beq_zero_iff.1 zero
  · simp only [replayed, chosen, Bool.false_eq_true, ↓reduceIte, candidate, ops,
      configured_decode, Option.getD_some]
    exact configured_decode_rounding (plan := plan) (code := code) _ hformat hrange
      (configured_decode _)

/-- The rational coefficient of the configured zero literal is zero. -/
private theorem toRat_zero_getD : (ExecFloat.Binary.toRat? (0 : Value)).getD 0 = 0 := by
  exact_mod_cast (Model.cast_toRat_eq_toReal _).trans
    (ExecFloat.Binary.toReal_zero (plan := plan) (code := code))

/-! ## Weight normalization -/

variable [Storage (ExecFloat (Configured.Family format code plan))]

/-- Flagging absorbed weights leaves the denominator of the weight normalizer equal to the
ordinary left-to-right sum. -/
private theorem normalizeWeights_denominator {n : Nat} (values : Tensor Value [n]) :
    (Tensor.foldl (fun (state : Value × Bool) value =>
      (state.1 + value, state.2 || (value != 0 && state.1 + value == state.1)))
      ((0 : Value), false) values).1 = Tensor.sum values := by
  refine (Tensor.Rep.map_foldl values _ _ id Prod.fst (· + ·) fun _ _ => ?_).trans ?_
  · rfl
  · rw [Tensor.Rep.map_id, Tensor.sum_eq_foldl]

/-- Encoding finite terms and adding their coefficients gives the exact rational sum. -/
private theorem foldl_encode_add {n : Nat} (values : Tensor Value [n])
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (values.getScalar i) = true) (start : ℚ) :
    Tensor.foldl (fun accumulated value => do
        let total ← accumulated
        let term ← @QuotientArithmetic.encode Value
          (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) value
        pure (QuotientCoefficients.add total term)) (some (.scalar start)) values =
      some (.scalar (start + ∑ i, (ExecFloat.Binary.toRat? (values.getScalar i)).getD 0)) := by
  rw [foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange, Fin.sum_univ_def]
  induction List.finRange n generalizing start with
  | nil => simp
  | cons i indices ih =>
      simp only [List.foldl_cons, List.map_cons, List.sum_cons,
        configured_encode_of_isFinite (hfinite i)]
      exact (ih _).trans (by rw [add_assoc])

/-- Every partial sum of a finite tensor sum is finite. -/
private theorem isFinite_of_mem_scanl_sum {n : Nat} {values : Tensor Value [n]}
    (hformat : format.isIEEE = true)
    (hsum : ExecFloat.Binary.isFinite (Tensor.sum values) = true) {v : Value}
    (hv : v ∈ (List.finRange n).scanl (fun accumulated j => accumulated + values.getScalar j) 0) :
    ExecFloat.Binary.isFinite v = true :=
  ExecFloat.Binary.isFinite_of_mem_finRange_scanl_add
    (by rwa [Tensor.sum_eq_foldl, foldl_eq_fin_foldl] at hsum) hv

/-- For finite nonnegative values with a positive exact total, each weight is either the ordinary
quotient by a finite nonzero denominator or the exact normalized weight rounded once. A
denominator that overflowed or absorbed a nonzero weight leads to the second case. -/
theorem normalizeWeights_getScalar {n : Nat} (values : Tensor Value [n])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (values.getScalar i) = true)
    (hnonneg : ∀ i, 0 ≤ ExecFloat.Binary.toReal (values.getScalar i))
    (htotal : 0 < ∑ i, ExecFloat.Binary.toReal (values.getScalar i)) (i : Fin n) :
    ((Softmax.normalizeWeights values).getScalar i =
          values.getScalar i / Tensor.sum values ∧
        ExecFloat.Binary.isFinite (Tensor.sum values) = true ∧
        ExecFloat.Binary.isZero (Tensor.sum values) = false) ∨
      (ExecFloat.Binary.isFinite ((Softmax.normalizeWeights values).getScalar i) =
          true ∧
        ExecFloat.Binary.toReal ((Softmax.normalizeWeights values).getScalar i) =
          Model.roundAt format (ExecFloat.Binary.toReal (values.getScalar i) /
            ∑ j, ExecFloat.Binary.toReal (values.getScalar j))) := by
  have total : ((∑ j, (ExecFloat.Binary.toRat? (values.getScalar j)).getD 0 : ℚ) : ℝ) =
      ∑ j, ExecFloat.Binary.toReal (values.getScalar j) := by
    simp only [Rat.cast_sum, cast_toRat_eq_toReal]
  unfold Softmax.normalizeWeights
  simp only [normalizeWeights_denominator, configured_supported hformat, Bool.not_true,
    Bool.false_eq_true, ↓reduceIte]
  split_ifs with keep
  · left
    simp only [Bool.and_eq_true, beq_iff_eq, bne, Bool.not_eq_true'] at keep
    obtain ⟨⟨-, flags⟩, nonzero⟩ := keep
    have finite := isFinite_of_addWithFlags_zero hformat flags
    refine ⟨by simp [div_def], finite, ?_⟩
    cases zero : ExecFloat.Binary.isZero (Tensor.sum values)
    · rfl
    · rw [ExecFloat.Binary.beq_zero_iff.2
        ⟨finite, (Model.isZero_eq_true_iff_toReal_eq_zero _ finite).1 zero⟩] at nonzero
      cases nonzero
  · right
    rw [configured_encode_of_isFinite ExecFloat.Binary.isFinite_zero, toRat_zero_getD,
      Option.bind_eq_bind, Option.bind_some, foldl_encode_add values hfinite, Option.bind_some,
      zero_add]
    have nonzero : (∑ j, (ExecFloat.Binary.toRat? (values.getScalar j)).getD 0) ≠ 0 := by
      intro h
      rw [h, Rat.cast_zero] at total
      linarith
    simp only [QuotientCoefficients.primal, beq_iff_eq, nonzero, ↓reduceIte, Option.pure_def,
      Option.getD_some, getScalar_ofFn, configured_encode_of_isFinite (hfinite i),
      QuotientCoefficients.div]
    have weight : ((((ExecFloat.Binary.toRat? (values.getScalar i)).getD 0 /
        ∑ j, (ExecFloat.Binary.toRat? (values.getScalar j)).getD 0 : ℚ)) : ℝ) =
        ExecFloat.Binary.toReal (values.getScalar i) /
          ∑ j, ExecFloat.Binary.toReal (values.getScalar j) := by
      rw [Rat.cast_div, cast_toRat_eq_toReal, total]
    have range : |(((ExecFloat.Binary.toRat? (values.getScalar i)).getD 0 /
        ∑ j, (ExecFloat.Binary.toRat? (values.getScalar j)).getD 0 : ℚ) : ℝ)| ≤
        Model.toReal (Model.posMaxFinite format) := by
      rw [weight, abs_of_nonneg (div_nonneg (hnonneg i) htotal.le)]
      exact ((div_le_one htotal).2 (Finset.single_le_sum (fun j _ => hnonneg j)
        (Finset.mem_univ i))).trans (Model.one_le_toReal_posMaxFinite format)
    exact (configured_replay_rounding _ _ hformat range).imp_right fun rounded =>
      rounded.trans (congrArg (Model.roundAt format) weight)

/-- Normalized finite nonnegative weights with a positive exact total are finite and lie in
`[0, 1]`, for every length and on both branches. -/
theorem normalizeWeights_mem_Icc {n : Nat} (values : Tensor Value [n])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (values.getScalar i) = true)
    (hnonneg : ∀ i, 0 ≤ ExecFloat.Binary.toReal (values.getScalar i))
    (htotal : 0 < ∑ i, ExecFloat.Binary.toReal (values.getScalar i)) (i : Fin n) :
    ExecFloat.Binary.isFinite ((Softmax.normalizeWeights values).getScalar i) = true ∧
      ExecFloat.Binary.toReal ((Softmax.normalizeWeights values).getScalar i) ∈
        Set.Icc 0 1 := by
  rcases normalizeWeights_getScalar values hformat hfinite hnonneg htotal i with
    ⟨equal, finiteSum, nonzeroSum⟩ | ⟨finite, rounded⟩
  · rw [equal]
    have dominates : ExecFloat.Binary.toReal (values.getScalar i) ≤
        ExecFloat.Binary.toReal (Tensor.sum values) := by
      rw [Tensor.sum_eq_foldl, foldl_eq_fin_foldl,
        Fin.foldl_eq_foldl_finRange] at finiteSum ⊢
      exact (ExecFloat.Binary.le_toReal_foldl_add finiteSum (fun j _ => hnonneg j)
        ExecFloat.Binary.toReal_zero.ge).2 i (List.mem_finRange i)
    have positive : 0 < ExecFloat.Binary.toReal (Tensor.sum values) := by
      refine ((hnonneg i).trans dominates).lt_of_ne fun zero => ?_
      exact Bool.false_ne_true
        (nonzeroSum.symm.trans ((Model.isZero_eq_true_iff_toReal_eq_zero _ finiteSum).2 zero.symm))
    have quotient : ExecFloat.Binary.toReal (values.getScalar i) /
        ExecFloat.Binary.toReal (Tensor.sum values) ∈ Set.Icc 0 1 :=
      ⟨div_nonneg (hnonneg i) positive.le, (div_le_one positive).2 dominates⟩
    have finite : ExecFloat.Binary.isFinite (values.getScalar i / Tensor.sum values) = true := by
      have divided := Model.isFinite_div_of_abs_div_le_posMaxFinite _ _ hformat (hfinite i)
        finiteSum nonzeroSum ((abs_of_nonneg quotient.1).le.trans
          (quotient.2.trans (Model.one_le_toReal_posMaxFinite format)))
      rwa [← ExecFloat.Binary.toModel_hDiv (values.getScalar i) (Tensor.sum values)] at divided
    exact ⟨finite, ExecFloat.Binary.toReal_div_eq_roundAt finiteSum finite ▸
      Model.roundAt_mem_Icc_zero_one format quotient⟩
  · exact ⟨finite, rounded ▸ Model.roundAt_mem_Icc_zero_one format
      ⟨div_nonneg (hnonneg i) htotal.le,
        (div_le_one htotal).2 (Finset.single_le_sum (fun j _ => hnonneg j) (Finset.mem_univ i))⟩⟩

/-- The finiteness of `normalizeWeights_mem_Icc`, in the form `roundoff` uses for
the finiteness of a weight. -/
@[roundoff_rule]
theorem isFinite_normalizeWeights {n : Nat} (values : Tensor Value [n])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (values.getScalar i) = true)
    (hnonneg : ∀ i, 0 ≤ ExecFloat.Binary.toReal (values.getScalar i))
    (htotal : 0 < ∑ i, ExecFloat.Binary.toReal (values.getScalar i)) (i : Fin n) :
    ExecFloat.Binary.isFinite ((Softmax.normalizeWeights values).getScalar i) = true :=
  (normalizeWeights_mem_Icc values hformat hfinite hnonneg htotal i).1

/-- The weights of `normalizeWeights_mem_Icc` are nonnegative, in the form
`roundoff` uses for the sign of a weight. -/
@[roundoff_rule]
theorem normalizeWeights_nonneg {n : Nat} (values : Tensor Value [n])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (values.getScalar i) = true)
    (hnonneg : ∀ i, 0 ≤ ExecFloat.Binary.toReal (values.getScalar i))
    (htotal : 0 < ∑ i, ExecFloat.Binary.toReal (values.getScalar i)) (i : Fin n) :
    0 ≤ ExecFloat.Binary.toReal ((Softmax.normalizeWeights values).getScalar i) :=
  (normalizeWeights_mem_Icc values hformat hfinite hnonneg htotal i).2.1

/-- Normalized finite nonnegative weights with a positive exact total approximate the exact
normalized weights with a mixed error when the `n - 1` roundings of a left-to-right sum stay below
a factor of two. The replay rounds once, and the bound also covers the ordinary quotients by that
sum. -/
@[roundoff_rule]
theorem normalizeWeights_mixedError {n : Nat} (values : Tensor Value [n])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (values.getScalar i) = true)
    (hnonneg : ∀ i, 0 ≤ ExecFloat.Binary.toReal (values.getScalar i))
    (htotal : 0 < ∑ i, ExecFloat.Binary.toReal (values.getScalar i))
    (hlength : (1 + Model.unitRoundoffAt format) ^ (n - 1) < 2) (i : Fin n) :
    MixedError
      (ExecFloat.Binary.toReal ((Softmax.normalizeWeights values).getScalar i))
      (ExecFloat.Binary.toReal (values.getScalar i) /
        ∑ j, ExecFloat.Binary.toReal (values.getScalar j))
      ((1 + Model.unitRoundoffAt format) *
          (((1 + Model.unitRoundoffAt format) ^ (n - 1) - 1) /
            (2 - (1 + Model.unitRoundoffAt format) ^ (n - 1))) +
        Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have hu := (Model.unitRoundoffAt_pos format).le
  rcases normalizeWeights_getScalar values hformat hfinite hnonneg htotal i with
    ⟨equal, finiteSum, -⟩ | ⟨-, rounded⟩
  · have finite := (normalizeWeights_mem_Icc values hformat hfinite hnonneg htotal i).1
    rw [equal] at finite ⊢
    have denominator := ExecFloat.Binary.mixedError_toReal_fin_foldl_add_of_toReal_eq_zero
      (hformat := hformat) (by exact ExecFloat.Binary.toReal_zero)
      (fun _ => isFinite_of_mem_scanl_sum hformat finiteSum) hfinite
      (fun j => MixedError.refl _) le_rfl hnonneg
    rw [← foldl_eq_fin_foldl, ← Tensor.sum_eq_foldl] at denominator
    simp only [add_zero, mul_one, Finset.sum_const_zero, mul_zero] at denominator
    exact (ExecFloat.Binary.mixedError_toReal_div finiteSum finite
      (MixedError.refl _) denominator (by linarith)).mono (le_of_eq (by ring_nf)) (by simp)
  · refine (ExecFloat.Binary.mixedError_toReal_of_eq_roundAt rounded).mono ?_ le_rfl
    have spread : 0 ≤ ((1 + Model.unitRoundoffAt format) ^ (n - 1) - 1) /
        (2 - (1 + Model.unitRoundoffAt format) ^ (n - 1)) :=
      div_nonneg (sub_nonneg.2 (one_le_pow₀ (by linarith))) (by linarith)
    nlinarith

/-- Absolute form of `normalizeWeights_mixedError`, with the total written as a tensor sum, which
the absolute analysis of `roundoff` reads as a fold of the stored weights. -/
@[roundoff_rule]
theorem abs_normalizeWeights_sub_le {n : Nat} (values : Tensor Value [n])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (values.getScalar i) = true)
    (hnonneg : ∀ i, 0 ≤ ExecFloat.Binary.toReal (values.getScalar i))
    (htotal : 0 < Tensor.sum (Tensor.map ExecFloat.Binary.toReal values))
    (hlength : (1 + Model.unitRoundoffAt format) ^ (n - 1) < 2) (i : Fin n) :
    |ExecFloat.Binary.toReal ((Softmax.normalizeWeights values).getScalar i) -
        ExecFloat.Binary.toReal (values.getScalar i) /
          Tensor.sum (Tensor.map ExecFloat.Binary.toReal values)| ≤
      ((1 + Model.unitRoundoffAt format) *
            (((1 + Model.unitRoundoffAt format) ^ (n - 1) - 1) /
              (2 - (1 + Model.unitRoundoffAt format) ^ (n - 1))) +
          Model.unitRoundoffAt format) *
        |ExecFloat.Binary.toReal (values.getScalar i) /
          Tensor.sum (Tensor.map ExecFloat.Binary.toReal values)| +
        Model.underflowErrorAt format := by
  simp only [Tensor.sum_vec, getScalar_map] at htotal ⊢
  exact (normalizeWeights_mixedError values hformat hfinite hnonneg htotal hlength i).bound

/-- Under the conditions of `normalizeWeights_mixedError`, the mass defect of the normalized
weights is at most their relative error plus `n` underflow errors. -/
theorem abs_sum_normalizeWeights_sub_one_le {n : Nat} (values : Tensor Value [n])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (values.getScalar i) = true)
    (hnonneg : ∀ i, 0 ≤ ExecFloat.Binary.toReal (values.getScalar i))
    (htotal : 0 < ∑ i, ExecFloat.Binary.toReal (values.getScalar i))
    (hlength : (1 + Model.unitRoundoffAt format) ^ (n - 1) < 2) :
    |∑ i, ExecFloat.Binary.toReal ((Softmax.normalizeWeights values).getScalar i) - 1| ≤
      (1 + Model.unitRoundoffAt format) *
          (((1 + Model.unitRoundoffAt format) ^ (n - 1) - 1) /
            (2 - (1 + Model.unitRoundoffAt format) ^ (n - 1))) +
        Model.unitRoundoffAt format + n * Model.underflowErrorAt format :=
  abs_sum_sub_one_le
    (fun i => normalizeWeights_mixedError values hformat hfinite hnonneg htotal hlength i)
    (fun i => div_nonneg (hnonneg i) htotal.le)
    (by rw [← Finset.sum_div, div_self htotal.ne'])

/-! ## Softmax pullback -/

/-- The replay forms the centered contraction exactly in the stored weights and the upstream
coefficients. -/
theorem softmaxPullbackCoefficients_eq {n : Nat} (weights : Tensor Value [n])
    (upstream : Fin n → ℚ) (origin : ℚ)
    (hweights : ∀ i, ExecFloat.Binary.isFinite (weights.getScalar i) = true) :
    Softmax.pullbackCoefficients? weights
        (fun i => some (.scalar (upstream i))) (some (.scalar origin)) =
      some fun i => some (.scalar ((ExecFloat.Binary.toRat? (weights.getScalar i)).getD 0 *
        ((upstream i - origin) - ∑ j, (upstream j - origin) *
          (ExecFloat.Binary.toRat? (weights.getScalar j)).getD 0))) := by
  have total (indices : List (Fin n)) (start : ℚ) :
      indices.foldl (fun accumulated i => do
        let total ← accumulated
        let weight ← @QuotientArithmetic.encode Value
          (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
          (weights.getScalar i)
        let gradient ← some (Coefficients.scalar (upstream i))
        pure (QuotientCoefficients.add total (QuotientCoefficients.mul
          (QuotientCoefficients.sub gradient (.scalar origin)) weight))) (some (.scalar start)) =
        some (.scalar (start + (indices.map fun j => (upstream j - origin) *
          (ExecFloat.Binary.toRat? (weights.getScalar j)).getD 0).sum)) := by
    induction indices generalizing start with
    | nil => simp
    | cons i indices ih =>
        simp only [List.foldl_cons, List.map_cons, List.sum_cons,
          configured_encode_of_isFinite (hweights i)]
        exact (ih _).trans (by rw [add_assoc])
  unfold Softmax.pullbackCoefficients?
  dsimp only
  rw [Option.bind_eq_bind, Option.bind_some,
    configured_encode_of_isFinite ExecFloat.Binary.isFinite_zero, toRat_zero_getD, Option.bind_some]
  refine (congrArg (Option.bind · _) (total _ 0)).trans ?_
  rw [zero_add, Option.bind_some, Option.pure_def]
  congr 1
  funext i
  simp [configured_encode_of_isFinite (hweights i), QuotientCoefficients.mul,
    QuotientCoefficients.sub, Fin.sum_univ_def]

/-- A replayed pullback coordinate within the largest finite magnitude is its exact coefficient
rounded once. -/
theorem replayPullback_getScalar {n : Nat} (weights ordinary : Tensor Value [n])
    (upstream : Fin n → ℚ) (origin : ℚ) (hformat : format.isIEEE = true)
    (hweights : ∀ i, ExecFloat.Binary.isFinite (weights.getScalar i) = true) (i : Fin n)
    (hrange : |((((ExecFloat.Binary.toRat? (weights.getScalar i)).getD 0 *
        ((upstream i - origin) - ∑ j, (upstream j - origin) *
          (ExecFloat.Binary.toRat? (weights.getScalar j)).getD 0) : ℚ)) : ℝ)| ≤
        Model.toReal (Model.posMaxFinite format)) :
    let replayed := ((Softmax.replayPullback? weights ordinary
      (fun i => some (.scalar (upstream i))) (some (.scalar origin))).getD ordinary).getScalar i
    ExecFloat.Binary.isFinite replayed = true ∧
      ExecFloat.Binary.toReal replayed =
        Model.roundAt format ((((ExecFloat.Binary.toRat? (weights.getScalar i)).getD 0 *
          ((upstream i - origin) - ∑ j, (upstream j - origin) *
            (ExecFloat.Binary.toRat? (weights.getScalar j)).getD 0) : ℚ)) : ℝ) := by
  unfold Softmax.replayPullback?
  dsimp only
  rw [softmaxPullbackCoefficients_eq weights upstream origin hweights, Option.bind_eq_bind,
    Option.bind_some, Option.pure_def, Option.getD_some, getScalar_ofFn]
  exact configured_replay_rounding _ _ hformat hrange

/-- The pullback's replay from the encoded upstream values, with its coefficient read in reals. -/
private theorem replayPullback_ofFn_getScalar {n : Nat}
    (weights ordinary : Tensor Value [n]) (gradient : Fin n → Value) (anchor : Value)
    (hformat : format.isIEEE = true)
    (hweights : ∀ j, ExecFloat.Binary.isFinite (weights.getScalar j) = true)
    (hgradient : ∀ j, ExecFloat.Binary.isFinite (gradient j) = true)
    (hanchor : ExecFloat.Binary.isFinite anchor = true) (i : Fin n)
    (hrange : |ExecFloat.Binary.toReal (weights.getScalar i) *
        ((ExecFloat.Binary.toReal (gradient i) - ExecFloat.Binary.toReal anchor) -
          ∑ j, (ExecFloat.Binary.toReal (gradient j) - ExecFloat.Binary.toReal anchor) *
            ExecFloat.Binary.toReal (weights.getScalar j))| ≤
        Model.toReal (Model.posMaxFinite format)) :
    let replayed := ((Softmax.replayPullback? weights ordinary
      (fun j => QuotientArithmetic.encode (gradient j))
      (QuotientArithmetic.encode anchor)).getD ordinary).getScalar i
    ExecFloat.Binary.isFinite replayed = true ∧
      ExecFloat.Binary.toReal replayed =
        Model.roundAt format (ExecFloat.Binary.toReal (weights.getScalar i) *
          ((ExecFloat.Binary.toReal (gradient i) - ExecFloat.Binary.toReal anchor) -
            ∑ j, (ExecFloat.Binary.toReal (gradient j) - ExecFloat.Binary.toReal anchor) *
              ExecFloat.Binary.toReal (weights.getScalar j))) := by
  simp only [configured_encode_of_isFinite (hgradient _), configured_encode_of_isFinite hanchor]
  have coefficient : ((((ExecFloat.Binary.toRat? (weights.getScalar i)).getD 0 *
      (((ExecFloat.Binary.toRat? (gradient i)).getD 0 -
          (ExecFloat.Binary.toRat? anchor).getD 0) -
        ∑ j, ((ExecFloat.Binary.toRat? (gradient j)).getD 0 -
          (ExecFloat.Binary.toRat? anchor).getD 0) *
          (ExecFloat.Binary.toRat? (weights.getScalar j)).getD 0) : ℚ)) : ℝ) =
      ExecFloat.Binary.toReal (weights.getScalar i) *
        ((ExecFloat.Binary.toReal (gradient i) - ExecFloat.Binary.toReal anchor) -
          ∑ j, (ExecFloat.Binary.toReal (gradient j) - ExecFloat.Binary.toReal anchor) *
            ExecFloat.Binary.toReal (weights.getScalar j)) := by
    simp only [Rat.cast_mul, Rat.cast_sub, Rat.cast_sum, cast_toRat_eq_toReal]
  exact (replayPullback_getScalar weights _ _ _ hformat hweights i
    (coefficient ▸ hrange)).imp_right fun rounded =>
      rounded.trans (congrArg (Model.roundAt format) coefficient)

/-- Absolute form of the pullback replay, with the contraction written as a tensor sum, which the
absolute analysis of `roundoff` reads as a fold of the stored weights and upstream values. -/
@[roundoff_rule]
theorem abs_replayPullback_sub_le {n : Nat}
    (weights ordinary gradient : Tensor Value [n]) (anchor : Value)
    (hformat : format.isIEEE = true)
    (hweights : ∀ j, ExecFloat.Binary.isFinite (weights.getScalar j) = true)
    (hgradient : ∀ j, ExecFloat.Binary.isFinite (gradient.getScalar j) = true)
    (hanchor : ExecFloat.Binary.isFinite anchor = true) (i : Fin n)
    (hrange : |ExecFloat.Binary.toReal (weights.getScalar i) *
        ((ExecFloat.Binary.toReal (gradient.getScalar i) - ExecFloat.Binary.toReal anchor) -
          Tensor.sum (zipWith (fun upstream weight => (ExecFloat.Binary.toReal upstream -
            ExecFloat.Binary.toReal anchor) * ExecFloat.Binary.toReal weight) gradient weights))| ≤
        Model.toReal (Model.posMaxFinite format)) :
    |ExecFloat.Binary.toReal (((Softmax.replayPullback? weights ordinary
        (fun j => QuotientArithmetic.encode (gradient.getScalar j))
        (QuotientArithmetic.encode anchor)).getD ordinary).getScalar i) -
      ExecFloat.Binary.toReal (weights.getScalar i) *
        ((ExecFloat.Binary.toReal (gradient.getScalar i) - ExecFloat.Binary.toReal anchor) -
          Tensor.sum (zipWith (fun upstream weight => (ExecFloat.Binary.toReal upstream -
            ExecFloat.Binary.toReal anchor) * ExecFloat.Binary.toReal weight) gradient weights))| ≤
      Model.halfUlpAt format (ExecFloat.Binary.toReal (weights.getScalar i) *
        ((ExecFloat.Binary.toReal (gradient.getScalar i) - ExecFloat.Binary.toReal anchor) -
          Tensor.sum (zipWith (fun upstream weight => (ExecFloat.Binary.toReal upstream -
            ExecFloat.Binary.toReal anchor) * ExecFloat.Binary.toReal weight)
              gradient weights))) := by
  simp only [Tensor.sum_vec, getScalar_zipWith] at hrange ⊢
  rw [(replayPullback_ofFn_getScalar weights ordinary gradient.getScalar anchor hformat
    hweights hgradient hanchor i hrange).2]
  exact Model.abs_roundAt_sub_le _ _

/-- Pullback replay for upstream values given coordinate by coordinate, such as the reads of one
row of a larger tensor. For a stored vector, the absolute analysis tries
`abs_replayPullback_sub_le` first. -/
@[roundoff_rule]
theorem abs_replayPullback_ofFn_sub_le {n : Nat}
    (weights ordinary : Tensor Value [n]) (gradient : Fin n → Value) (anchor : Value)
    (hformat : format.isIEEE = true)
    (hweights : ∀ j, ExecFloat.Binary.isFinite (weights.getScalar j) = true)
    (hgradient : ∀ j, ExecFloat.Binary.isFinite (gradient j) = true)
    (hanchor : ExecFloat.Binary.isFinite anchor = true) (i : Fin n)
    (hrange : |ExecFloat.Binary.toReal (weights.getScalar i) *
        ((ExecFloat.Binary.toReal (gradient i) - ExecFloat.Binary.toReal anchor) -
          Tensor.sum (zipWith (fun upstream weight => (ExecFloat.Binary.toReal upstream -
            ExecFloat.Binary.toReal anchor) * ExecFloat.Binary.toReal weight)
              (Tensor.ofFn gradient) weights))| ≤
        Model.toReal (Model.posMaxFinite format)) :
    |ExecFloat.Binary.toReal (((Softmax.replayPullback? weights ordinary
        (fun j => QuotientArithmetic.encode (gradient j))
        (QuotientArithmetic.encode anchor)).getD ordinary).getScalar i) -
      ExecFloat.Binary.toReal (weights.getScalar i) *
        ((ExecFloat.Binary.toReal (gradient i) - ExecFloat.Binary.toReal anchor) -
          Tensor.sum (zipWith (fun upstream weight => (ExecFloat.Binary.toReal upstream -
            ExecFloat.Binary.toReal anchor) * ExecFloat.Binary.toReal weight)
              (Tensor.ofFn gradient) weights))| ≤
      Model.halfUlpAt format (ExecFloat.Binary.toReal (weights.getScalar i) *
        ((ExecFloat.Binary.toReal (gradient i) - ExecFloat.Binary.toReal anchor) -
          Tensor.sum (zipWith (fun upstream weight => (ExecFloat.Binary.toReal upstream -
            ExecFloat.Binary.toReal anchor) * ExecFloat.Binary.toReal weight)
              (Tensor.ofFn gradient) weights))) := by
  simp only [Tensor.sum_vec, getScalar_zipWith, getScalar_ofFn] at hrange ⊢
  rw [(replayPullback_ofFn_getScalar weights ordinary gradient anchor hformat hweights
    hgradient hanchor i hrange).2]
  exact Model.abs_roundAt_sub_le _ _

/-- Either the pullback keeps its ordinary result, and then every coordinate is finite, or a
coordinate whose exact replay coefficient lies within the largest finite magnitude is that
coefficient rounded once. The replay applies even when an ordinary intermediate overflowed. -/
theorem softmaxPullback_getScalar {n : Nat} (weights gradient : Tensor Value [n]) (anchor : Value)
    (hformat : format.isIEEE = true)
    (hweights : ∀ j, ExecFloat.Binary.isFinite (weights.getScalar j) = true)
    (hgradient : ∀ j, ExecFloat.Binary.isFinite (gradient.getScalar j) = true)
    (hanchor : ExecFloat.Binary.isFinite anchor = true) (i : Fin n)
    (hrange : |ExecFloat.Binary.toReal (weights.getScalar i) *
        ((ExecFloat.Binary.toReal (gradient.getScalar i) - ExecFloat.Binary.toReal anchor) -
          ∑ j, (ExecFloat.Binary.toReal (gradient.getScalar j) - ExecFloat.Binary.toReal anchor) *
            ExecFloat.Binary.toReal (weights.getScalar j))| ≤
        Model.toReal (Model.posMaxFinite format)) :
    (Softmax.pullback weights gradient anchor =
        (weights * ((gradient - Tensor.full _ anchor) -
          Tensor.full _ (Tensor.sum ((gradient - Tensor.full _ anchor) * weights))))
              ∧
      ∀ j, ExecFloat.Binary.isFinite
        ((Softmax.pullback weights gradient anchor).getScalar j) = true) ∨
    (ExecFloat.Binary.isFinite
        ((Softmax.pullback weights gradient anchor).getScalar i) = true ∧
      ExecFloat.Binary.toReal
          ((Softmax.pullback weights gradient anchor).getScalar i) =
        Model.roundAt format (ExecFloat.Binary.toReal (weights.getScalar i) *
          ((ExecFloat.Binary.toReal (gradient.getScalar i) - ExecFloat.Binary.toReal anchor) -
            ∑ j, (ExecFloat.Binary.toReal (gradient.getScalar j) - ExecFloat.Binary.toReal anchor) *
              ExecFloat.Binary.toReal (weights.getScalar j)))) := by
  unfold Softmax.pullback
  simp only [configured_supported hformat, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  split_ifs with keep
  · left
    refine ⟨rfl, fun j => ?_⟩
    simp only [Bool.not_eq_true', List.any_eq_false, bne_iff_ne, ne_eq, Decidable.not_not] at keep
    exact isFinite_of_addWithFlags_zero hformat
      (keep j (List.mem_finRange j))
  · exact .inr (replayPullback_ofFn_getScalar weights _ gradient.getScalar anchor hformat
      hweights hgradient hanchor i hrange)

/-- A replayed pullback coordinate approximates the exact softmax VJP `p i * (g i - ∑ j, p j * g j)`
with the rounding error of one operation and the propagated weight errors `e`. A kept ordinary
pullback is finite, and the ordinary analysis bounds its error. -/
theorem softmaxPullback_mixedError {n : Nat} (weights gradient : Tensor Value [n]) (anchor : Value)
    (p e : Fin n → ℝ) (hformat : format.isIEEE = true)
    (hweights : ∀ j, ExecFloat.Binary.isFinite (weights.getScalar j) = true)
    (hgradient : ∀ j, ExecFloat.Binary.isFinite (gradient.getScalar j) = true)
    (hanchor : ExecFloat.Binary.isFinite anchor = true) (hp : ∑ j, p j = 1)
    (he : ∀ j, |ExecFloat.Binary.toReal (weights.getScalar j) - p j| ≤ e j) (i : Fin n)
    (hrange : |ExecFloat.Binary.toReal (weights.getScalar i) *
        ((ExecFloat.Binary.toReal (gradient.getScalar i) - ExecFloat.Binary.toReal anchor) -
          ∑ j, (ExecFloat.Binary.toReal (gradient.getScalar j) - ExecFloat.Binary.toReal anchor) *
            ExecFloat.Binary.toReal (weights.getScalar j))| ≤
        Model.toReal (Model.posMaxFinite format)) :
    (Softmax.pullback weights gradient anchor =
        (weights * ((gradient - Tensor.full _ anchor) -
          Tensor.full _ (Tensor.sum ((gradient - Tensor.full _ anchor) * weights))))
              ∧
      ∀ j, ExecFloat.Binary.isFinite
        ((Softmax.pullback weights gradient anchor).getScalar j) = true) ∨
    (ExecFloat.Binary.isFinite
        ((Softmax.pullback weights gradient anchor).getScalar i) = true ∧
      MixedError
        (ExecFloat.Binary.toReal
          ((Softmax.pullback weights gradient anchor).getScalar i))
        (p i * (ExecFloat.Binary.toReal (gradient.getScalar i) -
          ∑ j, p j * ExecFloat.Binary.toReal (gradient.getScalar j)))
        (Model.unitRoundoffAt format)
        ((1 + Model.unitRoundoffAt format) *
            (e i * (|ExecFloat.Binary.toReal (gradient.getScalar i) -
                  ExecFloat.Binary.toReal anchor| +
                ∑ j, |ExecFloat.Binary.toReal (gradient.getScalar j) -
                  ExecFloat.Binary.toReal anchor| *
                  |ExecFloat.Binary.toReal (weights.getScalar j)|) +
              |p i| * ∑ j, e j *
                |ExecFloat.Binary.toReal (gradient.getScalar j) - ExecFloat.Binary.toReal anchor|) +
          Model.underflowErrorAt format)) := by
  rcases softmaxPullback_getScalar weights gradient anchor hformat hweights hgradient hanchor i
    hrange with kept | ⟨finite, rounded⟩
  · exact .inl kept
  · refine .inr ⟨finite, ?_⟩
    exact (ExecFloat.Binary.mixedError_toReal_of_eq_roundAt rounded).trans_of_abs_sub_le
      (abs_softmaxPullback_coefficient_sub_le _ p e
        (fun j => ExecFloat.Binary.toReal (gradient.getScalar j)) (ExecFloat.Binary.toReal anchor)
        hp he i)

end Configured

end TorchLean.Roundoff
