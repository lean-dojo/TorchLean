/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Stable.Once
public import TorchLeanExamples.DeepDives.Floats.ProbBugs.SurvivalComplement
import TorchLean.Verify.RuntimeApprox.TransformerBlock

/-!
# The variance rounded once

The population variance `(1 / n) Σᵢ (xᵢ - x̄)²` and the sample variance
`(1 / (n - 1)) Σᵢ (xᵢ - x̄)²` of `n` inputs are rational functions of the inputs, so each is a
rational target (`popVar`, `sampleVar`). The population variance is the variance of
`SumBugs.Variance`, where PyG's one-pass formula returns `-2` in binary32 (`popVar_real`).

In every IEEE binary format, the program that rounds the variance once has the certificate of
`Once.claim` on every input, for `n ≥ 2` (`cert_popVar`, `cert_sampleVar`). In every configured
binary format it returns `+0` at finite inputs that are all equal, for every `n ≥ 1` in the
population variance and every `n ≥ 2` in the sample variance (`popVar_zero`, `sampleVar_zero`).
-/

@[expose] public section

namespace TorchLean.Examples.Stable

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open TorchLean.Stable
open Arith (Val isFinite toReal valid)

/-- The population variance `(1 / n) Σᵢ (xᵢ - x̄)²` of `n` inputs, a rational target. -/
noncomputable def popVar (n : ℕ) : Target n :=
  Target.ofRat (fun v => (∑ i, (v i - (∑ j, v j) / n) ^ 2) / n) Set.univ
    (fun q => (∑ i, (q i - (∑ j, q j) / n) ^ 2) / n) fun q _ => by
      push_cast
      rfl

/-- The sample variance `(1 / (n - 1)) Σᵢ (xᵢ - x̄)²` of `n` inputs, a rational target. -/
noncomputable def sampleVar (n : ℕ) : Target n :=
  Target.ofRat (fun v => (∑ i, (v i - (∑ j, v j) / n) ^ 2) / (n - 1)) Set.univ
    (fun q => (∑ i, (q i - (∑ j, q j) / n) ^ 2) / (n - 1)) fun q _ => by
      push_cast
      rfl

/-- The population variance is the variance of `SumBugs.Variance` of the list of inputs. -/
theorem popVar_real {n : ℕ} (v : Fin n → ℝ) :
    (popVar n).F v = SumBugs.Variance.exact (List.ofFn v) := by
  simp [popVar, Target.ofRat, SumBugs.Variance.exact, List.sum_ofFn]

/-- The sum of squared deviations from the mean is the sum of squares less the squared sum over
`n`. -/
private theorem sum_sq_sub_mean {n : ℕ} (hn : n ≠ 0) (x : Fin n → ℝ) :
    ∑ i, (x i - (∑ j, x j) / n) ^ 2 = ∑ i, x i ^ 2 - (∑ j, x j) ^ 2 / n := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn
  have h := TransformerBlock.sum_sub_mean_sq x
  rw [Fintype.expect_eq_sum_div_card, Fintype.card_fin] at h
  rw [h]
  field_simp

/-- The alternating signs `(-1) ^ i` sum to `0` over an even count and to `1` over an odd one. -/
private theorem sum_alt (n : ℕ) :
    ∑ i : Fin n, (-1 : ℝ) ^ (i : ℕ) = if Even n then 0 else 1 := by
  rw [Fin.sum_univ_eq_sum_range (fun i => (-1 : ℝ) ^ i) n, neg_one_geom_sum]

/-- The sum of squares of the alternating signs is the count. -/
private theorem sum_alt_sq (n : ℕ) : ∑ i : Fin n, ((-1 : ℝ) ^ (i : ℕ)) ^ 2 = n := by
  simp only [← pow_mul, pow_mul', neg_one_sq, one_pow, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, mul_one]

/-- The sign `1` at even places and `-1` at odd ones has the real value `(-1) ^ i`. -/
private theorem alt_cast (i : ℕ) :
    (((if Even i then 1 else -1 : SignType)) : ℝ) = (-1) ^ i := by
  by_cases h : Even i
  · simp [h, h.neg_one_pow]
  · simp [h, (Nat.not_even_iff_odd.mp h).neg_one_pow]

/-- The population variance rounded once has the certificate of `Once.claim` on every input in
every IEEE binary format, for `n ≥ 2`. -/
theorem cert_popVar {n : ℕ} (hn : 2 ≤ n) :
    Cert (popVar n).prog (popVar n).F {d | valid d}
      (Once.claim (popVar n).F fun _ => (popVar n).D) := by
  have hn0 : n ≠ 0 := by omega
  have hN : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hNpos : (0 : ℝ) < n := by linarith
  set c : ℝ := if Even n then 0 else 1
  have hc : c ^ 2 ≤ 1 := by
    show (if Even n then (0 : ℝ) else 1) ^ 2 ≤ 1
    split_ifs <;> norm_num
  have hF : (popVar n).F (fun i => ((if Even (i : ℕ) then 1 else -1 : SignType) : ℝ)) =
      (n - c ^ 2 / n) / n := by
    simp only [alt_cast]
    show (∑ i : Fin n, ((-1 : ℝ) ^ (i : ℕ) - (∑ j : Fin n, (-1 : ℝ) ^ (j : ℕ)) / n) ^ 2) / n = _
    rw [sum_sq_sub_mean hn0, sum_alt_sq, sum_alt]
  have h1 : c ^ 2 / n ≤ 1 / 2 := by
    rw [div_le_iff₀ hNpos]
    linarith
  have h2 : 0 ≤ c ^ 2 / n := div_nonneg (sq_nonneg c) hNpos.le
  have h3 : 3 / 4 ≤ (n - c ^ 2 / n) / n := by
    rw [le_div_iff₀ hNpos]
    linarith
  have h4 : (n - c ^ 2 / n) / n ≤ 1 := by
    rw [div_le_one hNpos]
    linarith
  refine (popVar n).computes_prog.cert (fun i => if Even (i : ℕ) then 1 else -1) (Set.mem_univ _)
    ?_ ?_ <;> rw [hF, abs_of_nonneg] <;> linarith

/-- The sample variance rounded once has the certificate of `Once.claim` on every input in every
IEEE binary format, for `n ≥ 2`. -/
theorem cert_sampleVar {n : ℕ} (hn : 2 ≤ n) :
    Cert (sampleVar n).prog (sampleVar n).F {d | valid d}
      (Once.claim (sampleVar n).F fun _ => (sampleVar n).D) := by
  have hn0 : n ≠ 0 := by omega
  have hN : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hNpos : (0 : ℝ) < n := by linarith
  have hN0 : (n : ℝ) ≠ 0 := hNpos.ne'
  have hN1 : (n : ℝ) - 1 ≠ 0 := by linarith
  set c : ℝ := if Even n then 0 else 1
  have hc : c ^ 2 ≤ 1 := by
    show (if Even n then (0 : ℝ) else 1) ^ 2 ≤ 1
    split_ifs <;> norm_num
  -- The witness `0` at even places and `1` at odd ones is `(1 - (-1) ^ i) / 2`.
  have hw : ∀ i : ℕ, (((if Even i then 0 else 1 : SignType)) : ℝ) = (1 - (-1) ^ i) / 2 := by
    intro i
    by_cases h : Even i
    · simp [h, h.neg_one_pow]
    · simp [h, (Nat.not_even_iff_odd.mp h).neg_one_pow]
  have hsum : ∑ i : Fin n, (1 - (-1 : ℝ) ^ (i : ℕ)) / 2 = (n - c) / 2 := by
    rw [← Finset.sum_div, Finset.sum_sub_distrib, sum_alt]
    simp
    rfl
  have hsq : ∑ i : Fin n, ((1 - (-1 : ℝ) ^ (i : ℕ)) / 2) ^ 2 = (n - c) / 2 := by
    rw [← hsum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rcases Nat.even_or_odd (i : ℕ) with h | h
    · rw [h.neg_one_pow]
      norm_num
    · rw [h.neg_one_pow]
      norm_num
  have hF : (sampleVar n).F (fun i => ((if Even (i : ℕ) then 0 else 1 : SignType) : ℝ)) =
      (n ^ 2 - c ^ 2) / (4 * n * (n - 1)) := by
    simp only [hw]
    show (∑ i : Fin n, ((1 - (-1 : ℝ) ^ (i : ℕ)) / 2 -
      (∑ j : Fin n, (1 - (-1 : ℝ) ^ (j : ℕ)) / 2) / (n : ℝ)) ^ 2) / ((n : ℝ) - 1) = _
    rw [sum_sq_sub_mean hn0, hsq, hsum]
    field_simp
    ring
  have hD : 0 < 4 * (n : ℝ) * (n - 1) := by nlinarith
  have h3 : 1 / 4 < (n ^ 2 - c ^ 2) / (4 * n * (n - 1)) := by
    rw [lt_div_iff₀ hD]
    nlinarith
  have h4 : (n ^ 2 - c ^ 2) / (4 * n * (n - 1)) ≤ 1 := by
    rw [div_le_one hD]
    nlinarith
  refine (sampleVar n).computes_prog.cert (fun i => if Even (i : ℕ) then 0 else 1)
    (Set.mem_univ _) ?_ ?_ <;> rw [hF, abs_of_nonneg] <;> linarith

/-- At equal inputs the squared deviations vanish. -/
private theorem sum_sq_sub_mean_const {n : ℕ} (hn : n ≠ 0) (a : ℝ) :
    ∑ _i : Fin n, (a - (∑ _j : Fin n, a) / n) ^ 2 = 0 := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn
  simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, hn']

/-- In every configured binary format, the population variance rounded once returns `+0` at
finite inputs that are all equal, for every `n ≥ 1`. -/
theorem popVar_zero {n : ℕ} (hn : n ≠ 0) {d : Binary.Desc} {x : Fin n → Val d}
    (hx : ∀ i, isFinite (x i) = true) (h : ∀ i j, toReal (x i) = toReal (x j)) :
    Binary.toModel ((popVar n).prog.float d x) = Model.zero d.format false := by
  refine (popVar n).prog_zero ⟨hx, Set.mem_univ _⟩ ?_
  have i0 : Fin n := ⟨0, Nat.pos_of_ne_zero hn⟩
  have he : toReal ∘ x = fun _ => toReal (x i0) := funext fun i => h i i0
  show (∑ i, ((toReal ∘ x) i - (∑ j, (toReal ∘ x) j) / n) ^ 2) / n = 0
  rw [he, sum_sq_sub_mean_const hn, zero_div]

/-- In every configured binary format, the sample variance rounded once returns `+0` at finite
inputs that are all equal, for every `n ≥ 2`. -/
theorem sampleVar_zero {n : ℕ} (hn : 2 ≤ n) {d : Binary.Desc} {x : Fin n → Val d}
    (hx : ∀ i, isFinite (x i) = true) (h : ∀ i j, toReal (x i) = toReal (x j)) :
    Binary.toModel ((sampleVar n).prog.float d x) = Model.zero d.format false := by
  refine (sampleVar n).prog_zero ⟨hx, Set.mem_univ _⟩ ?_
  have i0 : Fin n := ⟨0, by omega⟩
  have he : toReal ∘ x = fun _ => toReal (x i0) := funext fun i => h i i0
  show (∑ i, ((toReal ∘ x) i - (∑ j, (toReal ∘ x) j) / n) ^ 2) / (n - 1) = 0
  rw [he, sum_sq_sub_mean_const (by omega), zero_div]

end TorchLean.Examples.Stable
