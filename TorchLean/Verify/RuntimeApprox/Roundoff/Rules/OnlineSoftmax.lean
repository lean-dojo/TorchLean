/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.OnlineSoftmax
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Online softmax rules for `roundoff`

The online schedule of `Tensor.softmax` and `Tensor.attention` folds `Softmax.onlineStep`
over the blocks of the allowed keys of a fiber. The mask selects that list, so `roundoff` does not
read the fold as a loop over stored entries, and the rules here are the hand bounds of
`TorchLean.Verify.RuntimeApprox.OnlineSoftmax` for the two iterators. `onlineWeight_mixedError` is
`onlineWeight_error` for the weight of one key, and `onlineAttend_mixedError` is
`onlineAverage_error` for one query with `round = id`, so the weights are not rounded before their
products with the values. Each writes its allowance relative to its reference and has an absolute
form whose reference writes its sums as folds over the indices, which the absolute analysis reads
as loops.

The references are the softmax weight and the softmax average of the allowed keys on the decoded
inputs. The scores are bounded by `maxFinite / 2`, and the blocks by the smaller of the block size
and the number of allowed keys, or by that number for size `0`. A blocked key, or a fiber without an
allowed key, is exactly zero, with a zero reference, zero budgets and no range premises. When the
scores are computed, as the scaled products of attention are, the analysis propagates their errors
through the reference.
-/

public section

namespace TorchLean.OnlineSoftmax

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics

/-- Every block of the keys has at most `min size N` of the `N` keys, or all of them when
`size = 0`. -/
private theorem length_le_of_mem_toChunks_min {β : Type} (size : Nat) (keys : List β) :
    ∀ block ∈ List.toChunks size keys,
      block.length ≤ if size = 0 then keys.length else min size keys.length := by
  intro block hblock
  have hflat : block.length ≤ keys.length := by
    rw [← List.flatten_toChunks size keys, List.length_flatten]
    exact List.le_sum_of_mem (List.mem_map_of_mem hblock)
  split_ifs with hsize
  · exact hflat
  · exact le_min
      (List.length_le_of_mem_toChunks (Nat.pos_of_ne_zero hsize) keys block hblock) hflat

/-- The block bound is at least one when there is a key. -/
private theorem one_le_blockBound {size N : Nat} (hN : 0 < N) :
    (1 : ℝ) ≤ ((if size = 0 then N else min size N : Nat) : ℝ) := by
  have : 1 ≤ if size = 0 then N else min size N := by
    split_ifs with hsize
    · exact hN
    · exact le_min (Nat.pos_of_ne_zero hsize) hN
  exact_mod_cast this

/-- `onlineWeight_error`'s bound with its allowance relative to the weight `W / S`. -/
private theorem weight_bound_eq {r a W S El Ej : ℝ} (hS : S ≠ 0) (hSE : S - El ≠ 0) :
    (1 + r) * ((W * Ej + a + W / S * El) / (S - El)) + r * (W / S) + a =
      (r + (1 + r) * ((El + S * Ej) / (S - El))) * (W / S) + (a + (1 + r) * (a / (S - El))) := by
  field_simp
  ring

/-- `onlineAverage_error`'s bound with its allowance relative to the average `A`. -/
private theorem average_bound_eq {r a A El Eo S : ℝ} :
    (1 + r) * ((Eo + A * El) / (S - El)) + r * A + a =
      (r + (1 + r) * (El / (S - El))) * A + ((1 + r) * (Eo / (S - El)) + a) := by
  ring

section Configured
variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)
local notation "u" => Model.unitRoundoffAt format
local notation "η" => Model.underflowErrorAt format
local notation "maxFinite" => Model.toReal (Model.posMaxFinite format)

/-- A weight shifted by the running maximum, times its allowance, is nonnegative when the score is
at most the maximum and the exponent's constant is nonnegative. -/
private theorem allowance_nonneg {s M c : ℝ} (hs : s ≤ M) (hc : 0 ≤ c) :
    0 ≤ Real.exp (s - M) * (Real.exp (u * (M - s) + c) - 1) := by
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  refine mul_nonneg (Real.exp_pos _).le (sub_nonneg.mpr (Real.one_le_exp ?_))
  nlinarith

/-- In an IEEE format and for every block size, the online softmax weight of an allowed key `j` is
finite and has the mixed error of `onlineWeight_error` against the softmax weight
`exp s_j / ∑ exp s_i` over the allowed keys. A blocked key has weight zero. Every premise applies
only when `j` is allowed: finite allowed scores of magnitude at most `maxFinite / 2` and the ranges
of `onlineWeight_error`. -/
@[roundoff_rule]
theorem onlineWeight_mixedError (hformat : format.isIEEE = true)
    (blocks : Softmax.BlockPartition) {n : Nat} (score : Fin n → Value) (allowed : Fin n → Bool)
    (j : Fin n) :
    let keys := Softmax.onlineKeys score (fun _ => 0) allowed
    let N := keys.length
    let B := if blocks.size = 0 then N else min blocks.size N
    let T := (List.toChunks blocks.size keys).length
    let M := keys.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal (score j))
    let S := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let W := Real.exp (ExecFloat.Binary.toReal (score j) - M)
    let El := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    let Ej := Real.exp (u * (M - ExecFloat.Binary.toReal (score j)) + u) - 1
    (allowed j = true → ∀ i, allowed i = true → ExecFloat.Binary.isFinite (score i) = true) →
    (allowed j = true → ∀ i, allowed i = true →
      2 * |ExecFloat.Binary.toReal (score i)| ≤ maxFinite) →
    (allowed j = true → (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) +
        T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η ≤
      maxFinite) →
    (allowed j = true → El < S) → (allowed j = true → (W + (W * Ej + η)) / (S - El) ≤ maxFinite) →
    ExecFloat.Binary.isFinite (Softmax.onlineWeight blocks score allowed j) = true ∧
      MixedError (ExecFloat.Binary.toReal (Softmax.onlineWeight blocks score allowed j))
        (if allowed j then Real.exp (ExecFloat.Binary.toReal (score j)) /
          ∑ i, (if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) else 0) else 0)
        (if allowed j then u + (1 + u) * ((El + S * Ej) / (S - El)) else 0)
        (if allowed j then η + (1 + u) * (η / (S - El)) else 0) := by
  intro keys N B T M S W El Ej hfinite hscore hrangel hden hquot
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  by_cases hj : allowed j = true
  · have hK : 2 * (maxFinite / 2) ≤ maxFinite := le_of_eq (by ring)
    have hscore' : ∀ i, allowed i = true → ExecFloat.Binary.isFinite (score i) = true ∧
        |ExecFloat.Binary.toReal (score i)| ≤ maxFinite / 2 :=
      fun i hi => ⟨hfinite hj i hi, by linarith [hscore hj i hi]⟩
    have hB : ∀ block ∈ List.toChunks blocks.size keys, block.length ≤ B :=
      length_le_of_mem_toChunks_min blocks.size keys
    obtain ⟨hfin, hbound⟩ := onlineWeight_error hformat blocks score allowed j hj hK hscore' hB
      (hrangel hj) (hden hj) (hquot hj)
    have href : W / S = Real.exp (ExecFloat.Binary.toReal (score j)) /
        ∑ i, if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) else 0 :=
      onlineWeight_reference_eq score allowed j
    have hjkey : (score j, (0 : Value)) ∈ keys := by
      simp only [keys, Softmax.onlineKeys, List.mem_map, List.mem_filter,
        List.mem_finRange, true_and]
      exact ⟨j, hj, rfl⟩
    have hMle : ∀ e ∈ keys, ExecFloat.Binary.toReal e.1 ≤ M := fun e he =>
      List.le_foldl_max_of_mem keys (fun e : Value × Value => ExecFloat.Binary.toReal e.1) he
    have hB1 : (1 : ℝ) ≤ B :=
      one_le_blockBound (List.length_pos_iff.mpr (List.ne_nil_of_mem hjkey))
    have hEl : 0 ≤ El := by
      refine add_nonneg (List.sum_nonneg fun x hx => ?_) ?_
      · obtain ⟨e, he, rfl⟩ := List.mem_map.mp hx
        exact allowance_nonneg (hMle e he) (by have := Nat.cast_nonneg (α := ℝ) T; nlinarith)
      · have hg := one_le_residualGrowth hu hη
        unfold blockResidual
        positivity
    have hEj : 0 ≤ Ej := sub_nonneg.mpr (Real.one_le_exp (by nlinarith [hMle _ hjkey]))
    have hW : 0 < W := Real.exp_pos _
    have hSE : 0 < S - El := by linarith [hden hj]
    have hS : 0 < S := by linarith
    simp only [hj, ↓reduceIte]
    refine ⟨hfin, ⟨by positivity, by positivity, ?_⟩⟩
    rw [← href, abs_of_pos (div_pos hW hS)]
    exact hbound.trans (le_of_eq (weight_bound_eq hS.ne' hSE.ne'))
  · have hj' : allowed j = false := by simpa using hj
    have hzero : Softmax.onlineWeight blocks score allowed j = 0 := by
      unfold Softmax.onlineWeight
      split <;> simp [hj']
    simp only [hj', Bool.false_eq_true, ↓reduceIte, hzero]
    exact ⟨ExecFloat.Binary.isFinite_zero, by
      rw [ExecFloat.Binary.toReal_zero]; exact MixedError.refl 0⟩

/-- The absolute form of `onlineWeight_mixedError`, with the normalizer of the reference as a fold
over the indices, which the absolute analysis reads as a loop. -/
@[roundoff_rule]
theorem abs_toReal_onlineWeight_sub_le (hformat : format.isIEEE = true)
    (blocks : Softmax.BlockPartition) {n : Nat} (score : Fin n → Value) (allowed : Fin n → Bool)
    (j : Fin n) :
    let keys := Softmax.onlineKeys score (fun _ => 0) allowed
    let N := keys.length
    let B := if blocks.size = 0 then N else min blocks.size N
    let T := (List.toChunks blocks.size keys).length
    let M := keys.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal (score j))
    let S := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let W := Real.exp (ExecFloat.Binary.toReal (score j) - M)
    let El := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    let Ej := Real.exp (u * (M - ExecFloat.Binary.toReal (score j)) + u) - 1
    let reference := if allowed j then Real.exp (ExecFloat.Binary.toReal (score j)) /
        (List.finRange n).foldl (fun total i =>
          total + if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) else 0) 0
      else 0
    (allowed j = true → ∀ i, allowed i = true → ExecFloat.Binary.isFinite (score i) = true) →
    (allowed j = true → ∀ i, allowed i = true →
      2 * |ExecFloat.Binary.toReal (score i)| ≤ maxFinite) →
    (allowed j = true → (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) +
        T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η ≤
      maxFinite) →
    (allowed j = true → El < S) → (allowed j = true → (W + (W * Ej + η)) / (S - El) ≤ maxFinite) →
    |ExecFloat.Binary.toReal (Softmax.onlineWeight blocks score allowed j) - reference| ≤
      (if allowed j then u + (1 + u) * ((El + S * Ej) / (S - El)) else 0) * |reference| +
        (if allowed j then η + (1 + u) * (η / (S - El)) else 0) := by
  intro keys N B T M S W El Ej reference hfinite hscore hrangel hden hquot
  simp only [reference, Roundoff.foldl_finRange_add_eq_sum]
  exact (onlineWeight_mixedError hformat blocks score allowed j hfinite hscore hrangel hden
    hquot).2.bound

/-- `onlineAttend_mixedError` for the list `keys` of the allowed keys. -/
private theorem onlineAttend_mixedError_of_keys (hformat : format.isIEEE = true)
    (blocks : Softmax.BlockPartition) {n : Nat} (score value : Fin n → Value)
    (allowed : Fin n → Bool) (keys : List (Value × Value))
    (hkeys : Softmax.onlineKeys score value allowed = keys)
    (hfinite : ∀ i, allowed i = true → ExecFloat.Binary.isFinite (score i) = true)
    (hscore : ∀ i, allowed i = true → 2 * |ExecFloat.Binary.toReal (score i)| ≤ maxFinite)
    (hvalue : ∀ i, allowed i = true → ExecFloat.Binary.isFinite (value i) = true) :
    let N := keys.length
    let B := if blocks.size = 0 then N else min blocks.size N
    let T := (List.toChunks blocks.size keys).length
    let V := keys.foldl (fun m e => max m |ExecFloat.Binary.toReal e.2|) 0
    let M := keys.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal (keys.headD (0, 0)).1)
    let S := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let O := (keys.map fun e =>
      Real.exp (ExecFloat.Binary.toReal e.1 - M) * ExecFloat.Binary.toReal e.2).sum
    let El := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    let Eo := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        |ExecFloat.Binary.toReal e.2| *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T)) - 1)).sum +
      T * residualGrowth u η ^ T *
        blockResidual u η N B V (u * (B + 3 * T)) (productResidual u η 0 0 V)
    (0 < N → (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) +
        T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η ≤
      maxFinite) →
    (0 < N → (N : ℝ) * V * Real.exp (u * (B + 3 * T)) +
        T * residualGrowth u η ^ T *
          blockResidual u η N B V (u * (B + 3 * T)) (productResidual u η 0 0 V) ≤
      maxFinite) →
    (0 < N → El < S) → (0 < N → (|O| + Eo) / (S - El) ≤ maxFinite) →
    ExecFloat.Binary.isFinite (Softmax.onlineAttend blocks score value allowed) = true ∧
      MixedError
        (ExecFloat.Binary.toReal (Softmax.onlineAttend blocks score value allowed))
        (if 0 < N then (∑ i, if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) *
            ExecFloat.Binary.toReal (value i) else 0) /
          ∑ i, (if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) else 0) else 0)
        (if 0 < N then u + (1 + u) * (El / (S - El)) else 0)
        (if 0 < N then (1 + u) * (Eo / (S - El)) + η else 0) := by
  have hu : 0 ≤ u := (Model.unitRoundoffAt_pos format).le
  have hη : 0 ≤ η := (Model.underflowErrorAt_pos format).le
  have hmem : ∀ e ∈ keys, ∃ i, allowed i = true ∧ e = (score i, value i) := by
    intro e he
    rw [← hkeys] at he
    simp only [Softmax.onlineKeys, List.mem_map, List.mem_filter, List.mem_finRange,
      true_and] at he
    obtain ⟨i, hi, rfl⟩ := he
    exact ⟨i, hi, rfl⟩
  cases keys with
  | nil =>
    intro N B T V M S O El Eo _ _ _ _
    have hzero : Softmax.onlineAttend blocks score value allowed = 0 := by
      unfold Softmax.onlineAttend
      rw [hkeys]
    have hN : ¬0 < N := by simp [N]
    rw [hzero, ExecFloat.Binary.toReal_zero]
    simp only [hN, ↓reduceIte]
    exact ⟨ExecFloat.Binary.isFinite_zero, MixedError.refl 0⟩
  | cons first rest =>
    intro N B T V M S O El Eo hrangel hrangeo hden hquot
    have hpos : 0 < N := by simp [N]
    have hflat := List.flatten_toChunks blocks.size (first :: rest)
    have hV : 0 ≤ V := List.le_foldl_max_init (first :: rest)
      (fun e : Value × Value => |ExecFloat.Binary.toReal e.2|) 0
    have hkeyfacts : ∀ e ∈ (List.toChunks blocks.size (first :: rest)).flatten,
        ExecFloat.Binary.isFinite e.1 = true ∧ ExecFloat.Binary.isFinite e.2 = true ∧
          |ExecFloat.Binary.toReal e.1| ≤ maxFinite / 2 ∧ |ExecFloat.Binary.toReal e.2| ≤ V := by
      intro e he
      rw [hflat] at he
      have hVe := List.le_foldl_max_of_mem (first :: rest)
        (fun e : Value × Value => |ExecFloat.Binary.toReal e.2|) (acc := 0) he
      obtain ⟨i, hi, rfl⟩ := hmem e he
      exact ⟨hfinite i hi, hvalue i hi, by linarith [hscore i hi], hVe⟩
    have hfirst := hkeyfacts first (by rw [hflat]; exact List.mem_cons_self ..)
    have hB : ∀ block ∈ List.toChunks blocks.size (first :: rest), block.length ≤ B :=
      length_le_of_mem_toChunks_min blocks.size (first :: rest)
    have h := onlineAverage_error hformat id (uL := 0) (ηL := 0) le_rfl le_rfl
      (fun x hx h01 => ⟨hx, h01, by simp⟩) first.1
      (List.toChunks blocks.size (first :: rest)) (K := maxFinite / 2) (V := V)
      (B := B) (le_of_eq (by ring)) hV hfirst.1 hfirst.2.2.1 hB hkeyfacts
    rw [hflat] at h
    simp only [add_zero] at h
    obtain ⟨hfin, hbound⟩ := h (hrangel hpos) (hrangeo hpos) (hden hpos) (hquot hpos)
    have hattend : Softmax.onlineAttend blocks score value allowed =
        Softmax.onlineAverage (fun x => ExecFloat.Binary.CorrectlyRounded.exp x) id id
          first.1 (List.toChunks blocks.size (first :: rest)) := by
      unfold Softmax.onlineAttend
      rw [hkeys]
      simp only [mathFunctions_exp_eq hformat, Softmax.onlineStart_of_isInf_eq_false
        first rest (Model.isInf_eq_false_of_isFinite_eq_true _ hfirst.1)]
    have hshift (x : ℝ) : Real.exp (x - M) = Real.exp x * Real.exp (-M) := by
      rw [← Real.exp_add]; ring_nf
    have href : O / S = (∑ i, if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) *
          ExecFloat.Binary.toReal (value i) else 0) /
        ∑ i, if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) else 0 := by
      simp only [O, S, ← hkeys, Softmax.onlineKeys, List.map_map, Fin.sum_univ_def,
        List.sum_map_ite, Bool.decide_eq_true, List.sum_map_zero, add_zero]
      simp only [Function.comp_def, hshift, mul_right_comm _ (Real.exp (-M)),
        List.sum_map_mul_right]
      exact mul_div_mul_right _ _ (Real.exp_ne_zero _)
    have hMle : ∀ e ∈ first :: rest, ExecFloat.Binary.toReal e.1 ≤ M := fun e he =>
      List.le_foldl_max_of_mem (first :: rest)
        (fun e : Value × Value => ExecFloat.Binary.toReal e.1) he
    have hB1 : (1 : ℝ) ≤ B := one_le_blockBound hpos
    have hT : (0 : ℝ) ≤ T := Nat.cast_nonneg _
    have hg := one_le_residualGrowth hu hη
    have hEl : 0 ≤ El := by
      refine add_nonneg (List.sum_nonneg fun x hx => ?_) ?_
      · obtain ⟨e, he, rfl⟩ := List.mem_map.mp hx
        exact allowance_nonneg (hMle e he) (by nlinarith)
      · unfold blockResidual
        positivity
    have hEo : 0 ≤ Eo := by
      refine add_nonneg (List.sum_nonneg fun x hx => ?_) ?_
      · obtain ⟨e, he, rfl⟩ := List.mem_map.mp hx
        have := allowance_nonneg (format := format) (hMle e he)
          (by nlinarith : 0 ≤ u * (B + 3 * T))
        rw [mul_right_comm]
        exact mul_nonneg this (abs_nonneg _)
      · unfold blockResidual productResidual
        positivity
    have hSE : 0 < S - El := by linarith [hden hpos]
    rw [hattend]
    simp only [hpos, ↓reduceIte]
    refine ⟨hfin, ⟨by positivity, by positivity, ?_⟩⟩
    rw [← href]
    exact hbound.trans (le_of_eq average_bound_eq)

/-- In an IEEE format and for every block size, online softmax attention of one query and value
coordinate is finite and has the mixed error of `onlineAverage_error` with `round = id` against the
softmax average `∑ exp s_i v_i / ∑ exp s_i` of the allowed keys. Without an allowed key it is zero.
The premises are finite allowed scores of magnitude at most `maxFinite / 2`, finite allowed values
and, when some key is allowed, the ranges of `onlineAverage_error`. -/
@[roundoff_rule]
theorem onlineAttend_mixedError (hformat : format.isIEEE = true)
    (blocks : Softmax.BlockPartition) {n : Nat} (score value : Fin n → Value)
    (allowed : Fin n → Bool)
    (hfinite : ∀ i, allowed i = true → ExecFloat.Binary.isFinite (score i) = true)
    (hscore : ∀ i, allowed i = true → 2 * |ExecFloat.Binary.toReal (score i)| ≤ maxFinite)
    (hvalue : ∀ i, allowed i = true → ExecFloat.Binary.isFinite (value i) = true) :
    let keys := Softmax.onlineKeys score value allowed
    let N := keys.length
    let B := if blocks.size = 0 then N else min blocks.size N
    let T := (List.toChunks blocks.size keys).length
    let V := keys.foldl (fun m e => max m |ExecFloat.Binary.toReal e.2|) 0
    let M := keys.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal (keys.headD (0, 0)).1)
    let S := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let O := (keys.map fun e =>
      Real.exp (ExecFloat.Binary.toReal e.1 - M) * ExecFloat.Binary.toReal e.2).sum
    let El := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    let Eo := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        |ExecFloat.Binary.toReal e.2| *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T)) - 1)).sum +
      T * residualGrowth u η ^ T *
        blockResidual u η N B V (u * (B + 3 * T)) (productResidual u η 0 0 V)
    (0 < N → (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) +
        T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η ≤
      maxFinite) →
    (0 < N → (N : ℝ) * V * Real.exp (u * (B + 3 * T)) +
        T * residualGrowth u η ^ T *
          blockResidual u η N B V (u * (B + 3 * T)) (productResidual u η 0 0 V) ≤
      maxFinite) →
    (0 < N → El < S) → (0 < N → (|O| + Eo) / (S - El) ≤ maxFinite) →
    ExecFloat.Binary.isFinite (Softmax.onlineAttend blocks score value allowed) = true ∧
      MixedError
        (ExecFloat.Binary.toReal (Softmax.onlineAttend blocks score value allowed))
        (if 0 < N then (∑ i, if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) *
            ExecFloat.Binary.toReal (value i) else 0) /
          ∑ i, (if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) else 0) else 0)
        (if 0 < N then u + (1 + u) * (El / (S - El)) else 0)
        (if 0 < N then (1 + u) * (Eo / (S - El)) + η else 0) :=
  onlineAttend_mixedError_of_keys hformat blocks score value allowed _ rfl hfinite hscore hvalue

/-- The absolute form of `onlineAttend_mixedError`, with the sums of the reference as folds over
the indices, which the absolute analysis reads as loops. -/
@[roundoff_rule]
theorem abs_toReal_onlineAttend_sub_le (hformat : format.isIEEE = true)
    (blocks : Softmax.BlockPartition) {n : Nat} (score value : Fin n → Value)
    (allowed : Fin n → Bool)
    (hfinite : ∀ i, allowed i = true → ExecFloat.Binary.isFinite (score i) = true)
    (hscore : ∀ i, allowed i = true → 2 * |ExecFloat.Binary.toReal (score i)| ≤ maxFinite)
    (hvalue : ∀ i, allowed i = true → ExecFloat.Binary.isFinite (value i) = true) :
    let keys := Softmax.onlineKeys score value allowed
    let N := keys.length
    let B := if blocks.size = 0 then N else min blocks.size N
    let T := (List.toChunks blocks.size keys).length
    let V := keys.foldl (fun m e => max m |ExecFloat.Binary.toReal e.2|) 0
    let M := keys.foldl (fun m e => max m (ExecFloat.Binary.toReal e.1))
      (ExecFloat.Binary.toReal (keys.headD (0, 0)).1)
    let S := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M)).sum
    let O := (keys.map fun e =>
      Real.exp (ExecFloat.Binary.toReal e.1 - M) * ExecFloat.Binary.toReal e.2).sum
    let El := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T - 1)) - 1)).sum +
      T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η
    let Eo := (keys.map fun e => Real.exp (ExecFloat.Binary.toReal e.1 - M) *
        |ExecFloat.Binary.toReal e.2| *
        (Real.exp (u * (M - ExecFloat.Binary.toReal e.1) + u * (B + 3 * T)) - 1)).sum +
      T * residualGrowth u η ^ T *
        blockResidual u η N B V (u * (B + 3 * T)) (productResidual u η 0 0 V)
    let reference := if 0 < N then (List.finRange n).foldl (fun total i =>
        total + if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) *
          ExecFloat.Binary.toReal (value i) else 0) 0 /
      (List.finRange n).foldl (fun total i =>
        total + if allowed i then Real.exp (ExecFloat.Binary.toReal (score i)) else 0) 0 else 0
    (0 < N → (N : ℝ) * 1 * Real.exp (u * (B + 3 * T - 1)) +
        T * residualGrowth u η ^ T * blockResidual u η N B 1 (u * (B + 3 * T - 1)) η ≤
      maxFinite) →
    (0 < N → (N : ℝ) * V * Real.exp (u * (B + 3 * T)) +
        T * residualGrowth u η ^ T *
          blockResidual u η N B V (u * (B + 3 * T)) (productResidual u η 0 0 V) ≤
      maxFinite) →
    (0 < N → El < S) → (0 < N → (|O| + Eo) / (S - El) ≤ maxFinite) →
    |ExecFloat.Binary.toReal (Softmax.onlineAttend blocks score value allowed) -
        reference| ≤
      (if 0 < N then u + (1 + u) * (El / (S - El)) else 0) * |reference| +
        (if 0 < N then (1 + u) * (Eo / (S - El)) + η else 0) := by
  intro keys N B T V M S O El Eo reference hrangel hrangeo hden hquot
  simp only [reference, Roundoff.foldl_finRange_add_eq_sum]
  exact (onlineAttend_mixedError hformat blocks score value allowed hfinite hscore hvalue hrangel
    hrangeo hden hquot).2.bound

end Configured

end TorchLean.OnlineSoftmax
