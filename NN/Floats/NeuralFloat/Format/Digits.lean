/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Format.Magnitude

/-!
# Integer Digits, Scaling, and Slices

These definitions are the effective integer layer used by radix-based rounding algorithms.  Signed
division and remainder use `Int.tdiv` and `Int.tmod`, matching Flocq's quotient and remainder toward
zero rather than Lean's Euclidean `/` and `%` operations.
-/

@[expose] public section

namespace TorchLean.Floats

/-- Integer radix power, with the Flocq convention that negative powers are zero. -/
def neuralIntPower (β : NeuralRadix) (e : ℤ) : ℤ :=
  if 0 ≤ e then Int.ofNat (β.base ^ e.toNat) else 0

/-- On nonnegative exponents the radix power is the ordinary one. -/
@[simp] theorem neuralIntPower_of_nonneg (β : NeuralRadix) {e : ℤ} (he : 0 ≤ e) :
    neuralIntPower β e = Int.ofNat (β.base ^ e.toNat) := by
  simp [neuralIntPower, he]

/-- Negative exponents give zero, which is the Flocq convention rather than a truncation accident:
it keeps `neuralIntPower` an integer function and makes the slice lemmas below unconditional. -/
@[simp] theorem neuralIntPower_of_neg (β : NeuralRadix) {e : ℤ} (he : e < 0) :
    neuralIntPower β e = 0 := by
  have hnot : ¬0 ≤ e := by linarith
  simp [neuralIntPower, hnot]

/-- `β⁰ = 1`. -/
@[simp] theorem neuralIntPower_zero (β : NeuralRadix) : neuralIntPower β 0 = 1 := by
  simp [neuralIntPower]

/-- The signed radix digit at position `k`. -/
def neuralDigit (β : NeuralRadix) (n k : ℤ) : ℤ :=
  Int.tmod (Int.tdiv n (neuralIntPower β k)) (Int.ofNat β.base)

/-- Every digit of zero is zero. -/
@[simp] theorem neuralDigit_zero (β : NeuralRadix) (k : ℤ) : neuralDigit β 0 k = 0 := by
  simp [neuralDigit]

/-- Digits are odd in the value, because `tdiv` and `tmod` truncate toward zero on both signs. -/
@[simp] theorem neuralDigit_neg (β : NeuralRadix) (n k : ℤ) :
    neuralDigit β (-n) k = -neuralDigit β n k := by
  simp [neuralDigit]

/-- There are no digits below position zero: this layer describes integers, not fractions. -/
theorem neuralDigit_of_neg_index (β : NeuralRadix) (n : ℤ) {k : ℤ} (hk : k < 0) :
    neuralDigit β n k = 0 := by
  simp [neuralDigit, neuralIntPower_of_neg β hk]

/-- Every signed digit has absolute value strictly smaller than the radix. -/
theorem neuralDigit_abs_lt_base (β : NeuralRadix) (n k : ℤ) :
    |neuralDigit β n k| < Int.ofNat β.base := by
  have hb : (0 : ℤ) < Int.ofNat β.base := by
    exact Int.natCast_pos.mpr (Nat.zero_lt_of_lt β.base_valid)
  have hlower := Int.lt_tmod_of_pos
    (Int.tdiv n (neuralIntPower β k)) hb
  have hupper := Int.tmod_lt_of_pos
    (Int.tdiv n (neuralIntPower β k)) hb
  rw [abs_lt]
  exact ⟨by simpa [neuralDigit] using hlower, by simpa [neuralDigit] using hupper⟩

/-- Shift an integer left for nonnegative `k`, and right with truncation for negative `k`. -/
def neuralScale (β : NeuralRadix) (n k : ℤ) : ℤ :=
  if 0 ≤ k then n * neuralIntPower β k else Int.tdiv n (neuralIntPower β (-k))

/-- A nonnegative shift is exact multiplication by a radix power. -/
theorem neuralScale_of_nonneg (β : NeuralRadix) (n : ℤ) {k : ℤ} (hk : 0 ≤ k) :
    neuralScale β n k = n * neuralIntPower β k := by
  simp [neuralScale, hk]

/-- A negative shift truncates toward zero, which is where information is lost and hence where the
rounding proofs have to do real work. -/
theorem neuralScale_of_neg (β : NeuralRadix) (n : ℤ) {k : ℤ} (hk : k < 0) :
    neuralScale β n k = Int.tdiv n (neuralIntPower β (-k)) := by
  have hnot : ¬0 ≤ k := by linarith
  simp [neuralScale, hnot]

/-- Shifting zero gives zero, at any shift. -/
@[simp] theorem neuralScale_zero_value (β : NeuralRadix) (k : ℤ) : neuralScale β 0 k = 0 := by
  simp [neuralScale]

/-- A zero shift is the identity. -/
@[simp] theorem neuralScale_zero_shift (β : NeuralRadix) (n : ℤ) : neuralScale β n 0 = n := by
  simp [neuralScale]

/-- Shifting is odd in the value, in both the multiplying and the truncating branch. -/
@[simp] theorem neuralScale_neg (β : NeuralRadix) (n k : ℤ) :
    neuralScale β (-n) k = -neuralScale β n k := by
  unfold neuralScale
  split <;> simp

/-- Extract `width` radix digits beginning at `start`; negative widths produce zero. -/
def neuralSlice (β : NeuralRadix) (n start width : ℤ) : ℤ :=
  if 0 ≤ width then
    Int.tmod (neuralScale β n (-start)) (neuralIntPower β width)
  else 0

/-- Any slice of zero is zero. -/
@[simp] theorem neuralSlice_zero_value (β : NeuralRadix) (start width : ℤ) :
    neuralSlice β 0 start width = 0 := by
  simp [neuralSlice]

/-- A negative width selects no digits. Total rather than partial, so callers need no side
condition; the interesting bound is `neuralSlice_abs_lt_power`. -/
theorem neuralSlice_of_neg_width (β : NeuralRadix) (n start : ℤ) {width : ℤ}
    (hwidth : width < 0) : neuralSlice β n start width = 0 := by
  have hnot : ¬0 ≤ width := by linarith
  simp [neuralSlice, hnot]

/-- A nonnegative-width slice has absolute value below the corresponding radix power. -/
theorem neuralSlice_abs_lt_power (β : NeuralRadix) (n start : ℤ) {width : ℤ}
    (hwidth : 0 ≤ width) :
    |neuralSlice β n start width| < neuralIntPower β width := by
  have hpow : (0 : ℤ) < neuralIntPower β width := by
    rw [neuralIntPower_of_nonneg β hwidth]
    exact Int.natCast_pos.mpr (pow_pos (Nat.zero_lt_of_lt β.base_valid) width.toNat)
  have hlower := Int.lt_tmod_of_pos (neuralScale β n (-start)) hpow
  have hupper := Int.tmod_lt_of_pos (neuralScale β n (-start)) hpow
  rw [abs_lt]
  exact ⟨by simpa [neuralSlice, hwidth] using hlower,
    by simpa [neuralSlice, hwidth] using hupper⟩

/-- Slices are odd in the value, inherited from `neuralScale` and `Int.tmod`. -/
@[simp] theorem neuralSlice_neg (β : NeuralRadix) (n start width : ℤ) :
    neuralSlice β (-n) start width = -neuralSlice β n start width := by
  unfold neuralSlice
  split <;> simp

/-- Number of base-`β` digits in the absolute value of an integer. -/
def neuralDigits (β : NeuralRadix) (n : ℤ) : ℕ :=
  if n = 0 then 0 else Nat.log β.base n.natAbs + 1

/-- Zero is given zero digits rather than one, again following Flocq. -/
@[simp] theorem neuralDigits_zero (β : NeuralRadix) : neuralDigits β 0 = 0 := by
  simp [neuralDigits]

/-- The digit count depends only on the magnitude. -/
@[simp] theorem neuralDigits_neg (β : NeuralRadix) (n : ℤ) :
    neuralDigits β (-n) = neuralDigits β n := by
  by_cases hn : n = 0
  · subst n
    simp
  · simp [neuralDigits, hn]

/-- A nonzero integer has at least one digit, the converse of `neuralDigits_zero`. -/
theorem neuralDigits_pos {β : NeuralRadix} {n : ℤ} (hn : n ≠ 0) :
    0 < neuralDigits β n := by
  simp [neuralDigits, hn]

/-- A nonzero integer lies between consecutive powers selected by its digit count. -/
theorem neuralDigits_bounds (β : NeuralRadix) {n : ℤ} (hn : n ≠ 0) :
    β.base ^ (neuralDigits β n - 1) ≤ n.natAbs ∧
      n.natAbs < β.base ^ neuralDigits β n := by
  have habs : n.natAbs ≠ 0 := Int.natAbs_ne_zero.mpr hn
  have hbase : 1 < β.base := lt_of_lt_of_le Nat.one_lt_two β.base_valid
  simp only [neuralDigits, if_neg hn]
  constructor
  · simpa using Nat.pow_log_le_self β.base habs
  · simpa [Nat.pow_succ] using Nat.lt_pow_succ_log_self hbase n.natAbs

/-- The radix digit count of a positive integer is the magnitude of its real embedding. -/
theorem neuralMagnitude_intCast_of_pos (β : NeuralRadix) {n : ℤ} (hn : 0 < n) :
    neuralMagnitude β (n : ℝ) = (neuralDigits β n : ℤ) := by
  have hn0 : n ≠ 0 := ne_of_gt hn
  have hdPos := neuralDigits_pos (β := β) hn0
  obtain ⟨d, hd⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hdPos)
  have hb := neuralDigits_bounds β hn0
  rw [hd] at hb ⊢
  have hnabs : (n.natAbs : ℤ) = n := Int.natAbs_of_nonneg hn.le
  have hlower : (β.base ^ d : ℤ) ≤ n := by
    rw [← hnabs]
    exact_mod_cast (by simpa using hb.1)
  have hupper : n < (β.base ^ (d + 1) : ℕ) := by
    rw [← hnabs]
    exact_mod_cast (by simpa [Nat.succ_eq_add_one] using hb.2)
  apply neuralMagnitude_eq_of_bpow_bounds β (n : ℝ) ((d : ℤ) + 1)
  · exact_mod_cast hn0
  · rw [abs_of_pos (by exact_mod_cast hn)]
    simpa [neuralBpow, NeuralRadix.toReal] using (show (β.base ^ d : ℝ) ≤ n by exact_mod_cast hlower)
  · rw [abs_of_pos (by exact_mod_cast hn)]
    change (n : ℝ) < β.toReal ^ ((d : ℤ) + 1)
    rw [show (d : ℤ) + 1 = ((d + 1 : ℕ) : ℤ) by norm_num, zpow_natCast]
    simpa [NeuralRadix.toReal] using (show (n : ℝ) < β.base ^ (d + 1) by
      exact_mod_cast hupper)

/-- A positive integer is strictly below the radix power selected by its digit count. -/
theorem intCast_lt_neuralBpow_digits (β : NeuralRadix) {n : ℤ} (hn : 0 < n) :
    (n : ℝ) < neuralBpow β (neuralDigits β n : ℤ) := by
  have h := abs_lt_neuralBpow_magnitude β (n : ℝ) (by exact_mod_cast (ne_of_gt hn))
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  simpa [abs_of_pos hnR, neuralMagnitude_intCast_of_pos β hn] using h

/-- The successor of a positive integer does not exceed its next radix-power boundary. -/
theorem intCast_add_one_le_neuralBpow_digits (β : NeuralRadix) {n : ℤ} (hn : 0 < n) :
    ((n + 1 : ℤ) : ℝ) ≤ neuralBpow β (neuralDigits β n : ℤ) := by
  have hb := (neuralDigits_bounds β (ne_of_gt hn)).2
  have hnabs : (n.natAbs : ℤ) = n := Int.natAbs_of_nonneg hn.le
  have hz : n + 1 ≤ (β.base ^ neuralDigits β n : ℕ) := by
    have hbZ : (n.natAbs : ℤ) < (β.base ^ neuralDigits β n : ℕ) := by
      exact_mod_cast hb
    rw [hnabs] at hbZ
    exact Int.add_one_le_iff.mpr hbZ
  rw [show (neuralDigits β n : ℤ) = ((neuralDigits β n : ℕ) : ℤ) by rfl,
    neuralBpow, zpow_natCast]
  simpa [NeuralRadix.toReal] using (show ((n + 1 : ℤ) : ℝ) ≤ β.base ^ neuralDigits β n by
    exact_mod_cast hz)

/-- The power bounds uniquely determine the digit count of a nonzero integer. -/
theorem neuralDigits_unique (β : NeuralRadix) {n : ℤ} (hn : n ≠ 0) {d : ℕ} (hd : 0 < d)
    (hlower : β.base ^ (d - 1) ≤ n.natAbs)
    (hupper : n.natAbs < β.base ^ d) : neuralDigits β n = d := by
  have hlog : Nat.log β.base n.natAbs = d - 1 :=
    Nat.log_eq_of_pow_le_of_lt_pow hlower (by simpa [Nat.sub_add_cancel hd] using hupper)
  simp [neuralDigits, hn, hlog, Nat.sub_add_cancel hd]

/-- Digit count is monotone with respect to integer absolute value away from zero. -/
theorem neuralDigits_mono_abs (β : NeuralRadix) {n m : ℤ} (hn : n ≠ 0)
    (hnm : n.natAbs ≤ m.natAbs) : neuralDigits β n ≤ neuralDigits β m := by
  have hm : m ≠ 0 := by
    intro hm
    subst m
    simp only [Int.natAbs_zero] at hnm
    have hnabs : n.natAbs = 0 := Nat.eq_zero_of_le_zero hnm
    exact hn (Int.natAbs_eq_zero.mp hnabs)
  simp only [neuralDigits, if_neg hn, if_neg hm]
  exact Nat.add_le_add_right (Nat.log_mono_right hnm) 1

/-- A nonzero product uses at most the sum of the operand digit counts. -/
theorem neuralDigits_mul_le (β : NeuralRadix) {n m : ℤ} (hn : n ≠ 0) (hm : m ≠ 0) :
    neuralDigits β (n * m) ≤ neuralDigits β n + neuralDigits β m := by
  have hnm : n * m ≠ 0 := mul_ne_zero hn hm
  have hnBounds := neuralDigits_bounds β hn
  have hmBounds := neuralDigits_bounds β hm
  have hprod : (n * m).natAbs < β.base ^ (neuralDigits β n + neuralDigits β m) := by
    rw [Int.natAbs_mul, pow_add]
    calc
      n.natAbs * m.natAbs < β.base ^ neuralDigits β n * m.natAbs :=
        Nat.mul_lt_mul_of_pos_right hnBounds.2 (Int.natAbs_pos.mpr hm)
      _ < β.base ^ neuralDigits β n * β.base ^ neuralDigits β m :=
        Nat.mul_lt_mul_of_pos_left hmBounds.2
          (pow_pos (Nat.zero_lt_of_lt β.base_valid) _)
  have hlog : Nat.log β.base (n * m).natAbs <
      neuralDigits β n + neuralDigits β m :=
    Nat.log_lt_of_lt_pow (Int.natAbs_ne_zero.mpr hnm) hprod
  simp only [neuralDigits, if_neg hnm]
  exact Nat.add_one_le_iff.mpr hlog

/-- A positive radix power has one more digit than its exponent. -/
theorem neuralDigits_power (β : NeuralRadix) (k : ℕ) :
    neuralDigits β (Int.ofNat (β.base ^ k)) = k + 1 := by
  have hbase : 1 < β.base := lt_of_lt_of_le Nat.one_lt_two β.base_valid
  have hpow : β.base ^ k ≠ 0 := pow_ne_zero _ (Nat.ne_of_gt (Nat.zero_lt_of_lt β.base_valid))
  have hpowInt : Int.ofNat (β.base ^ k) ≠ 0 := Int.ofNat_ne_zero.mpr hpow
  rw [neuralDigits, if_neg hpowInt]
  simp [Nat.log_pow hbase]

end TorchLean.Floats
