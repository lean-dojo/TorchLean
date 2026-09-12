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
# Round to Odd

For an inexact real input, `neuralOddRound` selects the odd member of the two adjacent integers.
Exact integers are unchanged.  Applied to a binary scaled mantissa, this is the usual round-to-odd
or jamming rule: discarded information is recorded by setting the least-significant retained bit.
-/

@[expose] public section

namespace TorchLean.Floats

/-- Preserve integers; otherwise select the odd integer among `floor x` and `floor x + 1`. -/
noncomputable def neuralOddRound (x : ℝ) : ℤ :=
  let f := ⌊x⌋
  if x = (f : ℝ) then f else if Odd f then f else f + 1

/-- Round-to-odd always selects one of the two adjacent integers. -/
theorem neuralOddRound_bounds (x : ℝ) :
    ⌊x⌋ ≤ neuralOddRound x ∧ neuralOddRound x ≤ ⌊x⌋ + 1 := by
  simp only [neuralOddRound]
  split_ifs <;> simp

/-- Exact integers are fixed by round-to-odd. -/
@[simp] theorem neuralOddRound_intCast (n : ℤ) : neuralOddRound (n : ℝ) = n := by
  simp [neuralOddRound]

/-- An inexact input receives an odd integer result. -/
theorem neuralOddRound_odd_of_inexact {x : ℝ} (hx : x ≠ (⌊x⌋ : ℝ)) :
    Odd (neuralOddRound x) := by
  simp only [neuralOddRound, hx, if_false]
  by_cases hodd : Odd ⌊x⌋
  · simp [hodd]
  · rw [if_neg hodd]
    exact odd_add_one.mpr (Int.not_odd_iff_even.mp hodd)

/-- Round-to-odd is monotone and fixes every integer. -/
instance neuralOddRoundValid : NeuralValidRnd neuralOddRound where
  id := neuralOddRound_intCast
  monotone := by
    intro x y hxy
    have hfloor : ⌊x⌋ ≤ ⌊y⌋ := Int.floor_mono hxy
    rcases hfloor.eq_or_lt with hfloorEq | hfloorLt
    · by_cases hxInt : x = (⌊x⌋ : ℝ)
      · rw [hxInt, neuralOddRound_intCast]
        rw [hfloorEq]
        exact (neuralOddRound_bounds y).1
      · have hyInt : y ≠ (⌊y⌋ : ℝ) := by
          intro hy
          have hxfloor : (⌊x⌋ : ℝ) ≤ x := Int.floor_le x
          have hxyFloor : x ≤ (⌊x⌋ : ℝ) := by simpa [hfloorEq] using hxy.trans_eq hy
          exact hxInt (le_antisymm hxyFloor hxfloor)
        have hxIntY : x ≠ (⌊y⌋ : ℝ) := by simpa [← hfloorEq] using hxInt
        simp only [neuralOddRound, hxIntY, hyInt, if_false, hfloorEq]
        exact le_rfl
    · exact (neuralOddRound_bounds x).2.trans
        ((Int.add_one_le_iff.mpr hfloorLt).trans (neuralOddRound_bounds y).1)

/-- An input below an even integer rounds to an integer strictly below it. -/
theorem neuralOddRound_lt_even {x : ℝ} {n : ℤ} (hxn : x < (n : ℝ)) (hn : Even n) :
    neuralOddRound x < n := by
  have hle : neuralOddRound x ≤ n := by
    have hmono := NeuralValidRnd.monotone (rnd := neuralOddRound) x (n : ℝ) hxn.le
    simpa using hmono
  apply lt_of_le_of_ne hle
  intro heq
  by_cases hx : x = (⌊x⌋ : ℝ)
  · have hround : neuralOddRound x = ⌊x⌋ := by
      unfold neuralOddRound
      rw [if_pos hx]
    have : x = (n : ℝ) := by rw [hx, ← hround, heq]
    exact hxn.ne this
  · have hodd : Odd n := by simpa [heq] using neuralOddRound_odd_of_inexact hx
    exact (Int.not_even_iff_odd.mpr hodd) hn

/-- An input above an even integer rounds to an integer strictly above it. -/
theorem even_lt_neuralOddRound {x : ℝ} {n : ℤ} (hnx : (n : ℝ) < x) (hn : Even n) :
    n < neuralOddRound x := by
  have hle : n ≤ neuralOddRound x := by
    have hmono := NeuralValidRnd.monotone (rnd := neuralOddRound) (n : ℝ) x hnx.le
    simpa using hmono
  apply lt_of_le_of_ne hle
  intro heq
  by_cases hx : x = (⌊x⌋ : ℝ)
  · have hround : neuralOddRound x = ⌊x⌋ := by
      unfold neuralOddRound
      rw [if_pos hx]
    have : (n : ℝ) = x := by rw [hx, ← hround, ← heq]
    exact hnx.ne this
  · have hodd : Odd n := by simpa [← heq] using neuralOddRound_odd_of_inexact hx
    exact (Int.not_even_iff_odd.mpr hodd) hn

/--
Round-to-odd with at least two extra binary digits prevents nearest-even double rounding.  This is
the integer-scale core of the format-level theorem.
-/
theorem neuralNearestEven_roundOdd_binary_extra (extra : ℕ) (x : ℝ) :
    neuralNearestEven
        ((neuralOddRound (x * (2 : ℝ) ^ (extra + 2)) : ℝ) /
          (2 : ℝ) ^ (extra + 2)) =
      neuralNearestEven x := by
  let n : ℤ := ⌊x⌋
  let K : ℤ := (2 : ℤ) ^ (extra + 2)
  let H : ℤ := (2 : ℤ) ^ (extra + 1)
  let z : ℝ := x * (K : ℝ)
  let a : ℤ := neuralOddRound z
  let y : ℝ := (a : ℝ) / (K : ℝ)
  have hKtwo : K = 2 * H := by
    simp [K, H, pow_succ]
    ring
  have hKposI : 0 < K := by positivity
  have hKpos : (0 : ℝ) < (K : ℝ) := by exact_mod_cast hKposI
  have hHposI : 0 < H := by positivity
  have hHpos : (0 : ℝ) < (H : ℝ) := by exact_mod_cast hHposI
  have hKEven : Even K := by
    refine ⟨H, ?_⟩
    rw [hKtwo]
    ring
  have hHEven : Even H := by
    refine ⟨(2 : ℤ) ^ extra, ?_⟩
    simp [H, pow_succ]
    ring
  have hnLower : ((n * K : ℤ) : ℝ) ≤ z := by
    dsimp [n, z]
    push_cast
    exact mul_le_mul_of_nonneg_right (Int.floor_le x) hKpos.le
  have haLower : n * K ≤ a := by
    have hmono := NeuralValidRnd.monotone
      (rnd := neuralOddRound) ((n * K : ℤ) : ℝ) z hnLower
    rw [neuralOddRound_intCast] at hmono
    exact hmono
  have hnUpper : z < (((n + 1) * K : ℤ) : ℝ) := by
    dsimp [n, z]
    push_cast
    have hxUpper : x < (⌊x⌋ : ℝ) + 1 := Int.lt_floor_add_one x
    nlinarith
  have hUpperEven : Even ((n + 1) * K) := hKEven.mul_left (n + 1)
  have haUpper : a < (n + 1) * K :=
    neuralOddRound_lt_even hnUpper hUpperEven
  have hyLower : (n : ℝ) ≤ y := by
    rw [le_div_iff₀ hKpos]
    exact_mod_cast haLower
  have hyUpper : y < (n : ℝ) + 1 := by
    rw [div_lt_iff₀ hKpos]
    have haUpper' : (a : ℝ) < (((n + 1) * K : ℤ) : ℝ) := by
      exact_mod_cast haUpper
    calc
      (a : ℝ) < (((n + 1) * K : ℤ) : ℝ) := haUpper'
      _ = ((n : ℝ) + 1) * (K : ℝ) := by
        push_cast
        rfl
  have hyFloor : ⌊y⌋ = n := (Int.floor_eq_iff).2 ⟨hyLower, hyUpper⟩
  have hMidEven : Even (n * K + H) := (hKEven.mul_left n).add hHEven
  have hKcast : (K : ℝ) = (2 : ℝ) ^ (extra + 2) := by
    simp [K]
  have hyDef :
      (neuralOddRound (x * (2 : ℝ) ^ (extra + 2)) : ℝ) /
          (2 : ℝ) ^ (extra + 2) = y := by
    simp [y, a, z, hKcast]
  rw [hyDef]
  by_cases hlt : x - (n : ℝ) < 1 / 2
  · have hzMid : z < ((n * K + H : ℤ) : ℝ) := by
      dsimp [z]
      push_cast
      rw [hKtwo]
      push_cast
      nlinarith [hHpos]
    have haMid : a < n * K + H := neuralOddRound_lt_even hzMid hMidEven
    have hyMid : y - (n : ℝ) < 1 / 2 := by
      apply (sub_lt_iff_lt_add).2
      apply (div_lt_iff₀ hKpos).2
      have haMid' : (a : ℝ) < ((n * K + H : ℤ) : ℝ) := by
        exact_mod_cast haMid
      calc
        (a : ℝ) < ((n * K + H : ℤ) : ℝ) := haMid'
        _ = (1 / 2 + (n : ℝ)) * (K : ℝ) := by
          rw [hKtwo]
          push_cast
          ring
    rw [neural_nearest_even_eq_floor_of_frac_lt_half x (by simpa [n] using hlt)]
    rw [neural_nearest_even_eq_floor_of_frac_lt_half y (by simpa [hyFloor] using hyMid)]
    exact hyFloor
  · by_cases hgt : x - (n : ℝ) > 1 / 2
    · have hMidZ : ((n * K + H : ℤ) : ℝ) < z := by
        dsimp [z]
        push_cast
        rw [hKtwo]
        push_cast
        nlinarith [hHpos]
      have hMidA : n * K + H < a := even_lt_neuralOddRound hMidZ hMidEven
      have hyMid : y - (n : ℝ) > 1 / 2 := by
        apply (lt_sub_iff_add_lt).2
        apply (lt_div_iff₀ hKpos).2
        have hMidA' : ((n * K + H : ℤ) : ℝ) < (a : ℝ) := by
          exact_mod_cast hMidA
        calc
          (1 / 2 + (n : ℝ)) * (K : ℝ) = ((n * K + H : ℤ) : ℝ) := by
            rw [hKtwo]
            push_cast
            ring
          _ < (a : ℝ) := hMidA'
      rw [neural_nearest_even_eq_ceil_of_frac_gt_half x (by simpa [n] using hgt)]
      rw [neural_nearest_even_eq_ceil_of_frac_gt_half y (by simpa [hyFloor] using hyMid)]
      rw [hyFloor]
    · have heq : x - (n : ℝ) = 1 / 2 := by linarith
      have hzEq : z = ((n * K + H : ℤ) : ℝ) := by
        dsimp [z]
        push_cast
        rw [hKtwo]
        push_cast
        nlinarith [hHpos]
      have haEq : a = n * K + H := by
        dsimp [a]
        rw [hzEq]
        exact neuralOddRound_intCast _
      have hyEq : y - (n : ℝ) = 1 / 2 := by
        apply (sub_eq_iff_eq_add).2
        apply (div_eq_iff hKpos.ne').2
        rw [haEq, hKtwo]
        push_cast
        nlinarith
      by_cases hnEven : Even n
      · rw [neural_nearest_even_eq_floor_of_frac_half_even x (by simpa [n] using heq) hnEven]
        rw [neural_nearest_even_eq_floor_of_frac_half_even y
          (by simpa [hyFloor] using hyEq) (by simpa [hyFloor] using hnEven)]
        exact hyFloor
      · rw [neural_nearest_even_eq_ceil_of_frac_half_odd x (by simpa [n] using heq) hnEven]
        rw [neural_nearest_even_eq_ceil_of_frac_half_odd y
          (by simpa [hyFloor] using hyEq) (by simpa [hyFloor] using hnEven)]
        rw [hyFloor]

/--
Round-to-odd on a finer binary grid prevents nearest-even double rounding on a coarser grid.

The coarse grid has positive spacing `step`; the intermediate grid is finer by `extra + 2` binary
digits. The theorem is independent of any particular floating-point exponent format, so it also
applies to fixed-point and affine-quantization grids.
-/
theorem neuralRoundAtScale_nearestEven_after_odd_binary_extra
    (extra : ℕ) (step x : ℝ) (hstep : 0 < step) :
    neuralRoundAtScale neuralNearestEven step hstep
        (neuralRoundAtScale neuralOddRound
          (step / (2 : ℝ) ^ (extra + 2)) (div_pos hstep (by positivity)) x) =
      neuralRoundAtScale neuralNearestEven step hstep x := by
  let K : ℝ := (2 : ℝ) ^ (extra + 2)
  have hK : K ≠ 0 := by
    dsimp [K]
    positivity
  have hinput : x / (step / K) = (x / step) * K := by
    field_simp [ne_of_gt hstep]
  have hnormalize :
      (neuralOddRound (x / (step / K)) : ℝ) * (step / K) / step =
        (neuralOddRound ((x / step) * K) : ℝ) / K := by
    rw [hinput]
    field_simp [ne_of_gt hstep]
  unfold neuralRoundAtScale
  rw [show (2 : ℝ) ^ (extra + 2) = K by rfl]
  rw [hnormalize, neuralNearestEven_roundOdd_binary_extra extra (x / step)]

/-- If `x` is not representable, round-to-odd selects an odd scaled integer mantissa. -/
theorem neuralOddRound_scaled_mantissa_odd_of_not_generic
    {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp] {x : ℝ}
    (hx : ¬neuralGenericFormat β fexp x) :
    Odd (neuralOddRound (neuralScaledMantissa β fexp x)) := by
  apply neuralOddRound_odd_of_inexact
  intro hs
  apply hx
  apply neural_generic_format_of_scaled_mantissa_int
    (n := ⌊neuralScaledMantissa β fexp x⌋)
  exact hs

/-- Round-to-odd produces a value in the selected generic format. -/
theorem neural_generic_format_round_odd
    {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp] (x : ℝ) :
    neuralGenericFormat β fexp
      (neuralRound (β := β) (fexp := fexp) neuralOddRound x) :=
  neural_generic_format_round neuralOddRound x

/--
Semantic round-to-odd specification.  An exact input is unchanged.  An inexact result is a
directed neighbor and has an odd integer mantissa at an explicit radix exponent.
-/
def NeuralRoundOddPoint (β : NeuralRadix) (F : ℝ → Prop) (x f : ℝ) : Prop :=
  F f ∧
    (f = x ∨
      ((NeuralRoundDownPoint F x f ∨ NeuralRoundUpPoint F x f) ∧
        ∃ m : ℤ, ∃ e : ℤ, f = (m : ℝ) * neuralBpow β e ∧ Odd m))

/-- Generic rounding with `neuralOddRound` satisfies the round-to-odd point specification. -/
theorem neuralRound_odd_point
    {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp] (x : ℝ) :
    NeuralRoundOddPoint β (neuralGenericFormat β fexp) x
      (neuralRound (β := β) (fexp := fexp) neuralOddRound x) := by
  refine ⟨neural_generic_format_round_odd x, ?_⟩
  by_cases hx : neuralGenericFormat β fexp x
  · exact Or.inl (neural_round_preserves_generic neuralOddRound x hx)
  · right
    constructor
    · rcases neuralRound_eq_floor_or_ceil
        (β := β) (fexp := fexp) neuralOddRound x with hfloor | hceil
      · exact Or.inl (by simpa [hfloor] using
          (neuralRound_floor_point (β := β) (fexp := fexp) x))
      · exact Or.inr (by simpa [hceil] using
          (neuralRound_ceil_point (β := β) (fexp := fexp) x))
    · refine ⟨neuralOddRound (neuralScaledMantissa β fexp x),
        neuralCexp β fexp x, rfl, ?_⟩
      exact neuralOddRound_scaled_mantissa_odd_of_not_generic hx

end TorchLean.Floats
