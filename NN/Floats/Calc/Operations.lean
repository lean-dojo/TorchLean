/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Format.Digits

/-!
# Exact Operations on Mantissa/Exponent Values

These operations manipulate `NeuralFloat` representations without rounding.  Addition aligns both
operands to their smaller exponent; multiplication adds exponents.  Their correctness theorems are
radix-parametric counterparts of Flocq's `Calc.Operations` results.
-/

@[expose] public section

namespace TorchLean.Floats

/-- Integer radix powers agree with real radix powers at nonnegative exponents. -/
theorem neuralIntPower_cast_eq_bpow (β : NeuralRadix) {e : ℤ} (he : 0 ≤ e) :
    (neuralIntPower β e : ℝ) = neuralBpow β e := by
  obtain ⟨k, rfl⟩ := Int.eq_ofNat_of_zero_le he
  simp [neuralIntPower, neuralBpow, NeuralRadix.toReal]

/-- Two integer mantissas expressed on a common exponent grid. -/
structure NeuralFloatAlignment where
  /-- First aligned mantissa. -/
  leftMantissa : ℤ
  /-- Second aligned mantissa. -/
  rightMantissa : ℤ
  /-- Common exponent, equal to the smaller input exponent. -/
  exponent : ℤ

namespace NeuralFloat

variable {β : NeuralRadix}

/-- Align two representations to the smaller exponent without changing their real values. -/
def align (f g : NeuralFloat β) : NeuralFloatAlignment :=
  if f.exponent ≤ g.exponent then
    { leftMantissa := f.mantissa
      rightMantissa := g.mantissa * neuralIntPower β (g.exponent - f.exponent)
      exponent := f.exponent }
  else
    { leftMantissa := f.mantissa * neuralIntPower β (f.exponent - g.exponent)
      rightMantissa := g.mantissa
      exponent := g.exponent }

/-- Alignment preserves both represented real values. -/
theorem align_toReal (f g : NeuralFloat β) :
    neuralToReal f = neuralToReal (β := β) {
      mantissa := (align f g).leftMantissa
      exponent := (align f g).exponent } ∧
    neuralToReal g = neuralToReal (β := β) {
      mantissa := (align f g).rightMantissa
      exponent := (align f g).exponent } := by
  by_cases hfg : f.exponent ≤ g.exponent
  · have hdiff : 0 ≤ g.exponent - f.exponent := sub_nonneg.mpr hfg
    constructor
    · simp [align, hfg]
    · simp only [align, if_pos hfg]
      unfold neuralToReal
      rw [Int.cast_mul, neuralIntPower_cast_eq_bpow β hdiff]
      rw [show g.exponent = (g.exponent - f.exponent) + f.exponent by ring]
      rw [neuralBpow.add_exp]
      ring_nf
  · have hgf : g.exponent ≤ f.exponent := le_of_not_ge hfg
    have hdiff : 0 ≤ f.exponent - g.exponent := sub_nonneg.mpr hgf
    constructor
    · simp only [align, if_neg hfg]
      unfold neuralToReal
      rw [Int.cast_mul, neuralIntPower_cast_eq_bpow β hdiff]
      rw [show f.exponent = (f.exponent - g.exponent) + g.exponent by ring]
      rw [neuralBpow.add_exp]
      ring_nf
    · simp [align, hfg]

/-- Alignment selects the minimum input exponent. -/
theorem align_exponent (f g : NeuralFloat β) :
    (align f g).exponent = min f.exponent g.exponent := by
  by_cases hfg : f.exponent ≤ g.exponent
  · simp [align, hfg]
  · have hgf : g.exponent ≤ f.exponent := le_of_not_ge hfg
    simp [align, hfg, min_eq_right hgf]

/-- Exact negation of a mantissa/exponent representation. -/
def negExact (f : NeuralFloat β) : NeuralFloat β :=
  { mantissa := -f.mantissa, exponent := f.exponent }

/-- Exact absolute value of a mantissa/exponent representation. -/
def absExact (f : NeuralFloat β) : NeuralFloat β :=
  { mantissa := |f.mantissa|, exponent := f.exponent }

/-- Exact addition after exponent alignment. -/
def addExact (f g : NeuralFloat β) : NeuralFloat β :=
  let aligned := align f g
  { mantissa := aligned.leftMantissa + aligned.rightMantissa
    exponent := aligned.exponent }

/-- Exact subtraction after exponent alignment. -/
def subExact (f g : NeuralFloat β) : NeuralFloat β :=
  addExact f (negExact g)

/-- Exact multiplication of mantissas with exponent addition. -/
def mulExact (f g : NeuralFloat β) : NeuralFloat β :=
  { mantissa := f.mantissa * g.mantissa
    exponent := f.exponent + g.exponent }

/-- Negation is exact: no format can lose information by flipping a sign. -/
@[simp] theorem toReal_negExact (f : NeuralFloat β) :
    neuralToReal (negExact f) = -neuralToReal f := by
  simp [negExact, neuralToReal]

/-- Absolute value is exact, for the same reason. -/
@[simp] theorem toReal_absExact (f : NeuralFloat β) :
    neuralToReal (absExact f) = |neuralToReal f| := by
  unfold absExact neuralToReal
  rw [Int.cast_abs, abs_mul, abs_of_pos (neuralBpow.pos β f.exponent)]

/-- Addition after aligning exponents is exact on the reals.

The alignment step is where an implementation would round; keeping it exact here and rounding once
afterwards is what makes `toReal_addRounded` a one-line consequence. -/
@[simp] theorem toReal_addExact (f g : NeuralFloat β) :
    neuralToReal (addExact f g) = neuralToReal f + neuralToReal g := by
  obtain ⟨hf, hg⟩ := align_toReal f g
  unfold addExact
  dsimp only
  rw [neuralToReal, Int.cast_add, add_mul]
  change
    neuralToReal (β := β) {
        mantissa := (align f g).leftMantissa
        exponent := (align f g).exponent } +
      neuralToReal (β := β) {
        mantissa := (align f g).rightMantissa
        exponent := (align f g).exponent } =
      neuralToReal f + neuralToReal g
  rw [← hf, ← hg]

/-- Subtraction is exact, being addition of a negation. -/
@[simp] theorem toReal_subExact (f g : NeuralFloat β) :
    neuralToReal (subExact f g) = neuralToReal f - neuralToReal g := by
  simp [subExact, sub_eq_add_neg]

/-- Multiplication is exact: mantissas multiply and exponents add, with no alignment needed. -/
@[simp] theorem toReal_mulExact (f g : NeuralFloat β) :
    neuralToReal (mulExact f g) = neuralToReal f * neuralToReal g := by
  unfold mulExact neuralToReal
  rw [Int.cast_mul, neuralBpow.add_exp]
  ring

end NeuralFloat
end TorchLean.Floats
