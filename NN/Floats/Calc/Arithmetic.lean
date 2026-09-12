/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Calc.Round

/-!
# Rounded Arithmetic on Mantissa/Exponent Values

This module gathers the generic representation-level arithmetic. Addition, subtraction, and
multiplication first use their exact `NeuralFloat` operations. Division and square root need not
have finite radix expansions, so they are formed over the reals and rounded directly into the
selected format.

These definitions model finite rounded arithmetic. IEEE exceptional behavior for zero divisors,
negative square roots, infinities, and NaNs belongs to `NN.Floats.IEEEExec`.
-/

@[expose] public section

namespace TorchLean.Floats
namespace NeuralFloat

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Exact addition followed by rounding into the selected format. -/
noncomputable def addRounded (rnd : ℝ → ℤ) (f g : NeuralFloat β) : NeuralFloat β :=
  neuralRoundedFloat (β := β) (fexp := fexp) rnd (neuralToReal (addExact f g))

/-- Exact subtraction followed by rounding into the selected format. -/
noncomputable def subRounded (rnd : ℝ → ℤ) (f g : NeuralFloat β) : NeuralFloat β :=
  neuralRoundedFloat (β := β) (fexp := fexp) rnd (neuralToReal (subExact f g))

/-- Exact multiplication followed by rounding into the selected format. -/
noncomputable def mulRounded (rnd : ℝ → ℤ) (f g : NeuralFloat β) : NeuralFloat β :=
  neuralRoundedFloat (β := β) (fexp := fexp) rnd (neuralToReal (mulExact f g))

/-- Exact real division followed by rounding into the selected format. -/
noncomputable def divRounded (rnd : ℝ → ℤ) (f g : NeuralFloat β) : NeuralFloat β :=
  neuralRoundedFloat (β := β) (fexp := fexp) rnd (neuralToReal f / neuralToReal g)

/-- Exact real square root followed by rounding into the selected format. -/
noncomputable def sqrtRounded (rnd : ℝ → ℤ) (f : NeuralFloat β) : NeuralFloat β :=
  neuralRoundedFloat (β := β) (fexp := fexp) rnd (Real.sqrt (neuralToReal f))

/-- The rounded add denotes the rounding of the exact real sum. -/
@[simp] theorem toReal_addRounded (rnd : ℝ → ℤ) (f g : NeuralFloat β) :
    neuralToReal (addRounded (fexp := fexp) rnd f g) =
      neuralRound (β := β) (fexp := fexp) rnd (neuralToReal f + neuralToReal g) := by
  unfold addRounded neuralRoundedFloat
  rw [toReal_addExact]
  rfl

/-- The rounded sub denotes the rounding of the exact real difference. -/
@[simp] theorem toReal_subRounded (rnd : ℝ → ℤ) (f g : NeuralFloat β) :
    neuralToReal (subRounded (fexp := fexp) rnd f g) =
      neuralRound (β := β) (fexp := fexp) rnd (neuralToReal f - neuralToReal g) := by
  unfold subRounded neuralRoundedFloat
  rw [toReal_subExact]
  rfl

/-- The rounded mul denotes the rounding of the exact real product. -/
@[simp] theorem toReal_mulRounded (rnd : ℝ → ℤ) (f g : NeuralFloat β) :
    neuralToReal (mulRounded (fexp := fexp) rnd f g) =
      neuralRound (β := β) (fexp := fexp) rnd (neuralToReal f * neuralToReal g) := by
  unfold mulRounded neuralRoundedFloat
  rw [toReal_mulExact]
  rfl

/-- The rounded div denotes the rounding of the exact real quotient. -/
@[simp] theorem toReal_divRounded (rnd : ℝ → ℤ) (f g : NeuralFloat β) :
    neuralToReal (divRounded (fexp := fexp) rnd f g) =
      neuralRound (β := β) (fexp := fexp) rnd (neuralToReal f / neuralToReal g) := by
  rfl

/-- The rounded square root denotes the rounding of the exact real square root.

Together with the four lemmas above this is the whole content of "correctly rounded arithmetic": each
operation is specified by the real operation followed by one rounding, so an error analysis never has
to look inside an implementation. -/
@[simp] theorem toReal_sqrtRounded (rnd : ℝ → ℤ) (f : NeuralFloat β) :
    neuralToReal (sqrtRounded (fexp := fexp) rnd f) =
      neuralRound (β := β) (fexp := fexp) rnd (Real.sqrt (neuralToReal f)) := by
  rfl

/-- A nonzero mantissa gives a nonzero represented real value. -/
theorem toReal_ne_zero_of_mantissa_ne_zero {f : NeuralFloat β} (hf : f.mantissa ≠ 0) :
    neuralToReal f ≠ 0 := by
  unfold neuralToReal
  exact mul_ne_zero (by exact_mod_cast hf) (ne_of_gt (neuralBpow.pos β f.exponent))

end NeuralFloat
end TorchLean.Floats
