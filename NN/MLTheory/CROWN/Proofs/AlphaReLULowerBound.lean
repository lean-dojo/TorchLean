/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaCROWN
public import NN.Spec.Core.Context.Real

/-!
# Scalar alpha-ReLU Lower Bounds

Soundness of the scalar lower relaxation used by alpha-CROWN. Keeping this elementary fact in one
module lets graph transfer proofs and alpha/beta phase proofs share the same argument.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Proofs

open Spec TorchLean
open NN.MLTheory.CROWN.Cert

/-- Scaling an input by a coefficient in `[0, 1]` never exceeds its ReLU. -/
theorem scaledInput_le_relu (alpha input : Real)
    (alphaNonnegative : 0 ≤ alpha) (alphaAtMostOne : alpha ≤ 1) :
    alpha * input ≤ Activation.Math.reluSpec (α := Real) input := by
  by_cases inputNonpositive : input ≤ 0
  · have : alpha * input ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos alphaNonnegative inputNonpositive
    simpa [Activation.Math.reluSpec_eq_max, max_eq_right inputNonpositive] using this
  · have inputNonnegative : 0 ≤ input := le_of_not_ge inputNonpositive
    have : alpha * input ≤ (1 : Real) * input :=
      mul_le_mul_of_nonneg_right alphaAtMostOne inputNonnegative
    simpa [Activation.Math.reluSpec_eq_max, max_eq_left inputNonnegative, one_mul] using this

/-- The alpha-CROWN scalar lower relaxation under-approximates ReLU throughout `[lower, upper]`. -/
theorem alphaRelaxLowerScalar_sound
    (lower upper alpha input : Real)
    (lowerBound : lower ≤ input) (upperBound : input ≤ upper)
    (alphaNonnegative : 0 ≤ alpha) (alphaAtMostOne : alpha ≤ 1) :
    let relaxation := alphaRelaxLowerScalar (α := Real) lower upper alpha
    relaxation.slope * input + relaxation.bias ≤
      Activation.Math.reluSpec (α := Real) input := by
  unfold alphaRelaxLowerScalar
  by_cases upperPositive : upper > 0
  · by_cases lowerPositive : lower > 0
    · have inputPositive : 0 < input := lt_of_lt_of_le lowerPositive lowerBound
      have inputNonnegative : 0 ≤ input := le_of_lt inputPositive
      simp [upperPositive, lowerPositive, Activation.Math.reluSpec_eq_max,
        max_eq_left inputNonnegative]
    · simpa [upperPositive, lowerPositive] using
        scaledInput_le_relu alpha input alphaNonnegative alphaAtMostOne
  · have inputNonpositive : input ≤ 0 := le_trans upperBound (le_of_not_gt upperPositive)
    simp [upperPositive, Activation.Math.reluSpec_eq_max, max_eq_right inputNonpositive]

end NN.MLTheory.CROWN.Proofs
