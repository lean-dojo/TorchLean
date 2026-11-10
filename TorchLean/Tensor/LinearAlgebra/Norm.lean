/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- Downstream Lipschitz proofs (`TorchLean.NN.Lipschitz.Network`) reach the
-- `Context ℝ` instance only through this public import, so keep it even though nothing here names
-- it.
public import TorchLean.Tensor.Context.Real -- shake: keep-downstream
public import TorchLean.Tensor.LinearAlgebra.Real
public import TorchLean.Tensor.LinearAlgebra.Algorithms

/-!
# Sums and regularized normalization of real tensors

Additivity of `Tensor.sum` and `normalizeL2_eq`, the direct real formula of the scaled
regularized normalization.

The proofs use coordinate extensionality and finite sums, so they describe the mathematical tensor
whatever its packed storage.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Tensor

open scoped BigOperators
open Rep (dot)

/-- Sum distributes over elementwise addition. -/
theorem sum_add {s : Shape} (a b : Tensor ℝ s) :
    Tensor.sum (a + b) = Tensor.sum a + Tensor.sum b := by
  rw [sum_eq_coord_sum, sum_eq_coord_sum, sum_eq_coord_sum]
  simp only [add_def, Tensor.zipWith_apply]
  exact Finset.sum_add_distrib

/-- For `0 < epsilon`, the scaled evaluation in `Tensor.normalizeL2` equals the direct real
quotient `x / √(∑ x² + epsilon)`, because the common scale of `x` and `epsilon` cancels. With
`mean` true the sum of squares is divided by `meanDenominator [n]`, the RMS form. -/
theorem normalizeL2_eq {n : Nat} (x : Tensor ℝ [n]) (epsilon : ℝ)
    (hepsilon : 0 < epsilon) (mean : Bool := false) :
    normalizeL2 x epsilon (mean := mean) =
      Tensor.map (fun value => value / Real.sqrt
        ((if mean then Tensor.sum (Tensor.square x) / (meanDenominator [n] : ℝ) else Tensor.sum
            (Tensor.square x)) +
          epsilon)) x := by
  let scale := max (Tensor.maxAbs x) (Real.sqrt |epsilon|)
  have hs : 0 < scale := lt_of_lt_of_le
    (Real.sqrt_pos.2 (abs_pos.2 hepsilon.ne')) (le_max_right _ _)
  let scaled := Tensor.map (fun value => value / scale) x
  have hsum : Tensor.sum (Tensor.square scaled) = Tensor.sum (Tensor.square x) / scale ^ 2 := by
    rw [sum_vec, sum_vec, Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i _
    simp only [scaled, Tensor.square, getScalar_map]
    ring
  have hnorm :
      (if mean then Tensor.sum (Tensor.square scaled) / (meanDenominator [n] : ℝ) else Tensor.sum
          (Tensor.square scaled)) =
        (if mean then Tensor.sum (Tensor.square x) / (meanDenominator [n] : ℝ) else Tensor.sum
            (Tensor.square x)) /
          scale ^ 2 := by
    cases mean <;> simp only [Bool.false_eq_true, reduceIte, hsum]
    ring
  apply Tensor.ext_vector
  intro i
  simp only [normalizeL2, getScalar_map, Context.stopGradient,
    Context.stopGradient?, MathFunctions.sqrt, MathFunctions.abs]
  change (x.getScalar i / scale) / Real.sqrt
      ((if mean then Tensor.sum (Tensor.square scaled) / (meanDenominator [n] : ℝ) else Tensor.sum
          (Tensor.square scaled)) +
        (epsilon / scale) / scale) = _
  rw [hnorm]
  let total := if mean then Tensor.sum (Tensor.square x) / (meanDenominator [n] :
      ℝ) else Tensor.sum (Tensor.square x)
  have hinside : total / scale ^ 2 + (epsilon / scale) / scale =
      (total + epsilon) / scale ^ 2 := by ring
  change (x.getScalar i / scale) /
    Real.sqrt (total / scale ^ 2 + (epsilon / scale) / scale) = _
  -- The positive scale comes out of the square root and cancels the input scaling.
  rw [hinside, Real.sqrt_div' _ (sq_nonneg scale), Real.sqrt_sq hs.le]
  exact div_div_div_cancel_right₀ hs.ne' _ _

end TorchLean.Tensor
