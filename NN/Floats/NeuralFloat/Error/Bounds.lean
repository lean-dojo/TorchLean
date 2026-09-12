/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding.Core
import Mathlib.Algebra.Order.Algebra
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Floating-Point Error Bounds

We collect small, compositional error bounds for TorchLean’s Flocq-style rounding model.

The lowest-level rounding interface lives in `NN/Floats/NeuralFloat/Rounding/Core.lean`: once you have a rounding
mode $r:\mathbb{R}\to\mathbb{Z}$ that satisfies the usual “round-to-nearest” property, you get the familiar
half-ULP bound for a single rounding step.

Here we package that core lemma into helper theorems that come up frequently in proofs, including
the $\operatorname{fl}(x)=x(1+\delta)$ factorization.

Bounds for dot products, matrix operations, and backward stability require a concrete evaluation
order and format hypotheses. They belong with the corresponding algorithm rather than in this
generic one-step rounding file.

## References

- N. J. Higham, "Accuracy and Stability of Numerical Algorithms", SIAM, 2nd ed., 2002.
- D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic",
  ACM Computing Surveys, 1991.
- IEEE Std 754-2019, "IEEE Standard for Floating-Point Arithmetic" (for the intended meaning of
  rounding modes and ULP terminology).
-/

@[expose] public section


namespace TorchLean.Floats.ErrorBounds

open TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-! ## Single Operation Error Bounds -/

/--
Relative error for a nonzero exact value.

The proof argument prevents a computation with a nonzero error at exact value zero from being
misreported as having zero relative error. Use an absolute-error statement when the exact value may
vanish.
-/
noncomputable def relativeError (exact computed : ℝ) (_ : exact ≠ 0) : ℝ :=
  abs (computed - exact) / abs exact

/--
Relative error bound for a single `neuralRound` step (ULP form).

This is the “divide the half-ULP absolute bound by `|x|`” version of the classic rounding model.
It is often the easiest lemma to use when a proof is naturally phrased in relative terms.
-/
theorem relative_error_round_ulp (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) (hx : x ≠ 0) :
    relativeError x (neuralRound (β := β) (fexp := fexp) rnd x) hx ≤
      neuralUlp β fexp x / (2 * abs x) := by
  have h_rel :
      relativeError x (neuralRound (β := β) (fexp := fexp) rnd x) hx =
        abs (neuralRound (β := β) (fexp := fexp) rnd x - x) / abs x := by
    rfl
  rw [h_rel]

  -- Keep the ULP expression opaque (avoid rewriting it to `bpow` via simp lemmas).
  set a : ℝ := neuralUlp β fexp x

  have h_bound := neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) x
  have h_div :
      abs (neuralRound (β := β) (fexp := fexp) rnd x - x) / abs x ≤ (a / 2) / abs x := by
    simpa [a] using div_le_div_of_nonneg_right h_bound (abs_nonneg x)

  have h_eq : (a / 2) / abs x = a / (2 * abs x) := by
    simp [div_div]

  simpa [a, h_eq] using h_div

/-! ## One-step bounds for common expressions -/

/--
Rounding error for a real addition.

This is the core half-ULP bound instantiated at the expression `x + y`.
-/
theorem round_add_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x y : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (x + y) - (x + y)) ≤
      neuralUlp β fexp (x + y) / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (x + y))

/--
Rounding error for a real multiplication.

This is the core half‑ULP bound instantiated at the expression `x * y`.
-/
theorem round_mul_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x y : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (x * y) - (x * y)) ≤
      neuralUlp β fexp (x * y) / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (x * y))

/-- Rounding error for a real division. -/
theorem round_div_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x y : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (x / y) - (x / y)) ≤
      neuralUlp β fexp (x / y) / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (x / y))

/-- Rounding error for a “fused multiply-add” style expression `x*y + z` (one rounding step). -/
theorem round_fma_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x y z : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (x * y + z) - (x * y + z)) ≤
      neuralUlp β fexp (x * y + z) / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (x * y + z))

/-- Rounding error for `Real.sqrt x` (one rounding step). -/
theorem round_sqrt_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (Real.sqrt x) - Real.sqrt x) ≤
      neuralUlp β fexp (Real.sqrt x) / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (Real.sqrt x))

/-- Rounding error for `1 / Real.sqrt x` (one rounding step). -/
theorem round_rsqrt_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (1 / Real.sqrt x) - (1 / Real.sqrt x)) ≤
      neuralUlp β fexp (1 / Real.sqrt x) / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (1 / Real.sqrt x))

/--
Relative error factorisation for rounding, with a ULP-based bound.

This is the standard model $\operatorname{fl}(x)=x(1+\delta)$, with $|\delta|$ bounded using the
ULP at $x$.
-/
theorem neural_round_relative_error_ulp (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) (hx : x
  ≠ 0) :
    ∃ δ : ℝ,
      abs δ ≤ neuralUlp β fexp x / (2 * abs x) ∧
      neuralRound (β := β) (fexp := fexp) rnd x = x * (1 + δ) := by
  refine ⟨(neuralRound (β := β) (fexp := fexp) rnd x - x) / x, ?_, ?_⟩
  · have h_abs := neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) x
    have h_div :
        abs (neuralRound (β := β) (fexp := fexp) rnd x - x) / abs x ≤
          (neuralUlp β fexp x / 2) / abs x :=
      div_le_div_of_nonneg_right h_abs (abs_nonneg x)
    have h_rhs :
        (neuralUlp β fexp x / 2) / abs x =
          neuralUlp β fexp x / (2 * abs x) := by
      simp [div_div]
    calc
      abs ((neuralRound (β := β) (fexp := fexp) rnd x - x) / x)
          = abs (neuralRound (β := β) (fexp := fexp) rnd x - x) / abs x := by
              simp [abs_div]
      _ ≤ (neuralUlp β fexp x / 2) / abs x := h_div
      _ = neuralUlp β fexp x / (2 * abs x) := h_rhs
  · have :
        x * (1 + (neuralRound (β := β) (fexp := fexp) rnd x - x) / x) =
          neuralRound (β := β) (fexp := fexp) rnd x := by
      field_simp [hx]; ring
    simpa using this.symm

end TorchLean.Floats.ErrorBounds
