/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding
import Mathlib.Algebra.Order.Algebra
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Attr.Register

/-!
# Floating-Point Error Bounds

We collect small, compositional error bounds for TorchLean’s Flocq-style rounding model.

The lowest-level rounding interface lives in `NN/Floats/NeuralFloat/Rounding.lean`: once you have a rounding
mode `rnd : ℝ → ℤ` that satisfies the usual “round-to-nearest” property, you get the familiar
half-ULP bound for a single rounding step.

Here we package that core lemma into a few helper theorems that come up frequently in proofs
(e.g. the `fl(x) = x(1+δ)` factorization), and a lightweight mixed-precision estimate based on
machine-epsilon style proxies.

More ambitious Higham/Goldberg style results (dot products, matvec, backward stability, etc.)
require a concrete definition of the computed kernels and additional format hypotheses; those are
belong in specialized files with explicit hypotheses.

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
Relative error as a scalar metric.

We define this as `|computed - exact| / |exact|`, with the convention `relative_error 0 _ = 0`.
This convention avoids a division-by-zero branch in downstream statements.
-/
noncomputable def relativeError (exact computed : ℝ) : ℝ :=
  if exact = 0 then 0 else abs (computed - exact) / abs exact

/--
Relative error bound for a single `neural_round` step (ULP form).

This is the “divide the half-ULP absolute bound by `|x|`” version of the classic rounding model.
It is often the easiest lemma to use when a proof is naturally phrased in relative terms.
-/
theorem relative_error_round_ulp (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) (hx : x ≠ 0) :
    relativeError x (neuralRound (β := β) (fexp := fexp) rnd x) ≤
      neuralUlp β fexp x TrainingPhase.forward / (2 * abs x) := by
  have h_rel :
      relativeError x (neuralRound (β := β) (fexp := fexp) rnd x) =
        abs (neuralRound (β := β) (fexp := fexp) rnd x - x) / abs x := by
    simp [relativeError, hx]
  rw [h_rel]

  -- Keep the ULP expression opaque (avoid rewriting it to `bpow` via simp lemmas).
  set a : ℝ := neuralUlp β fexp x TrainingPhase.forward

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

This is just the core half‑ULP bound instantiated at the expression `x + y`.
-/
theorem round_add_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x y : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (x + y) - (x + y)) ≤
      neuralUlp β fexp (x + y) TrainingPhase.forward / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (x + y))

/--
Rounding error for a real multiplication.

This is the core half‑ULP bound instantiated at the expression `x * y`.
-/
theorem round_mul_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x y : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (x * y) - (x * y)) ≤
      neuralUlp β fexp (x * y) TrainingPhase.forward / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (x * y))

/-- Rounding error for a real division. -/
theorem round_div_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x y : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (x / y) - (x / y)) ≤
      neuralUlp β fexp (x / y) TrainingPhase.forward / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (x / y))

/-- Rounding error for a “fused multiply-add” style expression `x*y + z` (one rounding step). -/
theorem round_fma_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x y z : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (x * y + z) - (x * y + z)) ≤
      neuralUlp β fexp (x * y + z) TrainingPhase.forward / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (x * y + z))

/-- Rounding error for `Real.sqrt x` (one rounding step). -/
theorem round_sqrt_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (Real.sqrt x) - Real.sqrt x) ≤
      neuralUlp β fexp (Real.sqrt x) TrainingPhase.forward / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (Real.sqrt x))

/-- Rounding error for `1 / Real.sqrt x` (one rounding step). -/
theorem round_rsqrt_abs_error (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd (1 / Real.sqrt x) - (1 / Real.sqrt x)) ≤
      neuralUlp β fexp (1 / Real.sqrt x) TrainingPhase.forward / 2 := by
  simpa using (neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) (1 / Real.sqrt x))

/--
Relative error factorisation for rounding, with a ULP-based bound.

This is the standard model `fl(x) = x(1+δ)`, with `|δ|` bounded using the ULP at `x`.
-/
theorem neural_round_relative_error_ulp (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) (hx : x
  ≠ 0) :
    ∃ δ : ℝ,
      abs δ ≤ neuralUlp β fexp x TrainingPhase.forward / (2 * abs x) ∧
      neuralRound (β := β) (fexp := fexp) rnd x = x * (1 + δ) := by
  refine ⟨(neuralRound (β := β) (fexp := fexp) rnd x - x) / x, ?_, ?_⟩
  · have h_abs := neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) x
    have h_div :
        abs (neuralRound (β := β) (fexp := fexp) rnd x - x) / abs x ≤
          (neuralUlp β fexp x TrainingPhase.forward / 2) / abs x :=
      div_le_div_of_nonneg_right h_abs (abs_nonneg x)
    have h_rhs :
        (neuralUlp β fexp x TrainingPhase.forward / 2) / abs x =
          neuralUlp β fexp x TrainingPhase.forward / (2 * abs x) := by
      simp [div_div]
    calc
      abs ((neuralRound (β := β) (fexp := fexp) rnd x - x) / x)
          = abs (neuralRound (β := β) (fexp := fexp) rnd x - x) / abs x := by
              simp [abs_div]
      _ ≤ (neuralUlp β fexp x TrainingPhase.forward / 2) / abs x := h_div
      _ = neuralUlp β fexp x TrainingPhase.forward / (2 * abs x) := h_rhs
  · have :
        x * (1 + (neuralRound (β := β) (fexp := fexp) rnd x - x) / x) =
          neuralRound (β := β) (fexp := fexp) rnd x := by
      field_simp [hx]; ring
    simpa using this.symm

/-! ## Mixed Precision Error Bounds -/

/--
When accumulating in higher precision (e.g., FP32 accumulation for BFloat16 inputs),
the error is dominated by the input precision, not accumulation precision.
-/
theorem mixed_precision_accumulation_benefit
    (n : ℕ)
    (input_prec accum_prec : NeuralPrecision)
    (h_n : n ≥ 1)
    (h_accum_better :
      NeuralPrecision.machineEpsilon accum_prec ≤ NeuralPrecision.machineEpsilon input_prec / n) :
    let error_with_accum := n * NeuralPrecision.machineEpsilon input_prec +
      NeuralPrecision.machineEpsilon accum_prec
    let error_without_accum := n * NeuralPrecision.machineEpsilon input_prec
    error_with_accum ≤ 2 * error_without_accum := by
  show
    n * NeuralPrecision.machineEpsilon input_prec + NeuralPrecision.machineEpsilon accum_prec ≤
      2 * (n * NeuralPrecision.machineEpsilon input_prec)
  have h_n_cast : (n : ℝ) ≥ 1 := by
    exact Nat.one_le_cast.mpr h_n
  have h_eps_pos : 0 ≤ NeuralPrecision.machineEpsilon input_prec := by
    unfold NeuralPrecision.machineEpsilon
    have h2 : (0 : ℝ) ≤ 2 := by norm_num
    simp
  have h_accum_le :
      NeuralPrecision.machineEpsilon accum_prec ≤ NeuralPrecision.machineEpsilon input_prec := by
    calc NeuralPrecision.machineEpsilon accum_prec
        ≤ NeuralPrecision.machineEpsilon input_prec / n := h_accum_better
      _ ≤ NeuralPrecision.machineEpsilon input_prec := by
          apply div_le_self h_eps_pos
          exact h_n_cast
  calc
    n * NeuralPrecision.machineEpsilon input_prec + NeuralPrecision.machineEpsilon accum_prec
      ≤ n * NeuralPrecision.machineEpsilon input_prec + NeuralPrecision.machineEpsilon input_prec
        := by
        -- Newer Mathlib/Lean versions are pickier about the `add_le_add_left` elaboration here;
        -- `linarith` keeps the proof robust.
        linarith [h_accum_le]
    _ = (n + 1) * NeuralPrecision.machineEpsilon input_prec := by ring
    _ ≤ (2 * n) * NeuralPrecision.machineEpsilon input_prec := by
        apply mul_le_mul_of_nonneg_right _ h_eps_pos
        have : (n : ℝ) + 1 ≤ 2 * n := by linarith
        exact this
    _ = 2 * (n * NeuralPrecision.machineEpsilon input_prec) := by ring

end TorchLean.Floats.ErrorBounds
