/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Core
public import NN.Floats.NeuralFloat.Rounding

/-!
# NeuralFloat Conversion Helpers

This module contains small utilities for *moving values between*:

- real semantics (`ℝ`), and
- the Flocq-style mantissa/exponent payload `NeuralFloat β`.

Important scope note:

- The canonical rounding semantics in this folder is `neural_round` (see `Rounding.lean`), which
  models **one** “compute in `ℝ`, then round” step and comes with the key half-ULP theorem
  `neural_error_bound_ulp` (under round-to-nearest assumptions).
- The helpers in this file are conversion utilities and examples: packaging the rounded result into a
  `NeuralFloat` record and attaching a small amount of error/ULP metadata.

If you are doing proofs about rounding error, prefer theorems in:

- `NN/Floats/NeuralFloat/Rounding.lean` (core half-ULP bound), and
- `NN/Floats/NeuralFloat/ErrorBounds.lean` (small derived bounds).
-/

@[expose] public section


namespace TorchLean.Floats.Conversion

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/--
Conversion result with error tracking.
-/
structure NeuralConversionResult (α : Type) where
  /-- Converted value. -/
  value : α
  /--
  Absolute error bound for the conversion step.

  For conversions that only *repackage* an already-rounded value, this may be a metadata field
  rather than a proved bound.
  -/
  error_bound : ℝ
  /-- Whether the conversion was exact (`error_bound = 0`). -/
  is_exact : Bool
  /--
  Optional ULP-scale metadata at the input point.

  This is not itself an error bound. For round-to-nearest, a typical one-step error bound is
  `ulp(x)/2`, proved separately (see `neural_error_bound_ulp`).
  -/
  ulp_error : Option ℝ
  /-- Training phase used when interpreting ULP scaling hooks. -/
  target_phase : TrainingPhase := TrainingPhase.forward

/--
Convert a `NeuralFloat β` payload to its real semantics, returning lightweight metadata.

In words: the returned `value` is exactly `neural_to_real f`.
The `error_bound` field is the *metadata* stored in the input record (it is not recomputed here).
-/
noncomputable def neuralFloatToReal (f : NeuralFloat β) : NeuralConversionResult ℝ :=
  { value := neuralToReal f,
    error_bound := f.error_bound,
    is_exact := f.error_bound = 0,
    ulp_error := none,  -- No `fexp` available: we cannot compute a principled ULP scale here.
    target_phase := f.phase }

/--
Round a real number `x` onto the `(β,fexp,rnd)` grid and package it as a `NeuralFloat β`.

In words:
The returned payload is the same one used by `neural_round` (it uses
`mantissa := rnd (scaled_mantissa x)` and `exponent := cexp x`), and
`error_bound = |neural_to_real(value) - x|` is the *actual* absolute error for this rounding step.

If you additionally assume round-to-nearest (`NeuralValidRndToNearest`), the theorem
`neural_error_bound_ulp` bounds this error by `ulp(x)/2`.
-/
noncomputable def realToNeuralFloat (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    (x : ℝ) (phase : TrainingPhase := TrainingPhase.forward) :
    NeuralConversionResult (NeuralFloat β) := by
  let m : ℤ := rnd (neuralScaledMantissa β fexp x)
  let e : ℤ := neuralCexp β fexp x
  let f : NeuralFloat β :=
    { mantissa := m
      exponent := e
      precision := NeuralPrecision.ieee_single
      error_bound := 0
      phase := phase
      layer_id := 0
      batch_id := 0 }
  let y : ℝ := neuralToReal (β := β) f
  let err : ℝ := abs (y - x)
  exact
    { value := { f with error_bound := err }
      error_bound := err
      is_exact := err = 0
      ulp_error := some (neuralUlp β fexp x phase)
      target_phase := phase }

/-- Convert list of reals to neural floats -/
noncomputable def listToNeuralFloats (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    (values : List ℝ) (phase : TrainingPhase := TrainingPhase.forward) : NeuralConversionResult
      (List (NeuralFloat β)) := by
  let conv := values.map (fun x => realToNeuralFloat (β := β) (fexp := fexp) rnd x phase)
  let converted := conv.map (·.value)
  let max_error := (conv.map (·.error_bound)).foldl max 0
  exact
    { value := converted
      error_bound := max_error
      is_exact := max_error = 0
      ulp_error := some max_error
      target_phase := phase }

/-- Convert neural floats back to reals -/
noncomputable def neuralFloatsToList (neural_floats : List (NeuralFloat β)) :
  NeuralConversionResult (List ℝ) := by
  let converted := neural_floats.map neuralToReal
  let max_error := (neural_floats.map (·.error_bound)).foldl max 0
  exact {
    value := converted,
    error_bound := max_error,
    is_exact := max_error = 0,
    ulp_error := some max_error,
    target_phase := TrainingPhase.forward
  }

/-- Test conversion round-trip accuracy -/
noncomputable def testRoundTripAccuracy (original : ℝ) (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (phase
  : TrainingPhase := TrainingPhase.forward) : ℝ := by
  let neural_conv := realToNeuralFloat (β := β) (fexp := fexp) rnd original phase
  let back_conv := neuralFloatToReal neural_conv.value
  exact abs (back_conv.value - original)

/-- Validate conversion preserves essential properties -/
noncomputable def validateNeuralConversion (original : ℝ) (converted : NeuralConversionResult ℝ)
    (tolerance : ℝ) : Bool :=
  converted.error_bound ≤ tolerance ∧
  abs (converted.value - original) ≤ tolerance

end TorchLean.Floats.Conversion
