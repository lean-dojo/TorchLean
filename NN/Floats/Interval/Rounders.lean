/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding

/-!
# Directed rounding (down/up) for Flocq-style formats

For interval propagation under a *discrete* numeric grid (float, fixed-point, quantization), one
typically wants **directed rounding** at interval endpoints:

- `down x` is a representable value with `down x ≤ x`,
- `up x` is a representable value with `x ≤ up x`.

In IEEE-754 hardware this corresponds to rounding modes “toward -∞” and “toward +∞”. In TorchLean’s
proof-oriented stack we model this with Flocq-style rounding on `ℝ` via `neural_round` together with
the floor/ceil rounding functions from `NN/Floats/NeuralFloat/Rounding.lean`.

This file is *format-generic*: it works for any radix `β` and exponent selection
function `fexp` satisfying `NeuralValidExp`.

References:
- IEEE 754-2019 (rounding modes; directed rounding).
- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
- Flocq (rounded arithmetic on reals).
-/

@[expose] public section


namespace TorchLean.Floats.Interval

open TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Format-directed rounding down to the `(β,fexp)` grid (via floor rounding of the scaled
  mantissa). -/
noncomputable def roundDown (x : ℝ) : ℝ :=
  neuralRound (β := β) (fexp := fexp) neuralFloorRound x

/-- Format-directed rounding up to the `(β,fexp)` grid (via ceil rounding of the scaled mantissa).
  -/
noncomputable def roundUp (x : ℝ) : ℝ :=
  neuralRound (β := β) (fexp := fexp) neuralCeilRound x

/--
Correctness of directed rounding down: `roundDown x` is an enclosure **lower bound**.

This is the format-generic analogue of the IEEE-754 fact that rounding “toward -∞” never exceeds
the exact real value.
-/
theorem roundDown_le (x : ℝ) : roundDown (β := β) (fexp := fexp) x ≤ x := by
  -- Unfold `neural_round` and compare scaled mantissas; `β^e` is positive.
  simp [roundDown, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal]
  set s : ℝ := neuralScaledMantissa β fexp x
  set e : ℤ := neuralCexp β fexp x
  have hx : s * neuralBpow β e = x := by
    -- `scaled_mantissa * bpow = x` is a helper lemma in `NeuralFloat/Rounding.lean`.
    simpa [s, e] using (TorchLean.Floats.neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x)
  have hb : 0 ≤ neuralBpow β e := neuralBpow.nonneg β e
  have hf : (⌊s⌋ : ℝ) ≤ s := Int.floor_le s
  -- Multiply the mantissa inequality by `β^e` and rewrite back to `x`.
  have : (⌊s⌋ : ℝ) * neuralBpow β e ≤ s * neuralBpow β e :=
    mul_le_mul_of_nonneg_right hf hb
  simpa [hx, s, e, neuralFloorRound] using this

/--
Correctness of directed rounding up: `roundUp x` is an enclosure **upper bound**.

This is the format-generic analogue of the IEEE-754 fact that rounding “toward +∞” is never below
the exact real value.
-/
theorem le_roundUp (x : ℝ) : x ≤ roundUp (β := β) (fexp := fexp) x := by
  simp [roundUp, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal]
  set s : ℝ := neuralScaledMantissa β fexp x
  set e : ℤ := neuralCexp β fexp x
  have hx : s * neuralBpow β e = x := by
    simpa [s, e] using (TorchLean.Floats.neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x)
  have hb : 0 ≤ neuralBpow β e := neuralBpow.nonneg β e
  have hc : s ≤ (⌈s⌉ : ℝ) := Int.le_ceil s
  have : s * neuralBpow β e ≤ (⌈s⌉ : ℝ) * neuralBpow β e :=
    mul_le_mul_of_nonneg_right hc hb
  simpa [hx, s, e, neuralCeilRound, mul_assoc, mul_left_comm, mul_comm] using this

/--
A focused “rounder” interface for enclosure-style interval arithmetic.

Monotonicity of `down`/`up` is not required: enclosure proofs use only
`down x ≤ x ≤ up x`.
-/
structure Rounder where
  /-- down. -/
  down : ℝ → ℝ
  /-- up. -/
  up : ℝ → ℝ
  /-- down le. -/
  down_le : ∀ x, down x ≤ x
  /-- le up. -/
  le_up : ∀ x, x ≤ up x

/-- Canonical rounder for the `(β,fexp)` format via `roundDown`/`roundUp`. -/
noncomputable def formatRounder (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp] : Rounder :=
  { down := fun x => roundDown (β := β) (fexp := fexp) x
    up := fun x => roundUp (β := β) (fexp := fexp) x
    down_le := fun x => roundDown_le (β := β) (fexp := fexp) x
    le_up := fun x => le_roundUp (β := β) (fexp := fexp) x }

end TorchLean.Floats.Interval
