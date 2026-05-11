/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import NN.Floats.NeuralFloat.Metadata
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# NeuralFloat core (Flocq-style rounded arithmetic)

TorchLean frequently reasons about floating-point behavior using a classical "rounded arithmetic on
`ℝ`" approach rather than a bit-level IEEE-754 model:

- represent a value as an integer mantissa `m : ℤ` and exponent `e : ℤ`,
- interpret it as a real number `m * β^e`,
- describe the *format* via an exponent-selection function `fexp : ℤ → ℤ` and a rounding operator.

This decomposition is the same one used by the Coq library **Flocq**. It makes many theorems
reusable across formats (fixed-point, unbounded floats, bounded IEEE-like floats) and aligns well
with ULP-style error bounds from numerical analysis.

TorchLean also cares about *mixed precision* training/inference (FP16/bfloat16/TF32/FP32/FP64).
In this folder, "precision" is not a promise about bit-level encoding; it's a parameter that
selects mantissa/exponent sizes and is used by the format and error-bound layers.

For executable, bit-level IEEE-754 semantics (NaN/Inf/signed zero), see `NN/Floats/IEEEExec/`.

## References

- Flocq project (documentation + sources): https://flocq.gitlabpages.inria.fr/flocq/
- S. Boldo, G. Melquiond, “Flocq: a unified Coq library for proving floating-point algorithms
  correct”
  (ARITH 2011), DOI: 10.1109/ARITH.2011.40
- IEEE Standard for Floating-Point Arithmetic (IEEE 754-2019)
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002
-/

@[expose] public section


namespace TorchLean.Floats

/--
Radix (base) for "floating-point-like" representations.

In practice we almost always use base 2 (`binary_radix`) because that's what hardware implements,
but keeping the base explicit helps make the model match the literature (and it keeps some proofs
parametric in the radix).
-/
structure NeuralRadix where
  /-- Radix base (e.g. `2` for binary). -/
  base : ℕ
  /-- Validity condition: the base is at least 2. -/
  base_valid : 2 ≤ base

/-- Standard binary radix (`β = 2`). -/
def binaryRadix : NeuralRadix := ⟨2, by norm_num⟩

/-- Decimal radix (`β = 10`, useful for small examples and exact decimal inputs). -/
def decimalRadix : NeuralRadix := ⟨10, by norm_num⟩

namespace NeuralRadix

variable (r : NeuralRadix)

/-- Coerce the radix base to `ℝ` (used by `bpow` and logarithms). -/
def toReal : ℝ := r.base

/-- The radix base is positive when viewed as a real number. -/
lemma pos : 0 < r.toReal := by
  have hb : 0 < r.base := lt_of_lt_of_le (by norm_num) r.base_valid
  simpa [toReal] using (Nat.cast_pos.mpr hb)

/-- The radix base is nonzero when viewed as a real number. -/
lemma ne_zero : r.toReal ≠ 0 := ne_of_gt (pos r)

/-- The radix base is strictly greater than 1 (for a valid radix, `2 ≤ base`). -/
lemma gt_one : 1 < r.toReal := by
  simp [toReal]
  exact Nat.one_lt_cast.mpr (Nat.succ_le_iff.mp r.base_valid)

end NeuralRadix

/--
An abstract floating-point value (mantissa/exponent) plus analysis metadata.

The core mathematical payload is `mantissa` and `exponent`. The other fields are there to support
mixed precision and simple error-tracking experiments used in some TorchLean demos:

- `precision`: a named format (FP16/FP32/…); see `NeuralPrecision`.
- `error_bound`: a conservative absolute error bound attached by a conversion/rounding pass.
- `phase`: forward/backward/update/inference (see `TrainingPhase.requires_high_precision`).
- `layer_id`/`batch_id`: lightweight tags used by “track where an error came from”
  utilities.
-/
structure NeuralFloat (β : NeuralRadix) where
  /-- Integer mantissa `m`. -/
  mantissa : ℤ
  /-- Integer exponent `e`. -/
  exponent : ℤ
  /-- Named precision metadata (e.g. FP16/FP32); does not affect `to_real`. -/
  precision : NeuralPrecision
  /-- Conservative absolute error bound metadata attached by conversions/rounding passes. -/
  error_bound : ℝ
  /-- Training phase metadata (forward/backward/update/inference). -/
  phase : TrainingPhase
  /-- Optional layer id tag (used in demos/experiments). -/
  layer_id : ℕ
  /-- Optional batch id tag (used in demos/experiments). -/
  batch_id : ℕ

namespace NeuralFloat

variable {β : NeuralRadix} (f : NeuralFloat β)

/-- Structural zero test (mantissa is exactly `0`). -/
def isZero : Prop := f.mantissa = 0

/-- Sign of the mantissa (matches the sign of `to_real` since `β^e > 0`). -/
def sign : ℤ := Int.sign f.mantissa

end NeuralFloat

/--
Base power: `β^e` as a real number.

This is Flocq's `bpow` concept: the scaling factor used to interpret mantissa/exponent pairs.
-/
noncomputable def neuralBpow (β : NeuralRadix) (e : ℤ) : ℝ := β.toReal ^ e

namespace neuralBpow

variable (β : NeuralRadix)

/-- Base powers are positive: `β^e > 0` for any exponent `e`. -/
lemma pos (e : ℤ) : 0 < neuralBpow β e := zpow_pos (NeuralRadix.pos β) e

/-- Base powers are nonnegative: `β^e ≥ 0` for any exponent `e`. -/
lemma nonneg (e : ℤ) : 0 ≤ neuralBpow β e := le_of_lt (pos β e)

/-- Base powers are never zero. -/
lemma ne_zero (e : ℤ) : neuralBpow β e ≠ 0 := ne_of_gt (pos β e)

/-- Exponent addition law: `β^(e1+e2) = β^e1 * β^e2`. -/
lemma add_exp (e1 e2 : ℤ) : neuralBpow β (e1 + e2) = neuralBpow β e1 * neuralBpow β e2 := by
  simp [neuralBpow, zpow_add₀ (NeuralRadix.ne_zero β)]

/-- Negating the exponent inverts the base power: `β^(-e) = (β^e)⁻¹`. -/
lemma neg_exp (e : ℤ) : neuralBpow β (-e) = (neuralBpow β e)⁻¹ := by
  simp [neuralBpow, zpow_neg]

end neuralBpow

/--
Interpret a `NeuralFloat` as a real number: `m * β^e`.
-/
noncomputable def neuralToReal {β : NeuralRadix} (f : NeuralFloat β) : ℝ :=
  f.mantissa * neuralBpow β f.exponent

namespace neuralToReal

variable {β : NeuralRadix} (f : NeuralFloat β)

/-- `to_real = 0` iff the mantissa is `0` (since `β^e ≠ 0`). -/
@[simp] lemma zero_iff : neuralToReal f = 0 ↔ f.mantissa = 0 := by
  simp only [neuralToReal]
  -- We need to show: f.mantissa * neural_bpow β f.exponent = 0 ↔ f.mantissa = 0
  have bpow_pos : 0 < neuralBpow β f.exponent := neuralBpow.pos β f.exponent
  have bpow_ne_zero : neuralBpow β f.exponent ≠ 0 := ne_of_gt bpow_pos
  -- Use mul_eq_zero and the fact that bpow_ne_zero
  rw [mul_eq_zero]
  simp only [bpow_ne_zero, or_false]
  -- Now we just need to show the equivalence for the integer cast
  simp only [Int.cast_eq_zero]

end neuralToReal

/--
Magnitude (base-`β`) of a real number.

This matches the usual definition `mag(x) = ⌊log_β(|x|)⌋ + 1` for `x ≠ 0`, and `0` for `x = 0`.
It is the bridge between a real input `x` and the exponent-selection function `fexp`.
-/
noncomputable def neuralMagnitude (β : NeuralRadix) (x : ℝ) : ℤ :=
  if x = 0 then 0
  else
    let base_mag := ⌊Real.log (abs x) / Real.log β.toReal⌋ + 1
    base_mag

/--
Validity predicate for exponent-selection functions.

In Flocq, many results are stated for an abstract `fexp : ℤ → ℤ` satisfying `Valid_exp`.
We keep essentially the same interface (see `flocq_valid`) so theorems can be stated using direct
analogues of Flocq results, and we add a couple of convenience properties that are often needed for
bounding arguments (`bounded_growth`, `monotone`).
-/
class NeuralValidExp (fexp : ℤ → ℤ) : Prop where
  flocq_valid : ∀ k : ℤ,
    (fexp k < k → fexp (k + 1) ≤ k) ∧
    (k ≤ fexp k → fexp (fexp k + 1) ≤ fexp k ∧ ∀ l, l ≤ fexp k → fexp l = fexp k)
  bounded_growth : ∀ k : ℤ, |fexp (k + 1) - fexp k| ≤ 1
  monotone : ∀ k1 k2 : ℤ, k1 ≤ k2 → fexp k1 ≤ fexp k2

/--
Canonical exponent (`cexp` in Flocq terminology).

Given a nonzero `x`, we first compute its magnitude `mag(x)` and then apply `fexp` to pick the
exponent used for scaling/rounding.
-/
noncomputable def neuralCexp (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp] (x : ℝ) : ℤ :=
  fexp (neuralMagnitude β x)

/--
Scaled mantissa (`x * β^{-cexp(x)}`).

Intuitively: rescale `x` so that rounding “happens around exponent 0”, which is where `rnd` acts.
-/
noncomputable def neuralScaledMantissa (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp] (x :
  ℝ) : ℝ :=
  x * neuralBpow β (-neuralCexp β fexp x)

/--
Generic format predicate (Flocq-style).

This says: `x` is exactly representable in the format picked out by `β` and `fexp`.
One way to read it is: the scaled mantissa is an integer (so there is no rounding error).
-/
def neuralGenericFormat (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp] (x : ℝ) : Prop :=
  x =
    neuralToReal (β := β)
      { mantissa := ⌊neuralScaledMantissa β fexp x⌋
      , exponent := neuralCexp β fexp x
      , precision := NeuralPrecision.ieee_single
      , error_bound := 0
      , phase := TrainingPhase.forward
      , layer_id := 0
      , batch_id := 0
      }

/--
Unit in the last place (`ulp`) associated with `x`.

This is the scale of the “one ulp” step at the exponent selected by `cexp`. For round-to-nearest,
many standard bounds have the shape `|round(x) - x| ≤ ulp(x)/2`.

TorchLean adds a small, *optional* twist: during numerically sensitive phases (see
`TrainingPhase.requires_high_precision`) we treat the bound as if it were one extra bit tighter.
This is a modeling hook for mixed-precision heuristics; it is not a replacement for a concrete
bit-level IEEE-754 semantics.
-/
noncomputable def neuralUlp (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp] (x : ℝ) (phase :
  TrainingPhase := TrainingPhase.forward) : ℝ :=
  if x = 0 then neuralBpow β (fexp 0)
  else
    let base_ulp := neuralBpow β (neuralCexp β fexp x)
    if phase.requiresHighPrecision then base_ulp / 2
    else base_ulp

namespace neuralUlp

variable (β : NeuralRadix) (fexp : ℤ → ℤ) [NeuralValidExp fexp]

/--
`neural_ulp` is always nonnegative.

Informally: an ulp is a step size on a real grid, so it cannot be negative.
-/
lemma nonneg (x : ℝ) (phase : TrainingPhase) : 0 ≤ neuralUlp β fexp x phase := by
  simp [neuralUlp]
  split_ifs
  · exact neuralBpow.nonneg β (fexp 0)
  · exact div_nonneg (neuralBpow.nonneg β _) (by norm_num)
  · exact neuralBpow.nonneg β _

/--
`neural_ulp` is strictly positive away from zero.

Informally: if `x ≠ 0` then the exponent selection `cexp(x)` picks some power of `β`, and the ulp
at that exponent is `β^{cexp(x)}` (or half of it in high-precision phases), hence positive.
-/
lemma pos_of_ne_zero (x : ℝ) (hx : x ≠ 0) (phase : TrainingPhase) : 0 < neuralUlp β fexp x phase :=
  by
  simp [neuralUlp, hx]
  -- Since x ≠ 0, we're in the else branch
  split_ifs with h_phase
  · -- High precision phase: base_ulp / 2
    apply div_pos
    · exact neuralBpow.pos β (neuralCexp β fexp x)
    · norm_num
  · -- Normal precision phase: base_ulp
    exact neuralBpow.pos β (neuralCexp β fexp x)

/-- `neural_ulp` at zero does not depend on the training phase. -/
@[simp] lemma zero (phase : TrainingPhase) :
    neuralUlp β fexp (0 : ℝ) phase = neuralBpow β (fexp 0) := by
  simp [neuralUlp]

/--
When `x ≠ 0` and the phase does **not** request high precision, `neural_ulp` is just the base grid
step `β^{cexp(x)}`.
-/
lemma eq_base_of_ne_zero_of_not_high_precision (x : ℝ) (hx : x ≠ 0)
    (phase : TrainingPhase) (hphase : phase.requiresHighPrecision = false) :
    neuralUlp β fexp x phase = neuralBpow β (neuralCexp β fexp x) := by
  simp [neuralUlp, hx, hphase]

/--
When `x ≠ 0` and the phase requests high precision, we use the same exponent scale but treat the
ULP as “one extra bit tighter” by dividing by 2.
-/
lemma eq_base_div_two_of_ne_zero_of_high_precision (x : ℝ) (hx : x ≠ 0)
    (phase : TrainingPhase) (hphase : phase.requiresHighPrecision = true) :
    neuralUlp β fexp x phase = neuralBpow β (neuralCexp β fexp x) / 2 := by
  simp [neuralUlp, hx, hphase]

/--
Forward-mode ULP simplification for `x ≠ 0`.

This matches the common “one ulp at exponent `cexp(x)`” intuition used in numerical analysis and
in everyday PyTorch/IEEE-754 error reasoning.
-/
@[simp] lemma forward_of_ne_zero (x : ℝ) (hx : x ≠ 0) :
    neuralUlp β fexp x TrainingPhase.forward = neuralBpow β (neuralCexp β fexp x) := by
  simpa [TrainingPhase.requiresHighPrecision] using
    (eq_base_of_ne_zero_of_not_high_precision (β := β) (fexp := fexp) x hx TrainingPhase.forward
      rfl)

/-- Inference-phase ULP simplification for `x ≠ 0` (same scale as forward). -/
@[simp] lemma inference_of_ne_zero (x : ℝ) (hx : x ≠ 0) :
    neuralUlp β fexp x TrainingPhase.inference = neuralBpow β (neuralCexp β fexp x) := by
  simpa [TrainingPhase.requiresHighPrecision] using
    (eq_base_of_ne_zero_of_not_high_precision (β := β) (fexp := fexp) x hx TrainingPhase.inference
      rfl)

/-- Backward-phase ULP simplification for `x ≠ 0` (uses the “tighter by 2” convention). -/
@[simp] lemma backward_of_ne_zero (x : ℝ) (hx : x ≠ 0) :
    neuralUlp β fexp x TrainingPhase.backward =
      neuralBpow β (neuralCexp β fexp x) / 2 := by
  simpa [TrainingPhase.requiresHighPrecision] using
    (eq_base_div_two_of_ne_zero_of_high_precision (β := β) (fexp := fexp) x hx
      TrainingPhase.backward rfl)

/-- Update-phase ULP simplification for `x ≠ 0` (uses the “tighter by 2” convention). -/
@[simp] lemma update_of_ne_zero (x : ℝ) (hx : x ≠ 0) :
    neuralUlp β fexp x TrainingPhase.update =
      neuralBpow β (neuralCexp β fexp x) / 2 := by
  simpa [TrainingPhase.requiresHighPrecision] using
    (eq_base_div_two_of_ne_zero_of_high_precision (β := β) (fexp := fexp) x hx TrainingPhase.update
      rfl)

end neuralUlp

end TorchLean.Floats
