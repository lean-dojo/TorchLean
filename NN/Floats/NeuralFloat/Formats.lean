/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Core
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Flocq-style formats (FIX / FLX / FLT)

We are **not** defining the executable IEEE-754 layer here.

What we do here is the same separation used by Flocq:

- `Core.lean` gives us mantissa/exponent arithmetic on `ℝ` plus `bpow`, `mag`, and `cexp`;
- this file defines *format families* via exponent-selection functions `fexp : ℤ → ℤ`.

Those `fexp`s let us talk about “a fixed-point grid”, “an unbounded float grid”, or “a bounded,
IEEE-like float grid with gradual underflow” without committing to a concrete bit encoding.

In particular:

- `FIX_*` models a fixed exponent (useful for quantization / fixed-point reasoning),
- `FLX_*` models an unbounded exponent float (a convenient intermediate model),
- `FLT_*` is the IEEE-like finite model (bounded exponent, gradual underflow).

If you want NaN/Inf/signed-zero and an *executable* kernel, that is `NN/Floats/IEEEExec/`.
For TorchLean-specific “which precision do we use in each phase?” configuration helpers, see
`NN/Floats/NeuralFloat/Metadata.lean`.

References:

- Flocq project: https://flocq.gitlabpages.inria.fr/flocq/
- S. Boldo, G. Melquiond, “Flocq: a unified Coq library for proving floating-point algorithms
  correct”
  (ARITH 2011), DOI: 10.1109/ARITH.2011.40
- IEEE Standard for Floating-Point Arithmetic (IEEE 754-2019)
-/

@[expose] public section


namespace TorchLean.Floats

variable {β : NeuralRadix}

/--
`FIX_exp emin` is the simplest exponent-selection function: it always returns the same exponent.

This is the Flocq “FIX” family. It is useful when you want to reason about values living on a
single, fixed grid `β^emin * ℤ` (think: fixed-point arithmetic / quantization).
-/
def FIXExp (emin : ℤ) : ℤ → ℤ := fun _ => emin

/--
`FIX_exp` satisfies the standard Flocq-style `Valid_exp` axioms (here: `NeuralValidExp`).

Even though the proof is trivial, having the instance is what lets later theorems reuse the same
generic lemmas for FIX/FLX/FLT.
-/
instance fixValidExp (emin : ℤ) : NeuralValidExp (FIXExp emin) where
  flocq_valid := by
    intro k
    constructor
    · intro h; simp [FIXExp] at h ⊢; exact Int.le_of_lt h
    · intro _; simp [FIXExp]
  bounded_growth := by simp [FIXExp]
  monotone := by simp [FIXExp]

/--
`FIX_format emin x` says “`x` is exactly representable on the fixed grid”.

This is phrased via an existential `NeuralFloat β` so that it composes smoothly with the rest of
the rounding model (`neural_to_real`, ULP bounds, etc.).
-/
def FIXFormat (emin : ℤ) (x : ℝ) : Prop :=
  ∃ f : NeuralFloat β, x = neuralToReal f ∧ f.exponent = emin

/--
`FLX_exp prec` is the unbounded-exponent family.

This is Flocq’s “FLX” family: it models a floating-point format with *no exponent bounds* but with
a mantissa precision parameter `prec`. It is a convenient intermediate model for proofs because it
removes underflow/overflow corner cases while still tracking mantissa rounding.
-/
def FLXExp (prec : ℤ) : ℤ → ℤ := fun e => e - prec

/--
`FLX_exp` satisfies `NeuralValidExp`.

The side-condition `0 < prec` matches the standard assumption that “precision is positive”.
-/
instance flxValidExp (prec : ℤ) (h : 0 < prec) : NeuralValidExp (FLXExp prec) where
  flocq_valid := by
    intro k
    constructor
    · intro H; simp [FLXExp] at H ⊢; linarith
    · intro H; simp [FLXExp] at H ⊢
      constructor
      · linarith
      · intros l hl; exfalso; linarith [h, H]
  bounded_growth := by simp [FLXExp]
  monotone := by
    intros k1 k2 hk
    simp [FLXExp]
    linarith

/--
Exact representability predicate for `FLX`.

Heuristically: there exists a mantissa/exponent pair with mantissa bounded by the precision, and
`x = m * β^e`.
-/
def FLXFormat (prec : ℤ) (x : ℝ) : Prop :=
  ∃ f : NeuralFloat β, x = neuralToReal f ∧ Int.natAbs f.mantissa < β.base ^ prec.natAbs

/--
`FLT_exp emin prec` is the bounded-exponent family with gradual underflow.

This is Flocq’s “FLT” family: it is the closest match to the *finite* fragment of IEEE-754
floating-point (no NaNs/Inf), where the exponent is bounded below by `emin` and the effective
precision is `prec`. Gradual underflow is captured by taking `max (e - prec) emin`.
-/
def FLTExp (emin prec : ℤ) : ℤ → ℤ := fun e => max (e - prec) emin

/--
`FLT_exp` satisfies `NeuralValidExp`.

This is where most format-bridge lemmas live when connecting proofs to float32-style bounds
(e.g. via `NN/Floats/FP32`).
-/
instance fltValidExp (emin prec : ℤ) (h : 0 < prec) : NeuralValidExp (FLTExp emin prec) where
  flocq_valid := by
    intro k
    constructor
    · intro hk
      have hk' : max (k - prec) emin < k := by simpa [FLTExp] using hk
      have hprec1 : (1 : ℤ) ≤ prec := by linarith [h]
      have hemin_le : emin ≤ k :=
        le_of_lt (lt_of_le_of_lt (le_max_right (k - prec) emin) hk')
      have hleft_le : k + 1 - prec ≤ k := by linarith [hprec1]
      simpa [FLTExp] using (max_le_iff).2 ⟨hleft_le, hemin_le⟩
    · intro hk
      have hk' : k ≤ max (k - prec) emin := by simpa [FLTExp] using hk
      have hk_cases : k ≤ k - prec ∨ k ≤ emin := (le_max_iff).1 hk'
      have hprec0 : 0 ≤ prec := le_of_lt h
      have hprec1 : (1 : ℤ) ≤ prec := by linarith [h]
      cases hk_cases with
      | inl hk_le =>
          exfalso
          have : k - prec < k := by linarith [h]
          exact (not_le_of_gt this) hk_le
      | inr hk_le_emin =>
          have hk_fexp : FLTExp emin prec k = emin := by
            apply max_eq_right
            have : k - prec ≤ emin - prec := sub_le_sub_right hk_le_emin prec
            exact this.trans (sub_le_self emin hprec0)
          constructor
          · have hleft : emin + 1 - prec ≤ emin := by linarith [hprec1]
            -- rewrite `fexp k` to `emin`, then unfold `FLT_exp` and use `max_le_iff`.
            simp [hk_fexp]
            dsimp [FLTExp]
            exact (max_le_iff).2 ⟨hleft, le_rfl⟩
          · intro l hl
            have hl' : l ≤ emin := by simpa [hk_fexp] using hl
            have hle : l - prec ≤ emin := (sub_le_self l hprec0).trans hl'
            -- Rewrite the RHS `fexp k` to `emin` and show `max (l - prec) emin = emin`.
            rw [hk_fexp]
            dsimp [FLTExp]
            exact max_eq_right hle
  bounded_growth := by
    intro k
    simp [FLTExp]
    -- |max (k + 1 - prec) emin - max (k - prec) emin| ≤ 1
    -- This follows from the properties of max function
    have h1 : max (k + 1 - prec) emin ≤ max (k - prec) emin + 1 := by
      simp [max_def]
      split_ifs with h2 h3
      · -- Both cases where the first argument is chosen
        linarith
      · -- Mixed cases
        linarith
      · -- Mixed cases
        linarith
      · -- Both cases where emin is chosen
        linarith
    have h2 : max (k - prec) emin ≤ max (k + 1 - prec) emin + 1 := by
      simp [max_def]
      split_ifs with h4 h5
      · -- Both cases where the first argument is chosen
        linarith
      · -- Mixed cases
        linarith
      · -- Mixed cases
        linarith
      · -- Both cases where emin is chosen
        linarith
    exact abs_sub_le_iff.mpr ⟨by linarith, by linarith⟩
  monotone := by
    intros k1 k2 hk
    -- We need to show: FLT_exp emin prec k1 ≤ FLT_exp emin prec k2
    -- That is: max (k1 - prec) emin ≤ max (k2 - prec) emin
    simp only [FLTExp]
    -- This follows from the fact that k1 ≤ k2 implies k1 - prec ≤ k2 - prec
    have h1 : k1 - prec ≤ k2 - prec := by linarith [hk]
    exact max_le_max h1 (le_refl emin)

/--
Exact representability predicate for `FLT`.

This version includes:

- a mantissa size bound (precision),
- and the lower exponent bound `emin ≤ exponent` (no values smaller than the min normal/subnormal
  scale, depending on the choice of `emin` and rounding).
-/
def FLTFormat (emin prec : ℤ) (x : ℝ) : Prop :=
  ∃ f : NeuralFloat β, x = neuralToReal f ∧
    Int.natAbs f.mantissa < β.base ^ prec.natAbs ∧ emin ≤ f.exponent

end TorchLean.Floats
