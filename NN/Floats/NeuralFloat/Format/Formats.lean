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

- `Core.lean` gives us mantissa/exponent arithmetic on $\mathbb{R}$ plus `bpow`, `mag`, and `cexp`;
- this file defines *format families* via exponent-selection functions `fexp : ℤ → ℤ`.

Those `fexp`s let us talk about “a fixed-point grid”, “an unbounded float grid”, or “a float grid
with a lower exponent bound and gradual underflow” without committing to a concrete bit encoding.

In particular:

- `FIX_*` models a fixed exponent (useful for quantization / fixed-point reasoning),
- `FLX_*` models an unbounded exponent float (a convenient intermediate model),
- `FLT_*` has a lower exponent bound and gradual underflow, but no upper bound or overflow.

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
A positive number of radix digits for FLX, FLT, and FTZ formats.

Raw exponent functions remain available for algebraic proofs, but callers accepting an integer
configuration should first use `NeuralFormatPrecision.ofInt?`.  This prevents a zero or negative
precision from being silently reinterpreted through an absolute-value conversion.
-/
structure NeuralFormatPrecision where
  /-- Number of radix digits retained by the format. -/
  digits : ℕ
  /-- A floating-point precision has at least one radix digit. -/
  digits_pos : 0 < digits
  deriving DecidableEq, Repr

namespace NeuralFormatPrecision

/-- Convert a proof-carrying format precision to the integer parameter used by exponent formulas. -/
def toInt (precision : NeuralFormatPrecision) : ℤ := precision.digits

/-- Check a natural-number precision at a configuration boundary. -/
def ofNat? (digits : ℕ) : Option NeuralFormatPrecision :=
  if h : 0 < digits then some ⟨digits, h⟩ else none

/-- Check an integer precision, rejecting zero and every negative value. -/
def ofInt? (digits : ℤ) : Option NeuralFormatPrecision :=
  if h : 0 < digits then
    some ⟨digits.toNat, by
      have hcast : (digits.toNat : ℤ) = digits := Int.toNat_of_nonneg h.le
      have : (0 : ℤ) < (digits.toNat : ℤ) := by simpa [hcast] using h
      exact_mod_cast this⟩
  else none

/-- A checked precision remains positive after conversion to the integer exponent parameter. -/
@[simp] theorem toInt_pos (precision : NeuralFormatPrecision) : 0 < precision.toInt := by
  change (0 : ℤ) < (precision.digits : ℤ)
  exact_mod_cast precision.digits_pos

/-- Natural precision validation fails exactly at zero. -/
@[simp] theorem ofNat?_eq_none_iff (digits : ℕ) :
    ofNat? digits = none ↔ digits = 0 := by
  simp [ofNat?]

/-- Integer precision validation fails exactly for nonpositive inputs. -/
@[simp] theorem ofInt?_eq_none_iff (digits : ℤ) :
    ofInt? digits = none ↔ digits ≤ 0 := by
  simp [ofInt?]

end NeuralFormatPrecision

/--
`FIX_exp emin` is the simplest exponent-selection function: it always returns the same exponent.

This is the Flocq “FIX” family. It is useful when you want to reason about values living on a
single, fixed grid $\beta^{\mathtt{emin}}\mathbb{Z}$ (think: fixed-point arithmetic or
quantization).
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

/-- A fixed-point grid has a monotone exponent function: it is constant, so monotone. -/
instance fixMonotoneExp (emin : ℤ) : NeuralMonotoneExp (FIXExp emin) where
  monotone := by simp [FIXExp]

/-- The ULP at zero for a fixed-point grid is its fixed grid step. -/
theorem neuralUlp_zero_FIX (emin : ℤ) :
    neuralUlp β (FIXExp emin) 0 = neuralBpow β emin := by
  rw [neuralUlp.zero]
  cases hopt : neuralNegligibleExp (FIXExp emin) with
  | none =>
      have hnone := (neuralNegligibleExp_eq_none_iff (FIXExp emin)).mp hopt
      exfalso
      apply hnone
      exact ⟨emin, by simp [IsNeuralNegligibleExp, FIXExp]⟩
  | some n => simp [FIXExp]

/--
`FIX_format emin x` says “`x` is exactly representable on the fixed grid”.

This is phrased via an existential `NeuralFloat β` so that it composes smoothly with the rest of
the rounding model (`neuralToReal`, ULP bounds, etc.).
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
abbrev flxValidExp (prec : ℤ) (h : 0 < prec) : NeuralValidExp (FLXExp prec) where
  flocq_valid := by
    intro k
    constructor
    · intro H; simp [FLXExp] at H ⊢; linarith
    · intro H; simp [FLXExp] at H ⊢
      constructor
      · linarith
      · intros l hl; exfalso; linarith [h, H]

/-- `FLXExp prec` satisfies the generic exponent axioms exactly when `prec` is positive. -/
theorem neuralValidExp_FLX_iff (prec : ℤ) : NeuralValidExp (FLXExp prec) ↔ 0 < prec := by
  constructor
  · intro hvalid
    by_contra hprec
    have hnonpos : prec ≤ 0 := le_of_not_gt hprec
    have hsecond := (hvalid.flocq_valid 0).2 (by simp [FLXExp]; linarith)
    have := hsecond.1
    simp [FLXExp] at this
    linarith
  · exact flxValidExp prec

namespace NeuralFormatPrecision

/-- The unbounded exponent selector associated with a checked precision. -/
def flxExp (precision : NeuralFormatPrecision) : ℤ → ℤ := FLXExp precision.toInt

/-- A checked precision automatically discharges the FLX exponent-validity obligation. -/
instance flxExpValid (precision : NeuralFormatPrecision) : NeuralValidExp precision.flxExp :=
  flxValidExp precision.toInt precision.toInt_pos

end NeuralFormatPrecision


/--
Exact representability predicate for `FLX`.

Heuristically, there exists a mantissa/exponent pair with mantissa bounded by the precision, and
$x=m\beta^e$.
-/
def FLXFormat (prec : ℤ) (x : ℝ) : Prop :=
  0 < prec ∧
    ∃ f : NeuralFloat β, x = neuralToReal f ∧ Int.natAbs f.mantissa < β.base ^ prec.toNat

/-- Nonpositive precision is rejected by the explicit FLX format predicate. -/
theorem not_FLXFormat_of_nonpos (prec : ℤ) (hprec : prec ≤ 0) (x : ℝ) :
    ¬FLXFormat (β := β) prec x := by
  simp [FLXFormat, not_lt_of_ge hprec]

/-- The unbounded FLX exponent function has no negligible exponent. -/
theorem neuralNegligibleExp_FLX (prec : ℤ) (hprec : 0 < prec) :
    neuralNegligibleExp (FLXExp prec) = none := by
  rw [neuralNegligibleExp_eq_none_iff]
  rintro ⟨n, hn⟩
  simp [IsNeuralNegligibleExp, FLXExp] at hn
  linarith

/-- Consequently, the generic ULP of zero is zero for FLX. -/
theorem neuralUlp_zero_FLX (prec : ℤ) (hprec : 0 < prec) :
    @neuralUlp β (FLXExp prec) (flxValidExp prec hprec) 0 = 0 := by
  simp [neuralUlp, neuralNegligibleExp_FLX prec hprec]

/--
`FLT_exp emin prec` is the lower-exponent-bounded family with gradual underflow.

This is Flocq’s “FLT” family. The exponent is bounded below by `emin`, but it has no upper bound, so
this rounded-real format models gradual underflow but not overflow, infinities, or NaNs. Gradual
underflow is captured by taking `max (e - prec) emin`.
-/
def FLTExp (emin prec : ℤ) : ℤ → ℤ := fun e => max (e - prec) emin

/--
`FLT_exp` satisfies `NeuralValidExp`.

This is where most format-bridge lemmas live when connecting proofs to float32-style bounds
(e.g. via `NN/Floats/FP32`).
-/
abbrev fltValidExp (emin prec : ℤ) (h : 0 < prec) : NeuralValidExp (FLTExp emin prec) where
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

/-- `FLTExp emin prec` satisfies the exponent axioms exactly for positive precision. -/
theorem neuralValidExp_FLT_iff (emin prec : ℤ) : NeuralValidExp (FLTExp emin prec) ↔ 0 < prec := by
  constructor
  · intro hvalid
    by_contra hprec
    have hnonpos : prec ≤ 0 := le_of_not_gt hprec
    have hk : emin ≤ FLTExp emin prec emin := by simp [FLTExp]
    have hnext := ((hvalid.flocq_valid emin).2 hk).1
    have hlower : FLTExp emin prec (FLTExp emin prec emin + 1) ≥
        FLTExp emin prec emin + 1 := by
      apply le_max_of_le_left
      linarith
    linarith
  · exact fltValidExp emin prec

namespace NeuralFormatPrecision

/-- The gradual-underflow exponent selector associated with a checked precision. -/
def fltExp (precision : NeuralFormatPrecision) (emin : ℤ) : ℤ → ℤ :=
  FLTExp emin precision.toInt

/-- A checked precision automatically discharges the gradual-underflow validity obligation. -/
instance fltExpValid (precision : NeuralFormatPrecision) (emin : ℤ) :
    NeuralValidExp (precision.fltExp emin) :=
  fltValidExp emin precision.toInt precision.toInt_pos

end NeuralFormatPrecision


/-- `FLT` exponents are monotone, since `max` is monotone in its first argument. -/
abbrev fltMonotoneExp (emin prec : ℤ) : NeuralMonotoneExp (FLTExp emin prec) where
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
- and the lower exponent bound $\mathtt{emin}\le\mathtt{exponent}$ (no values smaller than the
  minimum normal/subnormal
  scale, depending on the choice of `emin` and rounding).
-/
def FLTFormat (emin prec : ℤ) (x : ℝ) : Prop :=
  0 < prec ∧
    ∃ f : NeuralFloat β, x = neuralToReal f ∧
      Int.natAbs f.mantissa < β.base ^ prec.toNat ∧ emin ≤ f.exponent

/-- Nonpositive precision is rejected by the explicit FLT format predicate. -/
theorem not_FLTFormat_of_nonpos (emin prec : ℤ) (hprec : prec ≤ 0) (x : ℝ) :
    ¬FLTFormat (β := β) emin prec x := by
  simp [FLTFormat, not_lt_of_ge hprec]

/-- FLT has a negligible-exponent witness at `emin`. -/
theorem exists_neuralNegligibleExp_FLT (emin prec : ℤ) :
    ∃ n, IsNeuralNegligibleExp (FLTExp emin prec) n := by
  refine ⟨emin, ?_⟩
  exact le_max_right _ _

/-- The ULP at zero for FLT is the smallest grid step $\beta^{\mathtt{emin}}$. -/
theorem neuralUlp_zero_FLT (emin prec : ℤ) (hprec : 0 < prec) :
    @neuralUlp β (FLTExp emin prec) (fltValidExp emin prec hprec) 0 = neuralBpow β emin := by
  rw [neuralUlp.zero]
  cases hopt : neuralNegligibleExp (FLTExp emin prec) with
  | none =>
      have hnone := (neuralNegligibleExp_eq_none_iff (FLTExp emin prec)).mp hopt
      exact False.elim (hnone (exists_neuralNegligibleExp_FLT emin prec))
  | some n =>
      have hn := neuralNegligibleExp_spec hopt
      have hp : 0 ≤ prec := hprec.le
      have hnE : n ≤ emin := by
        rcases (le_max_iff.mp hn) with hbad | hgood
        · exfalso
          linarith
        · exact hgood
      have hselected : FLTExp emin prec n = emin := by
        apply max_eq_right
        exact (sub_le_self n hp).trans hnE
      simp [hselected]

end TorchLean.Floats
