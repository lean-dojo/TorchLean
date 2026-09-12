/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Format.Generic
public import NN.Floats.NeuralFloat.Format.Formats

/-!
# Exact Representations for Rounded Arithmetic

The error theorems for addition, multiplication, division, and square root need more than a bound:
they track the grid on which an intermediate value is exactly representable.  This file contains
the representation lemmas shared by those developments.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/--
Rounding a mantissa/exponent value can be represented using its original exponent.

If the canonical exponent is no larger than the stored exponent, the input is already generic and
rounding fixes it. Otherwise the rounded canonical mantissa is shifted by an integral radix power.
This is the Lean counterpart of Flocq's `round_repr_same_exp`.
-/
theorem neuralRound_toReal_exists_same_exponent (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    (f : NeuralFloat β) :
    ∃ m : ℤ,
      neuralRound (β := β) (fexp := fexp) rnd (neuralToReal f) =
        neuralToReal ({ mantissa := m, exponent := f.exponent } : NeuralFloat β) := by
  let x := neuralToReal f
  let e' := neuralCexp β fexp x
  by_cases he : e' ≤ f.exponent
  · refine ⟨f.mantissa, ?_⟩
    have hx : neuralGenericFormat β fexp x :=
      neural_generic_format_of_toReal_of_cexp_le f x rfl he
    simpa [x] using neural_round_preserves_generic rnd x hx
  · have he' : f.exponent < e' := lt_of_not_ge he
    obtain ⟨scale, hscale⟩ := neuralBpow_eq_natCast_of_nonneg β
      (e' - f.exponent) (sub_nonneg.mpr he'.le)
    refine ⟨rnd (neuralScaledMantissa β fexp x) * Int.ofNat scale, ?_⟩
    unfold neuralRound neuralToReal
    change
      (rnd (neuralScaledMantissa β fexp x) : ℝ) * neuralBpow β e' =
        ((rnd (neuralScaledMantissa β fexp x) * Int.ofNat scale : ℤ) : ℝ) *
          neuralBpow β f.exponent
    rw [show e' = (e' - f.exponent) + f.exponent by ring]
    rw [neuralBpow.add_exp, hscale]
    have hcast : (((Int.ofNat scale : ℤ) : ℝ)) = (scale : ℝ) := by norm_num
    rw [Int.cast_mul, hcast]
    ring

/-- The sum of two mantissa/exponent values has a representation at the smaller exponent. -/
theorem neuralToReal_add_exists_min_exponent (f g : NeuralFloat β) :
    ∃ h : NeuralFloat β,
      neuralToReal f + neuralToReal g = neuralToReal h ∧
      h.exponent = min f.exponent g.exponent := by
  rcases le_total f.exponent g.exponent with hfg | hgf
  · obtain ⟨scale, hscale⟩ := neuralBpow_eq_natCast_of_nonneg β
      (g.exponent - f.exponent) (sub_nonneg.mpr hfg)
    let h : NeuralFloat β :=
      { mantissa := f.mantissa + g.mantissa * Int.ofNat scale
        exponent := f.exponent }
    refine ⟨h, ?_, by simp [h, min_eq_left hfg]⟩
    unfold neuralToReal h
    rw [show g.exponent = (g.exponent - f.exponent) + f.exponent by ring]
    rw [neuralBpow.add_exp, hscale]
    have hcast : (((Int.ofNat scale : ℤ) : ℝ)) = (scale : ℝ) := by norm_num
    rw [Int.cast_add, Int.cast_mul, hcast]
    ring
  · obtain ⟨scale, hscale⟩ := neuralBpow_eq_natCast_of_nonneg β
      (f.exponent - g.exponent) (sub_nonneg.mpr hgf)
    let h : NeuralFloat β :=
      { mantissa := f.mantissa * Int.ofNat scale + g.mantissa
        exponent := g.exponent }
    refine ⟨h, ?_, by simp [h, min_eq_right hgf]⟩
    unfold neuralToReal h
    rw [show f.exponent = (f.exponent - g.exponent) + g.exponent by ring]
    rw [neuralBpow.add_exp, hscale]
    have hcast : (((Int.ofNat scale : ℤ) : ℝ)) = (scale : ℝ) := by norm_num
    rw [Int.cast_add, Int.cast_mul, hcast]
    ring

/--
An FLX sum is representable when it fits in `prec` radix digits relative to both operand
representations. This is the common-grid lemma used by division and square-root residual proofs.
-/
theorem neural_generic_format_FLX_add_of_repr_bounds (prec : ℤ) (hprec : 0 < prec)
    (f g : NeuralFloat β) (x y : ℝ)
    (hx : x = neuralToReal f) (hy : y = neuralToReal g)
    (hxf : abs (x + y) < neuralBpow β (prec + f.exponent))
    (hyg : abs (x + y) < neuralBpow β (prec + g.exponent)) :
    @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) (x + y) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  by_cases hzero : x + y = 0
  · rw [hzero]
    exact neural_generic_format_zero
  obtain ⟨h, hsum, hexp⟩ := neuralToReal_add_exists_min_exponent f g
  have hrepr : x + y = neuralToReal h := by simpa [hx, hy] using hsum
  apply neural_generic_format_of_toReal_of_cexp_le h (x + y) hrepr
  have hmagF : neuralMagnitude β (x + y) ≤ prec + f.exponent :=
    neuralMagnitude_le_of_abs_lt_bpow β (x + y) _ hzero hxf
  have hmagG : neuralMagnitude β (x + y) ≤ prec + g.exponent :=
    neuralMagnitude_le_of_abs_lt_bpow β (x + y) _ hzero hyg
  rw [hexp]
  simp only [neuralCexp, FLXExp]
  exact le_min (by linarith) (by linarith)

end TorchLean.Floats
