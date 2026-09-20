/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Lyapunov.Certificate
public import NN.Spec.Core.Context.Real

/-!
# Consequences of valid Lyapunov bounds

This module derives Lyapunov inequalities from a certificate whose bounds have already been proved
valid for the stated functions.

Design:
- `LyapunovCert` packages bounds on a candidate Lyapunov function `V` and its derivative `V̇`
  over a boxed region.
- `NeuralLyapunov` is an abstract interface for `V` and `V̇` (typically defined from a network).
- `LyapunovCert.ValidFor` records the substantive enclosure theorem. A graph checker may prove it;
  an external producer cannot obtain it merely by writing numbers to JSON.

The bottom portion specializes to `ℝ` so that strict inequalities like `V_lo > 0 ⟹ V(x) > 0` can be
discharged by simple order transitivity (`0 < V_lo` and `V_lo ≤ V(x)`).
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Lyapunov

open Spec TorchLean
open NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α] {n : Nat}

/-- `V` is bounded below on the certified region. -/
theorem v_bounded_below (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (hcert : cert.ValidFor lyap)
    (x : Tensor α [n]) (hx : Box.contains cert.region x) :
    lyap.value x ≥ cert.vLower :=
  hcert.valueBounds x hx |>.1

/-- `V` is bounded above on the certified region. -/
theorem v_bounded_above (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (hcert : cert.ValidFor lyap)
    (x : Tensor α [n]) (hx : Box.contains cert.region x) :
    lyap.value x ≤ cert.vUpper :=
  hcert.valueBounds x hx |>.2

/-- `V̇` is bounded below on the certified region. -/
theorem vdot_bounded_below (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (hcert : cert.ValidFor lyap)
    (x : Tensor α [n]) (hx : Box.contains cert.region x) :
    lyap.orbitalDerivative x ≥ cert.derivativeLower :=
  hcert.orbitalDerivativeBounds x hx |>.1

/-- `V̇` is bounded above on the certified region. -/
theorem vdot_bounded_above (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (hcert : cert.ValidFor lyap)
    (x : Tensor α [n]) (hx : Box.contains cert.region x) :
    lyap.orbitalDerivative x ≤ cert.derivativeUpper :=
  hcert.orbitalDerivativeBounds x hx |>.2

/-- Quantitative bounds on `V` and `Vdot` over the certified region. -/
theorem quantitative_bounds (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (hcert : cert.ValidFor lyap) :
    (∀ x, Box.contains cert.region x →
      cert.vLower ≤ lyap.value x ∧ lyap.value x ≤ cert.vUpper) ∧
    (∀ x, Box.contains cert.region x →
      cert.derivativeLower ≤ lyap.orbitalDerivative x ∧
        lyap.orbitalDerivative x ≤ cert.derivativeUpper) :=
  ⟨hcert.valueBounds, hcert.orbitalDerivativeBounds⟩

end NN.MLTheory.CROWN.Lyapunov

/-!
# Specialization to ℝ

For proofs involving strict positivity and negativity, we specialize to `ℝ`, whose linear order
supports the required transitivity arguments.
-/

namespace NN.MLTheory.CROWN.Lyapunov

open Spec TorchLean

/-- Concrete real-valued certificate format for JSON/importer-facing workflows. -/
structure RealCert (n : Nat) where
  /-- Lower bound for the Lyapunov candidate `V`. -/
  vLower : ℝ
  /-- Upper bound for the Lyapunov candidate `V`. -/
  vUpper : ℝ
  /-- Lower bound for the orbital derivative `Vdot`. -/
  derivativeLower : ℝ
  /-- Upper bound for the orbital derivative `Vdot`. -/
  derivativeUpper : ℝ
  /-- Lower endpoint of the certified input region, componentwise. -/
  regionLower : Fin n → ℝ
  /-- Upper endpoint of the certified input region, componentwise. -/
  regionUpper : Fin n → ℝ

/-- Convert the importer-friendly `RealCert` record into the canonical `LyapunovCert`. -/
noncomputable def RealCert.toCert {n : Nat} (rc : RealCert n) : LyapunovCert ℝ n := {
  region := {
    lo := Tensor.dim (fun i => Tensor.scalar (rc.regionLower i))
    hi := Tensor.dim (fun i => Tensor.scalar (rc.regionUpper i))
  }
  vLower := rc.vLower
  vUpper := rc.vUpper
  derivativeLower := rc.derivativeLower
  derivativeUpper := rc.derivativeUpper
}

end NN.MLTheory.CROWN.Lyapunov

namespace NN.MLTheory.CROWN.Lyapunov.Real

open Spec TorchLean
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Lyapunov

variable {n : Nat}

/-- For `ℝ`: `V` is positive when the certified lower bound is positive. -/
theorem v_positive (lyap : NeuralLyapunov ℝ n) (cert : LyapunovCert ℝ n)
    (hcert : cert.ValidFor lyap)
    (h_pos : cert.vLower > 0) (x : Tensor ℝ [n])
    (hx : Box.contains cert.region x) : lyap.value x > 0 := by
  have h : cert.vLower ≤ lyap.value x := by
    simpa using (v_bounded_below lyap cert hcert x hx)
  exact lt_of_lt_of_le h_pos h

/-- For `ℝ`: `V̇` is negative when its certified upper bound is negative. -/
theorem vdot_negative (lyap : NeuralLyapunov ℝ n) (cert : LyapunovCert ℝ n)
    (hcert : cert.ValidFor lyap)
    (h_neg : cert.derivativeUpper < 0) (x : Tensor ℝ [n])
    (hx : Box.contains cert.region x) : lyap.orbitalDerivative x < 0 := by
  have h : lyap.orbitalDerivative x ≤ cert.derivativeUpper :=
    vdot_bounded_above lyap cert hcert x hx
  exact lt_of_le_of_lt h h_neg

/-- Positivity and decay follow from valid strict certificate margins. -/
theorem lyapunov_conditions (lyap : NeuralLyapunov ℝ n) (cert : LyapunovCert ℝ n)
    (hcert : cert.ValidFor lyap)
    (h_V_pos : cert.vLower > 0) (h_Vdot_neg : cert.derivativeUpper < 0) :
    (∀ x, Box.contains cert.region x → lyap.value x > 0) ∧
    (∀ x, Box.contains cert.region x → lyap.orbitalDerivative x < 0) :=
  ⟨v_positive lyap cert hcert h_V_pos, vdot_negative lyap cert hcert h_Vdot_neg⟩

end NN.MLTheory.CROWN.Lyapunov.Real
