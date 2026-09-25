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

This module turns a certificate whose bounds have already been proved valid for the stated
functions into sign conditions on the certified region. These are sign facts, not a stability
theorem: `NeuralLyapunov` stores `V` and `V̇` as independent functions with no dynamics tying them
together, and `V > 0` on the whole region excludes a zero-valued equilibrium from that region.

Design:
- `LyapunovCert` packages bounds on a candidate Lyapunov function `V` and its derivative `V̇`
  over a boxed region.
- `NeuralLyapunov` is an abstract interface for `V` and `V̇` (typically defined from a network).
- `LyapunovCert.ValidFor` records the substantive enclosure theorem. A graph checker may prove it;
  an external producer cannot obtain it merely by writing numbers to JSON.

The interval bounds themselves are the fields of `LyapunovCert.ValidFor`. The results below
specialize to `ℝ` so that strict inequalities like `V_lo > 0 ⟹ V(x) > 0` follow by order
transitivity (`0 < V_lo` and `V_lo ≤ V(x)`).
-/

@[expose] public section


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
  exact lt_of_lt_of_le h_pos (hcert.valueBounds x hx).1

/-- For `ℝ`: `V̇` is negative when its certified upper bound is negative. -/
theorem vdot_negative (lyap : NeuralLyapunov ℝ n) (cert : LyapunovCert ℝ n)
    (hcert : cert.ValidFor lyap)
    (h_neg : cert.derivativeUpper < 0) (x : Tensor ℝ [n])
    (hx : Box.contains cert.region x) : lyap.orbitalDerivative x < 0 := by
  exact lt_of_le_of_lt (hcert.orbitalDerivativeBounds x hx).2 h_neg

/-- Strict certificate margins give `V > 0` and `V̇ < 0` at every point of the region. The two
functions are not linked to any dynamics here, so this is a sign condition, not stability. -/
theorem lyapunov_conditions (lyap : NeuralLyapunov ℝ n) (cert : LyapunovCert ℝ n)
    (hcert : cert.ValidFor lyap)
    (h_V_pos : cert.vLower > 0) (h_Vdot_neg : cert.derivativeUpper < 0) :
    (∀ x, Box.contains cert.region x → lyap.value x > 0) ∧
    (∀ x, Box.contains cert.region x → lyap.orbitalDerivative x < 0) :=
  ⟨v_positive lyap cert hcert h_V_pos, vdot_negative lyap cert hcert h_Vdot_neg⟩

end NN.MLTheory.CROWN.Lyapunov.Real
