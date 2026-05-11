/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Lyapunov.Oracle

/-!
# Lyapunov certificate checking via CROWN bounds

This module provides a small certificate-checking interface for Lyapunov stability arguments.

Design:
- `LyapunovCert` packages bounds on a candidate Lyapunov function `V` and its derivative `V̇`
  over a boxed region.
- `NeuralLyapunov` is an abstract interface for `V` and `V̇` (typically defined from a network).
- The trusted boundary for this Lyapunov workflow is the oracle axiom `crown_oracle`, quarantined in
  `NN.MLTheory.CROWN.Lyapunov.Oracle`. Downstream theorems in this file only depend on that axiom.

The bottom portion specializes to `ℝ` so that strict inequalities like `V_lo > 0 ⟹ V(x) > 0` can be
discharged by simple order transitivity (`0 < V_lo` and `V_lo ≤ V(x)`).
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Lyapunov

open _root_.Spec
open NN.MLTheory.CROWN

variable {α : Type} [Context α] {n : Nat}

/-- `V` is bounded below on the certified region. -/
theorem v_bounded_below (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (w : CrownOracleWitness lyap cert)
    (x : Tensor α (.dim n .scalar)) (hx : Box.contains cert.region x) :
    lyap.V x ≥ cert.V_lo :=
  (crown_oracle lyap cert w).1 x hx |>.1

/-- `V` is bounded above on the certified region. -/
theorem v_bounded_above (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (w : CrownOracleWitness lyap cert)
    (x : Tensor α (.dim n .scalar)) (hx : Box.contains cert.region x) :
    lyap.V x ≤ cert.V_hi :=
  (crown_oracle lyap cert w).1 x hx |>.2

/-- `V̇` is bounded below on the certified region. -/
theorem vdot_bounded_below (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (w : CrownOracleWitness lyap cert)
    (x : Tensor α (.dim n .scalar)) (hx : Box.contains cert.region x) :
    lyap.Vdot x ≥ cert.Vdot_lo :=
  (crown_oracle lyap cert w).2 x hx |>.1

/-- `V̇` is bounded above on the certified region. -/
theorem vdot_bounded_above (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (w : CrownOracleWitness lyap cert)
    (x : Tensor α (.dim n .scalar)) (hx : Box.contains cert.region x) :
    lyap.Vdot x ≤ cert.Vdot_hi :=
  (crown_oracle lyap cert w).2 x hx |>.2

theorem V_bounded_below (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (w : CrownOracleWitness lyap cert)
    (x : Tensor α (.dim n .scalar)) (hx : Box.contains cert.region x) :
    lyap.V x ≥ cert.V_lo :=
  v_bounded_below lyap cert w x hx

theorem V_bounded_above (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (w : CrownOracleWitness lyap cert)
    (x : Tensor α (.dim n .scalar)) (hx : Box.contains cert.region x) :
    lyap.V x ≤ cert.V_hi :=
  v_bounded_above lyap cert w x hx

theorem Vdot_bounded_below (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (w : CrownOracleWitness lyap cert)
    (x : Tensor α (.dim n .scalar)) (hx : Box.contains cert.region x) :
    lyap.Vdot x ≥ cert.Vdot_lo :=
  vdot_bounded_below lyap cert w x hx

theorem Vdot_bounded_above (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (w : CrownOracleWitness lyap cert)
    (x : Tensor α (.dim n .scalar)) (hx : Box.contains cert.region x) :
    lyap.Vdot x ≤ cert.Vdot_hi :=
  vdot_bounded_above lyap cert w x hx

/-- Quantitative bounds on `V` and `Vdot` over the certified region. -/
theorem quantitative_bounds (lyap : NeuralLyapunov α n) (cert : LyapunovCert α n)
    (w : CrownOracleWitness lyap cert) :
    (∀ x, Box.contains cert.region x → cert.V_lo ≤ lyap.V x ∧ lyap.V x ≤ cert.V_hi) ∧
    (∀ x, Box.contains cert.region x → cert.Vdot_lo ≤ lyap.Vdot x ∧ lyap.Vdot x ≤ cert.Vdot_hi) :=
  crown_oracle lyap cert w

end NN.MLTheory.CROWN.Lyapunov

/-!
# Specialization to ℝ

For proofs involving transitivity (V_lo > 0 implies V > 0), we specialize to ℝ
which has the full Preorder/LinearOrder structure.
-/

open Spec in
open NN.MLTheory.CROWN in

/-- Concrete real-valued certificate format for JSON/importer-facing workflows. -/
structure RealCert (n : Nat) where
  /-- Lower bound for the Lyapunov candidate `V`. -/
  V_lo : ℝ
  /-- Upper bound for the Lyapunov candidate `V`. -/
  V_hi : ℝ
  /-- Lower bound for the orbital derivative `Vdot`. -/
  Vdot_lo : ℝ
  /-- Upper bound for the orbital derivative `Vdot`. -/
  Vdot_hi : ℝ
  /-- Lower endpoint of the certified input region, componentwise. -/
  region_lo : Fin n → ℝ
  /-- Upper endpoint of the certified input region, componentwise. -/
  region_hi : Fin n → ℝ

open Spec in
open NN.MLTheory.CROWN in
open NN.MLTheory.CROWN.Lyapunov in

/-- Convert the importer-friendly `RealCert` record into the canonical `LyapunovCert`. -/
noncomputable def mkCert {n : Nat} (rc : RealCert n) : NN.MLTheory.CROWN.Lyapunov.LyapunovCert ℝ n
  := {
  region := {
    lo := Tensor.dim (fun i => Tensor.scalar (rc.region_lo i))
    hi := Tensor.dim (fun i => Tensor.scalar (rc.region_hi i))
  }
  V_lo := rc.V_lo
  V_hi := rc.V_hi
  Vdot_lo := rc.Vdot_lo
  Vdot_hi := rc.Vdot_hi
}

namespace NN.MLTheory.CROWN.Lyapunov.Real

open _root_.Spec
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Lyapunov

variable {n : Nat}

/-- For `ℝ`: `V` is positive when `V_lo > 0`. -/
theorem v_positive (lyap : NeuralLyapunov ℝ n) (cert : LyapunovCert ℝ n)
    (w : CrownOracleWitness lyap cert)
    (h_pos : cert.V_lo > 0) (x : Tensor ℝ (.dim n .scalar))
    (hx : Box.contains cert.region x) : lyap.V x > 0 := by
  have h : cert.V_lo ≤ lyap.V x := by
    simpa using (v_bounded_below lyap cert w x hx)
  exact lt_of_lt_of_le h_pos h

/-- For `ℝ`: `V̇` is negative when `Vdot_hi < 0`. -/
theorem vdot_negative (lyap : NeuralLyapunov ℝ n) (cert : LyapunovCert ℝ n)
    (w : CrownOracleWitness lyap cert)
    (h_neg : cert.Vdot_hi < 0) (x : Tensor ℝ (.dim n .scalar))
    (hx : Box.contains cert.region x) : lyap.Vdot x < 0 := by
  have h : lyap.Vdot x ≤ cert.Vdot_hi :=
    vdot_bounded_above lyap cert w x hx
  exact lt_of_le_of_lt h h_neg

theorem V_positive (lyap : NeuralLyapunov ℝ n) (cert : LyapunovCert ℝ n)
    (w : CrownOracleWitness lyap cert)
    (h_pos : cert.V_lo > 0) (x : Tensor ℝ (.dim n .scalar))
    (hx : Box.contains cert.region x) : lyap.V x > 0 :=
  v_positive lyap cert w h_pos x hx

theorem Vdot_negative (lyap : NeuralLyapunov ℝ n) (cert : LyapunovCert ℝ n)
    (w : CrownOracleWitness lyap cert)
    (h_neg : cert.Vdot_hi < 0) (x : Tensor ℝ (.dim n .scalar))
    (hx : Box.contains cert.region x) : lyap.Vdot x < 0 :=
  vdot_negative lyap cert w h_neg x hx

/-- For ℝ: Main Lyapunov conditions -/
theorem lyapunov_conditions (lyap : NeuralLyapunov ℝ n) (cert : LyapunovCert ℝ n)
    (w : CrownOracleWitness lyap cert)
    (h_V_pos : cert.V_lo > 0) (h_Vdot_neg : cert.Vdot_hi < 0) :
    (∀ x, Box.contains cert.region x → lyap.V x > 0) ∧
    (∀ x, Box.contains cert.region x → lyap.Vdot x < 0) :=
  ⟨v_positive lyap cert w h_V_pos, vdot_negative lyap cert w h_Vdot_neg⟩

end NN.MLTheory.CROWN.Lyapunov.Real
