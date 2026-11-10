/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.Lyapunov.Certificate
public import TorchLean.Tensor.Context.Real

/-!
# Consequences of valid Lyapunov bounds

A certificate whose bounds are proved valid for its functions gives sign conditions on the certified
region. A positive lower bound on `V` gives `V > 0` (`value_pos`), a negative upper bound on `V̇`
gives `V̇ < 0` (`orbitalDerivative_neg`), and `sign_conditions` states both. These sign facts stop
short of a stability theorem, because `Candidate` stores `V` and `V̇` as independent functions with
no dynamics tying them together. A positive `V` on the whole region also keeps any zero-valued
equilibrium out of that region.

Design:
- `Cert` packages bounds on a candidate Lyapunov function `V` and its derivative `V̇`
  over a boxed region.
- `Candidate` is an abstract interface for `V` and `V̇` (typically defined from a network).
- `Cert.ValidFor` is the enclosure statement itself. A graph checker may prove it, but an
  external producer cannot obtain it by writing numbers to JSON.
- `RealCert` is a flat real-valued record, and `RealCert.toCert` turns it into a `Cert ℝ n`.

The theorems specialize to `ℝ`, where a strict margin passes through a bound by transitivity:
`0 < V_lo` and `V_lo ≤ V(x)` give `0 < V(x)`.
-/

@[expose] public section


namespace TorchLean.Examples.Lyapunov

/-- Flat real-valued certificate: four bounds and the two corners of the region. -/
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

/-- Repackages `rc` as a `Cert ℝ n` whose box runs from `regionLower` to `regionUpper`. -/
noncomputable def RealCert.toCert {n : Nat} (rc : RealCert n) : Cert ℝ n := {
  region := {
    lo := Tensor.dim (fun i => Tensor.scalar (rc.regionLower i))
    hi := Tensor.dim (fun i => Tensor.scalar (rc.regionUpper i))
  }
  vLower := rc.vLower
  vUpper := rc.vUpper
  derivativeLower := rc.derivativeLower
  derivativeUpper := rc.derivativeUpper
}

end TorchLean.Examples.Lyapunov

namespace TorchLean.Examples.Lyapunov.Cert

open CROWN

variable {n : Nat}

/-- If `cert` is valid for `lyap` and `0 < cert.vLower`, then `lyap.value x > 0` at every point `x`
of the region. -/
theorem value_pos (lyap : Candidate ℝ n) (cert : Cert ℝ n)
    (hcert : cert.ValidFor lyap)
    (h_pos : cert.vLower > 0) (x : Tensor ℝ [n])
    (hx : Box.Contains cert.region x) : lyap.value x > 0 :=
  lt_of_lt_of_le h_pos (hcert.value_bounds x hx).1

/-- If `cert` is valid for `lyap` and `cert.derivativeUpper < 0`, then
`lyap.orbitalDerivative x < 0` at every point `x` of the region. -/
theorem orbitalDerivative_neg (lyap : Candidate ℝ n) (cert : Cert ℝ n)
    (hcert : cert.ValidFor lyap)
    (h_neg : cert.derivativeUpper < 0) (x : Tensor ℝ [n])
    (hx : Box.Contains cert.region x) : lyap.orbitalDerivative x < 0 :=
  lt_of_le_of_lt (hcert.orbitalDerivative_bounds x hx).2 h_neg

/-- A valid certificate with `0 < cert.vLower` and `cert.derivativeUpper < 0` gives `V > 0` and
`V̇ < 0` at every point of the region. These are the sign conditions of a Lyapunov argument, with
no dynamics linking `V̇` to `V`. -/
theorem sign_conditions (lyap : Candidate ℝ n) (cert : Cert ℝ n)
    (hcert : cert.ValidFor lyap)
    (h_V_pos : cert.vLower > 0) (h_Vdot_neg : cert.derivativeUpper < 0) :
    (∀ x, Box.Contains cert.region x → lyap.value x > 0) ∧
    (∀ x, Box.Contains cert.region x → lyap.orbitalDerivative x < 0) :=
  ⟨value_pos lyap cert hcert h_V_pos, orbitalDerivative_neg lyap cert hcert h_Vdot_neg⟩

end TorchLean.Examples.Lyapunov.Cert
